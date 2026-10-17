"""Local retrieval service for bm25 / e5 / qwen3_emb_4b, and its client.

One process per run owns the searcher (the Lucene JVM, or the FAISS index + the GPU
query encoder).  Requests are served ONE AT A TIME on the main thread: searches are
milliseconds next to LLM calls, pyjnius stays on one thread, and results cannot depend on
request interleaving.  Every hit carries the document text from corpus.jsonl, so all
retrievers return byte-identical text for the same doc id.

    POST /search   {"query": str, "top_k": int}
                -> {"hits": [{"id", "score", "contents"}], "elapsed_s": float}
    GET  /health   {"ok": true, "retriever", "corpus", "index_manifest_sha256", "docs"}
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from ..corpus import DocStore, corpus_dir
from ..utils import PipelineError, log, sha256_json
from .common import load_index_manifest


def make_searcher(base: dict, retriever: str, corpus: str, smoke: bool, device: str):
    kind = base["retrievers"][retriever]["kind"]
    if kind == "bm25":
        from .bm25 import BM25Searcher
        return BM25Searcher(base, corpus, smoke)
    if kind == "dense":
        from .dense import DenseSearcher
        return DenseSearcher(base, retriever, corpus, smoke, device)
    raise PipelineError(f"{retriever} ({kind}) is not served by this service")


def display_function(base: dict, corpus: str):
    """What an agent is SHOWN of a document.  With `display_truncation` (BCP) this is the
    official BrowseComp-Plus / AgentIR snippet rule (searcher/searchers/base.py): encode
    without special tokens, keep the first `max_tokens`, decode skipping special tokens."""
    trunc = base["corpora"][corpus]["display_truncation"]
    if trunc is None:
        return lambda text: text
    from transformers import AutoTokenizer

    from ..hub import snapshot_dir
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(base, trunc["tokenizer"]["model"],
                                                         trunc["tokenizer"]["revision"])))
    n = trunc["max_tokens"]

    def show(text: str) -> str:
        ids = tok.encode(text, add_special_tokens=False)
        return tok.decode(ids[:n], skip_special_tokens=True) if len(ids) > n else text
    return show


def serve(base: dict, retriever: str, corpus: str, smoke: bool, port: int, device: str) -> None:
    manifest = load_index_manifest(base, retriever, corpus, smoke)
    searcher = make_searcher(base, retriever, corpus, smoke, device)
    store = DocStore(corpus_dir(base, corpus, smoke))
    show = display_function(base, corpus)
    health = {"ok": True, "retriever": retriever, "corpus": corpus, "smoke": smoke,
              "docs": manifest["docs"], "index_manifest_sha256": sha256_json(manifest),
              "display_truncation": base["corpora"][corpus]["display_truncation"]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, health)

        def do_POST(self):
            if self.path != "/search":
                return self._send(404, {"error": "unknown path"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t0 = time.time()
                hits = searcher.search(req["query"], int(req["top_k"]))
                out = [{"id": d, "score": s, "contents": show(store.get(d)["contents"])} for d, s in hits]
                self._send(200, {"hits": out, "elapsed_s": time.time() - t0})
            except Exception as exc:  # reported to the caller as a TOOL error, never hidden
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    # Requests are served one at a time, but every client thread must be able to WAIT: with
    # the default listen backlog (5) a run's 64 concurrent searches overflowed it and got
    # "connection reset by peer".
    HTTPServer.request_queue_size = 1024
    server = HTTPServer(("127.0.0.1", port), Handler)
    log(f"retrieval service {retriever}/{corpus}{' (smoke)' if smoke else ''} on :{port}")
    server.serve_forever()


class RetrievalError(RuntimeError):
    pass


class RetrievalClient:
    def __init__(self, url: str, timeout_s: float):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def health(self) -> dict:
        return self.session.get(self.url + "/health", timeout=30).json()

    def search(self, query: str, top_k: int) -> list[dict]:
        try:
            r = self.session.post(self.url + "/search", json={"query": query, "top_k": top_k},
                                  timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise RetrievalError(f"retrieval service unreachable: {exc}") from exc
        if r.status_code != 200:
            raise RetrievalError(r.json().get("error", r.text[:300]))
        return r.json()["hits"]
