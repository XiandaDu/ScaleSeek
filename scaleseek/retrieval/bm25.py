"""BM25 over Lucene (Pyserini), k1/b fixed in base.yaml and never swept (BASELINES.md §8)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time

from ..corpus import corpus_dir, load_corpus_manifest
from ..utils import PipelineError, git_state, log, now_iso, read_json, repo_root, write_json
from .pyserini_offline import install
from .common import hash_tree, index_dir


def build(base: dict, corpus: str, smoke: bool) -> dict:
    cfg = base["retrievers"]["bm25"]
    cman = load_corpus_manifest(base, corpus, smoke)
    src = corpus_dir(base, corpus, smoke) / "bm25_input"
    out = index_dir(base, "bm25", corpus, smoke)
    lucene = out / "lucene"
    if (out / "manifest.json").exists():
        old = read_json(out / "manifest.json")
        if old["corpus_sha256"] == cman["sha256"] and old["params"] == {"k1": cfg["k1"], "b": cfg["b"]}:
            log("bm25 index exists and matches the corpus")
            return old
    if lucene.exists():
        shutil.rmtree(lucene)            # never resume into a half-written Lucene index
    out.mkdir(parents=True, exist_ok=True)
    # Positions/docvectors/raw are not stored: hits are mapped back to corpus.jsonl by id,
    # so every retriever shows the agent the identical document text.
    cmd = [sys.executable, "-m", "scaleseek.retrieval.pyserini_offline",
           "pyserini.index.lucene", "--collection", "JsonCollection",
           "--input", str(src), "--index", str(lucene),
           "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(cfg["index_threads"])]
    log("bm25 index: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=repo_root())
    install()
    from pyserini.search.lucene import LuceneSearcher
    n = LuceneSearcher(str(lucene)).num_docs
    if n != cman["docs"]:
        raise PipelineError(f"bm25 index has {n} docs, corpus has {cman['docs']}")
    manifest = {"retriever": "bm25", "corpus": cman["corpus"], "corpus_sha256": cman["sha256"],
                "harness_commit": git_state(repo_root())["commit"],
                "docs": n, "params": {"k1": cfg["k1"], "b": cfg["b"]},
                "build_command": cmd, "build_seconds": round(time.time() - t0, 1),
                "created": now_iso(), "files": hash_tree(lucene)}
    write_json(out / "manifest.json", manifest)
    log(f"bm25 index ok: {n} docs")
    return manifest


class BM25Searcher:
    def __init__(self, base: dict, corpus: str, smoke: bool):
        install()
        from pyserini.search.lucene import LuceneSearcher

        cfg = base["retrievers"]["bm25"]
        self.searcher = LuceneSearcher(str(index_dir(base, "bm25", corpus, smoke) / "lucene"))
        self.searcher.set_bm25(k1=cfg["k1"], b=cfg["b"])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(h.docid, float(h.score)) for h in self.searcher.search(query, k)]
