"""Exact phrase search, the `find` tool of the new agent (not a baseline retriever).

The index is built from a case- and accent-folded copy of the corpus, with token positions,
no stemming and no stopword removal: a quoted phrase matches that token sequence (case,
accents and punctuation ignored) and nothing else.  Hits are ranked by BM25 over the
phrases plus the optional hint words, and passages whose TITLE contains a phrase come
first, so the article about an entity outranks passages that merely mention it.  Hits map
back to corpus.jsonl by id, so the agent sees the original text.

Query string: required phrases in double quotes, then free hint words, e.g.
    "teenage wolfpack" "directed by" film
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

from ..corpus import DocStore, corpus_dir, load_corpus_manifest
from ..utils import PipelineError, git_state, log, now_iso, repo_root, write_json
from .common import hash_tree, index_dir
from .pyserini_offline import install

# Letters NFKD does not decompose.
_LETTERS = str.maketrans({"ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d",
                          "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th", "ı": "i"})
N_PARTS = 32


def fold(text: str) -> str:
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text.translate(_LETTERS))
        text = "".join(c for c in text if not unicodedata.combining(c))
    return text.casefold()


def _fold_part(args: tuple) -> int:
    cdir, part, out = args
    store = DocStore(cdir)
    lo, hi = store.n * part // N_PARTS, store.n * (part + 1) // N_PARTS
    with open(out / f"part-{part:02d}.jsonl", "w") as f:
        for i, row in enumerate(store.iter_rows(lo, hi)):
            f.write(json.dumps({"id": store.id_of(lo + i), "contents": fold(row["contents"])}) + "\n")
    return hi - lo


def build(base: dict, corpus: str, smoke: bool) -> dict:
    cfg = base["retrievers"]["exact"]
    cman = load_corpus_manifest(base, corpus, smoke)
    cdir = corpus_dir(base, corpus, smoke)
    src, out = cdir / "exact_input", index_dir(base, "exact", corpus, smoke)
    lucene = out / "lucene"
    t0 = time.time()
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir()
    with Pool(cfg["index_threads"]) as pool:
        n = sum(pool.map(_fold_part, [(cdir, p, src) for p in range(N_PARTS)]))
    log(f"exact: folded {n} docs in {time.time() - t0:.0f} s")
    shutil.rmtree(lucene, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "scaleseek.retrieval.pyserini_offline",
           "pyserini.index.lucene", "--collection", "JsonCollection",
           "--input", str(src), "--index", str(lucene),
           "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(cfg["index_threads"]),
           "--storePositions", "--stemmer", "none", "--keepStopwords"]
    log("exact index: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root())
    shutil.rmtree(src)
    searcher = ExactSearcher(base, corpus, smoke)
    if searcher.reader.numDocs() != cman["docs"]:
        raise PipelineError(f"exact index has {searcher.reader.numDocs()} docs, corpus has {cman['docs']}")
    manifest = {"retriever": "exact", "corpus": cman["corpus"], "corpus_sha256": cman["sha256"],
                "harness_commit": git_state(repo_root())["commit"], "docs": cman["docs"],
                "params": {"k1": cfg["k1"], "b": cfg["b"]}, "build_command": cmd,
                "build_seconds": round(time.time() - t0, 1), "created": now_iso(), "files": hash_tree(lucene)}
    write_json(out / "manifest.json", manifest)
    log(f"exact index ok: {cman['docs']} docs")
    return manifest


class ExactSearcher:
    def __init__(self, base: dict, corpus: str, smoke: bool):
        install()
        from pyserini.analysis import Analyzer, get_lucene_analyzer   # sets the JVM classpath: before jnius
        from jnius import autoclass, cast

        cfg = base["retrievers"]["exact"]
        path = index_dir(base, "exact", corpus, smoke) / "lucene"
        self.analyzer = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
        self.Term = autoclass("org.apache.lucene.index.Term")
        self.TermQuery = autoclass("org.apache.lucene.search.TermQuery")
        self.PhraseBuilder = autoclass("org.apache.lucene.search.PhraseQuery$Builder")
        self.BoolBuilder = autoclass("org.apache.lucene.search.BooleanQuery$Builder")
        occur = autoclass("org.apache.lucene.search.BooleanClause$Occur")
        self.MUST, self.SHOULD = occur.MUST, occur.SHOULD
        directory = autoclass("org.apache.lucene.store.FSDirectory").open(autoclass("java.io.File")(str(path)).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(directory)
        self.searcher = autoclass("org.apache.lucene.search.IndexSearcher")(cast("org.apache.lucene.index.IndexReader", self.reader))
        self.searcher.setSimilarity(autoclass("org.apache.lucene.search.similarities.BM25Similarity")(cfg["k1"], cfg["b"]))
        self.stored = self.searcher.storedFields()
        self.pool = cfg["rerank_pool"]

    def _clause(self, tokens: list[str]):
        if len(tokens) == 1:
            return self.TermQuery(self.Term("contents", tokens[0]))
        b = self.PhraseBuilder()
        for t in tokens:
            b.add(self.Term("contents", t))
        return b.build()

    def _query(self, phrases: list[list[str]], hint: list[str]):
        b = self.BoolBuilder()
        for p in phrases:
            b.add(self._clause(p), self.MUST)
        for t in hint:
            b.add(self._clause([t]), self.SHOULD)
        return b.build()

    def find(self, query: str, k: int, store: DocStore, relax: bool = False) -> dict:
        """{"hits": [(id, score)], "count": passages containing every phrase,
            "phrase_counts": count of each phrase alone (in query order), only when
            several phrases never co-occur,
            "relaxed_count": with `relax` and no exact match, passages containing every
            word of the phrases in any order -- the hits are then those passages}"""
        phrases = [self.analyzer.analyze(fold(p)) for p in re.findall(r'"([^"]*)"', query)]
        hint = self.analyzer.analyze(fold(re.sub(r'"[^"]*"', " ", query)))
        if not phrases or not all(phrases):
            raise ValueError("every phrase needs at least one word or number")
        q = self._query(phrases, hint)
        count = self.searcher.count(q)
        out = {"count": count, "phrase_counts": None, "relaxed_count": None, "hits": []}
        if count == 0 and len(phrases) > 1:
            out["phrase_counts"] = [self.searcher.count(self._clause(p)) for p in phrases]
        if count == 0 and relax:
            q = self._query([[w] for w in dict.fromkeys(w for p in phrases for w in p)], hint)
            count = out["relaxed_count"] = self.searcher.count(q)
        if count == 0:
            return out
        top = self.searcher.search(q, max(k, self.pool)).scoreDocs
        hits = [(self.stored.document(sd.doc).get("id"), float(sd.score)) for sd in top]

        def in_title(doc_id: str) -> bool:
            # word tokens in Python: a Lucene analyze() per title costs ~70 ms through the JVM
            title = " " + " ".join(re.findall(r"\w+", fold(store.get(doc_id)["contents"].split("\n", 1)[0]))) + " "
            return any(" " + " ".join(p) + " " in title for p in phrases)
        # stable sort: title matches first, BM25 order within each group
        out["hits"] = sorted(hits, key=lambda h: not in_title(h[0]))[:k]
        return out
