"""BM25 retriever over the wiki-18 corpus via Pyserini/Lucene.

The index is built once by scripts/build_bm25_index.py and lives at
$BM25_INDEX_DIR/index. Load is lazy (on first retrieve() call).

Key advantage over bm25s: LuceneSearcher.set_bm25(k1, b) applies per-query,
so the agent can control k1/b dynamically without rebuilding the index.

Usage:
    from eval.bm25_retriever import BM25Retriever
    ret = BM25Retriever()
    hits = ret.retrieve("Albert Einstein Nobel Prize", top_k=3, k1=1.2, b=0.75)
    # -> [{"doc_id": "...", "score": float, "text": str}, ...]
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional


class BM25Retriever:
    """Lazy-loading BM25 retriever backed by Pyserini LuceneSearcher."""

    def __init__(self, index_dir: Optional[str] = None):
        env_dir = os.environ.get("BM25_INDEX_DIR", "")
        self._index_dir = Path(index_dir or env_dir)
        self._searcher = None
        # set_bm25() mutates searcher similarity state, so retrieve() must be
        # atomic when the eval runs examples concurrently.
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._searcher is not None:
            return
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError(
                "Install Pyserini:\n"
                "  conda install -c conda-forge openjdk=21\n"
                "  pip install pyserini\n"
                "Then build the index: python scripts/build_bm25_index.py"
            )
        lucene_index = self._index_dir / "index"
        if not lucene_index.exists():
            raise FileNotFoundError(
                f"Lucene index not found at {lucene_index}.\n"
                "Build it: python scripts/build_bm25_index.py"
            )
        self._searcher = LuceneSearcher(str(lucene_index))
        print(f"[BM25Retriever] loaded index: {lucene_index} ({self._searcher.num_docs:,} docs)")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[dict]:
        """Return top_k passages for query.

        k1 and b are applied per-call (Pyserini supports this natively).

        Returns:
            [{"doc_id": str, "score": float, "text": str}, ...]
        """
        self._load()
        top_k = max(1, min(top_k, 1000))
        with self._lock:
            self._searcher.set_bm25(k1, b)
            raw_hits = self._searcher.search(query, k=top_k)
            hits = []
            for h in raw_hits:
                try:
                    raw = self._searcher.doc(h.docid).raw()
                    contents = json.loads(raw).get("contents", "")
                except Exception:
                    contents = ""
                hits.append({
                    "doc_id": h.docid,
                    "score": float(h.score),
                    "text": contents,
                })
        return hits

    @property
    def num_docs(self) -> int:
        self._load()
        return self._searcher.num_docs

    @property
    def metadata(self) -> dict:
        path = self._index_dir / "index_manifest.json"
        if not path.exists():
            return {"backend": "bm25_lucene", "manifest_missing": True}
        return json.loads(path.read_text())
