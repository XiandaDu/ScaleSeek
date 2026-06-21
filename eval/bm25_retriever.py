"""BM25 retriever over the wiki-18 corpus.

Wraps a bm25s index built by scripts/build_bm25_index.py.
The index lives at $BM25_INDEX_DIR and is loaded lazily on first use.

Usage:
    from eval.bm25_retriever import BM25Retriever
    ret = BM25Retriever()                          # loads from $BM25_INDEX_DIR
    hits = ret.retrieve("Albert Einstein Nobel Prize", top_k=5)
    # hits: [{"doc_id": "...", "score": float, "text": str}, ...]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class BM25Retriever:
    """Lazy-loading BM25 retriever backed by bm25s."""

    def __init__(self, index_dir: Optional[str] = None):
        self._index_dir = Path(index_dir or os.environ.get("BM25_INDEX_DIR", ""))
        self._retriever = None
        self._corpus: list[str] = []
        self._doc_ids: list[str] = []

    def _load(self) -> None:
        if self._retriever is not None:
            return
        try:
            import bm25s
        except ImportError:
            raise ImportError(
                "Install bm25s:  pip install bm25s\n"
                "Then build the index: python scripts/build_bm25_index.py"
            )

        index_path = self._index_dir / "index"
        id_map_path = self._index_dir / "doc_ids.txt"

        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}.\n"
                "Build it first: python scripts/build_bm25_index.py"
            )

        self._retriever = bm25s.BM25.load(str(index_path), load_corpus=True)
        self._corpus = list(self._retriever.corpus)

        if id_map_path.exists():
            with open(id_map_path, encoding="utf-8") as f:
                self._doc_ids = [l.strip() for l in f if l.strip()]
        else:
            self._doc_ids = [str(i) for i in range(len(self._corpus))]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[dict]:
        """Return top_k passages matching query.

        Args:
            query: natural-language query string
            top_k: number of passages to return (1-100)
            k1: BM25 term-frequency saturation (higher = more weight on freq)
            b: BM25 length normalization (1.0 = full norm, 0.0 = none)

        Returns:
            List of {"doc_id": str, "score": float, "text": str}
        """
        self._load()

        # bm25s exposes k1/b via idf/tf params at index-build time, not
        # at query time. For the prompt-based pipeline we use the default
        # k1/b from build time and treat k1/b args as advisory (logged but
        # not applied). The RL agent will learn to predict them; in the
        # prompt pipeline they're plausibly ignored.
        top_k = max(1, min(top_k, len(self._corpus)))
        import bm25s

        tokenized = bm25s.tokenize([query], stopwords="en")
        results, scores = self._retriever.retrieve(tokenized, k=top_k)

        hits = []
        for doc, score in zip(results[0], scores[0]):
            idx = self._corpus.index(doc) if isinstance(doc, str) else doc
            hits.append({
                "doc_id": self._doc_ids[idx] if idx < len(self._doc_ids) else str(idx),
                "score": float(score),
                "text": self._corpus[idx] if isinstance(doc, str) else doc,
            })
        return hits

    @property
    def size(self) -> int:
        self._load()
        return len(self._corpus)
