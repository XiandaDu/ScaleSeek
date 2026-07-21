"""Shared lazy FAISS index reader used by E5 and Qwen3 embedding retrievers."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class DenseFaissIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index = None
        self.doc_ids: list[str] | None = None
        self.offsets = None
        self.texts_path: Path | None = None
        self.lock = threading.RLock()

    def load(self) -> None:
        if self.index is not None:
            return
        with self.lock:
            if self.index is not None:
                return
            import faiss
            import numpy as np
            idx_path = self.index_dir / "index.faiss"
            if not idx_path.exists():
                raise FileNotFoundError(f"Dense index not found at {idx_path}")
            self.index = faiss.read_index(str(idx_path))
            self.doc_ids = (self.index_dir / "doc_ids.txt").read_text().splitlines()
            if len(self.doc_ids) != self.index.ntotal:
                raise RuntimeError("doc_ids and FAISS index sizes differ")
            self.texts_path = self.index_dir / "doc_texts.jsonl"
            offsets_path = self.index_dir / "offsets.npy"
            if offsets_path.exists():
                self.offsets = np.load(offsets_path)
            else:
                offsets = [0]
                with self.texts_path.open("rb") as fh:
                    for line in fh:
                        offsets.append(offsets[-1] + len(line))
                if len(offsets) - 1 != self.index.ntotal:
                    raise RuntimeError("doc_texts and FAISS index sizes differ")
                self.offsets = np.asarray(offsets, dtype=np.uint64)
                np.save(offsets_path, self.offsets)

    def text(self, row: int) -> str:
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        with self.texts_path.open("rb") as fh:
            fh.seek(start)
            raw = fh.read(end - start)
        try:
            return json.loads(raw).get("text", "")
        except Exception:
            return ""

    def search(self, vector, top_k: int) -> list[dict]:
        self.load()
        top_k = max(1, min(int(top_k), 1000))
        with self.lock:
            scores, rows = self.index.search(vector, top_k)
        return [
            {"doc_id": self.doc_ids[int(row)], "score": float(score),
             "text": self.text(int(row))}
            for score, row in zip(scores[0], rows[0]) if row >= 0
        ]

    @property
    def num_docs(self) -> int:
        self.load()
        return int(self.index.ntotal)
