"""E5 dense retriever over the wiki-18 corpus via FAISS.

Drop-in replacement for BM25Retriever (same .retrieve() contract) backed by
the sq8_flat index built by scripts/build_e5_index.py. This is the retriever
Search-R1 was trained with (intfloat/e5-base-v2) — used for the paper-faithful
search_r1 rerun, the search_o1 dense backend, and the RAG(E5) row.

Index dir layout (from build_e5_index.py):
    index.faiss      — IndexScalarQuantizer sq8, inner product, L2-normed vecs
    doc_ids.txt      — one doc_id per line, positional
    doc_texts.jsonl  — one {"text": ...} per line, positional (~7 GB — NOT
                       loaded to RAM; an offsets.npy sidecar is built on first
                       load and rows are seek-read on demand)

Usage:
    ret = E5Retriever()          # $E5_INDEX_DIR
    hits = ret.retrieve("Albert Einstein Nobel Prize", top_k=3)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

_E5_MODEL_ID = "intfloat/e5-base-v2"
_E5_MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
_QUERY_PREFIX = "query: "


def format_query(query: str) -> str:
    return _QUERY_PREFIX + query


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class E5Retriever:
    """Lazy-loading E5 dense retriever (FAISS sq8_flat + e5-base-v2 encoder)."""

    def __init__(self, index_dir: Optional[str] = None, device: Optional[str] = None):
        env_dir = os.environ.get("E5_INDEX_DIR", "")
        self._index_dir = Path(index_dir or env_dir)
        self._device = device or os.environ.get("E5_DEVICE", "cuda")
        self._index = None
        self._doc_ids: list[str] | None = None
        self._offsets = None          # np.uint64 file offsets into doc_texts.jsonl
        self._texts_path: Path | None = None
        self._model = None
        self._tokenizer = None
        # faiss search is read-thread-safe; the torch encode is serialized
        self._lock = threading.Lock()

    def _load(self) -> None:
        # Double-checked under the lock: with concurrent first calls (eval runs
        # 16+ workers) an unguarded lazy load races and N threads each
        # faiss.read_index the 16 GB file -> host OOM (observed 2026-07-13).
        if self._index is not None:
            return
        with self._lock:
            if self._index is not None:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        import faiss
        import numpy as np
        idx_path = self._index_dir / "index.faiss"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"E5 index not found at {idx_path}.\n"
                "Build it: python scripts/build_e5_index.py --corpus ... --out <dir>"
            )
        print(f"[E5Retriever] loading {idx_path} (~16 GB, one-time) ...", flush=True)
        self._index = faiss.read_index(str(idx_path))
        self._doc_ids = (self._index_dir / "doc_ids.txt").read_text().split()
        if len(self._doc_ids) != self._index.ntotal:
            raise RuntimeError(
                f"doc_ids ({len(self._doc_ids):,}) != index.ntotal "
                f"({self._index.ntotal:,}) — index dir inconsistent/incomplete")
        self._texts_path = self._index_dir / "doc_texts.jsonl"
        off_path = self._index_dir / "offsets.npy"
        if off_path.exists():
            self._offsets = np.load(off_path)
        else:
            print("[E5Retriever] building doc_texts offsets sidecar (one-time scan) ...",
                  flush=True)
            offs = np.empty(self._index.ntotal + 1, dtype=np.uint64)
            pos = 0
            with open(self._texts_path, "rb") as f:
                i = 0
                for line in f:
                    offs[i] = pos
                    pos += len(line)
                    i += 1
            if i != self._index.ntotal:
                raise RuntimeError(
                    f"doc_texts rows ({i:,}) != index.ntotal ({self._index.ntotal:,})")
            offs[i] = pos
            np.save(off_path, offs)
            self._offsets = offs
        print(f"[E5Retriever] ready: {self._index.ntotal:,} docs", flush=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        print(f"[E5Retriever] loading {_E5_MODEL_ID} on {self._device} ...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            _E5_MODEL_ID, revision=_E5_MODEL_REVISION)
        want = torch.float16 if self._device != "cpu" else torch.float32
        model = None
        for kw in ({"dtype": want}, {"torch_dtype": want}):
            try:
                model = AutoModel.from_pretrained(
                    _E5_MODEL_ID, revision=_E5_MODEL_REVISION,
                    attn_implementation="sdpa", **kw).to(self._device)
                break
            except TypeError:
                continue
        if model is None:
            model = AutoModel.from_pretrained(
                _E5_MODEL_ID, revision=_E5_MODEL_REVISION).to(self._device).to(want)
        model.eval()
        self._model = model

    def _encode_query(self, query: str):
        import torch
        self._load_model()
        enc = self._tokenizer([format_query(query)], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._model(**enc, return_dict=True).last_hidden_state
            reps = masked_mean_pool(hidden, enc["attention_mask"])
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps.cpu().float().numpy()

    def _read_text(self, row: int) -> str:
        start, end = int(self._offsets[row]), int(self._offsets[row + 1])
        with open(self._texts_path, "rb") as f:
            f.seek(start)
            line = f.read(end - start)
        try:
            return json.loads(line).get("text", "")
        except Exception:
            return ""

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        self._load()
        top_k = max(1, min(top_k, 1000))
        with self._lock:
            qvec = self._encode_query(query)
            scores, rows = self._index.search(qvec, top_k)
        hits = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            hits.append({
                "doc_id": self._doc_ids[row],
                "score": float(score),
                "text": self._read_text(int(row)),
            })
        return hits

    @property
    def num_docs(self) -> int:
        self._load()
        return self._index.ntotal

    @property
    def metadata(self) -> dict:
        path = self._index_dir / "index_manifest.json"
        if not path.exists():
            return {"backend": "e5", "manifest_missing": True}
        return json.loads(path.read_text())
