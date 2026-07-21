"""Qwen3-Embedding-4B retriever using the official instruction and last-token pool."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from .dense_index import DenseFaissIndex

MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


def last_token_pool(last_hidden_states, attention_mask):
    """Official Qwen3 embedding pooling, supporting left or right padding."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[range(batch_size), sequence_lengths]


class Qwen3EmbeddingRetriever:
    def __init__(self, index_dir: Optional[str] = None, device: Optional[str] = None):
        self._index_dir = Path(index_dir or os.environ.get("QWEN3_EMB_INDEX_DIR", ""))
        self._device = device or os.environ.get("QWEN3_EMB_DEVICE", "cuda")
        self._store = DenseFaissIndex(self._index_dir)
        self._model = self._tokenizer = None
        self._encode_lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                        padding_side="left")
        dtype = torch.bfloat16 if self._device != "cpu" else torch.float32
        self._model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype,
            attn_implementation="sdpa").to(self._device).eval()

    def _encode(self, query: str):
        import torch
        self._load_model()
        batch = self._tokenizer([format_query(query)], padding=True, truncation=True,
                                max_length=8192, return_tensors="pt")
        batch = {key: value.to(self._device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
            vector = last_token_pool(hidden, batch["attention_mask"])
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector.float().cpu().numpy()

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        # Model initialization and forward passes share mutable CUDA state.
        # Serialize them when the evaluation runner uses multiple workers.
        with self._encode_lock:
            vector = self._encode(query)
        return self._store.search(vector, top_k)

    @property
    def num_docs(self) -> int:
        return self._store.num_docs

    @property
    def metadata(self) -> dict:
        path = self._index_dir / "index_manifest.json"
        if not path.exists():
            return {"backend": "qwen3_emb_4b", "manifest_missing": True}
        return json.loads(path.read_text())
