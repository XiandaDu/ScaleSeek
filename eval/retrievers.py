"""Backend-neutral retriever protocol and frozen Phase-1 factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]: ...

    @property
    def num_docs(self) -> int: ...

    @property
    def metadata(self) -> dict: ...


def merge_ranked_hits(shards: list[list[dict]], top_k: int = 3) -> list[dict]:
    """Deterministic global top-k over shard results, deduplicated by doc_id."""
    best: dict[str, dict] = {}
    for hits in shards:
        for hit in hits:
            doc_id = str(hit["doc_id"])
            current = best.get(doc_id)
            if current is None or float(hit["score"]) > float(current["score"]):
                best[doc_id] = hit
    return sorted(best.values(), key=lambda hit: (-float(hit["score"]), str(hit["doc_id"])))[:top_k]


@dataclass
class FixedBM25Retriever:
    """Expose a backend-neutral API while freezing the comparison parameters."""

    inner: object
    k1: float = 1.2
    b: float = 0.75

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        return self.inner.retrieve(query, top_k=top_k, k1=self.k1, b=self.b)

    @property
    def num_docs(self) -> int:
        return self.inner.num_docs

    @property
    def metadata(self) -> dict:
        return self.inner.metadata


def build_retriever(name: str, *, index_dir: str | None = None,
                    device: str | None = None) -> Retriever:
    """Build one of the three comparable local retrievers."""
    if name == "bm25":
        from .bm25_retriever import BM25Retriever
        return FixedBM25Retriever(BM25Retriever(index_dir=index_dir))
    if name == "e5":
        from .e5_retriever import E5Retriever
        return E5Retriever(index_dir=index_dir, device=device)
    if name == "qwen3_emb_4b":
        from .qwen3_embedding_retriever import Qwen3EmbeddingRetriever
        return Qwen3EmbeddingRetriever(index_dir=index_dir, device=device)
    raise ValueError(f"Unknown retriever {name!r}; choose bm25, e5, or qwen3_emb_4b")
