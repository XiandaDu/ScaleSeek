"""Dense retrievers (e5, qwen3_emb_4b): encoder, sharded corpus encoding, FAISS HNSW.

Encoding follows each model card exactly as configured in base.yaml: prefixes, pooling
(masked mean for e5, last token for Qwen3-Embedding with left padding), L2 normalisation
and max lengths.  Shard i of N covers rows [N_docs*i/N, N_docs*(i+1)/N) of corpus.jsonl
and stores float16 vectors in corpus order, so FAISS row r IS corpus line r -- the doc id
alignment across retrievers is by construction and re-checked by `check_retrievers`.

The HNSW index is IndexHNSWFlat (inner product) with M / efConstruction / efSearch from
base.yaml, i.e. the GrepSeek paper's 32 / 128 / 128.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..corpus import DocStore, corpus_dir, load_corpus_manifest
from ..hub import local_snapshot
from ..utils import (PipelineError, git_state, log, now_iso, read_json, repo_root, sha256_file,
                     write_json)
from .common import index_dir


class Encoder:
    def __init__(self, base: dict, retriever: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.cfg = base["retrievers"][retriever]
        self.torch = torch
        self.device = device
        path = local_snapshot(base, self.cfg["model"], self.cfg["revision"])
        side = "left" if self.cfg["pooling"] == "last_token" else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side=side)
        self.model = AutoModel.from_pretrained(str(path), dtype=getattr(torch, self.cfg["encode_dtype"]),
                                               attn_implementation="sdpa").to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Untruncated token count of each document as encode() would tokenize it."""
        out = self.tokenizer([self.cfg["doc_prefix"] + t for t in texts], add_special_tokens=True,
                             truncation=False, return_length=True)
        return list(out["length"])

    def encode(self, texts: list[str], is_query: bool) -> tuple[np.ndarray, int]:
        """L2-normalised float32 vectors, and how many inputs hit the length cap."""
        torch = self.torch
        prefix = self.cfg["query_prefix"] if is_query else self.cfg["doc_prefix"]
        max_len = self.cfg["query_max_length"] if is_query else self.cfg["doc_max_length"]
        batch = self.tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"]
            if self.cfg["pooling"] == "mean":
                summed = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(1)
                reps = summed / mask.sum(1, keepdim=True).to(hidden.dtype)
            else:  # last_token, Qwen3-Embedding's official last_token_pool
                if bool((mask[:, -1].sum() == mask.shape[0]).item()):
                    reps = hidden[:, -1]
                else:
                    lengths = mask.sum(1) - 1
                    reps = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
            reps = reps.float()
            if self.cfg["normalize"]:
                reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        at_cap = int((mask.sum(1) >= max_len).sum().item())
        return reps.cpu().numpy(), at_cap


def release_gpu() -> None:
    """Drop models and return cached CUDA memory, so a later step in the same process (or
    a subprocess such as the official AgentIR encoder) gets the whole GPU."""
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ============================================================================= encoding
def shard_bounds(n_docs: int, shard: int, n_shards: int) -> tuple[int, int]:
    return n_docs * shard // n_shards, n_docs * (shard + 1) // n_shards


def _shard_path(out: Path, shard: int, n_shards: int) -> Path:
    return out / "shards" / f"emb_{shard:04d}_of_{n_shards:04d}.npy"


def encode_shard(base: dict, retriever: str, corpus: str, shard: int, n_shards: int,
                 smoke: bool, device: str) -> dict:
    cfg = base["retrievers"][retriever]
    cman = load_corpus_manifest(base, corpus, smoke)
    out = index_dir(base, retriever, corpus, smoke)
    path = _shard_path(out, shard, n_shards)
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta["corpus_sha256"] == cman["sha256"] and path.exists():
            log(f"shard {shard}/{n_shards} already encoded")
            return meta
    store = DocStore(corpus_dir(base, corpus, smoke))
    lo, hi = shard_bounds(store.n, shard, n_shards)
    texts = [row["contents"] for row in store.iter_rows(lo, hi)]
    enc = Encoder(base, retriever, device)
    # Exact token lengths (prefix + text, capped at doc_max_length), then longest first:
    # a batch holds as many documents as fit `encode_batch_tokens` PADDED tokens.  (A
    # chars/3 estimate under-counted number- and table-heavy passages and ran out of GPU
    # memory on 7 of 64 Wiki-18 shards.)
    lengths = np.array([min(cfg["doc_max_length"], n) for n in enc.token_lengths(texts)])
    order = np.argsort(-lengths, kind="stable")
    embs = np.zeros((len(texts), enc.dim), dtype=np.float16)
    at_cap, i, nb, t0 = 0, 0, 0, time.time()
    while i < len(order):
        idx = order[i:i + max(1, cfg["encode_batch_tokens"] // int(lengths[order[i]]))]
        vecs, cap = enc.encode([texts[j] for j in idx], is_query=False)
        embs[idx] = vecs.astype(np.float16)
        at_cap += cap
        i += len(idx)
        nb += 1
        if nb % 200 == 0:
            log(f"  shard {shard}: {i}/{len(texts)} ({i / max(time.time() - t0, 1e-6):.0f} docs/s)")
    dim = enc.dim
    del enc               # the caller must drop its reference for the memory to go
    release_gpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, embs)
    tmp.rename(path)
    meta = {"retriever": retriever, "corpus_sha256": cman["sha256"], "shard": shard,
            "n_shards": n_shards, "rows": [lo, hi], "dim": dim, "dtype": "float16",
            "docs_at_length_cap": at_cap, "seconds": round(time.time() - t0, 1),
            "sha256": sha256_file(path), "created": now_iso()}
    write_json(meta_path, meta)
    log(f"shard {shard}/{n_shards} done: {hi - lo} docs, {at_cap} at the length cap")
    return meta


def build_hnsw(base: dict, retriever: str, corpus: str, n_shards: int, smoke: bool,
               threads: int) -> dict:
    import faiss

    cfg = base["retrievers"][retriever]
    cman = load_corpus_manifest(base, corpus, smoke)
    out = index_dir(base, retriever, corpus, smoke)
    metas = []
    for s in range(n_shards):
        mp = _shard_path(out, s, n_shards).with_suffix(".json")
        if not mp.exists():
            raise PipelineError(f"shard {s}/{n_shards} missing ({mp})")
        meta = read_json(mp)
        if meta["corpus_sha256"] != cman["sha256"]:
            raise PipelineError(f"shard {s} was encoded from another corpus")
        metas.append(meta)
    if metas[0]["rows"][0] != 0 or metas[-1]["rows"][1] != cman["docs"] or any(
            a["rows"][1] != b["rows"][0] for a, b in zip(metas, metas[1:])):
        raise PipelineError("shards do not tile the corpus")
    dim = metas[0]["dim"]
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(dim, cfg["hnsw"]["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg["hnsw"]["ef_construction"]
    t0 = time.time()
    for s, meta in enumerate(metas):
        x = np.load(_shard_path(out, s, n_shards), mmap_mode="r")
        for a in range(0, x.shape[0], 500_000):
            index.add(np.ascontiguousarray(x[a:a + 500_000], dtype=np.float32))
        log(f"  hnsw: added shard {s + 1}/{n_shards}, ntotal={index.ntotal}, "
            f"{time.time() - t0:.0f}s")
    if index.ntotal != cman["docs"]:
        raise PipelineError(f"hnsw has {index.ntotal} vectors, corpus has {cman['docs']}")
    path = out / "hnsw.faiss"
    faiss.write_index(index, str(path))
    manifest = {"retriever": retriever, "corpus": cman["corpus"], "corpus_sha256": cman["sha256"],
                "harness_commit": git_state(repo_root())["commit"],
                "docs": int(index.ntotal), "dim": dim,
                "params": {k: cfg[k] for k in ("model", "revision", "query_prefix", "doc_prefix",
                                               "pooling", "normalize", "query_max_length",
                                               "doc_max_length", "encode_dtype")},
                "hnsw": dict(cfg["hnsw"]), "metric": "inner_product", "faiss": faiss.__version__,
                "docs_at_length_cap": sum(m["docs_at_length_cap"] for m in metas),
                "shards": [{k: m[k] for k in ("shard", "rows", "sha256")} for m in metas],
                "build_seconds": round(time.time() - t0, 1), "build_threads": threads,
                "index_sha256": sha256_file(path), "index_bytes": path.stat().st_size,
                "created": now_iso()}
    write_json(out / "manifest.json", manifest)
    log(f"hnsw ok: {retriever} {index.ntotal} x {dim}")
    return manifest


class DenseSearcher:
    def __init__(self, base: dict, retriever: str, corpus: str, smoke: bool, device: str):
        import faiss

        cfg = base["retrievers"][retriever]
        self.index = faiss.read_index(str(index_dir(base, retriever, corpus, smoke) / "hnsw.faiss"))
        self.index.hnsw.efSearch = cfg["hnsw"]["ef_search"]
        self.encoder = Encoder(base, retriever, device)
        self.store = DocStore(corpus_dir(base, corpus, smoke))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec, _ = self.encoder.encode([query], is_query=True)
        scores, rows = self.index.search(vec, k)
        return [(self.store.id_of(int(r)), float(s)) for s, r in zip(scores[0], rows[0]) if r >= 0]


def describe(base: dict, retriever: str) -> str:
    return json.dumps(base["retrievers"][retriever], sort_keys=True)
