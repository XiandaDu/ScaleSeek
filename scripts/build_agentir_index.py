#!/usr/bin/env python3
"""Build FAISS dense index over the full 21M-passage corpus using AgentIR-4B.

Run ONCE offline (≈12–24h on a single GPU). The resulting index is then read
by eval/agentir_retriever.py at eval time, with no BM25 involvement.

Usage:
    source setup_env.sh
    python scripts/build_agentir_index.py \\
        --corpus /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \\
        --out /data/rech/mofengra/data/agentir_index \\
        --device cuda \\
        --batch-size 512

Output files:
    index.faiss     — FAISS ANN index (default: hnsw_sq8, ~55–60 GB)
    doc_ids.txt     — one doc_id per line; positional index matches FAISS IDs
    doc_texts.jsonl — one {"text": "..."} per line (same positional order)

Index type notes:
    --index-type hnsw_sq8 (default) — SQ8-compressed HNSW: 1 byte/dim instead of
                           4, ~4x less RAM. Recommended for the full 21M corpus.
    --index-type hnsw    — uncompressed HNSW (fp32); best recall but ~4x the RAM.
    --index-type ivfflat — uncompressed IVF; adds in batches, slower CPU search.
    --index-type flat    — exact search; same RAM as hnsw but ~10x slower search.

Memory: AgentIR-4B embedding dim is 2560. Peak build RAM ≈ N × dim × bytes/dim:
    21M × 2560 × 4B ≈ 215 GB  (flat / hnsw / ivfflat — uncompressed fp32)
    21M × 2560 × 1B ≈  54 GB  (hnsw_sq8 — SQ8, default; + ~5 GB HNSW graph)
    Prefer hnsw_sq8 unless the machine has >250 GB RAM to spare.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_AGENTIR_MODEL_ID = "Tevatron/AgentIR-4B"


def load_model(device: str):
    print(f"[AgentIR] loading {_AGENTIR_MODEL_ID} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(_AGENTIR_MODEL_ID)
    want = torch.float16 if device != "cpu" else torch.float32
    # transformers v5 renamed torch_dtype -> dtype and IGNORES the old kwarg
    # (silently loading fp32 — 2-4x slower + 2x memory). Try new name first.
    model = None
    for kw in ({"dtype": want}, {"torch_dtype": want}):
        try:
            model = AutoModel.from_pretrained(
                _AGENTIR_MODEL_ID, attn_implementation="sdpa", **kw).to(device)
            break
        except TypeError:
            continue
    if model is None:
        model = AutoModel.from_pretrained(_AGENTIR_MODEL_ID).to(device).to(want)
    model.eval()
    p = next(model.parameters())
    print(f"[AgentIR] model loaded. dtype={p.dtype} (must be float16 on GPU!)")
    return model, tokenizer


def _encode_once(texts: list[str], model, tokenizer, device: str, max_length: int) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = model(**enc, return_dict=True).last_hidden_state
        reps = hidden[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    out = reps.cpu().float().numpy()
    del enc, hidden, reps
    return out


_WORKING_CHUNK: int | None = None  # adaptive sub-batch size learned from OOMs


def encode_batch(texts: list[str], model, tokenizer, device: str,
                 max_length: int = 512) -> np.ndarray:
    """Encode a batch OOM-safely.

    IMPORTANT: the retry happens OUTSIDE the except block (iterative, not
    recursive). Retrying inside `except` keeps the OOM traceback — and with it
    references to the half-built activation tensors — alive across the retry,
    so memory is never actually freed and the process death-spirals to OOM at
    batch size 1. We also remember the working chunk size so later batches
    start there instead of re-triggering OOMs.
    """
    import gc
    global _WORKING_CHUNK
    chunk = min(_WORKING_CHUNK or len(texts), len(texts))
    outs: list[np.ndarray] = []
    i = 0
    while i < len(texts):
        n = min(chunk, len(texts) - i)
        oom = False
        try:
            outs.append(_encode_once(texts[i:i + n], model, tokenizer, device, max_length))
        except torch.cuda.OutOfMemoryError:
            oom = True
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
        if oom:
            # traceback is dropped here (except block exited) -> tensors freeable
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if n == 1:
                raise RuntimeError("OOM even at batch size 1 after cache clear")
            chunk = max(1, n // 2)
            _WORKING_CHUNK = chunk
            print(f"  [oom] chunk -> {chunk}", flush=True)
            continue
        i += n
    return outs[0] if len(outs) == 1 else np.vstack(outs)


def main():
    parser = argparse.ArgumentParser(
        description="Build AgentIR-4B FAISS index over full corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", required=True,
                        help="Path to wiki_corpus.jsonl")
    parser.add_argument("--out", required=True,
                        help="Output directory for index files")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Passages per GPU batch (auto-halves on CUDA OOM)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max tokens per passage (lower to cut GPU memory)")
    parser.add_argument("--index-type", default="sq8_flat",
                        choices=["flat", "hnsw", "hnsw_sq8", "sq8_flat", "ivfflat", "ivfpq"],
                        help="FAISS index type. sq8_flat (default): SQ8-compressed flat "
                             "index — adds are ~free (quantize+append; HNSW insertion is "
                             "CPU-bound at ~100/s and dominates the build), search is "
                             "exact brute-force over SQ8 codes (fine at eval-scale query "
                             "volumes with OMP threads). hnsw_sq8: ANN, fast queries but "
                             "multi-hour graph construction. ivfpq: lowest RAM.")
    parser.add_argument("--hnsw-m", type=int, default=32,
                        help="HNSW M (graph connections per node)")
    parser.add_argument("--nlist", type=int, default=65536,
                        help="IVFFlat/IVFPQ number of Voronoi cells")
    parser.add_argument("--pq-m", type=int, default=64,
                        help="IVFPQ sub-quantizers (bytes/vector). dim must be divisible by it.")
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Encode only N passages (for smoke tests / sharding)")
    parser.add_argument("--skip-passages", type=int, default=0,
                        help="Skip the first N corpus lines (shard offset). Combine "
                             "with --max-passages to build one shard of a multi-GPU "
                             "sharded build; eval merges results across shard indexes.")
    args = parser.parse_args()

    try:
        import faiss
    except ImportError:
        sys.exit(
            "Install faiss:\n"
            "  conda install -c pytorch faiss-gpu   # GPU\n"
            "  conda install -c pytorch faiss-cpu   # CPU-only"
        )

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"Corpus not found: {corpus_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count passages in this shard's window (respects --skip/--max)
    print("Counting passages ...")
    n_total = 0
    with open(corpus_path) as f:
        for i, _ in enumerate(f):
            if i < args.skip_passages:
                continue
            n_total += 1
            if args.max_passages and n_total >= args.max_passages:
                break
    print(f"  {n_total:,} passages to encode "
          f"(skip={args.skip_passages:,}, max={args.max_passages or 'all'}).")

    model, tokenizer = load_model(args.device)

    # Infer embedding dimension from a dummy call
    dummy = encode_batch(["hello"], model, tokenizer, args.device, args.max_length)
    dim = dummy.shape[1]
    print(f"  Embedding dim: {dim}")
    del dummy

    # Build index skeleton
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    elif args.index_type == "hnsw_sq8":
        # SQ8-compressed HNSW: stores 1 byte/dim instead of 4 → ~4x less RAM.
        # The scalar quantizer must be trained on a sample before adding.
        index = faiss.IndexHNSWSQ(
            dim, faiss.ScalarQuantizer.QT_8bit, args.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = 200
    elif args.index_type == "sq8_flat":
        # Flat SQ8: 1 byte/dim storage, add == quantize+append (no graph build),
        # search == exact scan over SQ8 codes. Needs the same SQ training sample.
        index = faiss.IndexScalarQuantizer(
            dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT,
        )
    elif args.index_type == "ivfpq":
        # PQ-compressed IVF: ~pq_m bytes/vector (21M × 64B ≈ 1.3 GB) — the low-RAM
        # option for the full corpus. Trains both the coarse quantizer and the PQ.
        if dim % args.pq_m != 0:
            sys.exit(f"--pq-m ({args.pq_m}) must divide the embedding dim ({dim}).")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.pq_m, 8,
                                 faiss.METRIC_INNER_PRODUCT)
    else:  # ivfflat
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT)

    doc_ids_path = out_dir / "doc_ids.txt"
    texts_path = out_dir / "doc_texts.jsonl"

    ids_f = open(doc_ids_path, "w")
    texts_f = open(texts_path, "w")

    # IVF*/SQ8 types need a training sample collected before adding passages.
    needs_training = args.index_type in ("ivfflat", "ivfpq", "hnsw_sq8", "sq8_flat")
    training_vecs: list[np.ndarray] = []
    if args.index_type in ("ivfflat", "ivfpq"):
        sample_target = min(args.nlist * 40, 2_000_000)
    elif args.index_type in ("hnsw_sq8", "sq8_flat"):
        sample_target = 262_144  # ample for SQ8 per-dimension min/max calibration
    else:
        sample_target = 0
    trained = not needs_training

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    pending_vecs: list[np.ndarray] = []   # only used before IVFFlat is trained
    total_encoded = 0

    with open(corpus_path) as f:
        for line_no, line in enumerate(f):
            if line_no < args.skip_passages:
                continue
            # stop queueing once done+queued reaches the shard target — the
            # post-loop flush then encodes exactly n_total (no shard overlap)
            if total_encoded + len(batch_texts) >= n_total:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc_id = str(obj.get("id", total_encoded))
            text = obj.get("contents", "")
            batch_texts.append(text[:512])
            batch_ids.append(doc_id)

            if len(batch_texts) < args.batch_size:
                continue

            vecs = encode_batch(batch_texts, model, tokenizer, args.device, args.max_length)

            if not trained:
                training_vecs.append(vecs)
                pending_vecs.append((vecs, batch_texts, batch_ids))
                n_collected = sum(v.shape[0] for v in training_vecs)
                if n_collected >= sample_target:
                    train_arr = np.vstack(training_vecs)
                    print(f"  Training {args.index_type} on {len(train_arr):,} vectors ...")
                    index.train(train_arr)
                    trained = True
                    del training_vecs, train_arr
                    for pv, pt, pi in pending_vecs:
                        index.add(pv)
                        for bid in pi:
                            ids_f.write(bid + "\n")
                        for t in pt:
                            texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    del pending_vecs
                    pending_vecs = []
            else:
                index.add(vecs)
                for bid in batch_ids:
                    ids_f.write(bid + "\n")
                for t in batch_texts:
                    texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

            n_add = len(batch_texts)
            total_encoded += n_add
            batch_texts, batch_ids = [], []
            # print when a 500K boundary is CROSSED (a `% 500_000 == 0` check never
            # fires when the batch size doesn't divide 500000 exactly)
            if (total_encoded // 500_000) != ((total_encoded - n_add) // 500_000):
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  [{total_encoded:,}/{n_total:,}] encoded", flush=True)

    # Flush final partial batch
    if batch_texts:
        vecs = encode_batch(batch_texts, model, tokenizer, args.device)
        if not trained:
            # Edge case: never hit sample_target; train on what we have
            all_vecs = np.vstack([v for v, _, _ in pending_vecs] + [vecs])
            print(f"  Training {args.index_type} on {len(all_vecs):,} vectors (small corpus)...")
            index.train(all_vecs)
            trained = True
            for pv, pt, pi in pending_vecs:
                index.add(pv)
                for bid in pi:
                    ids_f.write(bid + "\n")
                for t in pt:
                    texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        index.add(vecs)
        for bid in batch_ids:
            ids_f.write(bid + "\n")
        for t in batch_texts:
            texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        total_encoded += len(batch_texts)

    ids_f.close()
    texts_f.close()

    idx_path = out_dir / "index.faiss"
    print(f"  Writing FAISS index to {idx_path} ...")
    faiss.write_index(index, str(idx_path))

    idx_gb = idx_path.stat().st_size / 1e9
    txt_gb = texts_path.stat().st_size / 1e9
    print(f"\nDone. {total_encoded:,} passages indexed.")
    print(f"  index.faiss    : {idx_gb:.1f} GB")
    print(f"  doc_texts.jsonl: {txt_gb:.1f} GB")
    print(f"  doc_ids.txt    : {doc_ids_path.stat().st_size / 1e6:.0f} MB")
    print(f"\nSet AGENTIR_INDEX_DIR={out_dir} and run eval with --agent agentir_rag")


if __name__ == "__main__":
    main()
