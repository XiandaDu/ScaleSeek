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
    index.faiss     — FAISS ANN index (default: HNSW, ~30–60 GB)
    doc_ids.txt     — one doc_id per line; positional index matches FAISS IDs
    doc_texts.jsonl — one {"text": "..."} per line (same positional order)

Index type notes:
    --index-type hnsw    (default) — good recall/speed trade-off; requires all
                           vectors in RAM during build (~120–170 GB for 21M docs)
    --index-type ivfflat — lower RAM during build (adds in batches); similar
                           recall to hnsw but slower search without GPU FAISS
    --index-type flat    — exact search; same RAM as hnsw but ~10x slower search

Memory: AgentIR-4B hidden size is 3072–4096. Float32 = 4 bytes/dim.
    21M × 4096 × 4B = 336 GB (flat/hnsw build peak)
    21M × 4096 × 2B = 168 GB (fp16 encoding, then upcasted for FAISS)
    Use --index-type ivfflat if the machine has less than 200 GB RAM.
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
    model = AutoModel.from_pretrained(
        _AGENTIR_MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    print("[AgentIR] model loaded.")
    return model, tokenizer


def encode_batch(texts: list[str], model, tokenizer, device: str) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = model(**enc, return_dict=True).last_hidden_state
        reps = hidden[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps.cpu().float().numpy()


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
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Passages per GPU batch (lower if OOM)")
    parser.add_argument("--index-type", default="hnsw",
                        choices=["flat", "hnsw", "ivfflat"],
                        help="FAISS index type")
    parser.add_argument("--hnsw-m", type=int, default=32,
                        help="HNSW M (graph connections per node)")
    parser.add_argument("--nlist", type=int, default=65536,
                        help="IVFFlat number of Voronoi cells")
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Encode only first N passages (for smoke tests)")
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

    # Count passages (respects --max-passages for testing)
    print("Counting passages ...")
    n_total = 0
    with open(corpus_path) as f:
        for _ in f:
            n_total += 1
            if args.max_passages and n_total >= args.max_passages:
                break
    print(f"  {n_total:,} passages to encode.")

    model, tokenizer = load_model(args.device)

    # Infer embedding dimension from a dummy call
    dummy = encode_batch(["hello"], model, tokenizer, args.device)
    dim = dummy.shape[1]
    print(f"  Embedding dim: {dim}")
    del dummy

    # Build index skeleton
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    else:  # ivfflat
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT)

    doc_ids_path = out_dir / "doc_ids.txt"
    texts_path = out_dir / "doc_texts.jsonl"

    ids_f = open(doc_ids_path, "w")
    texts_f = open(texts_path, "w")

    # For IVFFlat: collect a training sample before adding passages
    training_vecs: list[np.ndarray] = []
    sample_target = min(args.nlist * 40, 2_000_000) if args.index_type == "ivfflat" else 0
    ivf_trained = args.index_type != "ivfflat"

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    pending_vecs: list[np.ndarray] = []   # only used before IVFFlat is trained
    total_encoded = 0

    with open(corpus_path) as f:
        for line in f:
            if total_encoded >= n_total:
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

            vecs = encode_batch(batch_texts, model, tokenizer, args.device)

            if not ivf_trained:
                training_vecs.append(vecs)
                pending_vecs.append((vecs, batch_texts, batch_ids))
                n_collected = sum(v.shape[0] for v in training_vecs)
                if n_collected >= sample_target:
                    train_arr = np.vstack(training_vecs)
                    print(f"  Training IVFFlat on {len(train_arr):,} vectors ...")
                    index.train(train_arr)
                    ivf_trained = True
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

            total_encoded += len(batch_texts)
            batch_texts, batch_ids = [], []
            if total_encoded % 500_000 == 0:
                print(f"  [{total_encoded:,}/{n_total:,}] encoded", flush=True)

    # Flush final partial batch
    if batch_texts:
        vecs = encode_batch(batch_texts, model, tokenizer, args.device)
        if not ivf_trained:
            # Edge case: never hit sample_target; train on what we have
            all_vecs = np.vstack([v for v, _, _ in pending_vecs] + [vecs])
            print(f"  Training IVFFlat on {len(all_vecs):,} vectors (small corpus)...")
            index.train(all_vecs)
            ivf_trained = True
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
