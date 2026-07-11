#!/usr/bin/env python3
"""Precompute AgentIR dense retrievals shard-by-shard under a small RAM budget.

The full 21M SQ8 index needs ~54 GB RAM — more than this node has. Instead we
load ONE shard index at a time (~9 GB for a 3.5M-passage shard), batch-search
ALL queries against it, release it, and move on; per-query results are merged
across shards by score at the end (embeddings are L2-normalized so inner-product
scores are directly comparable). Exact search, no approximation.

Usage (needs ONE free GPU briefly for query embedding; shards searched on CPU):
    python scripts/precompute_agentir_retrieval.py \
        --dataset popqa_full -n 1500 \
        --index-root /data/rech/mofengra/data/agentir_index_v2 \
        --top-k 5 --device cuda \
        --out results/popqa_full_agentir_retrieval.jsonl

Then run the reader with the cache:
    python -m eval.run_eval --dataset popqa_full --agent agentir_rag \
        --agentir-cache results/popqa_full_agentir_retrieval.jsonl \
        --output results/popqa_full_agentir.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.datasets import load_dataset  # noqa: E402

_AGENTIR_MODEL_ID = "Tevatron/AgentIR-4B"
_QUERY_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating "
    "the user's reasoning\nQuery: "
)


def load_model(device: str):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_AGENTIR_MODEL_ID)
    want = torch.float16 if device != "cpu" else torch.float32
    try:
        model = AutoModel.from_pretrained(_AGENTIR_MODEL_ID, dtype=want,
                                          attn_implementation="sdpa").to(device)
    except TypeError:
        model = AutoModel.from_pretrained(_AGENTIR_MODEL_ID, torch_dtype=want).to(device)
    model.eval()
    print(f"[precompute] model loaded, dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def embed_queries(questions: list[str], model, tokenizer, device: str,
                  batch: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(questions), batch):
        texts = [_QUERY_PREFIX + q for q in questions[i:i + batch]]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc, return_dict=True).last_hidden_state
            rep = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=-1)
        out.append(rep.cpu().float().numpy())
        if (i // batch) % 20 == 0:
            print(f"  embedded {i + len(texts)}/{len(questions)}", flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("-n", "--n", type=int, default=None)
    ap.add_argument("--index-root", required=True,
                    help="Dir containing shard*/index.faiss (+doc_ids.txt/doc_texts.jsonl)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import faiss

    examples = load_dataset(args.dataset, limit=args.n)
    print(f"[precompute] {len(examples)} queries from {args.dataset}")

    model, tokenizer = load_model(args.device)
    Q = embed_queries([e["question"] for e in examples], model, tokenizer, args.device)
    del model, tokenizer
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"[precompute] queries embedded: {Q.shape}; freeing GPU, searching shards on CPU")

    shard_dirs = sorted(d for d in Path(args.index_root).glob("shard*")
                        if (d / "index.faiss").exists())
    if not shard_dirs:
        sys.exit(f"no shard*/index.faiss under {args.index_root}")
    print(f"[precompute] {len(shard_dirs)} shards: {[d.name for d in shard_dirs]}")

    # per-query global top-k across shards: keep (score, docid, shard_idx, local_idx)
    merged: list[list] = [[] for _ in examples]
    needed_texts: dict[tuple, list[int]] = {}

    for si, sd in enumerate(shard_dirs):
        print(f"[precompute] shard {sd.name}: loading index ...", flush=True)
        index = faiss.read_index(str(sd / "index.faiss"))
        doc_ids = (sd / "doc_ids.txt").read_text().splitlines()
        scores, idxs = index.search(Q, args.top_k)
        for qi in range(len(examples)):
            for s, li in zip(scores[qi].tolist(), idxs[qi].tolist()):
                if li < 0:
                    continue
                merged[qi].append((s, doc_ids[li] if li < len(doc_ids) else str(li), si, li))
        del index
        gc.collect()
        print(f"[precompute] shard {sd.name}: searched ({index_ntotal(sd)} vecs)", flush=True)

    # trim to global top-k, then fetch texts shard-by-shard (line offsets)
    for qi in range(len(examples)):
        merged[qi].sort(key=lambda t: t[0], reverse=True)
        merged[qi] = merged[qi][: args.top_k]
        for (_, _, si, li) in merged[qi]:
            needed_texts.setdefault((si, li), [])

    for si, sd in enumerate(shard_dirs):
        wanted = sorted(li for (s, li) in needed_texts if s == si)
        if not wanted:
            continue
        texts = {}
        with open(sd / "doc_texts.jsonl", encoding="utf-8") as f:
            wi = 0
            for ln, line in enumerate(f):
                if wi >= len(wanted):
                    break
                if ln == wanted[wi]:
                    try:
                        texts[ln] = json.loads(line).get("text", "")
                    except Exception:
                        texts[ln] = ""
                    wi += 1
        for li, t in texts.items():
            needed_texts[(si, li)] = t
        print(f"[precompute] shard {sd.name}: fetched {len(texts)} texts", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for qi, ex in enumerate(examples):
            hits = [{"doc_id": d, "score": s,
                     "text": needed_texts.get((si, li), "") or ""}
                    for (s, d, si, li) in merged[qi]]
            f.write(json.dumps({"id": ex["id"], "question": ex["question"],
                                "golden_answers": ex["golden_answers"],
                                "hits": hits}, ensure_ascii=False) + "\n")
    print(f"[precompute] wrote {len(examples)} rows -> {out}")


def index_ntotal(sd: Path) -> str:
    try:
        return f"{sum(1 for _ in open(sd / 'doc_ids.txt')):,}"
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
