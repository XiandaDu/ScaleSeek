#!/usr/bin/env python3
"""Precompute AgentIR-4B reranked passages for a dataset, save to JSONL cache.

Usage (run WITHOUT vLLM active — needs all 4 GPUs):
    source setup_env.sh
    python scripts/precompute_agentir.py \
        --dataset popqa \
        --out results/popqa_agentir_cache.jsonl \
        --bm25-top-n 50 \
        --top-k 5

Then run eval using the cache:
    python -m eval.run_eval \
        --dataset popqa \
        --agent agentir_rag \
        --agentir-cache results/popqa_agentir_cache.jsonl \
        --output results/popqa_agentir_rag_4b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.bm25_retriever import BM25Retriever
from eval.datasets import load_dataset, ALL_DATASETS

_AGENTIR_MODEL_ID = "Tevatron/AgentIR-4B"
_QUERY_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating "
    "the user's reasoning\nQuery: "
)


def load_model(device):
    print(f"[AgentIR] loading {_AGENTIR_MODEL_ID} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(_AGENTIR_MODEL_ID)
    model = AutoModel.from_pretrained(
        _AGENTIR_MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("[AgentIR] loaded.")
    return model, tokenizer


def embed(texts: list[str], model, tokenizer, device: str,
          is_query: bool = False, batch_size: int = 64) -> torch.Tensor:
    all_reps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        if is_query:
            batch = [_QUERY_PREFIX + t for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc, return_dict=True).last_hidden_state
            reps = hidden[:, -1]
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        all_reps.append(reps.cpu())
    return torch.cat(all_reps, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--split", default=None)
    parser.add_argument("-n", "--n", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bm25-top-n", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-computed IDs
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} already done.")

    print(f"Loading {args.dataset} ...")
    examples = load_dataset(args.dataset, split=args.split, limit=args.n)
    examples = [ex for ex in examples if ex["id"] not in done_ids]
    print(f"  {len(examples)} examples to process.")

    retriever = BM25Retriever()
    model, tokenizer = load_model(args.device)

    with open(out_path, "a") as fout:
        for i, ex in enumerate(examples):
            # 1. BM25 candidates
            hits = retriever.retrieve(ex["question"], top_k=args.bm25_top_n)
            if not hits:
                row = {"id": ex["id"], "question": ex["question"],
                       "golden_answers": ex["golden_answers"], "passages": []}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # 2. AgentIR-4B rerank
            doc_texts = [h["text"][:1024] for h in hits]
            q_vec = embed([ex["question"]], model, tokenizer, args.device, is_query=True)[0]
            d_vecs = embed(doc_texts, model, tokenizer, args.device, is_query=False)
            scores = (d_vecs @ q_vec).tolist()
            ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
            top_hits = [h for _, h in ranked[:args.top_k]]

            row = {
                "id": ex["id"],
                "question": ex["question"],
                "golden_answers": ex["golden_answers"],
                "passages": [{"text": h["text"], "doc_id": h["doc_id"], "score": h["score"]}
                             for h in top_hits],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(examples)}] done", flush=True)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
