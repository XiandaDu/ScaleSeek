#!/usr/bin/env python3
"""ScaleSeek evaluation runner (prompt-based, temporary pipeline).

Usage:
    source setup_env.sh

    # ScaleSeek adaptive agent
    python -m eval.run_eval \\
        --dataset hotpotqa \\
        --agent scalaseek \\
        --n 500 \\
        --output results/hotpotqa_scalaseek.jsonl

    # BM25-RAG baseline
    python -m eval.run_eval --dataset nq --agent bm25_rag --n 1000 \\
        --output results/nq_bm25_rag.jsonl

    # Direct-answer baseline (no retrieval)
    python -m eval.run_eval --dataset triviaqa --agent direct \\
        --output results/triviaqa_direct.jsonl

Agents:
    scalaseek   — adaptive BM25 + workspace DCI (ScaleSeek prompt agent)
    bm25_rag    — single BM25 retrieve then answer
    direct      — LLM only, no retrieval

Datasets:
    nq  triviaqa  popqa  hotpotqa  2wikimultihopqa  musique  bamboogle
    bright  browsecomp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .datasets import load_dataset, ALL_DATASETS
from .metrics import score_example, aggregate


def _make_client(host: str, port: int):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Install openai:  pip install openai")
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_metrics(metrics: dict, n_answered: int, n_total: int) -> None:
    print(f"\n{'─'*40}")
    print(f"  n         : {n_total} (answered: {n_answered})")
    print(f"  EM        : {metrics['em']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"{'─'*40}\n")


def run_eval(args: argparse.Namespace) -> None:
    # --- Load dataset ---
    print(f"Loading {args.dataset} (split={args.split or 'default'}, "
          f"offset={args.offset}, limit={args.n or 'all'}) ...")
    examples = load_dataset(
        args.dataset,
        split=args.split,
        limit=args.n,
        offset=args.offset,
    )
    print(f"  {len(examples)} examples loaded.")

    if not examples:
        sys.exit("No examples found. Check --dataset and --split.")

    # --- LLM client ---
    host = args.host or os.environ.get("LLM_HOST", "127.0.0.1")
    port = int(args.port or os.environ.get("LLM_PORT", 8000))
    model = args.model or os.environ.get("LLM_MODEL", "agent")
    client = _make_client(host, port)
    print(f"LLM: {host}:{port}  model={model}")

    # --- BM25 retriever (lazy, only loaded if needed) ---
    retriever = None
    if args.agent in ("scalaseek", "bm25_rag"):
        from .bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        print(f"BM25 index: {retriever._index_dir}")

    # --- Run ---
    output_path = Path(args.output) if args.output else None
    results: list[dict] = []
    scores: list[dict] = []

    t0 = time.perf_counter()
    print(f"\nRunning agent={args.agent!r} on {len(examples)} examples ...\n")

    for i, ex in enumerate(examples):
        if args.agent == "scalaseek":
            from .agent import run_agent
            record = run_agent(
                ex,
                client=client,
                model=model,
                retriever=retriever,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.agent == "bm25_rag":
            from .agent import run_bm25_rag
            record = run_bm25_rag(
                ex,
                client=client,
                model=model,
                retriever=retriever,
                top_k=args.bm25_top_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.agent == "direct":
            from .agent import run_direct
            record = run_direct(
                ex,
                client=client,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            sys.exit(f"Unknown agent: {args.agent!r}")

        row = record.to_dict()
        sc = score_example(record.prediction, record.gold_answers)
        row["em"] = sc["em"]
        row["f1"] = sc["f1"]
        results.append(row)
        scores.append(sc)

        elapsed = time.perf_counter() - t0
        avg_s = elapsed / (i + 1)
        eta = avg_s * (len(examples) - i - 1)
        print(
            f"  [{i+1:4d}/{len(examples)}]  "
            f"em={sc['em']:.0f}  f1={sc['f1']:.2f}  "
            f"finish={record.finish_reason:<12s}  "
            f"ws={record.final_workspace_size:2d}  "
            f"ETA {eta/60:.1f}min",
            flush=True,
        )

        # Stream to disk so results aren't lost on crash
        if output_path and (i + 1) % 50 == 0:
            _save_jsonl(results, output_path)

    # --- Save & report ---
    if output_path:
        _save_jsonl(results, output_path)
        print(f"\nResults saved → {output_path}")

    n_answered = sum(1 for r in results if r.get("prediction") is not None)
    metrics = aggregate(scores)
    _print_metrics(metrics, n_answered, len(results))

    # per-finish_reason breakdown
    from collections import Counter
    reasons = Counter(r["finish_reason"] for r in results)
    print("Finish reasons:", dict(reasons))

    if args.agent in ("scalaseek", "bm25_rag"):
        avg_ws = sum(r["final_workspace_size"] for r in results) / len(results)
        avg_bm25 = sum(r.get("n_bm25_calls", 0) for r in results) / len(results)
        print(f"Avg workspace size : {avg_ws:.1f}")
        print(f"Avg BM25 calls     : {avg_bm25:.2f}")

    total_s = time.perf_counter() - t0
    print(f"Wall time          : {total_s/60:.1f} min ({total_s/len(results):.1f}s/example)")


def main():
    parser = argparse.ArgumentParser(
        description="ScaleSeek eval runner (prompt-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Dataset name")
    parser.add_argument("--split", default=None,
                        help="Dataset split (default depends on dataset)")
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="Max number of examples to evaluate")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N examples (for sharding)")

    # Agent
    parser.add_argument("--agent", default="scalaseek",
                        choices=["scalaseek", "bm25_rag", "direct"],
                        help="Agent type")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max agent turns (scalaseek only)")
    parser.add_argument("--bm25-top-k", type=int, default=5,
                        help="BM25 top-k for bm25_rag baseline")

    # LLM
    parser.add_argument("--host", default=None, help="vLLM host (default: $LLM_HOST)")
    parser.add_argument("--port", default=None, help="vLLM port (default: $LLM_PORT)")
    parser.add_argument("--model", default=None, help="Model name (default: $LLM_MODEL)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output
    parser.add_argument("--output", default=None,
                        help="JSONL output file (optional; printed to stdout if omitted)")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
