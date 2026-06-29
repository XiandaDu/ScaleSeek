#!/usr/bin/env python3
"""ScaleSeek evaluation runner.

Usage:
    source setup_env.sh

    # ScaleSeek adaptive agent (BM25 + workspace DCI)
    python -m eval.run_eval --dataset hotpotqa --agent scaleseek --n 500 \\
        --output results/hotpotqa_scaleseek.jsonl

    # BM25-RAG baseline
    python -m eval.run_eval --dataset nq --agent bm25_rag --n 1000 \\
        --output results/nq_bm25_rag.jsonl

    # Direct-answer baseline (no retrieval)
    python -m eval.run_eval --dataset triviaqa --agent direct \\
        --output results/triviaqa_direct.jsonl

    # DCI baseline — prompt-based grep on full corpus (no BM25 first stage)
    python -m eval.run_eval --dataset hotpotqa --agent dci \\
        --corpus-path /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \\
        --output results/hotpotqa_dci.jsonl

    # AgentIR dense retrieval (requires prebuilt FAISS index)
    python -m eval.run_eval --dataset nq --agent agentir_rag \\
        --agentir-index-dir /data/rech/mofengra/data/agentir_index \\
        --output results/nq_agentir.jsonl

    # GrepSeek trained model (separate vLLM on --grepseek-port)
    python -m eval.run_eval --dataset hotpotqa --agent grepseek \\
        --grepseek-port 8002 \\
        --corpus-path /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \\
        --output results/hotpotqa_grepseek.jsonl

    # Search-R1 (separate vLLM on --search-r1-port)
    python -m eval.run_eval --dataset hotpotqa --agent search_r1 \\
        --search-r1-port 8001 --output results/hotpotqa_search_r1.jsonl

Agents:
    scaleseek   — adaptive BM25 + workspace DCI (ScaleSeek prompt agent)
    bm25_rag    — single BM25 retrieve then answer
    direct      — LLM only, no retrieval
    dci         — prompt-based DCI: grep on full corpus (Beyond Semantic Similarity)
    agentir_rag — AgentIR-4B dense retrieval from full-corpus FAISS index
    search_r1   — Search-R1 (Qwen2.5-3B GRPO) on separate vLLM port
    grepseek    — GrepSeek trained model on separate vLLM port

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

_ALL_AGENTS = [
    "scaleseek", "bm25_rag", "direct",
    "dci", "agentir_rag", "search_r1", "grepseek",
]


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

    # --- LLM client (main vLLM, port 8000) ---
    host = args.host or os.environ.get("LLM_HOST", "127.0.0.1")
    port = int(args.port or os.environ.get("LLM_PORT", 8000))
    model = args.model or os.environ.get("LLM_MODEL", "agent")
    client = _make_client(host, port)
    print(f"LLM: {host}:{port}  model={model}")

    # --- Secondary clients ---
    sr1_client = None
    sr1_model = None
    if args.agent == "search_r1":
        sr1_host = args.search_r1_host or os.environ.get("SEARCH_R1_HOST", host)
        sr1_port = int(args.search_r1_port or os.environ.get("SEARCH_R1_PORT", 8001))
        sr1_model = args.search_r1_model or os.environ.get("SEARCH_R1_MODEL", "search_r1")
        sr1_client = _make_client(sr1_host, sr1_port)
        print(f"Search-R1: {sr1_host}:{sr1_port}  model={sr1_model}")

    gs_client = None
    gs_model = None
    if args.agent == "grepseek":
        gs_host = args.grepseek_host or os.environ.get("GREPSEEK_HOST", host)
        gs_port = int(args.grepseek_port or os.environ.get("GREPSEEK_PORT", 8002))
        gs_model = args.grepseek_model or os.environ.get("GREPSEEK_MODEL", "grepseek")
        gs_client = _make_client(gs_host, gs_port)
        print(f"GrepSeek: {gs_host}:{gs_port}  model={gs_model}")

    # --- Corpus path (for DCI and GrepSeek) ---
    corpus_path = args.corpus_path or os.environ.get(
        "CORPUS_PATH",
        "/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl",
    )
    if args.agent in ("dci", "grepseek") and not Path(corpus_path).exists():
        sys.exit(
            f"Corpus not found: {corpus_path}\n"
            "Set --corpus-path or CORPUS_PATH env var."
        )

    # --- AgentIR FAISS index ---
    agentir_index_dir: Optional[Path] = None
    if args.agent == "agentir_rag":
        raw = args.agentir_index_dir or os.environ.get("AGENTIR_INDEX_DIR", "")
        if not raw:
            sys.exit(
                "AgentIR requires a prebuilt FAISS index.\n"
                "Build it: python scripts/build_agentir_index.py --corpus ... --out <dir>\n"
                "Then pass --agentir-index-dir <dir> or set AGENTIR_INDEX_DIR."
            )
        agentir_index_dir = Path(raw)

    # --- BM25 retriever (lazy, only loaded when needed) ---
    retriever = None
    need_bm25 = args.agent in ("scaleseek", "bm25_rag", "search_r1")
    if need_bm25:
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
        if args.agent == "scaleseek":
            from .agent import run_agent
            record = run_agent(
                ex, client=client, model=model, retriever=retriever,
                max_turns=args.max_turns, max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.agent == "bm25_rag":
            from .agent import run_bm25_rag
            record = run_bm25_rag(
                ex, client=client, model=model, retriever=retriever,
                top_k=args.bm25_top_k, max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.agent == "direct":
            from .agent import run_direct
            record = run_direct(
                ex, client=client, model=model,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
        elif args.agent == "dci":
            from .dci_agent import run_dci
            record = run_dci(
                ex, client=client, model=model, corpus_path=corpus_path,
                max_turns=args.max_turns, max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.agent == "agentir_rag":
            from .agentir_retriever import run_agentir_rag
            record = run_agentir_rag(
                ex, client=client, model=model,
                index_dir=agentir_index_dir,
                top_k=args.bm25_top_k,
                agentir_device=args.agentir_device,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
        elif args.agent == "search_r1":
            from .search_r1_agent import run_search_r1
            record = run_search_r1(
                ex, client=sr1_client, model=sr1_model, retriever=retriever,
                max_turns=args.max_turns, max_tokens=args.max_tokens,
                bm25_top_k=args.bm25_top_k, temperature=args.temperature,
            )
        elif args.agent == "grepseek":
            from .grepseek_agent import run_grepseek
            record = run_grepseek(
                ex, client=gs_client, model=gs_model, corpus_path=corpus_path,
                max_turns=args.max_turns, max_tokens=args.max_tokens,
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
            f"ETA {eta/60:.1f}min",
            flush=True,
        )

        if output_path and (i + 1) % 50 == 0:
            _save_jsonl(results, output_path)

    # --- Save & report ---
    if output_path:
        _save_jsonl(results, output_path)
        print(f"\nResults saved → {output_path}")

    n_answered = sum(1 for r in results if r.get("prediction") is not None)
    metrics = aggregate(scores)
    _print_metrics(metrics, n_answered, len(results))

    from collections import Counter
    reasons = Counter(r["finish_reason"] for r in results)
    print("Finish reasons:", dict(reasons))

    total_s = time.perf_counter() - t0
    print(f"Wall time          : {total_s/60:.1f} min ({total_s/len(results):.1f}s/example)")


def main():
    parser = argparse.ArgumentParser(
        description="ScaleSeek eval runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--split", default=None)
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="Max examples to evaluate")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N examples")

    # Agent
    parser.add_argument("--agent", default="scaleseek", choices=_ALL_AGENTS)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=5,
                        help="Top-k for bm25_rag and agentir_rag")

    # Corpus (for dci and grepseek)
    parser.add_argument("--corpus-path", default=None,
                        help="Path to wiki_corpus.jsonl (default: $CORPUS_PATH)")

    # Search-R1
    parser.add_argument("--search-r1-host", default=None)
    parser.add_argument("--search-r1-port", default=None,
                        help="vLLM port for Search-R1 (default: $SEARCH_R1_PORT or 8001)")
    parser.add_argument("--search-r1-model", default=None)

    # GrepSeek
    parser.add_argument("--grepseek-host", default=None)
    parser.add_argument("--grepseek-port", default=None,
                        help="vLLM port for GrepSeek model (default: $GREPSEEK_PORT or 8002)")
    parser.add_argument("--grepseek-model", default=None)

    # AgentIR
    parser.add_argument("--agentir-index-dir", default=None,
                        help="Directory with prebuilt FAISS index (default: $AGENTIR_INDEX_DIR)")
    parser.add_argument("--agentir-device", default="cpu",
                        help="Device for AgentIR-4B embedding (cpu or cuda)")

    # Main LLM
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output
    parser.add_argument("--output", default=None,
                        help="JSONL output file")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
