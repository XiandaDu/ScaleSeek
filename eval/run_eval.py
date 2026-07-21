#!/usr/bin/env python3
"""Evaluation runner for locally verified Direct/RAG/search agents.

DCI, DR-DCI, RISE and AgentIR must be launched with
``scripts/run_official_baseline.py``. Formal runs use ``--full-eval``; see
RUNBOOK.md for frozen models, revisions, retrievers and action budgets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .datasets import load_dataset, validate_complete_dataset, ALL_DATASETS
from .config import load_baselines, resolved_method
from .provenance import prompt_hash, generator_revision, harness_metadata
from .metrics import score_example, aggregate

_ALL_AGENTS = ["scaleseek", "rag", "direct", "search_r1", "search_o1"]


def validate_resume_row(row: dict, expected_provenance: dict,
                        dataset_manifest: dict | None = None) -> None:
    for key, expected in expected_provenance.items():
        if row.get(key) != expected:
            raise ValueError(f"resume row {row.get('id')} has mismatched {key}")
    if dataset_manifest is not None and row.get("dataset_manifest") != dataset_manifest:
        raise ValueError(f"resume row {row.get('id')} has a mismatched dataset manifest")


def validate_exact_id_set(result_ids: list[str], expected_ids: set[str]) -> None:
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != expected_ids:
        raise ValueError("output ID set does not exactly match the dataset manifest")


def validate_retriever_manifest(name: str, manifest: dict) -> None:
    """Reject an index that does not implement the frozen retrieval contract."""
    cfg = load_baselines()["global"]
    expected = {
        "bm25": {"backend": "bm25_lucene", **cfg["fixed_bm25"]},
        "e5": {
            "backend": "e5",
            "model_id": cfg["retrievers"]["e5"]["repo_id"],
            "model_revision": cfg["retrievers"]["e5"]["revision"],
            "pooling": "masked_mean",
            "normalize": True,
        },
        "qwen3_emb_4b": {
            "backend": "qwen3_emb_4b",
            "model_id": cfg["retrievers"]["qwen3_emb_4b"]["repo_id"],
            "model_revision": cfg["retrievers"]["qwen3_emb_4b"]["revision"],
            "pooling": "last_token",
            "normalize": True,
        },
    }[name]
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"{name} index manifest violates frozen settings: {mismatches}")


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
    if args.full_eval and (args.n is not None or args.offset != 0):
        sys.exit("--full-eval forbids --n and non-zero --offset")
    if args.full_eval and not args.output:
        sys.exit("--full-eval requires --output")
    if args.full_eval and args.agent == "scaleseek" and (
            args.retriever != "bm25" or args.max_tool_response_tokens != 2048):
        sys.exit("Formal ScaleSeek requires BM25 and a 2048-token tool-response budget")
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
    dataset_manifest = validate_complete_dataset(args.dataset, examples) if args.full_eval else {
        "dataset": args.dataset, "count": len(examples), "mode": "smoke_or_partial"}
    expected_full_ids = {str(ex["id"]) for ex in examples} if args.full_eval else None
    method_cfg = resolved_method(args.agent)
    try:
        model_revision = generator_revision(
            args.agent, search_r1_checkpoint=args.search_r1_tokenizer)
    except ValueError as exc:
        sys.exit(str(exc))
    method_prompt_hash = prompt_hash(args.agent)
    method_harness = harness_metadata(args.agent)
    if args.full_eval and args.generator_revision != model_revision:
        sys.exit("--full-eval requires --generator-revision equal to the frozen method revision: "
                 + model_revision)
    print(f"  dataset_manifest={dataset_manifest}")
    print(f"  method_config_id={method_cfg['method_config_id']}")

    # --- Resume: skip examples already completed in a prior partial output.
    # Rows with a transient failure (api_error/exception) are re-run, not kept.
    output_path = Path(args.output) if args.output else None
    if args.full_eval and output_path.exists() and not args.resume:
        sys.exit("Full-eval output already exists; pass --resume or choose a new path")
    prior_done: dict[str, dict] = {}
    if args.resume and output_path and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("id") is not None and \
                        r.get("finish_reason") not in ("api_error", "exception"):
                    expected_provenance = {
                        "method_config_id": method_cfg["method_config_id"],
                        "generator_revision": model_revision,
                        "prompt_sha256": method_prompt_hash,
                        "harness_source_sha256": method_harness["harness_source_sha256"],
                    }
                    try:
                        validate_resume_row(
                            r, expected_provenance,
                            dataset_manifest if args.full_eval else None)
                    except ValueError as exc:
                        sys.exit(str(exc))
                    prior_done[str(r["id"])] = r
        before = len(examples)
        examples = [ex for ex in examples if str(ex.get("id", "")) not in prior_done]
        print(f"  [resume] {len(prior_done)} already done; "
              f"{before - len(examples)} skipped, {len(examples)} left to run.")
        if not examples:
            if args.full_eval and set(prior_done) != expected_full_ids:
                sys.exit("Resume file does not exactly cover the full dataset manifest")
            print("  [resume] nothing left to run; rewriting merged output.")
            _save_jsonl(list(prior_done.values()), output_path)
            print(f"Results saved → {output_path}")
            return

    # --- LLM client (main vLLM, port 8000) ---
    host = args.host or os.environ.get("LLM_HOST", "127.0.0.1")
    port = int(args.port or os.environ.get("LLM_PORT", 8000))
    model = args.model or os.environ.get("LLM_MODEL", "agent")
    client = None
    if args.agent != "search_r1":
        client = _make_client(host, port)
        print(f"LLM: {host}:{port}  model={model}")

    # --- Secondary clients ---
    sr1_client = None
    sr1_model = None
    sr1_tokenizer = None
    if args.agent == "search_r1":
        sr1_host = args.search_r1_host or os.environ.get("SEARCH_R1_HOST", host)
        sr1_port = int(args.search_r1_port or os.environ.get("SEARCH_R1_PORT", 8001))
        sr1_model = args.search_r1_model or os.environ.get("SEARCH_R1_MODEL", "search_r1")
        sr1_client = _make_client(sr1_host, sr1_port)
        print(f"Search-R1: {sr1_host}:{sr1_port}  model={sr1_model}")
        tok_name = args.search_r1_tokenizer or args.search_r1_model
        if not tok_name:
            sys.exit("Search-R1 requires --search-r1-tokenizer (the exact 7B or 14B checkpoint)")
        try:
            from transformers import AutoTokenizer
            sr1_tokenizer = AutoTokenizer.from_pretrained(
                tok_name, revision=model_revision)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Cannot load Search-R1 checkpoint tokenizer {tok_name!r}: {exc}")

    # --- retriever (lazy, only loaded when needed) ---
    retriever = None
    retriever_manifest = None
    need_bm25 = args.agent in ("scaleseek", "rag", "search_r1", "search_o1")
    if need_bm25:
        if args.agent == "scaleseek":
            from .bm25_retriever import BM25Retriever
            retriever = BM25Retriever()
        else:
            from .retrievers import build_retriever
            retriever = build_retriever(args.retriever, device=args.retriever_device)
        print(f"Retriever: {args.retriever if args.agent != 'scaleseek' else 'bm25-adaptive'}")
        retriever_manifest = retriever.metadata
        if args.full_eval and (retriever_manifest.get("manifest_missing")
                               or retriever_manifest.get("partial_smoke")
                               or not retriever_manifest.get("corpus_manifest_id")):
            sys.exit("Formal retrieval requires a complete index_manifest.json tied to a corpus manifest")
        if args.full_eval:
            try:
                validate_retriever_manifest(
                    "bm25" if args.agent == "scaleseek" else args.retriever,
                    retriever_manifest)
            except ValueError as exc:
                sys.exit(str(exc))

    # --- Exact checkpoint tokenizer where the local harness needs it. ---
    tokenizer = None
    if args.agent in {"scaleseek", "search_o1"}:
        tok_name = args.tokenizer or os.environ.get("LLM_TOKENIZER")
        if args.agent == "search_o1" and not tok_name:
            sys.exit("Search-O1 requires --tokenizer Qwen/Qwen3.5-9B")
        if args.full_eval and tok_name != "Qwen/Qwen3.5-9B":
            sys.exit("Formal Qwen methods require --tokenizer Qwen/Qwen3.5-9B")
        if tok_name:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    tok_name, revision=model_revision)
                print(f"Tokenizer: {tok_name} (exact tool-response token budgeting)")
            except Exception as e:  # noqa: BLE001
                print(f"warning: could not load tokenizer {tok_name!r} ({e}); "
                      "falling back to ~4 char/token estimate")
        else:
            print("Tokenizer: none — using ~4 char/token estimate for the tool-response "
                  "budget (set --tokenizer or LLM_TOKENIZER for exact counts)")
    if args.full_eval and args.agent == "scaleseek" \
            and os.environ.get("SCALESEEK_PROMPT", "scaleseek_prompt") != "scaleseek_prompt":
        sys.exit("Formal ScaleSeek evaluation forbids prompt ablation overrides")

    # --- Run ---
    results: list[dict] = []
    scores: list[dict] = []

    t0 = time.perf_counter()

    def _dispatch(ex):
        if args.agent == "scaleseek":
            from .agent import run_agent
            return run_agent(
                ex, client=client, model=model, retriever=retriever,
                max_turns=8, max_tokens=8192,
                temperature=0.0, tokenizer=tokenizer,
                max_tool_response_tokens=args.max_tool_response_tokens)
        elif args.agent == "rag":
            from .agent import run_rag
            return run_rag(
                ex, client=client, model=model, retriever=retriever,
                top_k=args.retrieval_top_k,
                max_tokens=8192, temperature=0.0)
        elif args.agent == "direct":
            from .agent import run_direct
            return run_direct(
                ex, client=client, model=model,
                max_tokens=8192, temperature=0.0)
        elif args.agent == "search_r1":
            from .search_r1_agent import run_search_r1
            return run_search_r1(
                ex, client=sr1_client, model=sr1_model, retriever=retriever,
                tokenizer=sr1_tokenizer, action_budget=4, max_tokens=1024,
                retrieval_top_k=args.retrieval_top_k, temperature=0.7)
        elif args.agent == "search_o1":
            from .search_o1_agent import run_search_o1
            return run_search_o1(
                ex, client=client, model=model, retriever=retriever,
                dataset_name=args.dataset, tokenizer=tokenizer,
                max_turns=15, max_tokens=8192,
                retrieval_top_k=args.retrieval_top_k, temperature=0.7,
                top_p=0.8, sampling_top_k=20)
        else:
            sys.exit(f"Unknown agent: {args.agent!r}")

    def _finalize(record, ex):
        row = record.to_dict()
        row["dataset_manifest"] = dataset_manifest
        row["method_config_id"] = method_cfg["method_config_id"]
        row["generator_revision"] = model_revision
        row["prompt_sha256"] = method_prompt_hash
        row["retriever_manifest"] = retriever_manifest
        row.update(method_harness)
        sc = score_example(record.prediction, record.gold_answers)
        row["em"], row["f1"] = sc["em"], sc["f1"]
        return row, sc

    def _dispatch_with_retries(ex):
        record = None
        last_error = None
        for attempt in range(args.retries + 1):
            try:
                record = _dispatch(ex)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if record.finish_reason != "api_error":
                return record
        if record is None and last_error is not None:
            from .agent import AgentRecord
            return AgentRecord(
                id=str(ex.get("id", "")), question=ex.get("question", ""),
                gold_answers=list(ex.get("golden_answers", [])),
                finish_reason="exception", error=str(last_error))
        return record

    n = len(examples)
    results = [None] * n
    scores = [None] * n
    conc = max(1, args.concurrency)
    print(f"\nRunning agent={args.agent!r} on {n} examples (concurrency={conc}) ...\n")

    def _record_result(i, record):
        row, sc = _finalize(record, examples[i])
        results[i], scores[i] = row, sc
        return row, sc

    def _progress(done, row):
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done) * (n - done)
        print(f"  [{done:4d}/{n}]  em={row['em']:.0f}  f1={row['f1']:.2f}  "
              f"finish={row['finish_reason']:<12s}  ETA {eta/60:.1f}min", flush=True)

    if conc == 1:
        for i, ex in enumerate(examples):
            row, _ = _record_result(i, _dispatch_with_retries(ex))
            _progress(i + 1, row)
            if output_path and (i + 1) % 50 == 0:
                _save_jsonl(list(prior_done.values())
                            + [r for r in results if r is not None], output_path)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = {pool.submit(_dispatch_with_retries, ex): i for i, ex in enumerate(examples)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    record = fut.result()
                    row, _ = _record_result(i, record)
                except Exception as e:  # noqa: BLE001
                    ex = examples[i]
                    row = {"id": str(ex.get("id", "")), "question": ex.get("question", ""),
                           "gold_answers": ex.get("golden_answers", []), "prediction": None,
                           "finish_reason": "exception", "error": str(e), "em": 0.0, "f1": 0.0,
                           "dataset_manifest": dataset_manifest,
                           "method_config_id": method_cfg["method_config_id"],
                           "generator_revision": model_revision,
                           "prompt_sha256": method_prompt_hash,
                           "retriever_manifest": retriever_manifest}
                    row.update(method_harness)
                    results[i], scores[i] = row, {"em": 0.0, "f1": 0.0}
                done += 1
                if done % 20 == 0 or done == n:
                    _progress(done, row)
                if output_path and done % 100 == 0:
                    _save_jsonl(list(prior_done.values())
                                + [r for r in results if r is not None], output_path)

    # --- Save & report ---
    if output_path:
        results = list(prior_done.values()) + results
        _save_jsonl(results, output_path)
        print(f"\nResults saved → {output_path}")

    if args.full_eval:
        result_ids = [str(row["id"]) for row in results]
        try:
            validate_exact_id_set(result_ids, expected_full_ids)
        except ValueError as exc:
            sys.exit(str(exc))
    n_answered = sum(1 for r in results if r.get("prediction") is not None)
    scores = [{"em": float(r.get("em", 0)), "f1": float(r.get("f1", 0))} for r in results]
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
    parser.add_argument("--full-eval", action="store_true",
                        help="Require the complete canonical split; forbids --n/--offset")

    # Agent
    parser.add_argument("--agent", default="scaleseek", choices=_ALL_AGENTS)
    parser.add_argument("--retriever", choices=["bm25", "e5", "qwen3_emb_4b"],
                        default="bm25", help="Comparable retriever backend")
    parser.add_argument("--retrieval-top-k", type=int, choices=[3], default=3,
                        help="Frozen local-retriever top-k")
    parser.add_argument("--retriever-device", default=None)
    parser.add_argument("--tokenizer", default=None,
                        help="Exact Qwen3.5 tokenizer for Search-O1/ScaleSeek and "
                             "ScaleSeek tool-response token budgeting")
    parser.add_argument("--max-tool-response-tokens", type=int, default=2048,
                        help="Token budget per tool response for the scaleseek agent "
                             "(whole passages dropped to fit). Default 2048.")

    # Search-R1
    parser.add_argument("--search-r1-host", default=None)
    parser.add_argument("--search-r1-port", default=None,
                        help="vLLM port for Search-R1 (default: $SEARCH_R1_PORT or 8001)")
    parser.add_argument("--search-r1-model", default=None)
    parser.add_argument("--search-r1-tokenizer", default=None,
                        help="Exact Search-R1 7B/14B checkpoint; required")

    # Main LLM
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--generator-revision", default=None,
                        help="Required in --full-eval; must match frozen config")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of examples to run in parallel against the vLLM "
                             "server (the server batches them). 1 = sequential.")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries per example after transient API errors")

    # Output
    parser.add_argument("--output", default=None,
                        help="JSONL output file")
    parser.add_argument("--resume", action="store_true",
                        help="Skip examples already completed in --output "
                             "(api_error/exception rows are re-run). Merges prior "
                             "results into the final file.")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
