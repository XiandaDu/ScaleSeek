#!/usr/bin/env python3
"""Convert official DR-DCI (Pi harness) run artifacts -> ScaleSeek results schema.

We run DR-DCI unchanged (github.com/EigenTom/DR-DCI) with only the model endpoint
pointed at our vLLM, per the "同 prompt 同超参、只换模型端点" decision. Its per-query
artifacts land in outputs/bcplus_eval/<run_name>/. This script maps them into the
same JSONL rows our eval emits, so scripts/compute_metrics.py can score EM/F1,
latency, and workspace Gold R@W / Qrel R@W with the identical metric definitions
used for every other baseline.

Because the exact artifact field names are only knowable from a real run, this
script:
  1. runs with --schema-probe first to print the keys of one artifact, then
  2. maps fields via CLI flags (edit once, no code change).

Emitted row (our AgentRecord.to_dict shape, subset):
    {id, question, gold_answers, prediction, finish_reason,
     workspace_doc_ids, n_tool_calls, total_time_s, turns}

Usage:
    python scripts/convert_dr_dci_output.py \
        --run-dir outputs/bcplus_eval/<run_name> --schema-probe
    python scripts/convert_dr_dci_output.py \
        --run-dir outputs/bcplus_eval/<run_name> \
        --out results/bcp_dr_dci.jsonl \
        --id-field query_id --answer-field final_answer \
        --workspace-field pulled_docids --time-field wall_time_s
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def _iter_artifacts(run_dir: Path):
    """Yield parsed per-query artifacts. Handles either one JSON file per query
    or a single JSONL/JSON list — adjust the glob if the layout differs."""
    files = sorted(glob.glob(str(run_dir / "**" / "*.json"), recursive=True)) \
        + sorted(glob.glob(str(run_dir / "**" / "*.jsonl"), recursive=True))
    for fp in files:
        try:
            if fp.endswith(".jsonl"):
                for line in open(fp, encoding="utf-8"):
                    if line.strip():
                        yield fp, json.loads(line)
            else:
                obj = json.load(open(fp, encoding="utf-8"))
                if isinstance(obj, list):
                    for o in obj:
                        yield fp, o
                else:
                    yield fp, obj
        except Exception as e:
            print(f"  [skip] {fp}: {e}", file=sys.stderr)


def _get(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--schema-probe", action="store_true",
                    help="Print the keys of the first artifact and exit.")
    ap.add_argument("--id-prefix", default="browsecomp_plus_",
                    help="Prefix so ids match qrels.json keys from build_browsecomp_plus.py")
    # field mappings (dotted paths allowed) — set after --schema-probe
    ap.add_argument("--id-field", default="query_id")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--answer-field", default="final_answer")
    ap.add_argument("--workspace-field", default="pulled_docids",
                    help="List of doc-ids materialized into the workspace (W_T)")
    ap.add_argument("--time-field", default="wall_time_s")
    ap.add_argument("--tools-field", default="num_tool_calls")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")

    if args.schema_probe:
        for fp, obj in _iter_artifacts(run_dir):
            print(f"artifact: {fp}")
            print("keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1500])
            return
        sys.exit("no artifacts found under run dir")

    out = Path(args.out or (run_dir / "converted_results.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as w:
        for _, obj in _iter_artifacts(run_dir):
            if not isinstance(obj, dict):
                continue
            qid = _get(obj, args.id_field)
            if qid is None:
                continue
            ws = _get(obj, args.workspace_field) or []
            row = {
                "id": f"{args.id_prefix}{qid}",
                "question": _get(obj, args.question_field) or "",
                "gold_answers": [],  # BCP scoring uses qrels / LLM-judge, not EM golds
                "prediction": _get(obj, args.answer_field),
                "finish_reason": "answer" if _get(obj, args.answer_field) else "no_answer",
                "workspace_doc_ids": [str(d) for d in ws],
                "n_tool_calls": _get(obj, args.tools_field) or 0,
                "n_bm25_calls": 0,
                "total_time_s": _get(obj, args.time_field) or 0.0,
                "turns": [],
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out}")
    print("score with:\n"
          f"  python scripts/compute_metrics.py --results {out} "
          "--dataset bcp --agent dr_dci --bcp-qrels <...>/qrels.json --bcp-doclen <...>/doclen.json")


if __name__ == "__main__":
    main()
