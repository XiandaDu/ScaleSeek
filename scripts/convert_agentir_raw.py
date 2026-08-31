#!/usr/bin/env python3
"""Aggregate the AgentIR oss_client per-query files for the normalizer.

Data plumbing only: each run_*.json holds query_id, a Responses-style result
array, and tool counters -- but neither the question nor the gold answers,
which the client never echoes. This joins them back from the canonical
normalized dataset by id, takes the LAST output_text entry as final_text
(the official 'Explanation/Exact Answer/Confidence' block; the Exact Answer
line is extracted later by normalize --answer-regex, same as DR-DCI/RISE),
and emits one results.jsonl row per query.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--dataset-jsonl", type=Path, required=True,
                    help="Canonical normalized dataset ({id, question, golden_answers})")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    with args.dataset_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                by_id[str(row["id"])] = row

    files = sorted(args.raw_dir.glob("run_*.json"))
    if not files:
        sys.exit(f"no run_*.json under {args.raw_dir}")
    out_rows = []
    dupes = 0
    seen: set[str] = set()
    for f in files:
        r = json.loads(f.read_text())
        qid = str(r.get("query_id", ""))
        if qid in seen:
            dupes += 1
            continue
        seen.add(qid)
        final_text = ""
        for entry in reversed(r.get("result") or []):
            if entry.get("type") == "output_text":
                final_text = entry.get("output") or ""
                break
        canon = by_id.get(qid, {})
        gold = canon.get("golden_answers") or []
        if isinstance(gold, str):
            gold = [gold]
        out_rows.append({
            "query_id": qid,
            "question": canon.get("question", ""),
            "gold_answers": gold,
            "final_text": final_text,
            "status": r.get("status", "unknown"),
            "n_turns": len(r.get("result") or []),
            "n_tool_calls": int(sum((r.get("tool_call_counts") or {}).values())),
            "source_file": f.name,
        })
    args.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows))
    print(f"[convert] {len(out_rows)} rows -> {args.output} "
          f"(skipped {dupes} duplicate query_ids)")


if __name__ == "__main__":
    main()
