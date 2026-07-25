#!/usr/bin/env python3
"""Analyze puzzle-testbed eval records for FEEDBACK-DRIVEN b adaptation.

Reads eval-record JSONL from run_scaleseek_smoke_eval.py (with `turns`, `em`) and,
per student, reports on the (held-out) puzzles:
  EM               answer correct
  recall           the answer token appeared in some tool response (gold retrieved)
  retry rate       fraction of examples with a 2nd bm25_retrieve (reacted to failure)
  lowered-b rate   fraction whose retry used b < 0.75 (the correct fix)
  mean workspace   final workspace size (tight is better)
  b values         distinct b values emitted across bm25 calls

    python scripts/analyze_puzzle_eval.py \
        omit=.puzzles/eval_teacher.jsonl heuristic=.puzzles/eval_heuristic.jsonl \
        search=.puzzles/eval_search.jsonl
"""
from __future__ import annotations

import collections
import json
import re
import sys

_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def analyze(path: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    em = recall = retry = lowered = 0
    ws_sum = ws_n = 0
    bvals: collections.Counter = collections.Counter()
    for r in rows:
        ans = (r.get("gold_answers") or r.get("golden_answers") or [""])[0].lower()
        turns = r.get("turns", [])
        # recall: answer token surfaced in any tool response
        if any(t.get("role") == "tool" and ans and ans in t.get("content", "").lower() for t in turns):
            recall += 1
        em += int(float(r.get("em", 0)) >= 1.0)
        # bm25 calls
        bm25 = []
        for t in turns:
            if t.get("role") != "assistant":
                continue
            m = _TC.search(t.get("content", ""))
            if not m:
                continue
            try:
                o = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if o.get("name") == "bm25_retrieve":
                bm25.append(o.get("arguments", {}))
                if "b" in o["arguments"]:
                    bvals[o["arguments"]["b"]] += 1
        if len(bm25) >= 2:
            retry += 1
            if any(float(c.get("b", 0.75)) < 0.75 for c in bm25[1:]):
                lowered += 1
        if r.get("final_workspace_size") is not None:
            ws_sum += r["final_workspace_size"]; ws_n += 1
    return {"n": n, "em": em / max(n, 1), "recall": recall / max(n, 1),
            "retry": retry / max(n, 1), "lowered": lowered / max(n, 1),
            "ws": ws_sum / max(ws_n, 1), "bvals": dict(bvals)}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_puzzle_eval.py label=records.jsonl [...]")
    items = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:]]
    print(f"{'student':<11}{'EM':<7}{'recall':<8}{'retry':<8}{'lowered-b':<11}{'mean_ws':<9}{'b values emitted'}")
    print("-" * 78)
    for label, path in items:
        s = analyze(path)
        print(f"{label:<11}{s['em']:<7.2f}{s['recall']:<8.2f}{s['retry']:<8.2f}"
              f"{s['lowered']:<11.2f}{s['ws']:<9.1f}{s['bvals'] or '{}'}")


if __name__ == "__main__":
    main()
