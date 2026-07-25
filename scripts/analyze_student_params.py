#!/usr/bin/env python3
"""Compare SFT students by the BM25 parameters they actually EMIT (+ EM).

Reads eval-record JSONL files written by run_scaleseek_smoke_eval.py (each record
carries `turns`, `em`, `f1`). For each student it reports EM and, from the raw
assistant `<tool_call>` JSON, how often the model SETS k1/b/top_k, how varied those
values are, and how often it uses mode=merge — i.e. whether the cold-start policy
taught it to control the knobs, the prerequisite for RL to refine them.

    python scripts/analyze_student_params.py \
        omit=.smoke_hard/eval_teacher.jsonl \
        heuristic=.smoke_hard/eval_heuristic.jsonl \
        search=.smoke_hard/eval_search.jsonl
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
    em = sum(float(r.get("em", 0)) for r in rows)
    f1 = sum(float(r.get("f1", 0)) for r in rows)
    calls = set_k1 = set_b = set_tk = merge = 0
    k1v: collections.Counter = collections.Counter()
    bv: collections.Counter = collections.Counter()
    tkv: collections.Counter = collections.Counter()
    for r in rows:
        for t in r.get("turns", []):
            if t.get("role") != "assistant":
                continue
            m = _TC.search(t.get("content", ""))
            if not m:
                continue
            try:
                o = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if o.get("name") != "bm25_retrieve":
                continue
            a = o.get("arguments", {})
            calls += 1
            if "k1" in a:
                set_k1 += 1; k1v[a["k1"]] += 1
            if "b" in a:
                set_b += 1; bv[a["b"]] += 1
            if "top_k" in a:
                set_tk += 1; tkv[a["top_k"]] += 1
            if a.get("mode") == "merge":
                merge += 1
    return {
        "n": n, "em": em / max(n, 1), "f1": f1 / max(n, 1),
        "calls": calls,
        "set_k1": set_k1, "set_b": set_b, "set_tk": set_tk, "merge": merge,
        "k1_vals": dict(k1v), "b_vals": dict(bv), "tk_vals": dict(tkv),
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_student_params.py label=records.jsonl [label2=...]")
    items = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:]]

    print(f"{'student':<12}{'EM':<7}{'F1':<7}{'bm25':<6}"
          f"{'set k1':<9}{'set b':<9}{'set top_k':<11}{'merge':<7}")
    print("-" * 74)
    stats = {}
    for label, path in items:
        s = analyze(path)
        stats[label] = s
        def frac(x):  # noqa: E306
            return f"{x}/{s['calls']}"
        print(f"{label:<12}{s['em']:<7.2f}{s['f1']:<7.2f}{s['calls']:<6}"
              f"{frac(s['set_k1']):<9}{frac(s['set_b']):<9}{frac(s['set_tk']):<11}{frac(s['merge']):<7}")

    print("\n=== distinct parameter values each student emitted ===")
    for label, _ in items:
        s = stats[label]
        print(f"  {label:<10} k1={s['k1_vals'] or '{}'}  b={s['b_vals'] or '{}'}  top_k={s['tk_vals'] or '{}'}")


if __name__ == "__main__":
    main()
