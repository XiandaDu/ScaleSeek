#!/usr/bin/env python3
"""Behavioural audit of verl rollout dumps (train_rollouts/*.jsonl, val_rollouts/*.jsonl)
and of run_scaleseek_smoke_eval records: how many tool calls, which tools, what
retrieval parameters the policy actually chooses, EM by number of calls, GRPO group
degeneracy (all rollouts of a question scoring identically => zero advantage).

    python scripts/analyze_rollouts.py <file.jsonl> [more.jsonl ...] [--show N]
"""
from __future__ import annotations
import argparse, collections, json, re, statistics, sys

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANS = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

def calls_from_text(txt: str) -> list[dict]:
    out = []
    for m in TC.finditer(txt):
        try:
            o = json.loads(m.group(1)); out.append({"name": o.get("name"), **(o.get("arguments") or {})})
        except Exception:
            out.append({"name": "<unparsable>"})
    return out

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    recs = []
    for r in rows:
        if "output" in r:                      # verl dump
            out = r["output"]; calls = calls_from_text(out)
            recs.append({"qid": r["gts"]["qid"], "src": r["gts"]["qid"].split("_")[0], "em": r.get("reward/em", r.get("score", 0.0)),
                         "score": r.get("score", 0.0), "fmt": r.get("reward/format_pass", None), "calls": calls,
                         "tokens": r.get("metrics/response_token_count"), "hit_max": r.get("reward/hit_max_context", 0),
                         "ws": r.get("metrics/workspace_size"), "text": out, "question": r["input"].split("Question:")[-1][:300]})
        else:                                   # smoke_eval record
            turns = r.get("turns", []); calls = []
            for t in turns:
                a = t.get("assistant") or t.get("content") or ""
                calls += calls_from_text(a) if isinstance(a, str) else []
            if not calls and r.get("bm25_calls"): calls = [{"name": "bm25_retrieve", **c} for c in r["bm25_calls"]]
            recs.append({"qid": r["id"], "src": r["id"].split("_")[0], "em": r.get("em", 0.0), "score": r.get("em", 0.0), "fmt": 1.0 if r.get("prediction") else 0.0,
                         "calls": calls, "tokens": None, "hit_max": 0, "ws": r.get("final_workspace_size"),
                         "text": json.dumps(turns, ensure_ascii=False)[:6000], "question": r["question"][:300]})
    return recs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    recs = [x for f in a.files for x in load(f)]
    n = len(recs); print(f"{n} rollouts from {a.files}")
    nc = collections.Counter(len(r["calls"]) for r in recs)
    print("tool calls per rollout:", dict(sorted(nc.items())))
    tools = collections.Counter(c["name"] for r in recs for c in r["calls"]); print("tools used:", dict(tools))
    for key in ("top_k", "k1", "b", "mode"):
        vals = collections.Counter(str(c.get(key, "<unset>")) for r in recs for c in r["calls"] if c["name"] == "bm25_retrieve")
        print(f"  bm25 {key}:", dict(vals.most_common(8)))
    print("EM overall: %.3f   format_pass: %.3f   hit_max: %.3f" % (
        statistics.mean(r["em"] for r in recs), statistics.mean(r["fmt"] or 0 for r in recs), statistics.mean(r["hit_max"] for r in recs)))
    by = collections.defaultdict(list)
    for r in recs: by[len(r["calls"])].append(r["em"])
    print("EM by #calls:", {k: (round(statistics.mean(v), 3), len(v)) for k, v in sorted(by.items())})
    by = collections.defaultdict(list)
    for r in recs: by[r["src"]].append(r["em"])
    print("EM by source:", {k: (round(statistics.mean(v), 3), len(v)) for k, v in sorted(by.items())})
    if recs[0]["tokens"] is not None:
        print("response tokens: mean %.0f  median %.0f  max %d" % (statistics.mean(r["tokens"] for r in recs), statistics.median(r["tokens"] for r in recs), max(r["tokens"] for r in recs)))
    groups = collections.defaultdict(list)
    for r in recs: groups[r["qid"]].append(r["score"])
    multi = {q: s for q, s in groups.items() if len(s) > 1}
    if multi:
        degenerate = sum(1 for s in multi.values() if max(s) - min(s) < 1e-6)
        print(f"GRPO groups: {len(multi)}  all-identical-score (zero advantage): {degenerate} ({100*degenerate/len(multi):.0f}%)  "
              f"all-zero: {sum(1 for s in multi.values() if max(s) < 1e-6)}")
    print("\n=== samples ===")
    shown = 0
    for want in ("em==1", "em==0"):
        for r in recs:
            if (r["em"] == 1) == (want == "em==1"):
                print(f"\n--- [{want}] {r['qid']}  calls={len(r['calls'])}  Q: {r['question'].strip()[:200]}")
                print(r["text"][:3500]); shown += 1
                if shown >= a.show * 2: break
            if shown >= a.show * 2: break
        if shown >= a.show * 2: break

if __name__ == "__main__":
    main()
