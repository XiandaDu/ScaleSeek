"""GrepSeek+ pilots vs the official harness on the same validation questions (500 NQ dev +
500 HotpotQA dev outside the test subset).  Strict EM with a paired exact McNemar test,
finish reasons, and what the variants did.  Usage: gpilot_cmp.py <pilot name>..."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def load(name):
    return {r["id"]: r for r in map(json.loads, open(P / name / "trajectories.jsonl"))}


def tools(r):
    return [json.loads(t["content"]) for t in r["trajectory"]["turns"] or [] if t["role"] == "tool"]


ref = load("base")
for name in ["base"] + sys.argv[1:]:
    run = load(name)
    print(f"\n== {name}  ({len(run)}/{len(ref)} records)  {json.load(open(P / name / 'config.json'))['variants']}")
    for src in ("nq", "hotpotqa", ""):
        sel = [i for i in ref if i.startswith(src) and i in run]
        a = {i: exact_match(final_answer(run[i]), run[i]["golden_answers"]) for i in sel}
        b = {i: exact_match(final_answer(ref[i]), ref[i]["golden_answers"]) for i in sel}
        mc = mcnemar_exact(a, b)
        rs = [run[i] for i in sel]
        fin = Counter(r["finish_reason"] for r in rs)
        answered = [i for i in sel if run[i]["finish_reason"] == "answer"]
        f1 = sum(f1_score(final_answer(r), r["golden_answers"]) for r in rs) / len(sel)
        print(f"  {src or 'all':8s} EM {100 * sum(a.values()) / len(sel):5.1f} (base {100 * sum(b.values()) / len(sel):5.1f}, "
              f"{100 * (sum(a.values()) - sum(b.values())) / len(sel):+5.1f}; won {mc['only_a']} lost {mc['only_b']} p={mc['p_value']:.3f})"
              f"  F1 {100 * f1:5.1f}  answered {100 * len(answered) / len(sel):5.1f}%"
              f"  EM|answered {100 * sum(a[i] for i in answered) / max(1, len(answered)):5.1f}"
              f"  finish {dict(fin.most_common())}")
    rs = list(run.values())
    payloads = [p for r in rs for p in tools(r)]
    notices = Counter()
    asked = heeded = 0
    for r in rs:
        turns = r["trajectory"]["turns"] or []
        for j, t in enumerate(turns):
            if t["role"] != "tool":
                continue
            n = json.loads(t["content"]).get("notice", "")
            for key in ("last tool call", "almost full", "left before", "case-insensitive", "match without"):
                notices[key] += key in n
        told = [j for j, t in enumerate(turns) if t["role"] == "tool" and "final answer now" in json.loads(t["content"]).get("notice", "")]
        if told:
            asked += 1
            nxt = turns[told[0] + 1] if told[0] + 1 < len(turns) else None
            heeded += bool(nxt and nxt["role"] == "assistant" and nxt["parse"]["answer"] is not None)
    empty = sum(not p["stdout"].strip() and not p["timed_out"] and p["exit_code"] in (0, 1) for p in payloads)
    print(f"  calls/q {len(payloads) / len(rs):.2f}  empty {100 * empty / max(1, len(payloads)):.1f}%  timed out {sum(p['timed_out'] for p in payloads)}"
          f"  truncated {100 * sum(p['truncated'] for p in payloads) / max(1, len(payloads)):.1f}%"
          f"  wall h {sum(r['wall_time_s'] or 0 for r in rs) / 3600:.1f} (sum over questions)")
    print(f"  notices {dict(notices)}   told to answer: {asked} questions, answered on the next turn: {heeded}")
