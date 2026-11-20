"""Headroom after GrepSeek+ on the validation questions, from the two existing runs of each configuration:
what two independent rollouts see and get right together, and where the remaining errors are."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, final_answer, normalize_answer

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def evidence(r):
    text = []
    for t in r["trajectory"]["turns"] or []:
        if t["role"] == "tool":
            for ln in json.loads(t["content"])["stdout"].split("\n"):
                try:
                    text.append(json.loads(ln)["contents"])
                except Exception:
                    text.append(ln)
    return " " + normalize_answer(" ".join(text)) + " "


def load(name):
    out = {}
    for r in map(json.loads, open(P / name / "trajectories.jsonl")):
        ev = evidence(r)
        golds = [normalize_answer(g) for g in r["golden_answers"] if len(normalize_answer(g)) >= 3]
        ans = final_answer(r)
        out[r["id"]] = dict(seen=any(f" {g} " in ev for g in golds), ok=exact_match(ans, r["golden_answers"]),
                            answered=ans is not None, grounded=bool(ans) and f" {normalize_answer(ans)} " in ev,
                            ans=normalize_answer(ans) if ans else None)
    return out


for a, b in (("base", "base_r2"), ("all", "all_r2")):
    A, B = load(a), load(b)
    n = len(A)
    one = (sum(x["ok"] for x in A.values()) + sum(x["ok"] for x in B.values())) / 2
    print(f"== {a} + {b}")
    print(f"  one rollout: EM {100 * one / n:.1f}, gold seen {100 * (sum(x['seen'] for x in A.values()) + sum(x['seen'] for x in B.values())) / 2 / n:.1f}%")
    print(f"  two rollouts: either correct {100 * sum(A[i]['ok'] or B[i]['ok'] for i in A) / n:.1f}, gold seen by either {100 * sum(A[i]['seen'] or B[i]['seen'] for i in A) / n:.1f}%,"
          f" same answer {100 * sum(A[i]['ans'] is not None and A[i]['ans'] == B[i]['ans'] for i in A) / n:.1f}% (EM when they agree {100 * sum(A[i]['ok'] for i in A if A[i]['ans'] is not None and A[i]['ans'] == B[i]['ans']) / max(1, sum(A[i]['ans'] is not None and A[i]['ans'] == B[i]['ans'] for i in A)):.1f},"
          f" when they differ {100 * sum(A[i]['ok'] for i in A if A[i]['ans'] != B[i]['ans']) / max(1, sum(A[i]['ans'] != B[i]['ans'] for i in A)):.1f})")
    for name, R in ((a, A), (b, B)):
        g = [x for x in R.values() if x["answered"]]
        gr, ug = [x for x in g if x["grounded"]], [x for x in g if not x["grounded"]]
        c = Counter("correct" if x["ok"] else "no answer" if not x["answered"] else "wrong, gold seen" if x["seen"] else "wrong, gold not seen" for x in R.values())
        print(f"  {name:8s} answer string in tool output: {100 * len(gr) / n:.1f}% of questions, EM {100 * sum(x['ok'] for x in gr) / len(gr):.1f};"
              f" not in output: {100 * len(ug) / n:.1f}%, EM {100 * sum(x['ok'] for x in ug) / len(ug):.1f}   | " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in c.most_common()))
