"""What is left after GrepSeek+ on the validation questions: are the wrong answers near misses (answer form),
and are the errors the two rollouts share retrieval failures?"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, normalize_answer

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def load(name):
    out = {}
    for r in map(json.loads, open(P / name / "trajectories.jsonl")):
        text = []
        for t in r["trajectory"]["turns"] or []:
            if t["role"] == "tool":
                for ln in json.loads(t["content"])["stdout"].split("\n"):
                    try:
                        text.append(json.loads(ln)["contents"])
                    except Exception:
                        text.append(ln)
        ev = " " + normalize_answer(" ".join(text)) + " "
        a = final_answer(r)
        na = normalize_answer(a) if a else None
        golds = [normalize_answer(g) for g in r["golden_answers"]]
        out[r["id"]] = dict(ok=exact_match(a, r["golden_answers"]), ans=na, f1=f1_score(a, r["golden_answers"]),
                            seen=any(len(g) >= 3 and f" {g} " in ev for g in golds),
                            over=bool(na) and any(g and f" {g} " in f" {na} " and g != na for g in golds),
                            under=bool(na) and any(g and f" {na} " in f" {g} " and g != na for g in golds))
    return out


def kind(x):
    if x["ok"]:
        return "correct"
    if x["ans"] is None:
        return "no answer"
    if x["over"]:
        return "wrong: answer contains gold"
    if x["under"]:
        return "wrong: gold contains answer"
    return "wrong: other, gold seen" if x["seen"] else "wrong: other, gold not seen"


A, B = load("all"), load("all_r2")
n = len(A)
for name, R in (("all", A), ("all_r2", B)):
    c = Counter(kind(x) for x in R.values())
    print(f"{name}: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in c.most_common())
          + f" | EM {100 * c['correct'] / n:.1f}, cover-EM (answer contains gold) {100 * (c['correct'] + c['wrong: answer contains gold']) / n:.1f},"
          f" F1 {100 * sum(x['f1'] for x in R.values()) / n:.1f}, wrong with F1>=0.5: {100 * sum(not x['ok'] and x['f1'] >= 0.5 for x in R.values()) / n:.1f}%")
for src in ("nq", "hotpotqa"):
    c = Counter(kind(A[i]) for i in A if i.startswith(src))
    m = sum(c.values())
    print(f"  all / {src}: " + ", ".join(f"{k} {100 * v / m:.1f}%" for k, v in c.most_common()))
agree = [i for i in A if A[i]["ans"] is not None and A[i]["ans"] == B[i]["ans"]]
aw = [i for i in agree if not A[i]["ok"]]
c = Counter(kind(A[i]) for i in aw)
print(f"both rollouts give the same answer: {100 * len(agree) / n:.1f}% of questions; same and wrong: {100 * len(aw) / n:.1f}% -> "
      + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in c.most_common())
      + f"; of these, gold seen by neither rollout {100 * sum(not A[i]['seen'] and not B[i]['seen'] for i in aw) / n:.1f}%")
dis = [i for i in A if i not in set(agree)]
print(f"different answers: {100 * len(dis) / n:.1f}%; one of the two right {100 * sum(A[i]['ok'] or B[i]['ok'] for i in dis) / n:.1f}%, both wrong {100 * sum(not A[i]['ok'] and not B[i]['ok'] for i in dis) / n:.1f}%")
