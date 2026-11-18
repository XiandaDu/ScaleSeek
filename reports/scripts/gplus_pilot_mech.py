"""Why the variants help: is the gold answer in what the tool returned, and what happens to the
questions the official harness left unanswered.  Usage: gpilot_mech.py <pilot name>..."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, final_answer, normalize_answer

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def load(name):
    return {r["id"]: r for r in map(json.loads, open(P / name / "trajectories.jsonl"))}


def seen(r):
    text = []
    for t in r["trajectory"]["turns"] or []:
        if t["role"] == "tool":
            for ln in json.loads(t["content"])["stdout"].split("\n"):
                try:
                    text.append(json.loads(ln)["contents"])
                except Exception:
                    text.append(ln)
    text = " " + normalize_answer(" ".join(text)) + " "
    return any(f" {normalize_answer(g)} " in text for g in r["golden_answers"] if len(normalize_answer(g)) >= 3)


ref = load("base")
ok_ref = {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in ref.items()}
unanswered = [i for i, r in ref.items() if r["finish_reason"] != "answer"]
print(f"base: {len(unanswered)} questions without an answer")
for name in ["base"] + sys.argv[1:]:
    run = load(name)
    ok = {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in run.items()}
    s = {i: seen(r) for i, r in run.items()}
    n = len(run)
    line = f"{name:16s} gold in tool output {100 * sum(s.values()) / n:5.1f}%"
    for src in ("nq", "hotpotqa"):
        sel = [i for i in run if i.startswith(src)]
        line += f"  ({src} {100 * sum(s[i] for i in sel) / len(sel):5.1f}%)"
    line += (f"  EM|seen {100 * sum(ok[i] for i in run if s[i]) / max(1, sum(s.values())):5.1f}"
             f"  EM|not seen {100 * sum(ok[i] for i in run if not s[i]) / max(1, n - sum(s.values())):5.1f}"
             f"  | base-unanswered: now answered {sum(run[i]['finish_reason'] == 'answer' for i in unanswered)}, correct {sum(ok[i] for i in unanswered)}"
             f"  | base-answered: EM {100 * sum(ok[i] for i in run if i not in set(unanswered)) / (n - len(unanswered)):5.1f}")
    print(line)
