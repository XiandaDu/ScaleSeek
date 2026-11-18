"""Does a two-number model of the agent predict the EM of every pilot configuration?
From the official-harness runs only: p1 = P(correct | answered, gold seen), p0 = P(correct | answered, gold not seen).
Prediction for a configuration = p1 * P(answered & seen) + p0 * P(answered & not seen), from that configuration's own
seen / answered rates.  Validation questions; also the 7 Wiki test sets (GrepSeek -> GrepSeek+)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, final_answer, normalize_answer

ROOT = Path("/scratch/a32du/scaleseek_v2")


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


def rows(path):
    return [(seen(r), r["finish_reason"] == "answer", exact_match(final_answer(r), r["golden_answers"]))
            for r in map(json.loads, open(path))]


def fit(rs):
    a1 = [c for s, a, c in rs if a and s]
    a0 = [c for s, a, c in rs if a and not s]
    return sum(a1) / len(a1), sum(a0) / len(a0)


def predict(rs, p1, p0):
    n = len(rs)
    return 100 * (p1 * sum(a and s for s, a, c in rs) + p0 * sum(a and not s for s, a, c in rs)) / n


P = ROOT / "pilots/grepseek"
data = {d.name: rows(d / "trajectories.jsonl") for d in sorted(P.iterdir()) if (d / "trajectories.jsonl").exists()}
p1, p0 = fit(data["base"] + data["base_r2"])
print(f"validation: p1 = {p1:.3f}, p0 = {p0:.3f} (from base + base_r2)")
err = []
print(f"{'configuration':16s} {'seen':>6s} {'answered':>8s} {'EM':>6s} {'predicted':>9s} {'error':>6s}")
for name, rs in data.items():
    n = len(rs)
    em = 100 * sum(c for _, _, c in rs) / n
    pr = predict(rs, p1, p0)
    if not name.startswith("base"):
        err.append((em, pr))
    print(f"{name:16s} {100 * sum(s for s, _, _ in rs) / n:5.1f}% {100 * sum(a for _, a, _ in rs) / n:7.1f}% {em:6.1f} {pr:9.1f} {pr - em:+6.1f}")
m = len(err)
mx, my = sum(e for e, _ in err) / m, sum(p for _, p in err) / m
cov = sum((e - mx) * (p - my) for e, p in err)
r = cov / (sum((e - mx) ** 2 for e, _ in err) ** .5 * sum((p - my) ** 2 for _, p in err) ** .5)
print(f"{m} configurations other than base: mean |error| {sum(abs(p - e) for e, p in err) / m:.2f} EM, max {max(abs(p - e) for e, p in err):.2f}, Pearson r {r:.3f}")

print("\ntest sets: parameters from GrepSeek on each dataset, prediction for GrepSeek+")
tot_em = tot_pr = tot_n = 0
for ds in ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]:
    a, b = rows(ROOT / "runs" / ds / "grepseek/trajectories.jsonl"), rows(ROOT / "runs" / ds / "grepseek_plus/trajectories.jsonl")
    q1, q0 = fit(a)
    em, pr, em0 = 100 * sum(c for _, _, c in b) / len(b), predict(b, q1, q0), 100 * sum(c for _, _, c in a) / len(a)
    tot_em, tot_pr, tot_n = tot_em + em * len(b), tot_pr + pr * len(b), tot_n + len(b)
    print(f"  {ds:16s} p1 {q1:.2f} p0 {q0:.2f}  GrepSeek {em0:5.1f}  GrepSeek+ actual {em:5.1f}  predicted {pr:5.1f}  ({pr - em:+.1f})")
print(f"  micro: actual {tot_em / tot_n:.1f}, predicted {tot_pr / tot_n:.1f}")
