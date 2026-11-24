"""Why GrepSeek looks ~5 points lower in our tables than in its paper: the same per-dataset
numbers under the two ways of averaging, the per-dataset differences against sampling
error, and the effect of our presence penalty (validation questions)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

ROOT = Path("/scratch/a32du/scaleseek_v2")
DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
FULL = dict(zip(DS, [3610, 11313, 14267, 7405, 12576, 2417, 125]))                 # the paper's evaluation sets (its Table 2)
PAPER = {"EM": dict(zip(DS, [40.47, 69.25, 43.63, 49.49, 46.27, 21.43, 51.20])),   # Table 6
         "F1": dict(zip(DS, [52.23, 76.73, 48.61, 62.31, 51.78, 30.06, 62.12]))}   # Table 1
ours, n_sub = {"EM": {}, "F1": {}}, {}
for ds in DS:
    recs = [json.loads(l) for l in open(ROOT / "runs" / ds / "grepseek/trajectories.jsonl")]
    n_sub[ds] = len(recs)
    ours["EM"][ds] = 100 * sum(exact_match(final_answer(r), r["golden_answers"]) for r in recs) / len(recs)
    ours["F1"][ds] = 100 * sum(f1_score(final_answer(r), r["golden_answers"]) for r in recs) / len(recs)


def agg(x, w):
    return sum(x[d] * w[d] for d in DS) / sum(w[d] for d in DS)


for m in ("EM", "F1"):
    print(f"\n{m}: per dataset (ours on {n_sub['nq']}-question subsets; +- = 95% interval of the subset estimate)")
    for ds in DS:
        p = ours[m][ds] / 100
        ci = 196 * (p * (1 - p) / n_sub[ds]) ** 0.5 if m == "EM" else float("nan")
        print(f"  {ds:16s} paper {PAPER[m][ds]:6.2f}   ours {ours[m][ds]:5.1f} +- {ci:3.1f}   ours - paper {ours[m][ds] - PAPER[m][ds]:+5.1f}"
              f"   weight: subsets {100 * n_sub[ds] / sum(n_sub.values()):4.1f}%  full sets {100 * FULL[ds] / sum(FULL.values()):4.1f}%")
    print(f"  mean per-dataset difference (ours - paper): {sum(ours[m][d] - PAPER[m][d] for d in DS) / 7:+.2f}")
    print(f"  averaged OUR way (micro over the subsets):      paper's numbers {agg(PAPER[m], n_sub):.2f}   ours {agg(ours[m], n_sub):.2f}")
    print(f"  averaged the PAPER's way (weighted by full sets): paper's numbers {agg(PAPER[m], FULL):.2f}   ours {agg(ours[m], FULL):.2f}")
    print(f"  macro over 7 datasets:                           paper's numbers {sum(PAPER[m].values()) / 7:.2f}   ours {sum(ours[m].values()) / 7:.2f}")

print("\nvalidation questions (500 NQ dev + 500 HotpotQA dev outside the test subset), official GrepSeek harness")
P = ROOT / "pilots/grepseek"
runs = {}
for name in ("base", "base_r2", "base_nopp", "base_nopp_r2"):
    f = P / name / "trajectories.jsonl"
    if f.exists():
        runs[name] = {r["id"]: r for r in map(json.loads, open(f))}
for name, R in runs.items():
    line = f"  {name:13s} presence_penalty {json.load(open(P / name / 'config.json'))['profile']['presence_penalty']}"
    for src in ("nq", "hotpotqa", ""):
        sel = [r for i, r in R.items() if i.startswith(src)]
        line += f"   {src or 'all'} EM {100 * sum(exact_match(final_answer(r), r['golden_answers']) for r in sel) / len(sel):5.1f} F1 {100 * sum(f1_score(final_answer(r), r['golden_answers']) for r in sel) / len(sel):5.1f}"
    fin = {}
    for r in R.values():
        fin[r["finish_reason"]] = fin.get(r["finish_reason"], 0) + 1
    print(line + f"   finish {fin}")
if "base_nopp" in runs:
    ok = {n: {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in R.items()} for n, R in runs.items()}
    pool = lambda a, b: {f"{i}#{k}": d[i] for k, d in enumerate((ok[a], ok[b])) for i in d}
    if "base_nopp_r2" in runs:
        A, B = pool("base_nopp", "base_nopp_r2"), pool("base", "base_r2")
        print(f"  pooled 2 x 1000: without penalty {100 * sum(A.values()) / len(A):.1f}, with penalty 2.0 {100 * sum(B.values()) / len(B):.1f}, {mcnemar_exact(A, B)}")
print(f"  (paper, full test sets: NQ EM {PAPER['EM']['nq']}, HotpotQA EM {PAPER['EM']['hotpotqa']}; our test subsets: NQ {ours['EM']['nq']:.1f}, HotpotQA {ours['EM']['hotpotqa']:.1f})")
