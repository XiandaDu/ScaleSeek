"""Leave-one-out ablation of GrepSeek+ on the validation questions, and run-to-run noise.
Strict EM; paired exact McNemar on the same questions."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def ok(name):
    return {r["id"]: exact_match(final_answer(r), r["golden_answers"]) for r in map(json.loads, open(P / name / "trajectories.jsonl"))}


def answered(name):
    rs = [json.loads(l) for l in open(P / name / "trajectories.jsonl")]
    return 100 * sum(r["finish_reason"] == "answer" for r in rs) / len(rs)


def em(d, src=""):
    sel = [i for i in d if i.startswith(src)]
    return 100 * sum(d[i] for i in sel) / len(sel)


def vs(a, b):
    mc = mcnemar_exact(a, b)
    return f"{em(a) - em(b):+5.1f} (won {mc['only_a']:3d} lost {mc['only_b']:3d} p={mc['p_value']:.3f})"


R = {n: ok(n) for n in ["base", "base_r2", "all", "all_r2", "all_no_budget", "all_no_dedup", "all_no_title", "all_no_relax", "all_no_fold",
                        "budget_dedup", "title", "ranked_budget"]}
print("run-to-run noise (same configuration, two runs)")
print(f"  base vs base_r2: {em(R['base']):.1f} / {em(R['base_r2']):.1f}   {vs(R['base_r2'], R['base'])}")
print(f"  all  vs all_r2 : {em(R['all']):.1f} / {em(R['all_r2']):.1f}   {vs(R['all_r2'], R['all'])}")
print("\nfull combination vs official harness, every pairing of the two runs of each")
for a in ("all", "all_r2"):
    for b in ("base", "base_r2"):
        print(f"  {a:7s} vs {b:8s}: {vs(R[a], R[b])}")
pool = lambda x, y: {f"{i}#{k}": d[i] for k, d in enumerate((R[x], R[y])) for i in d}
A, B = pool("all", "all_r2"), pool("base", "base_r2")
print(f"  pooled (2 x 1000): base {em(B):.1f} (nq {em(B, 'nq'):.1f}, hotpotqa {em(B, 'hotpotqa'):.1f})   all {em(A):.1f} (nq {em(A, 'nq'):.1f}, hotpotqa {em(A, 'hotpotqa'):.1f})   {vs(A, B)}")
print("\nleave one out (each vs the mean of the two full runs; paired test vs each full run)")
full = (em(R["all"]) + em(R["all_r2"])) / 2
print(f"  {'configuration':16s} {'nq':>6s} {'hotpot':>6s} {'all':>6s} {'answered':>8s}   vs all / vs all_r2 / vs base (mean of 2)")
for n in ["all", "all_r2", "all_no_budget", "all_no_dedup", "all_no_title", "all_no_relax", "all_no_fold", "budget_dedup", "title", "ranked_budget", "base", "base_r2"]:
    d = R[n]
    print(f"  {n:16s} {em(d, 'nq'):6.1f} {em(d, 'hotpotqa'):6.1f} {em(d):6.1f} {answered(n):7.1f}%   {em(d) - full:+5.1f} vs full mean"
          f" | {vs(d, R['all'])} | {vs(d, R['all_r2'])} | {em(d) - (em(R['base']) + em(R['base_r2'])) / 2:+5.1f} vs base mean")
