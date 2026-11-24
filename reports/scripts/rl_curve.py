"""Learning curves of RL runs on the same training batches (every run draws the same questions at
the same step): training EM per block of 10 steps, minus a reference run whose policy barely moved
(grpo_v2: lr 1e-6, no forced last turn).  A constant gap = no learning; a growing gap = learning.
Usage: rl_curve.py <run>..."""
import json
import sys
from pathlib import Path

D = Path("/scratch/a32du/scaleseek_v2/train_runs")


def steps(run):
    out = {}
    for p in (D / run / "train_rollouts").glob("*.jsonl"):
        rows = [json.loads(l) for l in open(p)]
        out[int(p.stem)] = (100 * sum(r["em"] for r in rows) / len(rows), 100 * sum(r["answered"] for r in rows) / len(rows),
                            sum(r["n_calls"] for r in rows) / len(rows))
    return out


ref = steps("grpo_v2")
runs = {name: steps(name) for name in sys.argv[1:]}
print(f"{'steps':>9s}  {'grpo_v2 EM':>10s}" + "".join(f"  | {name}: EM (gap) answered calls".rjust(44) for name in runs))
for lo in range(1, 150, 10):
    block = [s for s in range(lo, lo + 10) if s in ref]
    line = f"{lo:4d}-{lo + 9:<4d}  {sum(ref[s][0] for s in block) / max(1, len(block)):10.1f}"
    for name, st in runs.items():
        b = [s for s in block if s in st]
        if len(b) < 10:
            line += "  | " + f"(only {len(b)} steps)".rjust(40)
            continue
        em, gap = sum(st[s][0] for s in b) / len(b), sum(st[s][0] - ref[s][0] for s in b) / len(b)
        line += "  | " + f"{em:5.1f} ({gap:+5.1f})  {sum(st[s][1] for s in b) / len(b):5.1f}%  {sum(st[s][2] for s in b) / len(b):.2f}".rjust(40)
    print(line)
