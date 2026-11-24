"""RL rollout dumps (train_runs/<run>/train_rollouts/<step>.jsonl): per step the training EM, how many
rollouts reached the last turn (forced under `commit`) and how those did.
Usage: rl_rollout_stats.py <run> [--show]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek import agent

D = Path("/scratch/a32du/scaleseek_v2/train_runs") / sys.argv[1] / "train_rollouts"
NOTE = agent.budget_note(0).strip()
mean = lambda v: sum(v) / max(1, len(v))
for p in sorted(D.glob("*.jsonl"), key=lambda p: int(p.stem)):
    rows = [json.loads(l) for l in open(p)]
    last = lambda r: NOTE in r["output"]              # the note under the 5th tool turn: the next turn is the last
    forced, free = [r for r in rows if last(r)], [r for r in rows if not last(r)]
    print(f"step {int(p.stem):3d}  n {len(rows)}  EM {100 * mean([r['em'] for r in rows]):5.1f}  answered {100 * mean([r['answered'] for r in rows]):5.1f}%"
          f"  valid {100 * mean([r['valid'] for r in rows]):5.1f}%  calls {mean([r['n_calls'] for r in rows]):.2f}  tokens {mean([r['response_tokens'] for r in rows]):5.0f}"
          f"  | last turn {100 * len(forced) / len(rows):4.1f}%: EM {100 * mean([r['em'] for r in forced]):4.1f}, answered {100 * mean([r['answered'] for r in forced]):5.1f}%, cut {sum(r['n_cut'] for r in forced)}"
          f"  | earlier: EM {100 * mean([r['em'] for r in free]):4.1f}")
if "--show" in sys.argv:
    r = next(r for r in rows if last(r))
    print("\n... " + repr(r["output"][-500:]))
