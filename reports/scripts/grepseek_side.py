"""GrepSeek's side of the sampled `grepseek_wrong_this_right` cases: for every case another strong
run got right and GrepSeek got wrong, print GrepSeek's commands and what came back."""
import json
from pathlib import Path

A = Path("/scratch/a32du/scaleseek_v2/analysis/failures")
want = {}
for f in sorted((A / "cases" / "others").glob("*/*/grepseek_wrong_this_right/*.txt")):
    run, ds = f.parts[-4], f.parts[-3]
    want.setdefault((ds, f.stem), []).append(run)

for line in open(A / "grepseek_features.jsonl"):
    r = json.loads(line)
    runs = want.get((r["ds"], r["id"]))
    if not runs:
        continue
    print(f"\n=== {r['ds']} {r['id']}  others right: {', '.join(runs)}")
    print(f"Q: {r['question'][:300]}\nGOLD: {r['golds'][:4]}  PRED: {r['prediction']!r}  {r['category']}  finish={r['finish']}"
          f"  think_tok={r['think_tok']}")
    for c in r["calls"]:
        flags = "".join(k for k in ("empty", "truncated", "timed_out", "gold_hit") if c[k])
        print(f"  [{c['lines']:>3} lines {c['stdout_tok']:>5} tok {flags}] {c['cmd'][:220]}")
