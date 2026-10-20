"""Decoding pilot -- NOT a formal run.  One (method, dataset, retriever) with an alternative
presence_penalty, through the formal code path of the 903a200 worktree; output in
$ROOT/dev/pilots/<name>/.  Usage: decoding_pilot2.py <name> <method> <dataset> <retriever|none> <presence_penalty>"""
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/a32du/scaleseek_v2/harness/903a2003bdfd")
from scaleseek.config import load_base, resolve_run
from scaleseek.runner import eval_run

name, method, dataset, retriever, penalty = sys.argv[1:6]
base = load_base()
spec = resolve_run(base, method, dataset, None if retriever == "none" else retriever, "formal")
spec.profile["presence_penalty"] = float(penalty)
spec.run_dir = Path(base["paths"]["root"]) / "dev" / "pilots" / name
print(eval_run(spec))
