"""The validation table (500 NQ dev + 500 HotpotQA dev, used by no training or test run):
GrepSeek and GrepSeek+ (pilots/grepseek/) next to our agent's runs (train_data/rollouts/),
on the same questions.  Strict EM overall and by source, F1, and a paired exact McNemar
test against GrepSeek and against the untrained v3.
Usage: val_table.py <our run>...   (val_v3_base is always the first of ours)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

R = Path("/scratch/a32du/scaleseek_v2")


def load(path):
    """{question id: record}; our agent's ids carry the sample number (#0)."""
    return {r["id"].split("#")[0]: r for r in map(json.loads, open(R / path / "trajectories.jsonl"))}


runs = {label: load(f"pilots/grepseek/{run}") for label, run in (      # each configuration was run more than once
    ("GrepSeek", "base"), ("GrepSeek, run 2", "base_r2"),
    ("GrepSeek+", "all"), ("GrepSeek+, run 2", "all_r2"), ("GrepSeek+, run 3", "all_r3"))}
runs |= {name: load(f"train_data/rollouts/{name}") for name in ["val_v3_base"] + sys.argv[1:]}
ids = sorted(runs["GrepSeek"])
em = {name: {i: exact_match(final_answer(recs[i]), recs[i]["golden_answers"]) for i in ids}
      for name, recs in runs.items() if set(recs) == set(ids) or sys.exit(f"{name}: not the same questions")}
pair = lambda a, b: "{only_a} / {only_b}, p={p_value:.3g}".format(**mcnemar_exact(em[a], em[b]))
print(f"{len(ids)} questions, the same in every run")
print(f"{'run':24s}    EM    NQ HotpotQA    F1   won / lost vs GrepSeek      vs untrained v3")
for name, recs in runs.items():
    by = [100 * sum(em[name][i] for i in ids if i.startswith(s)) / sum(i.startswith(s) for i in ids) for s in ("", "nq", "hotpotqa")]
    f1 = 100 * sum(f1_score(final_answer(recs[i]), recs[i]["golden_answers"]) for i in ids) / len(ids)
    print(f"{name:24s} {by[0]:5.1f} {by[1]:5.1f} {by[2]:8.1f} {f1:5.1f}   {pair(name, 'GrepSeek'):26s}  {pair(name, 'val_v3_base')}")
