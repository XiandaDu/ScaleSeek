"""Is any evaluation question a training question?  Counts only (BCP text is never printed).

Training side (SFT + RL of ScaleSeek v3):
  pool      every NQ train + HotpotQA train question (what SFT and RL draw from)
  sft6000   the 6,000 questions the 9B model rolled out on (train_data/rollouts/self_v3)
  sft_rows  the 3,779 of them whose trajectory SFT trained on (train_data/sft_v3_self)
  rl_seen   the questions in the RL rollouts (train_runs/grpo_v3_sft/train_rollouts)
Evaluation side: the 8 test subsets and the 1,000 validation questions.
Compared by normalised question text (lower case, no punctuation, no articles).

    python reports/scripts/leak_check.py
"""
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scaleseek.config import load_base
from scaleseek.data import load_subset
from scaleseek.grepseek_pilot import questions as validation_questions
from scaleseek.hub import snapshot_dir
from scaleseek.metrics import normalize_answer as norm

base = load_base()
R = Path(base["paths"]["root"])
d = base["datasets"]["nq"]
snap = snapshot_dir(base, d["hf_repo"], d["revision"], repo_type="dataset")

by_id = {f"{source}_{r['id']}": norm(r["question"])          # id -> normalised question
         for source in ("nq", "hotpotqa") for r in map(json.loads, open(snap / source / "train.jsonl"))}
pool = set(by_id.values())

ids = {json.loads(l)["id"].split("#")[0] for l in open(R / "train_data/rollouts/self_v3/trajectories.jsonl")}
rows = {i.split("#")[0] for i in pq.read_table(R / "train_data/sft_v3_self/train.parquet").column("id").to_pylist()}
seen, unmatched = set(), 0
for p in (R / "train_runs/grpo_v3_sft/train_rollouts").glob("*.jsonl"):
    for r in map(json.loads, open(p)):
        q = norm(re.sub(r"\s*assistant\s*(<think>)?\s*$", "", r["input"].rsplit("Question: ", 1)[1]))
        unmatched += q not in pool
        seen.add(q)
train = {"pool": pool, "sft6000": {by_id[i] for i in ids}, "sft_rows": {by_id[i] for i in rows}, "rl_seen": seen}
print(f"train questions: {len(by_id)} ids; SFT rollouts on {len(ids)} ids, SFT rows from {len(rows)} ids; "
      f"RL rollout questions not in the train files: {unmatched}")
print("distinct normalised questions:", {k: len(v) for k, v in train.items()})

evals = {name: [e["question"] for e in load_subset(base, name)[0]] for name in base["datasets"]}
evals["validation"] = [e["question"] for e in validation_questions(base, 500)]
tests = set().union(*({norm(q) for q in qs} for name, qs in evals.items() if name != "validation"))
print(f"{'evaluation set':18s}{'n':>6s}" + "".join(f"{k:>10s}" for k in train))
for name, qs in evals.items():
    qs = {norm(q) for q in qs}
    print(f"{name:18s}{len(qs):6d}" + "".join(f"{len(qs & t):10d}" for t in train.values()))
    hit = qs & (train["sft6000"] | train["sft_rows"] | train["rl_seen"])
    if hit:                                                  # ids only
        print("    train ids:", sorted(i for i, q in by_id.items() if q in hit and (i in ids or q in seen)))
print("validation questions that are test questions:", len({norm(q) for q in evals["validation"]} & tests))
