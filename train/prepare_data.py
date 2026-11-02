"""RL data in verl's parquet format: NQ + HotpotQA train (GrepSeek's training data),
FlashRAG revision pinned in configs/base.yaml.  A held-out slice of the TRAIN splits is
the validation set, so no evaluation question is ever seen during training or selection.

    python train/prepare_data.py <out_dir> [--val-per-source 100]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleseek import agent
from scaleseek.config import load_base
from scaleseek.hub import snapshot_dir


def rows(path: Path, source: str) -> list[dict]:
    out = []
    for line in open(path):
        r = json.loads(line)
        golds = [g for g in r["golden_answers"] if str(g).strip()]
        if golds:
            out.append({"id": f"{source}_{r['id']}", "question": r["question"], "golds": golds, "source": source})
    return out


def to_verl(r: dict) -> dict:
    return {"data_source": r["source"], "agent_name": "scaleseek_agent",
            "prompt": [{"role": "system", "content": agent.SYSTEM_PROMPT},
                       {"role": "user", "content": agent.user_message(r["question"])}],
            "reward_model": {"style": "rule", "ground_truth": r["golds"]},
            "extra_info": {"id": r["id"], "question": r["question"]}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--val-per-source", type=int, default=100)
    a = p.parse_args()
    base = load_base()
    d = base["datasets"]["nq"]
    snap = snapshot_dir(base, d["hf_repo"], d["revision"], repo_type="dataset")
    rng = random.Random(20261102)
    train, val = [], []
    for source, sub in (("nq", "nq"), ("hotpotqa", "hotpotqa")):
        rs = rows(snap / sub / "train.jsonl", source)
        rng.shuffle(rs)
        val += rs[: a.val_per_source]
        train += rs[a.val_per_source:]
    rng.shuffle(train)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rs in (("train", train), ("dev", val)):
        pq.write_table(pa.Table.from_pylist([to_verl(r) for r in rs]), a.out_dir / f"{name}.parquet")
        print(name, len(rs))


if __name__ == "__main__":
    main()
