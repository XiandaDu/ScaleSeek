#!/usr/bin/env python3
"""Build the RL training file for continuing GRPO from a merged checkpoint.

The Nibi run (`grpo_9b`) reached global_step 190 with train_batch_size 64 and
only its merged HF weights survive (Allenda/ScaleSeek-Qwen3.5-9B-GRPO-step190);
the FSDP shards, optimizer state and dataloader state were not uploaded, so the
continuation is a fresh verl run warm-started from those weights.

verl draws training questions with its own sampler
(`verl.trainer.main_ppo.create_rl_sampler`: torchdata's stateful RandomSampler
over a `torch.Generator()` that `data.seed: null` leaves at torch's fixed default
seed), and each chain segment restored that sampler's position from the
checkpoint. So the questions steps 1..190 consumed are exactly the first
190*64 indices of that permutation. This script replays the identical sampler
on the identical train.jsonl (sha256 pinned in DATA_PROVENANCE.md) and writes
the remainder, so the continuation never re-trains on questions the policy
already saw.

    python scripts/carve_rl_continuation.py --steps_done 190 --batch 64 \
        --train $RL_DATA_DIR/train.jsonl --out $RL_DATA_DIR/train_from_step190.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TRAIN_SHA256 = "19b718c400f945674bdcae17f614392bbbee3d4e8c7478c09f2729600c386ebf"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps_done", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None,
                    help="data.seed of the original run (default: None = verl leaves the generator unseeded)")
    args = ap.parse_args()

    sha = sha256_file(args.train)
    if sha != EXPECTED_TRAIN_SHA256:
        raise SystemExit(f"train.jsonl sha256 {sha} != provenance {EXPECTED_TRAIN_SHA256}; "
                         "the replayed permutation would not match the original run")

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    # answerable_only filter in ScaleSeekDataset: all rows have golden_answers (prepare_rl_data guarantees it)
    rows = [r for r in rows if r.get("golden_answers")]
    n = len(rows)

    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import create_rl_sampler
    data_config = OmegaConf.create({"shuffle": True, "seed": args.seed, "sampler": None,
                                    "dataloader_num_workers": 1})
    sampler = create_rl_sampler(data_config, list(range(n)))
    order = list(iter(sampler))
    assert len(order) == n and len(set(order)) == n
    consumed = args.steps_done * args.batch
    seen = set(order[:consumed])
    remaining = [rows[i] for i in order[consumed:]]  # keep the original sampler order

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in remaining:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "source_train": str(args.train), "source_sha256": sha, "source_rows": n,
        "steps_done": args.steps_done, "batch": args.batch, "consumed": consumed,
        "remaining": len(remaining), "out": str(args.out), "out_sha256": sha256_file(args.out),
        "generator_initial_seed": None if args.seed is not None else "torch default (67280421310721)",
        "first_consumed_ids": [rows[i]["id"] for i in order[:5]],
    }
    Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
