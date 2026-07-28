#!/usr/bin/env python3
"""Prepare ScaleSeek RL training data.

Reads from already-downloaded eval datasets (run download_data.py first).
Writes train.jsonl + dev.jsonl in ScaleSeek canonical format:
    {"id": str, "question": str, "golden_answers": [str], "source": str}

Default mix (mirrors GrepSeek's training mix, extended to multi-hop):
    Train: NQ (train) + HotpotQA (train) + 2WikiMultihopQA (train) + MuSiQue (train)
    Val:   held out of those same train splits, per source

The evaluation splits (NQ test, HotpotQA/2Wiki/MuSiQue dev) are NEVER read here.
Validating on them — as this script did until 2026-07-28 — means the GRPO trainer
selects checkpoints (save_freq/test_freq=40) against the test set.

Note for the results table: nq / hotpotqa / 2wiki / musique are IN-DOMAIN for this
policy, while popqa / triviaqa / bamboogle are out-of-domain. Search-R1's public
checkpoint trains on nq+hotpotqa only, so a cross-method claim on 2wiki or musique
is confounded by the mix unless it is labelled as such.

Usage:
    source setup_env.sh
    python scripts/prepare_rl_data.py --out_dir $DATA/rl_data
    python scripts/prepare_rl_data.py --out_dir $DATA/rl_data/small \\
        --train_limit 20000 --val_limit_per_source 500

TBD decisions:
    - Final dataset mix and weighting
    - Whether to include single-hop (NQ, TriviaQA) in RL training
    - Curriculum ordering (start with single-hop, progress to multi-hop)
    - Hard-example selection / filtering strategy
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Resolve repo root so we can import eval.*
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from eval.datasets import load_dataset, FLASHRAG_DATASETS

# ---------------------------------------------------------------------------
# Dataset mix
# ---------------------------------------------------------------------------

# (dataset_name, RESERVED_eval_split, has_train_split)
#
# The second field is the split this project *evaluates* on. It is listed here so
# the guard below can refuse to read it — it is NOT a source. RL validation is
# carved out of the train split instead (see _split_train_val).
#
# Until 2026-07-28 this script used those very splits as the RL val set, i.e. it
# validated and checkpointed (test_freq/save_freq=40) on the evaluation data.
_SOURCES: list[tuple[str, str, bool]] = [
    ("nq",              "test",  True),
    ("hotpotqa",        "dev",   True),
    ("2wikimultihopqa", "dev",   True),
    ("musique",         "dev",   True),
]

# Datasets the main table reports but RL never trains on -> genuine OOD columns.
# Keep this in sync with the eval matrix; it is emitted into the run summary so a
# results table can mark in-domain vs out-of-domain without guesswork.
_OOD_EVAL_DATASETS = ["popqa", "triviaqa", "bamboogle"]


def _load(name: str, split: str | None, limit: int | None) -> list[dict]:
    """Load via eval.datasets (local first, HF fallback)."""
    rows = load_dataset(name, split=split, limit=limit)
    for r in rows:
        r["source"] = name
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} → {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out_dir", required=True,
                   help="Output directory for train.jsonl and dev.jsonl.")
    p.add_argument("--train_limit", type=int, default=None,
                   help="Total train examples cap (after mixing). Default: all.")
    p.add_argument("--val_limit_per_source", type=int, default=500,
                   help="Val examples per dataset source (default 500; 0=all).")
    p.add_argument("--sources", nargs="*", default=None,
                   help="Override dataset list. Default: nq hotpotqa 2wikimultihopqa musique")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_shuffle", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _SOURCES
    if args.sources:
        name_set = set(args.sources)
        sources = [(n, s, h) for n, s, h in _SOURCES if n in name_set]

    val_per = args.val_limit_per_source or None

    print(f"\nRL data directory : {out_dir}")
    print(f"Sources           : {[n for n,_,_ in sources]}")
    print(f"Train limit       : {args.train_limit or 'all'}")
    print(f"Val per source    : {val_per or 'all'}")
    print()

    # -- Train / val split ---------------------------------------------------
    # Both come from the TRAIN split of each source; the evaluation splits named
    # in _SOURCES are never read. Holding val out per source keeps every training
    # domain represented in the validation curve.
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for name, eval_split, has_train in sources:
        if not has_train:
            print(f"  [{name}] no train split available; skipping")
            continue
        rows = _load(name, split="train", limit=None)
        rows = [r for r in rows if r.get("golden_answers")]
        rng.shuffle(rows)                       # held-out choice must not track file order
        n_val = min(val_per or 0, len(rows) // 10) if val_per else 0
        held, kept = rows[:n_val], rows[n_val:]
        val_rows.extend(held)
        train_rows.extend(kept)
        print(f"  [{name}] train: {len(kept):,} rows  (+{len(held):,} held out for val; "
              f"eval split {eval_split!r} untouched)")

    if not args.no_shuffle:
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]

    # Hard guard: a train/val id collision would silently restore the leak.
    train_ids = {str(r["id"]) for r in train_rows}
    val_ids = {str(r["id"]) for r in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} ids appear in both train and val "
                 f"(e.g. {sorted(overlap)[:3]}); RL validation must be disjoint.")

    print(f"\nTotal train: {len(train_rows):,}")
    _write_jsonl(train_rows, out_dir / "train.jsonl")
    print(f"Total val  : {len(val_rows):,}  (held out of train; NOT the eval split)")
    _write_jsonl(val_rows, out_dir / "dev.jsonl")

    # -- Summary ------------------------------------------------------------
    summary = {
        "sources": [n for n, _, _ in sources],
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit_per_source": val_per,
        # Provenance for the results table: which eval columns are in-domain for
        # this policy and which are genuinely held out. A cross-method claim on an
        # in-domain column is confounded by the training mix (Search-R1's public
        # ckpt, for instance, trains on nq+hotpotqa only).
        "val_source": "held out of each source's train split",
        "eval_splits_never_read": {n: s for n, s, _ in sources},
        "in_domain_eval_datasets": [n for n, _, h in sources if h],
        "ood_eval_datasets": _OOD_EVAL_DATASETS,
    }
    summary_path = out_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
