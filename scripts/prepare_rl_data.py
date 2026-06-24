#!/usr/bin/env python3
"""Prepare ScaleSeek RL training data.

Reads from already-downloaded eval datasets (run download_data.py first).
Writes train.jsonl + dev.jsonl in ScaleSeek canonical format:
    {"id": str, "question": str, "golden_answers": [str], "source": str}

Default mix (mirrors GrepSeek's training mix, extended to multi-hop):
    Train: NQ (train) + HotpotQA (train) + 2WikiMultihopQA (train) + MuSiQue (train)
    Val:   NQ (test)  + HotpotQA (dev)   + 2WikiMultihopQA (dev)   + MuSiQue (dev)

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

# (dataset_name, eval_split, has_train_split)
_SOURCES: list[tuple[str, str, bool]] = [
    ("nq",              "test",  True),
    ("hotpotqa",        "dev",   True),
    ("2wikimultihopqa", "dev",   True),
    ("musique",         "dev",   True),
]


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

    # -- Train split --------------------------------------------------------
    train_rows: list[dict] = []
    for name, eval_split, has_train in sources:
        if not has_train:
            print(f"  [{name}] no train split available; skipping train")
            continue
        rows = _load(name, split="train", limit=None)
        rows = [r for r in rows if r.get("golden_answers")]
        print(f"  [{name}] train: {len(rows):,} rows")
        train_rows.extend(rows)

    if not args.no_shuffle:
        rng.shuffle(train_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]

    print(f"\nTotal train: {len(train_rows):,}")
    _write_jsonl(train_rows, out_dir / "train.jsonl")

    # -- Val split ----------------------------------------------------------
    val_rows: list[dict] = []
    for name, eval_split, _ in sources:
        rows = _load(name, split=eval_split, limit=None)
        rows = [r for r in rows if r.get("golden_answers")]
        if val_per:
            rows = rng.sample(rows, min(val_per, len(rows)))
        print(f"  [{name}] val ({eval_split}): {len(rows):,} rows")
        val_rows.extend(rows)

    if not args.no_shuffle:
        rng.shuffle(val_rows)

    print(f"\nTotal val  : {len(val_rows):,}")
    _write_jsonl(val_rows, out_dir / "dev.jsonl")

    # -- Summary ------------------------------------------------------------
    summary = {
        "sources": [n for n, _, _ in sources],
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit_per_source": val_per,
    }
    summary_path = out_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
