#!/usr/bin/env python
"""Convert ScaleSeek eval datasets -> official DCI-Agent-Lite harness format.

The official harness (`dr_dci_official/scripts/bcplus_eval/run_bcplus_eval.py`)
reads a jsonl of {query_id, query, answer}; multiple gold answers are joined with
" / " (that is the convention used by the shipped dci-bench files).

Ours is {id, question, golden_answers}. The query_id is kept **identical to our
`id`** so results can be paired 1:1 against our own runs in `results/*_dci.jsonl`.

Usage:
    python scripts/make_dcilite_datasets.py --datasets popqa        # full split
    python scripts/make_dcilite_datasets.py --datasets nq -n 50     # explicit slice
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.datasets import load_dataset  # noqa: E402

OUT_DIR = Path("/data/rech/mofengra/dr_dci_official/data/dci-bench/data/_scaleseek")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("-n", "--n", type=int, default=None,
                    help="Explicit slice size; default is the complete canonical split")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        examples = load_dataset(ds, limit=args.n)
        path = out_dir / (f"{ds}_full.jsonl" if args.n is None else f"{ds}_n{args.n}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                gold = ex.get("golden_answers") or []
                if isinstance(gold, str):
                    gold = [gold]
                f.write(json.dumps({
                    "query_id": str(ex["id"]),          # keep OUR id -> pairable
                    "query": ex["question"],
                    "answer": " / ".join(str(g) for g in gold),
                }, ensure_ascii=False) + "\n")
        print(f"{ds:18s} -> {path}  ({len(examples)} queries)")


if __name__ == "__main__":
    main()
