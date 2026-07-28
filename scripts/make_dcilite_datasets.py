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
    python scripts/make_dcilite_datasets.py --datasets popqa --fraction 10
        # deterministic 1/10 subset for the call-budget-bound harnesses
        # (DCI / DR-DCI / RISE only -- see TASK.md "1/10 规模例外")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.datasets import load_dataset  # noqa: E402
from eval.subsets import DECILE, select, subset_manifest  # noqa: E402

OUT_DIR = Path("/data/rech/mofengra/dr_dci_official/data/dci-bench/data/_scaleseek")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("-n", "--n", type=int, default=None,
                    help="Explicit slice size; default is the complete canonical split")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--fraction", type=int, default=None, metavar="D",
                    help="Emit the deterministic 1/D subset instead of the full "
                         "split (TASK.md sanctions D=10 for DCI/DR-DCI/RISE only). "
                         "Writes <ds>_decile1of<D>.jsonl plus a .manifest.json "
                         "fingerprinting the selected id set.")
    args = ap.parse_args()
    if args.fraction is not None and args.n is not None:
        ap.error("--fraction and -n are mutually exclusive")
    if args.fraction is not None and args.fraction != DECILE:
        ap.error(f"only --fraction {DECILE} is sanctioned by TASK.md; a different "
                 "denominator would create an undocumented evaluation scope")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        examples = load_dataset(ds, limit=args.n)
        total = len(examples)
        manifest = None
        if args.fraction is not None:
            examples = select(examples, dataset=ds, denominator=args.fraction)
            manifest = subset_manifest(examples, dataset=ds, total=total,
                                       denominator=args.fraction)
            stem = f"{ds}_decile1of{args.fraction}"
        elif args.n is None:
            stem = f"{ds}_full"
        else:
            stem = f"{ds}_n{args.n}"
        path = out_dir / f"{stem}.jsonl"
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
        if manifest is not None:
            mpath = out_dir / f"{stem}.manifest.json"
            mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"{'':18s} -> {mpath}  (of {total}; "
                  f"ids sha256 {manifest['subset_ids_sha256'][:16]}…)")


if __name__ == "__main__":
    main()
