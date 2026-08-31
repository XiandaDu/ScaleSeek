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
    python scripts/make_dcilite_datasets.py --datasets popqa --cap 1500
        # deterministic <=1500-question subset for the call-budget-bound
        # harnesses (DCI / DR-DCI / RISE only -- see TASK.md "评测规模上限").
        # Datasets at or below the cap are emitted complete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.datasets import load_dataset  # noqa: E402
from eval.subsets import MAX_EVAL_N, SANCTIONED_CAPS, select, subset_manifest  # noqa: E402

OUT_DIR = Path("/data/rech/mofengra/dr_dci_official/data/dci-bench/data/_scaleseek")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("-n", "--n", type=int, default=None,
                    help="Explicit slice size; default is the complete canonical split")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--cap", type=int, default=None, metavar="N",
                    help="Emit at most N questions, chosen deterministically "
                         "(TASK.md sanctions N=1500 for DCI/DR-DCI/RISE only). "
                         "A dataset at or below N is emitted complete. Writes "
                         "<ds>_cap<N>.jsonl plus a .manifest.json fingerprinting "
                         "the selected id set.")
    args = ap.parse_args()
    if args.cap is not None and args.n is not None:
        ap.error("--cap and -n are mutually exclusive")
    if args.cap is not None and args.cap not in SANCTIONED_CAPS:
        ap.error(f"only --cap in {sorted(SANCTIONED_CAPS)} is sanctioned; a different "
                 "cap would create an undocumented evaluation scope")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        examples = load_dataset(ds, limit=args.n)
        total = len(examples)
        manifest = None
        if args.cap is not None:
            examples = select(examples, dataset=ds, cap=args.cap)
            manifest = subset_manifest(examples, dataset=ds, total=total,
                                       cap=args.cap)
            stem = f"{ds}_cap{args.cap}"
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
            print(f"{'':18s} -> {mpath}  (of {total}; scope={manifest['eval_scope']}; "
                  f"ids sha256 {manifest['subset_ids_sha256'][:16]}…)")


if __name__ == "__main__":
    main()
