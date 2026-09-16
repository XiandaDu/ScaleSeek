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
import collections
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


def _parse_counts(items: list[str] | None) -> dict[str, int] | None:
    if not items:
        return None
    out = {}
    for it in items:
        for part in it.split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            out[k.strip()] = int(v)
    return out


def _sample_per_source(rows: list[dict], counts: dict[str, int], rng: random.Random) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    out = []
    for src, n in counts.items():
        pool = list(by.get(src, []))
        rng.shuffle(pool)
        out.extend(pool[:n])
    rng.shuffle(out)
    return out


def _apply_screen(rows: list[dict], stats_path: str, band: str, keep_hard: int) -> tuple[list[dict], dict]:
    """Keep the learnable band (lo < pass_rate < hi) plus reachable hard multi-hop.

    A GRPO group whose rollouts all score the same has zero advantage; pass_rate 0
    or 1 under the current policy is wasted compute unless the question is a hard
    multi-hop one whose supporting docs BM25 can actually surface (then the
    retrieval-process credit can still create a gradient).
    """
    lo, hi = (float(x) for x in band.split(","))
    stats = {}
    for l in open(stats_path, encoding="utf-8"):
        if l.strip():
            s = json.loads(l); stats[str(s["id"])] = s
    band_rows, hard_rows, unscreened = [], [], 0
    for r in rows:
        s = stats.get(str(r["id"]))
        if s is None:
            unscreened += 1
            continue
        p = float(s.get("pass_rate", 0.0))
        if lo < p < hi:
            band_rows.append(r)
        elif p <= lo and s.get("multi_hop") and float(s.get("reachable_coverage", 0.0)) >= 0.5:
            hard_rows.append(r)
    hard_rows = hard_rows[:keep_hard]
    kept = band_rows + hard_rows
    summary = {"stats_file": stats_path, "band": [lo, hi], "screened": len(rows) - unscreened,
               "unscreened_dropped": unscreened, "in_band": len(band_rows),
               "hard_multihop_kept": len(hard_rows), "kept": len(kept),
               "band_sources": dict(collections.Counter(r["source"] for r in band_rows)),
               "hard_sources": dict(collections.Counter(r["source"] for r in hard_rows))}
    print(f"  screen: {summary}")
    return kept, summary


def _load(name: str, split: str | None, limit: int | None, include_evidence: bool = False) -> list[dict]:
    """Load via eval.datasets (local first, HF fallback)."""
    rows = load_dataset(name, split=split, limit=limit, include_evidence=include_evidence)
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
    # ---- v2 options (2026-09-05). None of them change the output when absent, so
    # the v1 command in DATA_PROVENANCE.md still reproduces its sha256. ----
    p.add_argument("--with-evidence", action="store_true",
                   help="attach supporting titles / sub-answers (eval/evidence.py) to every row "
                        "that has them; needed by the v2 retrieval-process reward")
    p.add_argument("--mix", nargs="*", default=None, metavar="SRC=N",
                   help="per-source train caps after shuffling, e.g. 2wikimultihopqa=14000 musique=15000 "
                        "hotpotqa=15000 nq=6000 (multi-hop-heavy curriculum). Default: everything.")
    p.add_argument("--dev-ids", default=None,
                   help="reuse the dev ids of an existing dev.jsonl (keeps validation comparable "
                        "across data versions); these ids are excluded from train")
    p.add_argument("--sft-questions", nargs="*", default=None, metavar="SRC=N",
                   help="also write sft_questions.jsonl sampled per source from the train pool")
    p.add_argument("--screen-pool", nargs="*", default=None, metavar="SRC=N",
                   help="also write screen_pool.jsonl (per-source sample of the train pool) for "
                        "scripts/screen_difficulty.py")
    p.add_argument("--screen", default=None,
                   help="screen_stats.jsonl from scripts/screen_difficulty.py; train keeps only "
                        "questions in the learnable band plus reachable hard multi-hop ones")
    p.add_argument("--band", default="0.0,1.0",
                   help="keep questions with lo < pass_rate < hi (exclusive), default 0.0,1.0")
    p.add_argument("--keep-hard-multihop", type=int, default=1500,
                   help="additionally keep up to N pass_rate==0 multi-hop questions whose supporting "
                        "docs BM25 can reach (reachable_coverage >= 0.5)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _SOURCES
    if args.sources:
        name_set = set(args.sources)
        sources = [(n, s, h) for n, s, h in _SOURCES if n in name_set]

    val_per = args.val_limit_per_source or None
    mix = _parse_counts(args.mix)
    dev_ids = None
    if args.dev_ids:
        dev_ids = {str(json.loads(l)["id"]) for l in open(args.dev_ids, encoding="utf-8") if l.strip()}
        print(f"Dev ids           : {len(dev_ids)} reused from {args.dev_ids}")
    if mix:
        print(f"Train mix caps    : {mix}")

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
        rows = _load(name, split="train", limit=None, include_evidence=args.with_evidence)
        rows = [r for r in rows if r.get("golden_answers")]
        rng.shuffle(rows)                       # held-out choice must not track file order
        if dev_ids is not None:
            held = [r for r in rows if str(r["id"]) in dev_ids]
            kept = [r for r in rows if str(r["id"]) not in dev_ids]
        else:
            n_val = min(val_per or 0, len(rows) // 10) if val_per else 0
            held, kept = rows[:n_val], rows[n_val:]
        if mix is not None:
            kept = kept[:mix.get(name, len(kept))]
        val_rows.extend(held)
        train_rows.extend(kept)
        print(f"  [{name}] train: {len(kept):,} rows  (+{len(held):,} held out for val; "
              f"eval split {eval_split!r} untouched)")

    if not args.no_shuffle:
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if dev_ids is not None:
        # keep the exact v1 dev order so validation curves line up row for row
        order = {}
        for i, l in enumerate(open(args.dev_ids, encoding="utf-8")):
            if l.strip():
                order[str(json.loads(l)["id"])] = i
        val_rows.sort(key=lambda r: order.get(str(r["id"]), 1 << 30))

    pool_rows = list(train_rows)   # before screening: the curriculum pool
    screen_summary = None
    if args.screen:
        train_rows, screen_summary = _apply_screen(train_rows, args.screen, args.band, args.keep_hard_multihop)

    # Hard guard: a train/val id collision would silently restore the leak.
    train_ids = {str(r["id"]) for r in train_rows}
    val_ids = {str(r["id"]) for r in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} ids appear in both train and val "
                 f"(e.g. {sorted(overlap)[:3]}); RL validation must be disjoint.")

    print(f"\nTotal train: {len(train_rows):,}")
    _write_jsonl(train_rows, out_dir / "train.jsonl")
    if args.mix or args.screen or args.sft_questions or args.screen_pool:
        _write_jsonl(pool_rows, out_dir / "pool.jsonl")
    for flag, fname in ((args.sft_questions, "sft_questions.jsonl"), (args.screen_pool, "screen_pool.jsonl")):
        counts = _parse_counts(flag)
        if counts:
            _write_jsonl(_sample_per_source(pool_rows, counts, random.Random(args.seed + 1)), out_dir / fname)
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
        "with_evidence": args.with_evidence,
        "mix": mix, "dev_ids_from": args.dev_ids,
        "train_sources": dict(collections.Counter(r["source"] for r in train_rows)),
        "screen": screen_summary,
    }
    summary_path = out_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
