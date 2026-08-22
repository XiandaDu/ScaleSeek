#!/usr/bin/env python3
"""Re-shard the AgentIR tevatron pkl shards into smaller pieces.

Pure data plumbing: each official shard is a pickled (embeddings ndarray,
docid list) pair, and the official faiss loader does `for reps, lookup in
shards: index.add(reps); lookup += ...` -- so the LOAD PEAK is the full index
(~204G for 21M x 2560 float32) PLUS one live shard. With the official 4 x 51G
shards that peak is ~255G, physically above the 257G-class nodes (jobs 7511
and 7518 both OOM-killed exactly there), and both 500G nodes are out (octal40
kills every job silently, octal41 is admin-drained). 16 x ~12.75G shards cut
the peak to ~217G. Vectors, dtype, per-position docids and the add order
within every original shard are unchanged; shard boundaries carry no meaning
to the loader beyond memory footprint.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--pattern", default="corpus.*.pkl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pieces-per-shard", type=int, default=4)
    args = ap.parse_args()

    src = sorted(p for p in args.in_dir.glob(args.pattern)
                 if ".reshard." not in p.name)
    if not src:
        raise SystemExit(f"no shards match {args.in_dir}/{args.pattern}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_idx = 0
    total_rows = 0
    for path in src:
        print(f"[reshard] loading {path.name} ...", flush=True)
        with path.open("rb") as fh:
            reps, lookup = pickle.load(fh)
        n = len(lookup)
        assert reps.shape[0] == n, f"{path}: reps {reps.shape[0]} != lookup {n}"
        bounds = np.linspace(0, n, args.pieces_per_shard + 1, dtype=int)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            piece = args.out_dir / f"corpus.reshard.{out_idx:02d}.pkl"
            tmp = piece.with_suffix(".pkl.tmp")
            with tmp.open("wb") as fh:
                pickle.dump((np.ascontiguousarray(reps[lo:hi]),
                             list(lookup[lo:hi])), fh, protocol=4)
            tmp.rename(piece)
            print(f"[reshard]   {piece.name}: rows {lo}:{hi}", flush=True)
            out_idx += 1
        total_rows += n
        del reps, lookup
    print(f"[reshard] done: {out_idx} pieces, {total_rows:,} rows total", flush=True)


if __name__ == "__main__":
    main()
