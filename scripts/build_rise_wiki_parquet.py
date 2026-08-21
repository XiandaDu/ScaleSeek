#!/usr/bin/env python3
"""Package the structured Wiki article tree for the official RISE tooling.

Data plumbing only. The official pipeline consumes
  * a corpus parquet with columns (docid, text)   -> scripts/build_bm25_index.py
  * {"relpath_to_docid": {relpath: docid}}        -> run_rise.py --filename-map
while our TOC step (build_wiki_toc.py) produces a flat directory of .md files.
This walks that directory once and emits both, with docid = file stem and
relpath = file name, so BM25 hits resolve to exactly the files the agent can
read under --bc-plus-docs. The official build_filename_docid_map.py is a
documentation stub (it says to produce this JSON with the exporter that made
the tree); this script is that exporter's counterpart for our Wiki corpus.

Reads run in a thread pool: the tree lives on NFS where per-file open latency,
not bandwidth, dominates; 64 threads turn a ~4h sequential walk into ~20min.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", type=Path, required=True)
    ap.add_argument("--parquet-out", type=Path, required=True)
    ap.add_argument("--map-out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--row-group-size", type=int, default=100_000)
    args = ap.parse_args()

    names = sorted(e.name for e in os.scandir(args.docs_dir)
                   if e.name.endswith(".md"))
    if not names:
        raise SystemExit(f"no .md files under {args.docs_dir}")
    print(f"[parquet] {len(names):,} documents in {args.docs_dir}", flush=True)

    relpath_to_docid = {name: name[:-3] for name in names}
    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_map = args.map_out.with_suffix(".tmp")
    tmp_map.write_text(json.dumps({"relpath_to_docid": relpath_to_docid}),
                       encoding="utf-8")
    tmp_map.rename(args.map_out)
    print(f"[parquet] filename map -> {args.map_out}", flush=True)

    def read_one(name: str) -> str:
        return (args.docs_dir / name).read_text(encoding="utf-8", errors="replace")

    schema = pa.schema([("docid", pa.string()), ("text", pa.string())])
    args.parquet_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = args.parquet_out.with_suffix(".parquet.tmp")
    t0 = time.time()
    with pq.ParquetWriter(tmp_parquet, schema, compression="zstd") as writer, \
         ThreadPoolExecutor(max_workers=args.threads) as pool:
        for lo in range(0, len(names), args.row_group_size):
            chunk = names[lo:lo + args.row_group_size]
            texts = list(pool.map(read_one, chunk))
            writer.write_table(pa.table(
                {"docid": [relpath_to_docid[n] for n in chunk], "text": texts},
                schema=schema))
            done = lo + len(chunk)
            rate = done / (time.time() - t0)
            print(f"[parquet] {done:,}/{len(names):,} ({rate:.0f} docs/s, "
                  f"eta {(len(names)-done)/rate/60:.0f}min)", flush=True)
    tmp_parquet.rename(args.parquet_out)
    print(f"[parquet] wrote {args.parquet_out} in {(time.time()-t0)/60:.1f}min",
          flush=True)


if __name__ == "__main__":
    main()
