#!/usr/bin/env python3
"""Build a Pyserini/Lucene BM25 index over the wiki-18 corpus.

Run once before eval:
    source setup_env.sh
    python scripts/build_bm25_index.py

The corpus ($CORPUS_DIR/wiki_corpus.jsonl) is already in Pyserini's
JsonCollection format: one JSON object per line with "id" and "contents".
No preprocessing needed.

Building the full 21 M-passage index takes ~20-40 min with 16 threads.
The resulting index is ~15 GB.

Options:
    --threads N     indexing threads (default 16)
    --limit N       index only first N passages (for quick smoke-test)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _check_java() -> None:
    if shutil.which("java") is None:
        sys.exit(
            "Java not found. Install with:\n"
            "  conda install -c conda-forge openjdk=21"
        )
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    ver = result.stderr or result.stdout
    print(f"  java: {ver.splitlines()[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR"),
                        help="Directory containing wiki_corpus.jsonl ($CORPUS_DIR)")
    parser.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"),
                        help="Where to write the Lucene index ($BM25_INDEX_DIR)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only first N passages (smoke-test)")
    args = parser.parse_args()

    if not args.corpus_dir:
        sys.exit("Set $CORPUS_DIR or pass --corpus-dir")
    if not args.index_dir:
        sys.exit("Set $BM25_INDEX_DIR or pass --index-dir")

    corpus_dir = Path(args.corpus_dir)
    index_dir = Path(args.index_dir)
    corpus_file = corpus_dir / "wiki_corpus.jsonl"

    if not corpus_file.exists():
        sys.exit(f"Corpus not found: {corpus_file}")

    if (index_dir / "index").exists():
        print(f"Index already exists at {index_dir}/index. Delete to rebuild.")
        return

    print("Checking Java ...")
    _check_java()

    # If --limit, write a truncated copy to a temp dir for indexing
    if args.limit:
        print(f"--limit {args.limit}: writing truncated corpus to temp dir ...")
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_file = tmp_dir / "wiki_corpus.jsonl"
        with open(corpus_file, encoding="utf-8") as fin, \
             open(tmp_file, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= args.limit:
                    break
                fout.write(line)
        input_dir = tmp_dir
        print(f"  wrote {args.limit:,} passages → {tmp_file}")
    else:
        input_dir = corpus_dir

    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",       str(input_dir),
        "--index",       str(index_dir / "index"),
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",     str(args.threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",     # required to retrieve passage text at query time
    ]

    print(f"\nRunning Pyserini indexer:")
    print("  " + " ".join(cmd) + "\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if args.limit:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        sys.exit(f"Indexing failed (exit {result.returncode})")

    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  index : {index_dir}/index")

    # Quick sanity check
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(str(index_dir / "index"))
        n = searcher.num_docs
        print(f"  docs  : {n:,}")
    except Exception as e:
        print(f"  (sanity check failed: {e})")


if __name__ == "__main__":
    main()
