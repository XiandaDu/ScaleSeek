#!/usr/bin/env python3
"""Build a bm25s index over the wiki-18 corpus.

Run once before eval:
    source setup_env.sh
    python scripts/build_bm25_index.py

The index is saved to $BM25_INDEX_DIR and loaded on-demand by
eval/bm25_retriever.py. Building takes ~30-60 min and ~20 GB RAM for
the full 21 M-passage corpus.

Options:
    --limit N       index only the first N passages (for quick testing)
    --batch-size B  tokenization batch size (default 50000)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", default=os.environ.get("CORPUS_FILE"),
                        help="Path to wiki_corpus.jsonl")
    parser.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"),
                        help="Directory to save the index")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of passages (for testing)")
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    if not args.corpus_file:
        sys.exit("Set $CORPUS_FILE or pass --corpus-file")
    if not args.index_dir:
        sys.exit("Set $BM25_INDEX_DIR or pass --index-dir")

    corpus_file = Path(args.corpus_file)
    index_dir = Path(args.index_dir)

    if not corpus_file.exists():
        sys.exit(f"Corpus not found: {corpus_file}")

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index"
    id_map_path = index_dir / "doc_ids.txt"

    if index_path.exists() and id_map_path.exists():
        print(f"Index already exists at {index_dir}. Delete to rebuild.")
        return

    try:
        import bm25s
    except ImportError:
        sys.exit("Install bm25s:  pip install bm25s")

    import json

    print(f"Reading corpus: {corpus_file}")
    t0 = time.perf_counter()

    doc_ids: list[str] = []
    corpus: list[str] = []

    with open(corpus_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_ids.append(str(obj["id"]))
            corpus.append(obj["contents"])
            if (i + 1) % 1_000_000 == 0:
                print(f"  read {i+1:,} passages ({time.perf_counter()-t0:.0f}s)")

    print(f"Total passages: {len(corpus):,}  ({time.perf_counter()-t0:.1f}s)")

    print("Tokenizing ...")
    t1 = time.perf_counter()
    tokenized = bm25s.tokenize(corpus, stopwords="en", show_progress=True)
    print(f"Tokenized in {time.perf_counter()-t1:.1f}s")

    print("Building BM25 index ...")
    t2 = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=True)
    print(f"Indexed in {time.perf_counter()-t2:.1f}s")

    print(f"Saving index to {index_dir} ...")
    retriever.save(str(index_path), corpus=corpus)

    with open(id_map_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_ids))

    total = time.perf_counter() - t0
    print(f"Done. Total time: {total:.0f}s")
    print(f"  index  : {index_path}")
    print(f"  id map : {id_map_path}  ({len(doc_ids):,} docs)")


if __name__ == "__main__":
    main()
