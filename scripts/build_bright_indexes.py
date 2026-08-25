#!/usr/bin/env python3
"""Build one Lucene BM25 index per BRIGHT task.

BRIGHT retrieval is per-task by protocol: each of the 12 tasks ships its own
`<task>_documents.jsonl` and a query's golden_doc_ids only exist inside that
task's pool. Merging them into one index would (a) let a query retrieve
documents from an unrelated domain and (b) inflate the candidate pool ~10x,
making the numbers incomparable to any published BRIGHT result.

Writes, under --out-root:
    <task>/corpus/wiki_corpus.jsonl   the task's docs, unchanged
    <task>/corpus_manifest.json       checksum manifest
    <task>/index/                     Lucene index
    <task>/index_manifest.json        frozen-settings manifest

The corpus filename stays `wiki_corpus.jsonl` because build_bm25_index.py
looks for exactly that name; the content is BRIGHT's, not wiki.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bright-dir", required=True, help="$DATASETS/bright")
    ap.add_argument("--out-root", required=True, help="per-task index root")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks")
    args = ap.parse_args()

    bright = Path(args.bright_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    doc_files = sorted(bright.glob("*_documents.jsonl"))
    if not doc_files:
        sys.exit(f"no *_documents.jsonl under {bright}")

    summary = {}
    for df in doc_files:
        task = df.stem.replace("_documents", "")
        if args.tasks and task not in args.tasks:
            continue
        tdir = out_root / task
        corpus_dir = tdir / "corpus"
        index_dir = tdir / "index"
        manifest = tdir / "corpus_manifest.json"

        if index_dir.exists() and (tdir / "index_manifest.json").exists():
            print(f"[{task}] already built — skipping")
            summary[task] = "skipped"
            continue

        corpus_dir.mkdir(parents=True, exist_ok=True)
        target = corpus_dir / "wiki_corpus.jsonl"
        if not target.exists():
            shutil.copy(df, target)
        n_docs = sum(1 for _ in open(target, encoding="utf-8"))
        print(f"[{task}] {n_docs} docs -> {index_dir}")

        if not manifest.exists():
            subprocess.run([py, str(REPO / "scripts/build_corpus_manifest.py"),
                            "--corpus", str(target), "--out", str(manifest)],
                           check=True)
        subprocess.run([py, str(REPO / "scripts/build_bm25_index.py"),
                        "--corpus-dir", str(corpus_dir),
                        "--index-dir", str(tdir),
                        "--threads", str(args.threads),
                        "--corpus-manifest", str(manifest)], check=True)
        summary[task] = n_docs

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
