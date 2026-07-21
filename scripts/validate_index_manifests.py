#!/usr/bin/env python3
"""Ensure BM25/E5/Qwen indexes were built from the identical corpus manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus_manifest.read_text())
    expected = corpus["corpus_manifest_id"]
    for path in args.index_manifest:
        item = json.loads(path.read_text())
        if item.get("corpus_manifest_id") != expected:
            raise SystemExit(f"{path}: corpus manifest mismatch")
        if item.get("count") != corpus["count"]:
            raise SystemExit(f"{path}: document count mismatch")
        if item.get("ordered_doc_ids_sha256") is not None and \
                item["ordered_doc_ids_sha256"] != corpus["ordered_doc_ids_sha256"]:
            raise SystemExit(f"{path}: ordered doc-ID hash mismatch")
        print(f"OK {item.get('backend', 'unknown')}: {path}")


if __name__ == "__main__":
    main()
