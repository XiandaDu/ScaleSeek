#!/usr/bin/env python3
"""Write an index manifest for an index built before the manifest contract.

The manifest echoes the corpus manifest ID so validate_index_manifests.py can
prove all retriever backends share the identical corpus/doc-ID snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def dense_stats(index_dir: Path) -> tuple[int, str]:
    import faiss

    doc_ids = index_dir / "doc_ids.txt"
    ids_hash = hashlib.sha256()
    count = 0
    with doc_ids.open("rb") as fh:
        for raw in fh:
            doc_id = raw.rstrip(b"\n")
            if not doc_id:
                continue
            ids_hash.update(doc_id)
            ids_hash.update(b"\n")
            count += 1
    index = faiss.read_index(str(index_dir / "index.faiss"),
                             faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    if index.ntotal != count:
        raise SystemExit(f"index.faiss has {index.ntotal:,} vectors but "
                         f"doc_ids.txt has {count:,} IDs")
    return count, ids_hash.hexdigest()


def bm25_stats(index_dir: Path) -> tuple[int, None]:
    from pyserini.index.lucene import LuceneIndexReader

    reader = LuceneIndexReader(str(index_dir))
    return reader.stats()["documents"], None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["bm25", "e5", "qwen3_emb_4b"], required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus_manifest.read_text())
    if args.backend == "bm25":
        count, ordered_sha = bm25_stats(args.index_dir)
    else:
        count, ordered_sha = dense_stats(args.index_dir)
    if count != corpus["count"]:
        raise SystemExit(f"{args.backend}: index has {count:,} docs but corpus "
                         f"manifest has {corpus['count']:,}")
    if ordered_sha is not None and ordered_sha != corpus["ordered_doc_ids_sha256"]:
        raise SystemExit(f"{args.backend}: ordered doc-ID hash differs from corpus manifest")

    # Emit exactly the frozen-contract fields that eval.run_eval's
    # validate_retriever_manifest checks (mirrors build_e5_index.py's schema).
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "configs" / "baselines.yaml").read_text())["global"]
    if args.backend == "bm25":
        contract = {"backend": "bm25_lucene", **cfg["fixed_bm25"]}
    else:
        rcfg = cfg["retrievers"][args.backend]
        contract = {
            "backend": args.backend,
            "model_id": rcfg["repo_id"],
            "model_revision": rcfg["revision"],
            "pooling": rcfg["pooling"],
            "normalize": True,
            "max_length": rcfg["max_length"],
        }
    manifest = {
        "schema_version": 1,
        **contract,
        "index_dir": str(args.index_dir.resolve()),
        "corpus_manifest_id": corpus["corpus_manifest_id"],
        "count": count,
        "ordered_doc_ids_sha256": ordered_sha,
        "backfilled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
