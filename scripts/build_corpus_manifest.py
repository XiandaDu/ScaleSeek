#!/usr/bin/env python3
"""Stream a JSONL corpus once and write its content/doc-ID manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_hash = hashlib.sha256()
    ids_hash = hashlib.sha256()
    count = 0
    seen: set[str] = set()
    duplicates: list[str] = []
    with args.corpus.open("rb") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw_hash.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            doc_id = str(row.get("id", ""))
            if not doc_id:
                raise ValueError(f"empty doc id at line {line_no}")
            if doc_id in seen and len(duplicates) < 20:
                duplicates.append(doc_id)
            seen.add(doc_id)
            ids_hash.update(doc_id.encode())
            ids_hash.update(b"\n")
            count += 1
    if len(seen) != count:
        raise ValueError(f"duplicate doc IDs ({count - len(seen)}); examples={duplicates}")
    core = {"schema_version": 1, "corpus_path": str(args.corpus.resolve()),
            "count": count, "unique_doc_ids": len(seen),
            "corpus_sha256": raw_hash.hexdigest(),
            "ordered_doc_ids_sha256": ids_hash.hexdigest()}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    manifest = {**core, "corpus_manifest_id": hashlib.sha256(canonical).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
