#!/usr/bin/env python3
"""Deterministically reconstruct article files from Wiki-18 passages for RISE.

Exact title equality groups passages. Passage order is the original corpus line
order; articles are exported by ``(title.casefold(), title)``. No text is
summarized or rewritten. The output mapping preserves every passage/doc ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True,
                        help="Temporary/reusable SQLite grouping database")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS passages "
               "(title TEXT, ordinal INTEGER PRIMARY KEY, doc_id TEXT, body TEXT)")
    if db.execute("SELECT COUNT(*) FROM passages").fetchone()[0] == 0:
        batch = []
        with args.corpus.open(encoding="utf-8") as fh:
            for ordinal, line in enumerate(fh):
                row = json.loads(line)
                content = str(row.get("contents", ""))
                title, sep, body = content.partition("\n")
                if not sep:
                    title, body = content.strip() or f"untitled-{ordinal}", content
                batch.append((title, ordinal, str(row["id"]), body))
                if len(batch) == 10_000:
                    db.executemany("INSERT INTO passages VALUES (?,?,?,?)", batch)
                    db.commit(); batch = []
        if batch:
            db.executemany("INSERT INTO passages VALUES (?,?,?,?)", batch); db.commit()
    db.execute("CREATE INDEX IF NOT EXISTS passages_title ON passages(title)")
    db.commit()
    mapping_path = args.out / "passage_article_mapping.jsonl"
    article_hash = hashlib.sha256()
    n_articles = n_passages = 0
    with mapping_path.open("w", encoding="utf-8") as mapping:
        titles = db.execute("SELECT DISTINCT title FROM passages ORDER BY lower(title), title")
        for (title,) in titles:
            slug = hashlib.sha256(title.encode()).hexdigest() + ".md"
            parts = [f"# {title}\n"]
            rows = list(db.execute(
                "SELECT ordinal, doc_id, body FROM passages WHERE title=? ORDER BY ordinal", (title,)))
            for passage_index, (ordinal, doc_id, body) in enumerate(rows):
                parts.append("\n" + body.rstrip() + "\n")
                mapping.write(json.dumps({"doc_id": doc_id, "corpus_ordinal": ordinal,
                                          "title": title, "article_file": slug,
                                          "article_passage_index": passage_index},
                                         ensure_ascii=False) + "\n")
                n_passages += 1
            raw = "".join(parts).encode()
            (args.out / slug).write_bytes(raw)
            article_hash.update(slug.encode() + b"\0" + raw)
            n_articles += 1
    corpus_manifest = json.loads(args.corpus_manifest.read_text())
    if n_passages != corpus_manifest["count"]:
        raise SystemExit("Passage/article mapping count differs from corpus manifest")
    manifest = {"schema_version": 1, "corpus_manifest_id": corpus_manifest["corpus_manifest_id"],
                "article_count": n_articles, "passage_count": n_passages,
                "grouping": "exact first-line title", "passage_order": "source corpus ordinal",
                "article_order": "(title.casefold(), title)",
                "article_content_sha256": article_hash.hexdigest(),
                "mapping": mapping_path.name}
    (args.out / "article_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
