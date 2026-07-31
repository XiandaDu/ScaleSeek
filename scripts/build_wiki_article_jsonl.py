#!/usr/bin/env python3
"""Reconstruct ARTICLE-level wiki-18 as a single JsonCollection jsonl.

Same deterministic grouping as build_wiki_article_corpus.py (exact first-line
title equality, passage order = corpus ordinal, article order = casefolded
title) but materialized as ONE jsonl for Pyserini indexing instead of millions
of per-article .md files — the .md layout is RISE's file-navigation interface
and would blow /scratch's 1M-file quota (wiki-18 has ~millions of titles).

Why this corpus exists (2026-07-30): the SFT/RL training mix runs on wiki-18
passages (~102 words, uniform), where k1/b are measured-dead levers. The user
requires the training distribution to also contain a long-document regime so
the parameter-adaptation capability has non-zero support ("效果可以不好，但你
不能没有"), without touching any eval-reserved dataset (BCP/BRIGHT/PopQA/
TriviaQA/bamboogle). Article-level wiki-18 is that regime: same content, same
training questions, documents ranging from one passage to hundreds — length
variance gives b something to normalize and repeated mentions give k1 something
to saturate (cf. Pi-Serini's finding on ~2k-token documents).

Output:
    <out>/wiki_corpus.jsonl        {"id": "art_<sha1-16>", "contents": "<title>\\n<text>"}
                                   (filename kept as wiki_corpus.jsonl so
                                    build_bm25_index.py works unchanged)
    <out>/passage_article_mapping.jsonl   passage doc_id -> article id
    <out>/article_stats.json       length distribution (words)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="wiki-18 passage jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True,
                    help="scratch SQLite path (node-local $SLURM_TMPDIR recommended)")
    args = ap.parse_args()
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

    out_jsonl = args.out / "wiki_corpus.jsonl"
    mapping_path = args.out / "passage_article_mapping.jsonl"
    n_articles = n_passages = 0
    lengths: list[int] = []
    with out_jsonl.open("w", encoding="utf-8") as out, \
         mapping_path.open("w", encoding="utf-8") as mapping:
        titles = db.execute("SELECT DISTINCT title FROM passages ORDER BY lower(title), title")
        for (title,) in titles:
            art_id = "art_" + hashlib.sha256(title.encode()).hexdigest()[:16]
            rows = list(db.execute(
                "SELECT ordinal, doc_id, body FROM passages WHERE title=? ORDER BY ordinal",
                (title,)))
            body = "\n".join(b.rstrip() for _, _, b in rows)
            out.write(json.dumps({"id": art_id, "contents": f"{title}\n{body}"},
                                 ensure_ascii=False) + "\n")
            for passage_index, (ordinal, doc_id, _) in enumerate(rows):
                mapping.write(json.dumps(
                    {"doc_id": doc_id, "corpus_ordinal": ordinal, "title": title,
                     "article_id": art_id, "article_passage_index": passage_index},
                    ensure_ascii=False) + "\n")
                n_passages += 1
            lengths.append(len(body.split()))
            n_articles += 1

    lengths.sort()
    stats = {
        "articles": n_articles, "passages": n_passages,
        "words_p5": lengths[len(lengths) // 20],
        "words_median": lengths[len(lengths) // 2],
        "words_p95": lengths[-max(len(lengths) // 20, 1)],
        "words_max": lengths[-1],
    }
    (args.out / "article_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
