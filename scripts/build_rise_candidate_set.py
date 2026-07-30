#!/usr/bin/env python3
"""Choose which Wiki articles RISE's offline sectioning must actually structure.

The problem
-----------
RISE's interaction space is a *structured* corpus: one LLM call per document
produces the line-numbered TOC/section anchors the agent navigates. The official
release ships that asset for BrowseComp-Plus (100,195 docs). Wiki-18 grouped to
article level is ~3.2M documents, so structuring all of it is ~3.2M LLM calls --
24-72 days on one 2-GPU server, which is not a tuning problem.

What this changes, and what it does not
---------------------------------------
It does NOT shrink the retrieval corpus. `build_wiki_toc.py --candidates` still
emits every article into the structured directory, so BM25 still ranks over the
full 3.2M and RISE's boundary is unchanged -- structuring a subset must never be
confused with evaluating against a smaller corpus, which would hand the method an
easier task. Only the *TOC* is restricted: articles outside the candidate set are
written through with the official `build_final(text, [], empty_reason=...)`, i.e.
the same shape the official core emits whenever sectioning finds nothing.

Selection rule
--------------
An article is a candidate if it contains any of the top-K BM25 passages for any
evaluated question:

    evaluated questions -> wiki-18 passage BM25 (the index already built for every
    other method) -> doc_id -> article_file via passage_article_mapping.jsonl

This reuses existing assets (no second index build) and targets the documents the
agent can actually reach: RISE previews `--bm25-top-n-preview 10` per sub-query
and reads selectively, so coverage is driven by the head of the ranking, not by
the full `--bm25-k 1000` pull.

Known limitation, to be reported
--------------------------------
The agent issues its own sub-queries at runtime, which are not knowable in
advance; this set is computed from the question text. An article the agent
surfaces via some other sub-query will be present but carry an empty TOC. Measure
it after the run rather than assuming it away: the audit log records
`empty_reason`, so the fraction of *opened* documents that lacked a TOC is
recoverable and belongs in the RISE row's limitations.

    python scripts/build_rise_candidate_set.py \
        --questions $DATA/dcilite_datasets/popqa_cap1500.jsonl \
        --index-dir $BM25_INDEX_DIR --mapping $DATA/rise_wiki_articles/passage_article_mapping.jsonl \
        --top-k 100 --out $DATA/rise_wiki_articles/candidates.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.bm25_retriever import BM25Retriever  # noqa: E402

# Observed aggregate decode throughput for Qwen3.5-9B on 2 GPUs (7267, 07-30).
_TOK_PER_S = 768.0
_TOK_PER_ARTICLE = 800.0   # conservative mean TOC output length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True,
                    help="Evaluated questions (dci-lite jsonl: query_id/query/answer)")
    ap.add_argument("--index-dir", type=Path, required=True,
                    help="Wiki-18 *passage* BM25 index (the shared one)")
    ap.add_argument("--mapping", type=Path, required=True,
                    help="passage_article_mapping.jsonl from build_wiki_article_corpus.py")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Passages retrieved per question before mapping to articles")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-articles", type=int, default=250_000,
                    help="Refuse if the candidate set exceeds this. A silent "
                         "multi-week sectioning job is the failure mode here.")
    args = ap.parse_args()

    questions = []
    with args.questions.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                questions.append((str(row.get("query_id") or row.get("id")),
                                  row.get("query") or row.get("question")))
    print(f"[cand] {len(questions):,} questions, top_k={args.top_k}", flush=True)

    retriever = BM25Retriever(str(args.index_dir))
    wanted: set[str] = set()
    t0 = time.time()
    for i, (_qid, q) in enumerate(questions, 1):
        for hit in retriever.retrieve(q, top_k=args.top_k, k1=args.k1, b=args.b):
            wanted.add(str(hit["doc_id"]))
        if i % 200 == 0:
            print(f"[cand] retrieved {i:,}/{len(questions):,} "
                  f"({len(wanted):,} unique passages)", flush=True)
    print(f"[cand] {len(wanted):,} unique passages in {time.time()-t0:.0f}s", flush=True)

    # Stream the 21M-row mapping once and keep only the articles we need; a full
    # doc_id -> article dict would be several GB for no reason.
    articles: set[str] = set()
    seen = 0
    with args.mapping.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            if str(row["doc_id"]) in wanted:
                articles.add(row["article_file"])
    print(f"[cand] scanned {seen:,} mapping rows -> {len(articles):,} unique articles",
          flush=True)

    hours = len(articles) * _TOK_PER_ARTICLE / _TOK_PER_S / 3600
    print(f"[cand] projected sectioning cost: {len(articles):,} LLM calls "
          f"~= {hours:.1f} h on one 2-GPU server "
          f"({_TOK_PER_ARTICLE:.0f} tok/doc at {_TOK_PER_S:.0f} tok/s)", flush=True)

    ordered = sorted(articles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    manifest = {
        "questions": str(args.questions),
        "n_questions": len(questions),
        "passage_top_k": args.top_k,
        "bm25": {"k1": args.k1, "b": args.b},
        "n_unique_passages": len(wanted),
        "n_candidate_articles": len(articles),
        "candidates_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "projected_sectioning_hours": round(hours, 1),
        "note": ("Only these articles get an LLM-built TOC. Every other article is "
                 "still written into the structured corpus with an empty TOC, so the "
                 "BM25 retrieval boundary remains the full corpus."),
    }
    mpath = args.out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if len(articles) > args.max_articles:
        sys.exit(f"CANDIDATE-SET TOO LARGE: {len(articles):,} > {args.max_articles:,} "
                 f"(~{hours:.0f} h of sectioning). Lower --top-k and re-run; do not "
                 "raise --max-articles without deciding the time budget is acceptable.")


if __name__ == "__main__":
    main()
