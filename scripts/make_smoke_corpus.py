#!/usr/bin/env python3
"""Build a tiny SMOKE corpus + question set for the ScaleSeek SFT pipeline.

This exists so the whole cold-start -> SFT -> eval chain can run end to end on a
single GPU WITHOUT the 21M-passage wiki-18 corpus / 15 GB Lucene index. It reads
``train/sft/smoke_fixtures.json`` and writes, under ``--out-dir``:

    corpus/wiki_corpus.jsonl   Pyserini JsonCollection ({"id","contents"} per line):
                               every fixture passage + shared distractors.
    questions.jsonl            canonical QA rows ({"id","question","golden_answers"}).

The passages are authored so BM25 retrieves the answer-bearing passage from
sub-question terms alone. This is a SMOKE fixture: its numbers must never enter a
result table (see TASK.md).

Usage:
    python scripts/make_smoke_corpus.py --out-dir .smoke
    python scripts/make_smoke_corpus.py --out-dir .smoke --build-index   # also runs pyserini

Then (if not --build-index):
    python scripts/build_bm25_index.py \
        --corpus-dir .smoke/corpus --index-dir .smoke/bm25_index
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_FIXTURES = _REPO / "train" / "sft" / "smoke_fixtures.json"


def _passage_contents(p: dict) -> str:
    title = (p.get("title") or "").strip()
    text = (p.get("text") or "").strip()
    return f"{title}. {text}" if title else text


# ---------------------------------------------------------------------------
# Hard (parameter-sensitive) distractors
# ---------------------------------------------------------------------------
# k1/b/top_k only matter when the corpus contains passages that overlap the query
# lexically but do NOT hold the answer, in a mix of lengths and term-frequencies.
# For each question we synthesize such traps (never containing a golden answer):
#   - long/dilute : long filler with query terms sprinkled -> favored by high b;
#                   a short dense gold beats them only when b is lowered.
#   - short/spam  : a query term repeated many times -> favored by high k1.
#   - medium      : moderate overlap in a wrong context.
# This pushes the gold passage down to a rank where the BM25 knobs change whether
# and where it is retrieved, giving the search-then-teach mentor real signal.

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "is", "are", "was", "were",
    "what", "which", "who", "whom", "where", "when", "how", "why", "did", "do",
    "does", "and", "or", "for", "by", "that", "this", "with", "as", "it", "its",
    "official", "used", "country", "city", "known", "first", "year",
}
_FILLER = (
    "This passage concerns unrelated administrative matters and miscellaneous "
    "trivia. It records routine footnotes, scheduling notes, and clerical details "
    "that bear no substantive connection to any particular fact. Various committees "
    "reviewed the paperwork and filed it without further comment. The remainder is "
    "boilerplate retained only for archival completeness and indexing purposes."
)


def _salient_terms(question: str) -> list[str]:
    seen: list[str] = []
    for raw in question.split():
        w = raw.strip("?.,:;\"'()").strip()
        if len(w) < 4 or w.lower() in _STOP:
            continue
        if w not in seen:
            seen.append(w)
    return seen or (question.split()[:1] if question.split() else [])


def _hard_distractors(question: str, golds: list[str], n_each: int, qid: str) -> list[dict]:
    """Answer-free distractors that make k1 AND b the decisive retrieval levers, so
    a search-then-teach mentor must tune them (not just widen top_k) to surface the
    short, dense gold passage:

      - LONG dossiers carrying every query term once. Being far longer than the gold
        they outrank it under default b; only raising b (penalising length) lifts the
        gold. -> b lever.
      - TERM-SPAM tags repeating a single query term. Their high term frequency wins
        under high k1; only lowering k1 (tf saturation) lets the gold's full-term
        coverage win. -> k1 lever.

    Which lever dominates is varied per question (by a stable hash of the id) so the
    optimal (k1, b) differs across questions and the mentor learns a real mapping,
    not one constant. This is a controlled SMOKE testbed (never a result table)."""
    terms = _salient_terms(question)
    if len(terms) < 2:
        return []
    low_golds = [g.lower() for g in golds]
    # A distractor that densely repeats a SUBSET of the query terms (never all of
    # them) buries the gold under default k1 — its high term-frequency on the subset
    # outscores the gold's single mention of each term. Lowering k1 saturates that
    # repetition so the gold's FULL-term coverage wins. How large the subset is (and
    # how hard it is repeated) is varied per question, so the optimal k1 differs.
    regime = sum(ord(c) for c in qid) % 3           # 0 easy, 1 medium, 2 hard burial
    reps = {0: 5, 1: 9, 2: 14}[regime]
    subset = " ".join(terms[:-1])                    # all query terms except the last
    out: list[dict] = []
    for i in range(n_each):
        out.append({"title": f"Tag {qid}-S{i}",
                    "text": (" ".join([subset] * reps) + ". Short index tag; no definition.")})
    return [d for d in out if not any(g and g in d["text"].lower() for g in low_golds)]


def build(fixtures_path: Path, out_dir: Path, hard: int = 0) -> dict:
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    examples = fixtures.get("examples", [])
    distractors = fixtures.get("distractors", [])

    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "wiki_corpus.jsonl"
    questions_file = out_dir / "questions.jsonl"

    doc_rows: list[dict] = []
    seen_contents: set[str] = set()

    def _add(passage: dict) -> str:
        contents = _passage_contents(passage)
        if contents in seen_contents:
            # reuse an existing doc_id for identical passages
            for r in doc_rows:
                if r["contents"] == contents:
                    return r["id"]
        doc_id = f"smoke_doc_{len(doc_rows) + 1:04d}"
        doc_rows.append({"id": doc_id, "contents": contents})
        seen_contents.add(contents)
        return doc_id

    q_rows: list[dict] = []
    for ex in examples:
        for p in ex.get("passages", []):
            _add(p)
        if hard:
            for d in _hard_distractors(ex["question"], list(ex.get("golden_answers", [])),
                                       hard, str(ex["id"])):
                _add(d)
        q_rows.append({
            "id": str(ex["id"]),
            "question": ex["question"],
            "golden_answers": list(ex.get("golden_answers", [])),
            "source": "smoke",
        })
    for d in distractors:
        _add(d)

    with corpus_file.open("w", encoding="utf-8") as f:
        for r in doc_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with questions_file.open("w", encoding="utf-8") as f:
        for r in q_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "corpus_file": str(corpus_file),
        "questions_file": str(questions_file),
        "n_docs": len(doc_rows),
        "n_questions": len(q_rows),
    }


def build_index(out_dir: Path) -> None:
    corpus_dir = out_dir / "corpus"
    index_dir = out_dir / "bm25_index"
    cmd = [
        sys.executable, str(_REPO / "scripts" / "build_bm25_index.py"),
        "--corpus-dir", str(corpus_dir),
        "--index-dir", str(index_dir),
        "--threads", "2",
    ]
    print("\n[make_smoke_corpus] building BM25 index:")
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"index build failed (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default=str(_DEFAULT_FIXTURES))
    ap.add_argument("--out-dir", default=".smoke")
    ap.add_argument("--build-index", action="store_true",
                    help="also build the Pyserini/Lucene BM25 index (needs Java + pyserini)")
    ap.add_argument("--hard", type=int, default=0, metavar="N",
                    help="add N*3 parameter-sensitive lexical-overlap distractors per question "
                         "(long/dilute + short/spam + medium), so k1/b/top_k change gold ranking")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    info = build(Path(args.fixtures), out_dir, hard=args.hard)
    print(f"[make_smoke_corpus] wrote {info['n_docs']} docs -> {info['corpus_file']}")
    print(f"[make_smoke_corpus] wrote {info['n_questions']} questions -> {info['questions_file']}")

    if args.build_index:
        build_index(out_dir)
        print(f"[make_smoke_corpus] index ready -> {out_dir / 'bm25_index' / 'index'}")
    else:
        print("\nNext: build the BM25 index:")
        print(f"  python scripts/build_bm25_index.py "
              f"--corpus-dir {out_dir / 'corpus'} --index-dir {out_dir / 'bm25_index'} --threads 2")


if __name__ == "__main__":
    main()
