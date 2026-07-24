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


def build(fixtures_path: Path, out_dir: Path) -> dict:
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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    info = build(Path(args.fixtures), out_dir)
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
