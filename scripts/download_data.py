#!/usr/bin/env python3
"""Download and cache all ScaleSeek evaluation datasets.

FlashRAG datasets (7 standard benchmarks):
    nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle
    Source: RUC-NLPIR/FlashRAG_datasets on HuggingFace
    Format already matches ScaleSeek canonical: {id, question, golden_answers}

Additional benchmarks:
    bright       — xlangai/BRIGHT  (retrieval-intensive, multi-domain)
    browsecomp   — openai/browsecomp (complex web search questions)

Usage:
    source setup_env.sh
    python scripts/download_data.py               # download all
    python scripts/download_data.py --datasets nq triviaqa hotpotqa
    python scripts/download_data.py --verify      # check what's already cached
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"

FLASHRAG_DATASETS = {
    "nq":              ("nq",              "test"),
    "triviaqa":        ("triviaqa",        "test"),
    "popqa":           ("popqa",           "test"),
    "hotpotqa":        ("hotpotqa",        "dev"),
    "2wikimultihopqa": ("2wikimultihopqa", "dev"),
    "musique":         ("musique",         "dev"),
    "bamboogle":       ("bamboogle",       "test"),
}

# BRIGHT: retrieval benchmark with query/relevant-doc pairs.
# We download the task query files and treat them as QA queries where applicable.
BRIGHT_REPO = "xlangai/BRIGHT"
BRIGHT_TASKS = [
    "biology", "earth_science", "economics", "psychology",
    "robotics", "stackoverflow", "sustainable_living",
    "leetcode", "pony", "aops", "theoremqa_theorems", "theoremqa_questions",
]

BROWSECOMP_REPO = "openai/browsecomp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _datasets_dir() -> Path:
    d = os.environ.get("DATASETS") or os.path.join(
        os.environ.get("DATA", "/data/rech/mofengra/data"),
        "../../datasets",
    )
    return Path(d).resolve()


def _hf_cache() -> Optional[str]:
    return os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved {len(rows):,} rows → {path}")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# FlashRAG download
# ---------------------------------------------------------------------------

def download_flashrag(name: str, datasets_dir: Path, cache_dir: Optional[str]) -> Path:
    from datasets import load_dataset as hf_load

    subfolder, split = FLASHRAG_DATASETS[name]
    out_path = datasets_dir / name / f"{split}.jsonl"

    if out_path.exists():
        existing = _load_jsonl(out_path)
        print(f"  [{name}] already exists ({len(existing):,} rows) → {out_path}")
        return out_path

    print(f"  [{name}] downloading from {FLASHRAG_REPO}/{subfolder}/{split}.jsonl ...")
    ds = hf_load(
        FLASHRAG_REPO,
        data_files={split: f"{subfolder}/{split}.jsonl"},
        split=split,
        cache_dir=cache_dir,
    )

    rows = []
    for i, row in enumerate(ds):
        q = (row.get("question") or "").strip()
        if not q:
            continue
        golds = row.get("golden_answers") or row.get("answers") or []
        if isinstance(golds, str):
            golds = [golds]
        golds = [str(g).strip() for g in golds if str(g).strip()]
        if not golds:
            continue
        rid = str(row.get("id") or i)
        rows.append({"id": f"{name}_{rid}", "question": q, "golden_answers": golds})

    _save_jsonl(rows, out_path)
    return out_path


# ---------------------------------------------------------------------------
# BRIGHT download
# ---------------------------------------------------------------------------

def download_bright(datasets_dir: Path, cache_dir: Optional[str]) -> None:
    """Download BRIGHT retrieval benchmark.

    BRIGHT provides (query, relevant_doc_ids) pairs. We save:
      - queries as {id, question} JSONL per task
      - documents as {id, contents} JSONL per task (same format as wiki corpus)
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("  [bright] skip — datasets not installed")
        return

    bright_dir = datasets_dir / "bright"
    bright_dir.mkdir(parents=True, exist_ok=True)

    for task in BRIGHT_TASKS:
        query_path = bright_dir / f"{task}_queries.jsonl"
        if query_path.exists():
            n = sum(1 for _ in open(query_path))
            print(f"  [bright/{task}] already exists ({n} queries)")
            continue

        print(f"  [bright/{task}] downloading ...")
        try:
            # examples split has queries with relevance labels
            ds = hf_load(BRIGHT_REPO, task, split="examples", cache_dir=cache_dir)
            rows = []
            for row in ds:
                qid = str(row.get("id", ""))
                q = (row.get("query") or "").strip()
                if not q:
                    continue
                relevant = row.get("gold_ids") or row.get("relevant") or []
                rows.append({
                    "id": f"bright_{task}_{qid}",
                    "question": q,
                    "golden_doc_ids": list(relevant),
                })
            _save_jsonl(rows, query_path)

            # documents
            doc_path = bright_dir / f"{task}_documents.jsonl"
            if not doc_path.exists():
                docs_ds = hf_load(BRIGHT_REPO, task, split="documents", cache_dir=cache_dir)
                doc_rows = []
                for doc in docs_ds:
                    did = str(doc.get("id", ""))
                    content = (doc.get("content") or doc.get("text") or "").strip()
                    if content:
                        doc_rows.append({"id": did, "contents": content})
                _save_jsonl(doc_rows, doc_path)

        except Exception as e:
            print(f"  [bright/{task}] failed: {e}")


# ---------------------------------------------------------------------------
# BrowseComp download
# ---------------------------------------------------------------------------

def download_browsecomp(datasets_dir: Path, cache_dir: Optional[str]) -> None:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("  [browsecomp] skip — datasets not installed")
        return

    out_path = datasets_dir / "browsecomp" / "test.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"  [browsecomp] already exists ({n} rows)")
        return

    print(f"  [browsecomp] downloading from {BROWSECOMP_REPO} ...")
    try:
        ds = hf_load(BROWSECOMP_REPO, split="test", cache_dir=cache_dir)
        rows = []
        for i, row in enumerate(ds):
            q = (row.get("problem") or row.get("question") or "").strip()
            if not q:
                continue
            ans = row.get("answer") or row.get("solution") or ""
            golds = [ans.strip()] if isinstance(ans, str) and ans.strip() else []
            rows.append({
                "id": f"browsecomp_{i}",
                "question": q,
                "golden_answers": golds,
            })
        _save_jsonl(rows, out_path)
    except Exception as e:
        print(f"  [browsecomp] failed: {e}")
        print("  [browsecomp] try manually: https://huggingface.co/datasets/openai/browsecomp")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(datasets_dir: Path) -> None:
    print(f"\nDataset inventory in {datasets_dir}:\n")
    all_datasets = list(FLASHRAG_DATASETS) + ["bright", "browsecomp"]
    for name in all_datasets:
        d = datasets_dir / name
        if not d.exists():
            print(f"  {name:20s}  MISSING")
            continue
        files = list(d.glob("*.jsonl"))
        if not files:
            print(f"  {name:20s}  dir exists but no .jsonl files")
            continue
        total = 0
        for f in sorted(files):
            try:
                n = sum(1 for _ in open(f))
                total += n
                print(f"  {name:20s}  {f.name}  ({n:,} rows)")
            except Exception:
                print(f"  {name:20s}  {f.name}  (unreadable)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download ScaleSeek evaluation datasets")
    parser.add_argument(
        "--datasets", nargs="*",
        help="Which datasets to download (default: all). "
             f"FlashRAG: {', '.join(FLASHRAG_DATASETS)}. "
             "Extra: bright, browsecomp",
    )
    parser.add_argument("--verify", action="store_true",
                        help="Just print what is already downloaded and exit")
    parser.add_argument("--datasets-dir", default=None,
                        help="Override $DATASETS directory")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else _datasets_dir()
    cache_dir = _hf_cache()

    if args.verify:
        verify(datasets_dir)
        return

    requested = set(args.datasets) if args.datasets else (
        set(FLASHRAG_DATASETS) | {"bright", "browsecomp"}
    )

    print(f"Datasets directory : {datasets_dir}")
    print(f"HF cache           : {cache_dir or '(HF default)'}")
    print(f"Downloading        : {', '.join(sorted(requested))}\n")

    for name in sorted(FLASHRAG_DATASETS):
        if name not in requested:
            continue
        try:
            download_flashrag(name, datasets_dir, cache_dir)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}", file=sys.stderr)

    if "bright" in requested:
        download_bright(datasets_dir, cache_dir)

    if "browsecomp" in requested:
        download_browsecomp(datasets_dir, cache_dir)

    print("\nDone.")
    verify(datasets_dir)


if __name__ == "__main__":
    main()
