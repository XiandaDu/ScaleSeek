"""Dataset loaders for ScaleSeek evaluation.

All datasets normalize to the canonical schema:
    {"id": str, "question": str, "golden_answers": [str]}

Sources:
  - FlashRAG (7 benchmarks): RUC-NLPIR/FlashRAG_datasets
  - BRIGHT: xlangai/BRIGHT  — returned as (query, golden_doc_ids) pairs
  - BrowseComp: openai/browsecomp
  - Local JSONL: any file already downloaded by scripts/download_data.py
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Optional


FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
FLASHRAG_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"

FLASHRAG_DATASETS = {
    "nq":              ("nq",              "test"),
    "triviaqa":        ("triviaqa",        "test"),
    "popqa":           ("popqa",           "test"),
    "hotpotqa":        ("hotpotqa",        "dev"),
    "2wikimultihopqa": ("2wikimultihopqa", "dev"),
    "musique":         ("musique",         "dev"),
    "bamboogle":       ("bamboogle",       "test"),
}

ALL_DATASETS = list(FLASHRAG_DATASETS) + ["bright", "browsecomp", "browsecomp_plus"]

# Counts are gates, not sampling targets. Add a value only after pinning and
# verifying the complete upstream evaluation split.
# triviaqa pinned 2026-08-10 (phase 3 benchmark #2, Rorqual): FlashRAG revision
# bcafb8dd, raw sha256 8f0891dece17a6a6472c87bf0e8c555c80673853d8dc7780be4d47158ec85290.
EXPECTED_FULL_COUNTS = {"popqa": 14_267, "triviaqa": 11_313}
EXPECTED_NORMALIZED_SHA256 = {
    "popqa": "56269abde328a259e4e38a186941cfd755ab0d72d7fbd7a1e8801a8ea781bd42",
}


def _datasets_dir() -> Path:
    d = os.environ.get("DATASETS")
    if d:
        return Path(d)
    data = os.environ.get("DATA", "/data/rech/mofengra/data")
    return Path(data).parent / "datasets"


def _hf_cache() -> Optional[str]:
    return os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")


def _normalize_row(row: dict, prefix: str, idx: int) -> Optional[dict]:
    q = (row.get("question") or "").strip()
    if not q:
        return None
    # golden_answers > answers > answer (handles hotpotqa's singular field)
    golds = (row.get("golden_answers")
             or row.get("answers")
             or row.get("answer")
             or [])
    if isinstance(golds, str):
        golds = [golds]
    golds = [str(g).strip() for g in golds if str(g).strip()]
    if not golds:
        return None
    rid = str(row.get("id") or idx)
    if not rid.startswith(prefix):
        rid = f"{prefix}_{rid}"
    return {"id": rid, "question": q, "golden_answers": golds}


# Legacy directory names used by other projects on this server
_LOCAL_DIR_ALIASES: dict[str, str] = {
    "2wikimultihopqa": "2wiki",
    "triviaqa":        "trivialqa",
}

_LOCAL_NAME_OVERRIDES: dict[tuple[str, str], str] = {
    # (dataset_name, split) -> filename stem (relative to the dataset dir)
    ("hotpotqa",        "dev"):   "hotpot_dev",
    ("hotpotqa",        "train"): "hotpot_train",
    ("2wikimultihopqa", "dev"):   "2wiki_dev",
    ("triviaqa",        "test"):  "trivialqa_test",
}


def _resolve_local_path(datasets_dir: Path, name: str, split: str) -> Optional[Path]:
    """Try several filename conventions used by different projects."""
    # try canonical dir then legacy alias
    dirs = [datasets_dir / name]
    alias = _LOCAL_DIR_ALIASES.get(name)
    if alias:
        dirs.append(datasets_dir / alias)

    for d in dirs:
        if not d.exists():
            continue

        # check explicit overrides first (legacy naming from WXImpactRAG etc.)
        stem = _LOCAL_NAME_OVERRIDES.get((name, split))
        if stem:
            p = d / f"{stem}.jsonl"
            if p.exists():
                return p

        # canonical (from our download_data.py / FlashRAG naming)
        for filename in [f"{split}.jsonl", f"{name}_{split}.jsonl"]:
            p = d / filename
            if p.exists():
                return p

    return None


def load_local(name: str, split: Optional[str] = None,
               limit: Optional[int] = None, offset: int = 0) -> list[dict]:
    """Load from pre-downloaded JSONL in $DATASETS/<name>/."""
    datasets_dir = _datasets_dir()
    if split is None:
        _, split = FLASHRAG_DATASETS.get(name, (name, "test"))
    path = _resolve_local_path(datasets_dir, name, split)
    if path is None:
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex = _normalize_row(obj, name, i)
            if ex:
                rows.append(ex)
    return rows


def load_flashrag(name: str, split: Optional[str] = None,
                  limit: Optional[int] = None, offset: int = 0) -> list[dict]:
    """Download and load from HuggingFace (RUC-NLPIR/FlashRAG_datasets)."""
    from datasets import load_dataset as hf_load

    if name not in FLASHRAG_DATASETS:
        raise ValueError(f"Unknown FlashRAG dataset: {name!r}")
    subfolder, default_split = FLASHRAG_DATASETS[name]
    split = split or default_split
    cache = _hf_cache()

    ds = hf_load(
        FLASHRAG_REPO,
        revision=FLASHRAG_REVISION,
        data_files={split: f"{subfolder}/{split}.jsonl"},
        split=split,
        cache_dir=cache,
    )
    rows = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if limit is not None and len(rows) >= limit:
            break
        ex = _normalize_row(dict(row), name, i)
        if ex:
            rows.append(ex)
    return rows


def load_dataset(
    name: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    """Load a dataset, preferring local cache then falling back to HuggingFace.

    Returns list of {"id", "question", "golden_answers"}.
    For BRIGHT, also includes "golden_doc_ids".
    """
    if name == "bright":
        return load_bright(limit=limit, offset=offset)
    if name == "browsecomp":
        rows = load_local("browsecomp", split="test", limit=limit, offset=offset)
        if not rows:
            raise FileNotFoundError(
                "BrowseComp not found. Download manually from "
                "https://github.com/openai/simple-evals and place at "
                f"{_datasets_dir()}/browsecomp/test.jsonl"
            )
        return rows

    if name == "browsecomp_plus":
        path = _datasets_dir() / "browsecomp_plus" / "queries.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"BrowseComp-Plus not found at {path}. "
                "Run: python scripts/build_browsecomp_plus.py"
            )
        rows = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < offset:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if name not in FLASHRAG_DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Supported: {ALL_DATASETS}")

    rows = load_local(name, split=split, limit=limit, offset=offset)
    if rows:
        return rows
    return load_flashrag(name, split=split, limit=limit, offset=offset)


def validate_complete_dataset(name: str, rows: list[dict]) -> dict:
    """Reject duplicate IDs and known incomplete evaluation splits."""
    ids = [str(row.get("id", "")) for row in rows]
    if any(not rid for rid in ids):
        raise ValueError(f"{name}: empty normalized ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: {len(ids) - len(set(ids))} duplicate normalized IDs")
    expected = EXPECTED_FULL_COUNTS.get(name)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{name}: full split requires {expected:,} rows, got {len(rows):,}")
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ).encode()
    manifest = {
        "dataset": name,
        "count": len(rows),
        "unique_ids": len(set(ids)),
        "normalized_sha256": hashlib.sha256(payload).hexdigest(),
        "source": FLASHRAG_REPO if name in FLASHRAG_DATASETS else "local",
        "source_revision": FLASHRAG_REVISION if name in FLASHRAG_DATASETS else None,
        "split": FLASHRAG_DATASETS.get(name, (None, "test"))[1],
    }
    expected_hash = EXPECTED_NORMALIZED_SHA256.get(name)
    if expected_hash is not None and manifest["normalized_sha256"] != expected_hash:
        raise ValueError(f"{name}: normalized SHA256 differs from the pinned full split")
    return manifest


def load_bright(tasks: Optional[list[str]] = None,
                limit: Optional[int] = None, offset: int = 0) -> list[dict]:
    """Load BRIGHT queries from local cache (all tasks combined)."""
    datasets_dir = _datasets_dir()
    bright_dir = datasets_dir / "bright"
    if not bright_dir.exists():
        raise FileNotFoundError(
            f"BRIGHT not found at {bright_dir}. "
            "Run: python scripts/download_data.py --datasets bright"
        )
    rows = []
    for f in sorted(bright_dir.glob("*_queries.jsonl")):
        task = f.stem.replace("_queries", "")
        if tasks and task not in tasks:
            continue
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append(obj)
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


# (removed dead _load_browsecomp_hf_UNUSED — BrowseComp-Plus is wired via
#  scripts/build_browsecomp_plus.py which writes $DATASETS/browsecomp_plus/
#  {queries.jsonl,corpus.jsonl,qrels.json,doclen.json}; loaded above by name
#  "browsecomp_plus")
