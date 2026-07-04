"""Gold / qrel document sets for retrieval-metric scoring.

Two responsibilities:

1. A one-time **corpus title index** over wiki_corpus.jsonl: maps each article
   title -> the corpus doc-ids of its passages. Backed by sqlite so it can be
   queried with low memory (the 21M-passage corpus never has to sit in RAM).
   Build it once (heavy; run from scripts/compute_metrics.py --build-index or the
   RUNBOOK), then query the few gold titles per example.

2. Per-dataset **gold-title loaders**. wiki-18 QA datasets label gold evidence at
   *title* granularity, so we score Gold R@W / coverage at title granularity:
   a gold title counts as recalled if the agent surfaced >=1 corpus passage of it.
       - HotpotQA: supporting_facts titles from hotpot_dev_distractor_v1.json.
       - 2Wiki:    metadata.supporting_facts.title from 2wiki_dev.jsonl.
   BrowseComp-Plus ships native doc-id qrels (gold + evidence); those are scored in
   doc-id space (no title index needed) and wired in scripts/build_browsecomp_plus.py.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Title normalization + corpus title extraction
# ---------------------------------------------------------------------------

def normalize_title(t: str) -> str:
    t = (t or "").strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    t = t.replace("_", " ").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def corpus_title_of(contents: str) -> str:
    """wiki_corpus contents are '"<title>"\\n<passage text>' — take the 1st line."""
    nl = contents.find("\n")
    first = contents if nl < 0 else contents[:nl]
    return normalize_title(first)


# ---------------------------------------------------------------------------
# Corpus title index (sqlite)
# ---------------------------------------------------------------------------

def build_corpus_title_index(
    corpus_path: str, db_path: str, *, limit: Optional[int] = None,
    batch: int = 100_000,
) -> None:
    """Stream wiki_corpus.jsonl -> sqlite table (title_norm, doc_id), indexed on
    title_norm. One-time; ~21M rows. Safe to re-run (rebuilds from scratch)."""
    db_path = str(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE t (title_norm TEXT, doc_id TEXT)")
    rows, n = [], 0
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append((corpus_title_of(obj.get("contents", "")), str(obj.get("id"))))
            if len(rows) >= batch:
                con.executemany("INSERT INTO t VALUES (?,?)", rows)
                n += len(rows)
                rows = []
    if rows:
        con.executemany("INSERT INTO t VALUES (?,?)", rows)
        n += len(rows)
    con.commit()
    con.execute("CREATE INDEX idx_title ON t(title_norm)")
    con.commit()
    con.close()
    print(f"[qrels] built corpus title index: {n:,} passages -> {db_path}")


class CorpusTitleIndex:
    """Query wrapper: normalized title -> set of corpus doc-ids (with a cache)."""

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"corpus title index not found at {db_path}. Build it first:\n"
                "  python scripts/compute_metrics.py --build-title-index "
                "--corpus-path $CORPUS_FILE --title-index-db <db>"
            )
        self._con = sqlite3.connect(db_path)
        self._cache: dict[str, set[str]] = {}

    def docids_for_title(self, title_norm: str) -> set[str]:
        if title_norm in self._cache:
            return self._cache[title_norm]
        cur = self._con.execute("SELECT doc_id FROM t WHERE title_norm=?", (title_norm,))
        ids = {r[0] for r in cur.fetchall()}
        self._cache[title_norm] = ids
        return ids


# ---------------------------------------------------------------------------
# Per-dataset gold-title loaders  (keyed by the RAW dataset id)
# ---------------------------------------------------------------------------

def _datasets_dir() -> Path:
    d = os.environ.get("DATASETS")
    if d:
        return Path(d)
    data = os.environ.get("DATA", "/data/rech/mofengra/data")
    return Path(data).parent / "datasets"


def load_hotpotqa_gold_titles(datasets_dir: Optional[Path] = None) -> dict[str, list[str]]:
    datasets_dir = datasets_dir or _datasets_dir()
    path = datasets_dir / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    if not path.exists():
        return {}
    data = json.load(open(path, encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for ex in data:
        rid = str(ex.get("_id"))
        titles = sorted({normalize_title(t) for t, _ in ex.get("supporting_facts", [])})
        out[rid] = titles
    return out


def load_2wiki_gold_titles(datasets_dir: Optional[Path] = None) -> dict[str, list[str]]:
    datasets_dir = datasets_dir or _datasets_dir()
    path = datasets_dir / "2wiki" / "2wiki_dev.jsonl"
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            sf = (o.get("metadata") or {}).get("supporting_facts") or {}
            titles = sorted({normalize_title(t) for t in sf.get("title", [])})
            out[str(o.get("id"))] = titles
    return out


_GOLD_TITLE_LOADERS = {
    "hotpotqa": load_hotpotqa_gold_titles,
    "2wikimultihopqa": load_2wiki_gold_titles,
}

# Datasets whose gold evidence is title-level over the wiki-18 corpus.
TITLE_QREL_DATASETS = set(_GOLD_TITLE_LOADERS)


def strip_dataset_prefix(example_id: str, dataset: str) -> str:
    """Result ids are normalized as f'{dataset}_{raw}' unless raw already starts
    with the dataset name; recover the raw id used by the gold loaders."""
    pref = f"{dataset}_"
    return example_id[len(pref):] if example_id.startswith(pref) else example_id


def load_gold_titles(dataset: str, datasets_dir: Optional[Path] = None) -> dict[str, list[str]]:
    loader = _GOLD_TITLE_LOADERS.get(dataset)
    return loader(datasets_dir) if loader else {}


def supports_title_qrels(dataset: str) -> bool:
    return dataset in TITLE_QREL_DATASETS
