"""Index locations and manifests shared by every retriever."""

from __future__ import annotations

import os
from pathlib import Path

from ..corpus import load_corpus_manifest
from ..utils import PipelineError, read_json, sha256_file


def index_dir(base: dict, retriever: str, corpus: str, smoke: bool) -> Path:
    root = Path(base["paths"]["smoke"]) / "indexes" if smoke else Path(base["paths"]["indexes"])
    return root / (f"{corpus}_mini" if smoke else corpus) / retriever


def hash_tree(path: Path) -> dict:
    """sha256 + size of every file under `path` (index manifests)."""
    out = {}
    for root, _, files in os.walk(path):
        for f in sorted(files):
            p = Path(root) / f
            out[str(p.relative_to(path))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def load_index_manifest(base: dict, retriever: str, corpus: str, smoke: bool) -> dict:
    """The index exists, is complete, and was built from the corpus that is on disk now."""
    path = index_dir(base, retriever, corpus, smoke) / "manifest.json"
    if not path.exists():
        raise PipelineError(f"no index manifest {path}: build the {retriever} index first")
    man = read_json(path)
    cman = load_corpus_manifest(base, corpus, smoke)
    if man["corpus_sha256"] != cman["sha256"] or man["docs"] != cman["docs"]:
        raise PipelineError(f"{path}: built from corpus {man['corpus_sha256'][:12]} "
                            f"({man['docs']} docs) but the corpus is {cman['sha256'][:12]} ({cman['docs']})")
    return man
