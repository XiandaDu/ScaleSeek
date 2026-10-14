"""Make Pyserini importable on offline compute nodes, without touching site-packages.

Pyserini 2.4.0 reads its prebuilt-index catalogue (JSON files of
castorini/prebuilt-indexes at a commit pinned inside pyserini/prebuilt_index_info.py)
with urllib at IMPORT time.  We never use a prebuilt index, but the import must succeed.
`run.py prefetch` downloads exactly those JSON files on a login node; `install()` then
serves those pinned URLs -- and only those -- from the local copy, before pyserini is
imported.  Any other URL still fails, so nothing can silently reach the network.

    python -m scaleseek.retrieval.pyserini_offline pyserini.index.lucene <args>
"""

from __future__ import annotations

import io
import os
import re
import runpy
import sys
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path

PREFIX = "https://raw.githubusercontent.com/castorini/prebuilt-indexes/"


def catalog_dir() -> Path:
    return Path(os.environ["SCALESEEK_ROOT"]) / "data" / "pyserini_catalog"


def catalog_urls() -> list[str]:
    """The pinned catalogue URLs, read from pyserini's own source (not imported)."""
    import importlib.util

    spec = importlib.util.find_spec("pyserini")
    src = (Path(spec.origin).parent / "prebuilt_index_info.py").read_text()
    commit = re.search(r"PREBUILT_INDEXES_COMMIT = '([0-9a-f]{40})'", src).group(1)
    paths = sorted(set(re.findall(r"_prebuilt_indexes_url\('([^']+)'\)", src)))
    return [f"{PREFIX}{commit}/{p}" for p in paths]


def fetch(log) -> dict:
    """Login node only: download the pinned catalogue files."""
    out = {}
    for url in catalog_urls():
        dst = catalog_dir() / url[len(PREFIX):]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
        out[url] = str(dst)
    log(f"  pyserini catalogue: {len(out)} pinned JSON files")
    return out


class _LocalCatalog(urllib.request.BaseHandler):
    handler_order = 100          # consulted before the default (networked) HTTPS handler

    def https_open(self, req):
        url = req.full_url
        if not url.startswith(PREFIX):
            raise urllib.error.URLError(f"offline: {url} is not a pinned pyserini catalogue file")
        path = catalog_dir() / url[len(PREFIX):]
        if not path.exists():
            raise urllib.error.URLError(f"offline: {path} missing (run.py prefetch on a login node)")
        resp = urllib.response.addinfourl(io.BytesIO(path.read_bytes()), Message(), url, 200)
        resp.msg = "OK"
        return resp


def _stage_jars() -> None:
    """Inside a job, load the Anserini fat jar from node-local disk.  The JVM reads it with
    random access; through a congested Lustre that stalled for tens of minutes.  The
    official ANSERINI_CLASSPATH variable points pyserini at the copy (same bytes)."""
    import importlib.util
    import shutil

    tmp = os.environ.get("SLURM_TMPDIR")
    if not tmp or os.environ.get("ANSERINI_CLASSPATH"):
        return
    src = Path(importlib.util.find_spec("pyserini").origin).parent / "resources" / "jars"
    dst = Path(tmp) / "anserini_jars"
    dst.mkdir(exist_ok=True)
    for jar in src.glob("*.jar"):
        target = dst / jar.name
        if not target.exists() or target.stat().st_size != jar.stat().st_size:
            shutil.copyfile(jar, target)
    os.environ["ANSERINI_CLASSPATH"] = str(dst)


def install() -> None:
    _stage_jars()
    urllib.request.install_opener(urllib.request.build_opener(_LocalCatalog()))


if __name__ == "__main__":
    install()
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
