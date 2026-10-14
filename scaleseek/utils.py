"""Small helpers shared by every module: hashing, JSON(L) I/O, git state, logging."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


class PipelineError(RuntimeError):
    """A check failed; the pipeline must stop and a human must look (BASELINES.md §2)."""


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable serialisation used for every content hash (config ids, id lists, ...)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


# ----------------------------------------------------------------------------- JSON I/O
def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic write: a reader never sees a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def write_json_new(path: str | Path, obj: Any) -> None:
    """Write a file that must not exist yet (subset manifests are never overwritten)."""
    path = Path(path)
    if path.exists():
        raise PipelineError(f"refusing to overwrite {path}")
    write_json(path, obj)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"{path}:{n}: bad JSON ({exc})") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class JsonlAppender:
    """Append-only JSONL writer, flushed and fsynced per record (crash-safe resume)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


# ----------------------------------------------------------------------------- git
def git_state(repo: str | Path) -> dict:
    """Commit and cleanliness of a git checkout (the harness or an official repository)."""
    def _git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"path": str(repo), "commit": _git("rev-parse", "HEAD"), "dirty": bool(dirty),
            "dirty_files": dirty.splitlines()}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_agentir_env(base: dict) -> str:
    """Path of the AgentIR interpreter, building the job-local env on first use
    (setup_env.sh --job-agentir; the env lives on node-local disk, see setup_env.sh)."""
    py = base["paths"]["agentir_python"]
    if not os.path.exists(py):
        log("building the job-local AgentIR environment")
        subprocess.run(["bash", str(repo_root() / "setup_env.sh"), "--job-agentir"], check=True)
    return py

