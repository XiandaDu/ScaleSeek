"""Pinned assets: Hugging Face snapshots and official git checkouts.

`prefetch` is the ONLY network-facing code and runs on a login node (compute nodes have
no internet).  It does I/O only: no hashing of large files, no decompression -- that is
`run.py prepare`, a CPU job.  Every HF object is fetched at a 40-hex revision; every
repository at a 40-hex commit.  After a model snapshot lands, refs/main is pointed at the
pinned revision so that official code which loads a model BY NAME offline (e.g. AgentIR's
hard-coded "Qwen/Qwen3-0.6B" tokenizer) resolves to the pinned files and nothing else.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .config import load_all_methods
from .utils import PipelineError, git_state, log, now_iso, write_json

# Files needed from a repo used only as a tokenizer.
TOKENIZER_PATTERNS = ["*.json", "*.txt", "*.model", "*.jinja"]


def hub_dir(base: dict) -> Path:
    return Path(base["paths"]["hf_home"]) / "hub"


def _folder(repo_id: str, repo_type: str) -> str:
    prefix = {"model": "models", "dataset": "datasets"}[repo_type]
    return f"{prefix}--{repo_id.replace('/', '--')}"


def snapshot_dir(base: dict, repo_id: str, revision: str, repo_type: str = "model") -> Path:
    path = hub_dir(base) / _folder(repo_id, repo_type) / "snapshots" / revision
    if not path.is_dir():
        raise PipelineError(f"{repo_type} {repo_id}@{revision} is not in {hub_dir(base)}; "
                            "run `python run.py prefetch` on a login node")
    return path


# ============================================================================= inventory
def asset_inventory(base: dict) -> dict[str, list[dict]]:
    """Everything the configs need, derived from the configs (no separate list to drift)."""
    methods = load_all_methods(base)
    models: dict[tuple[str, str], dict] = {}

    def add_model(repo: str, rev: str, patterns: list[str] | None, used_by: str) -> None:
        key = (repo, rev)
        entry = models.setdefault(key, {"repo_id": repo, "revision": rev, "repo_type": "model",
                                        "allow_patterns": patterns, "used_by": []})
        if patterns is None:
            entry["allow_patterns"] = None      # a full download wins over a partial one
        entry["used_by"].append(used_by)

    for name, m in methods.items():
        add_model(m["generator"]["model"], m["generator"]["revision"], None, name)
        if m["method"] == "agentir":
            tok = m["agent"]["snippet_tokenizer"]
            add_model(tok["model"], tok["revision"], TOKENIZER_PATTERNS, f"{name}.snippet_tokenizer")
    for rname, r in base["retrievers"].items():
        if "model" in r:
            add_model(r["model"], r["revision"], None, f"retriever.{rname}")

    datasets: dict[tuple[str, str], dict] = {}
    for dname, d in base["datasets"].items():
        key = (d["hf_repo"], d["revision"])
        entry = datasets.setdefault(key, {"repo_id": d["hf_repo"], "revision": d["revision"],
                                          "repo_type": "dataset", "allow_patterns": [],
                                          "used_by": []})
        entry["allow_patterns"].append(d["file"])
        entry["used_by"].append(f"dataset.{dname}")
    for cname, c in base["corpora"].items():
        key = (c["hf_repo"], c["revision"])
        entry = datasets.setdefault(key, {"repo_id": c["hf_repo"], "revision": c["revision"],
                                          "repo_type": "dataset", "allow_patterns": [],
                                          "used_by": []})
        entry["allow_patterns"].append(c["file"])
        entry["used_by"].append(f"corpus.{cname}")

    repos: dict[str, dict] = {}
    for name, m in methods.items():
        if "official_repo" in m:
            r = m["official_repo"]
            if r["name"] in repos and repos[r["name"]]["commit"] != r["commit"]:
                raise PipelineError(f"repo {r['name']} pinned to two commits")
            repos[r["name"]] = dict(r)
    return {"models": list(models.values()), "datasets": list(datasets.values()),
            "repos": list(repos.values())}


# ============================================================================= downloads
def _point_refs_main(base: dict, repo_id: str, revision: str, repo_type: str) -> None:
    ref = hub_dir(base) / _folder(repo_id, repo_type) / "refs" / "main"
    if ref.exists():
        current = ref.read_text().strip()
        if current != revision:
            raise PipelineError(f"{ref} points at {current}, expected pinned {revision}")
        return
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)


def fetch_hf(base: dict, item: dict) -> Path:
    from huggingface_hub import snapshot_download

    log(f"  {item['repo_type']:7s} {item['repo_id']}@{item['revision'][:12]}"
        + (f"  {item['allow_patterns']}" if item["allow_patterns"] else ""))
    path = snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                             repo_type=item["repo_type"], allow_patterns=item["allow_patterns"],
                             cache_dir=str(hub_dir(base)))
    if item["repo_type"] == "model":
        _point_refs_main(base, item["repo_id"], item["revision"], "model")
    return Path(path)


def fetch_repo(base: dict, repo: dict) -> dict:
    dest = Path(base["paths"]["repos"]) / repo["name"]
    if not dest.exists():
        log(f"  git     {repo['url']}@{repo['commit'][:12]} -> {dest}")
        dest.mkdir(parents=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", repo["url"]],
                    ["git", "fetch", "-q", "--depth", "1", "origin", repo["commit"]],
                    ["git", "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=dest, check=True)
    state = git_state(dest)
    if state["commit"] != repo["commit"] or state["dirty"]:
        raise PipelineError(f"{dest}: at {state['commit']} dirty={state['dirty']}, "
                            f"pinned {repo['commit']}")
    return state


def prefetch(base: dict, only: str) -> dict:
    if os.environ.get("SLURM_JOB_ID"):
        raise PipelineError("prefetch needs the internet: run it on a login node, not in a job")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    inv = asset_inventory(base)
    record: dict = {"created": now_iso(), "only": only, "repos": [], "hf": []}
    if only in ("all", "repos"):
        log("official repositories")
        for repo in inv["repos"]:
            record["repos"].append(fetch_repo(base, repo))
        from .retrieval.pyserini_offline import fetch as fetch_pyserini_catalog
        record["pyserini_catalog"] = fetch_pyserini_catalog(log)
    if only in ("all", "models"):
        log("models")
        for item in inv["models"]:
            record["hf"].append({**item, "path": str(fetch_hf(base, item))})
    if only in ("all", "datasets"):
        log("datasets and corpora")
        for item in inv["datasets"]:
            record["hf"].append({**item, "path": str(fetch_hf(base, item))})
    out = Path(base["paths"]["data"]) / "prefetch" / f"prefetch_{only}.json"
    write_json(out, record)
    log(f"wrote {out}")
    return record


# ============================================================================= job-local copies
def _parallel_copy(src: Path, dst: Path, threads: int = 16, chunk: int = 64 << 20) -> None:
    """Copy one file with concurrent range reads: on a congested Lustre one stream gets
    ~20 MB/s while parallel streams add up to hundreds of MB/s."""
    import concurrent.futures

    size = src.stat().st_size
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)
    fin, fout = os.open(src, os.O_RDONLY), os.open(tmp, os.O_WRONLY)
    try:
        def piece(off: int) -> None:
            data = os.pread(fin, min(chunk, size - off), off)
            os.pwrite(fout, data, off)

        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            list(pool.map(piece, range(0, size, chunk)))
    finally:
        os.close(fin)
        os.close(fout)
    if tmp.stat().st_size != size:
        raise PipelineError(f"short copy of {src}")
    os.replace(tmp, dst)


def local_snapshot(base: dict, repo_id: str, revision: str, repo_type: str = "model") -> Path:
    """Inside a job: a node-local copy of a pinned snapshot (made once per job, then
    reused); outside a job: the snapshot itself.  Same files, same bytes -- only where
    they are read from changes, so provenance keeps recording `snapshot_dir`."""
    import fcntl

    src = snapshot_dir(base, repo_id, revision, repo_type)
    tmp = os.environ.get("SLURM_TMPDIR")
    if not tmp:
        return src
    dst = Path(tmp) / "models" / _folder(repo_id, repo_type) / revision
    done = dst / ".scaleseek_complete"
    if done.exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst.parent / f".{revision}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not done.exists():
            t0 = time.time()
            total = 0
            for path in sorted(src.rglob("*")):
                rel = path.relative_to(src)
                target = dst / rel
                if path.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                real = path.resolve()
                if not target.exists() or target.stat().st_size != real.stat().st_size:
                    _parallel_copy(real, target)
                total += real.stat().st_size
            done.write_text(revision)
            log(f"staged {repo_id}@{revision[:12]} to local disk: {total / 2**30:.1f} GiB in "
                f"{time.time() - t0:.0f}s")
    return dst
