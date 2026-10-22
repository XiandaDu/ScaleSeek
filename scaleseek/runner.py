"""One run = (dataset, method yaml, retriever): servers up, method over the subset, records.

<run_dir> = paths.runs/<dataset>/<method>[__<retriever>]   (paths.smoke/... for smoke)
    provenance.json      written at the first session; a resume must match it exactly
    sessions.jsonl       one line per job session (start/end, host, job id, harness commit)
    trajectories.jsonl   one record per question (scaleseek.methods.base)
    vllm.log, retriever.log, harness.log, official/   (official harness outputs, if any)

Formal runs refuse to start from a checkout with uncommitted changes, and refuse to resume
under a different harness commit or config id: one run = one code version.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import requests

from .config import RunSpec, load_prompts
from .corpus import load_corpus_manifest
from .data import load_subset
from .hub import local_snapshot, snapshot_dir
from .llm import LLMClient, VLLMServer, free_port
from .official import verify_method
from .retrieval.common import load_index_manifest
from .retrieval.service import RetrievalClient
from .utils import (JsonlAppender, PipelineError, git_state, iter_jsonl, log, now_iso, read_json,
                    repo_root, sha256_file, sha256_json, write_json, write_jsonl)

SERVED_BY_SERVICE = {"bm25", "e5", "qwen3_emb_4b"}


def job_gpus() -> list[str]:
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("vllm", "torch", "transformers", "pyserini", "faiss-cpu", "openai", "datasets"):
        out[pkg] = metadata.version(pkg)
    out["env_main_lock_sha256"] = sha256_file(repo_root() / "env" / "main.lock.txt")
    out["ripgrep"] = subprocess.run(["rg", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    return out


def check_context_window(model_path: Path, max_model_len: int) -> dict:
    cfg = json.loads((model_path / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    native = text_cfg.get("max_position_embeddings")
    rope = text_cfg.get("rope_scaling") or text_cfg.get("rope_parameters")
    if native is not None and max_model_len > native:
        raise PipelineError(f"{model_path.name}: max_model_len {max_model_len} exceeds the model's "
                            f"max_position_embeddings {native} (rope: {rope}); this needs a "
                            "decision (BASELINES.md does not allow silently extending it)")
    return {"max_position_embeddings": native, "rope": rope}


class RetrievalProcess:
    def __init__(self, spec: RunSpec, smoke: bool, gpu: str):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [sys.executable, str(repo_root() / "run.py"), "serve-retriever",
               "--retriever", spec.retriever, "--corpus", spec.corpus, "--port", str(self.port),
               "--device", "cuda:0"] + (["--smoke"] if smoke else [])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        self.log_path = spec.run_dir / "retriever.log"
        self._logf = open(self.log_path, "ab")
        log("retriever: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=self._logf, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)

    def wait(self, timeout_s: float) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise PipelineError(f"retrieval service exited {self.proc.returncode}; see {self.log_path}")
            try:
                h = requests.get(self.url + "/health", timeout=10)
                if h.status_code == 200:
                    log(f"retriever ready after {time.time() - t0:.0f}s")
                    return h.json()
            except requests.RequestException:
                pass
            time.sleep(5)
        raise PipelineError(f"retrieval service not ready after {timeout_s}s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._logf.close()


def build_provenance(spec: RunSpec, subset_manifest: dict, model_path: Path, smoke: bool,
                     server: VLLMServer, official: dict) -> dict:
    base = spec.base
    prov = {
        "run": spec.run_name, "dataset": spec.dataset, "method": spec.method_name,
        "retriever": spec.retriever, "mode": spec.mode, "config_id": spec.config_id,
        "created": now_iso(),
        "method_config": {k: v for k, v in spec.method.items() if not k.startswith("_")},
        "profile": spec.profile,
        "subset": {"ids_sha256": subset_manifest["ids_sha256"], "n": subset_manifest["n"],
                   "seed": subset_manifest["seed"],
                   "full_normalized_sha256": subset_manifest["full_normalized_sha256"],
                   "subset_sha256": subset_manifest["subset_sha256"]},
        "generator": {**spec.method["generator"], "snapshot": str(model_path),
                      "context_window": check_context_window(model_path, spec.profile["max_model_len"])},
        "corpus_sha256": load_corpus_manifest(base, spec.corpus, smoke)["sha256"],
        "index_manifest_sha256": (sha256_json(load_index_manifest(base, spec.retriever, spec.corpus, smoke))
                                  if spec.retriever else None),
        "prompts": {slot: ref["sha256"] for slot, ref in spec.method.get("prompts", {}).items()},
        "official": official,
        "server_command": server.command(),
        "seeds": {"sampling_seed": base["sampling"]["seed"], "vllm_seed": spec.method["server"]["seed"]},
        "software": software_versions(),
    }
    return prov


def _log_session(spec: RunSpec, row: dict) -> None:
    w = JsonlAppender(spec.run_dir / "sessions.jsonl")
    w.write(row)
    w.close()


def _session_start(spec: RunSpec) -> dict:
    harness = git_state(repo_root())
    if harness["dirty"] and spec.mode == "formal":
        raise PipelineError(f"formal runs need committed code; modified: {harness['dirty_files']}")
    return {"start": now_iso(), "t0": time.time(), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "harness_commit": harness["commit"]}


def eval_run(spec: RunSpec, smoke_n: int | None = None) -> dict:
    smoke = spec.mode == "smoke"
    base = spec.base
    method = spec.method
    examples, subset_manifest = load_subset(base, spec.dataset)
    if smoke:
        examples = examples[:smoke_n]
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    session = _session_start(spec)

    prompts = load_prompts(method)
    official = verify_method(base, method)
    model_path = snapshot_dir(base, method["generator"]["model"], method["generator"]["revision"])
    # Load from a node-local copy (same bytes; provenance records the pinned snapshot).
    serve_path = local_snapshot(base, method["generator"]["model"], method["generator"]["revision"])
    gpus = job_gpus()
    server = VLLMServer(serve_path, method["server"], spec.profile["max_model_len"],
                        spec.run_dir / "vllm.log", gpus)
    prov = build_provenance(spec, subset_manifest, model_path, smoke, server, official)
    prov["harness_commit"] = session["harness_commit"]
    prov_path = spec.run_dir / "provenance.json"
    if prov_path.exists():
        # One run = one configuration and one code version (smoke runs may change code).
        old = read_json(prov_path)
        keys = ["config_id", "subset", "generator", "corpus_sha256", "index_manifest_sha256",
                "prompts", "software"] + ([] if smoke else ["harness_commit"])
        for key in keys:
            if old[key] != prov[key]:
                raise PipelineError(f"cannot resume {spec.run_dir}: {key} changed "
                                    f"({old[key]!r} -> {prov[key]!r})")
    else:
        write_json(prov_path, prov)

    retr = None
    out_path = spec.run_dir / "trajectories.jsonl"
    if out_path.exists():
        have = {r["id"] for r in iter_jsonl(out_path)}
        if all(e["id"] in have for e in examples):
            log(f"{spec.run_name}: all {len(examples)} records present; nothing to run")
            return {"run_dir": str(spec.run_dir), "records": len(have), "missing": []}
    # The start is logged at once, so a session killed before `finally` still shows up
    # (as a start without an end) in sessions.jsonl.
    _log_session(spec, {"event": "start", **{k: v for k, v in session.items() if k != "t0"}})
    try:
        if spec.retriever in SERVED_BY_SERVICE:
            retr = RetrievalProcess(spec, smoke, gpus[0])
            retr.wait(base["runtime"]["retriever_start_timeout_s"])
        server.start(base["runtime"]["server_start_timeout_s"])
        kind = method["method"]
        if kind in ("direct", "rag", "search_r1", "search_o1"):
            from transformers import AutoTokenizer
            from .methods import METHODS
            from .methods.base import run_examples
            ctx = SimpleNamespace(
                spec=spec, prompts=prompts, llm=LLMClient(server, method["client"]),
                retrieval=RetrievalClient(retr.url, method["client"]["request_timeout_s"]) if retr else None,
                tokenizer=AutoTokenizer.from_pretrained(str(serve_path)),
                title_rule=load_corpus_manifest(base, spec.corpus, smoke)["title_rule"])
            result = run_examples(METHODS[kind](ctx), examples, out_path, method["client"]["concurrency"])
        elif kind == "grepseek":
            from .methods import grepseek
            pred = grepseek.run(spec, server, serve_path, examples, smoke)
            write_jsonl(out_path, grepseek.convert(spec, pred, examples))
            result = {"official_output": str(pred)}
        elif kind == "agentir":
            from .methods import agentir
            off = agentir.run(spec, server, examples, smoke)
            write_jsonl(out_path, agentir.convert(spec, off, examples))
            result = {"official_output": str(off)}
    finally:
        server.stop()
        if retr:
            retr.stop()
        _log_session(spec, {"event": "end", "start": session["start"], "end": now_iso(),
                            "seconds": round(time.time() - session["t0"], 1)})
    done = {r["id"] for r in iter_jsonl(out_path)}
    missing = [e["id"] for e in examples if e["id"] not in done]
    log(f"{spec.run_name}: {len(done)}/{len(examples)} records, {len(missing)} missing")
    return {"run_dir": str(spec.run_dir), "records": len(done), "missing": missing, **result}
