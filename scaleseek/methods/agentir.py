"""AgentIR: the official OSS-client scaffold, launched in the job-local AgentIR env.

The official command line (evaluation/scripts/run_oss_agentir.sh, with Qwen3.5-9B on our
vLLM server) is rebuilt from configs/methods/agentir.yaml + base.yaml and handed to
scaleseek/agentir_entry.py, which runs the official oss_client.main() with the
documented overrides.  Its run_*.json files (official format) and our sidecars are then
converted into the unified record format; the prediction is the `Exact Answer:` line of
the final response (official BrowseComp-Plus response format).  A failed model call ends
that question as a failure (official loop); if the vLLM server died we abort.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

from ..corpus import corpus_dir
from ..hub import local_snapshot
from ..llm import ServerDied, VLLMServer
from ..official import checkout_path
from ..retrieval.common import index_dir
from ..utils import PipelineError, ensure_agentir_env, log, read_json, repo_root, write_json

_EXACT = re.compile(r"^\s*[*_]*\s*exact\s+answer\s*[*_]*\s*:\s*[*_]*\s*(.*?)\s*[*_]*\s*$", re.IGNORECASE)


def exact_answer(text: str | None) -> str | None:
    """Value of the last `Exact Answer:` line (markdown bold tolerated), or None."""
    found = None
    for line in (text or "").splitlines():
        m = _EXACT.match(line)
        if m and m.group(1).strip():
            found = m.group(1).strip()
    return found


def official_args(spec, server: VLLMServer, tsv: Path, out_dir: Path, smoke: bool) -> list[str]:
    a = spec.method["agent"]
    base = spec.base
    r = base["retrievers"]["agentir_4b"]
    args = ["--query", str(tsv), "--model", server.served_name, "--output-dir", str(out_dir),
            "--model-url", server.base_url + "/v1",
            "--num-threads", str(a["num_threads"]),
            "--max-iterations", str(spec.profile["max_assistant_turns"]),
            "--reasoning-effort", a["reasoning_effort"],
            "--k", str(spec.method["retrieval"]["top_k"]),
            "--searcher-type", "faiss",
            "--index-path", str(index_dir(base, "agentir_4b", spec.corpus, smoke) / "corpus*.pkl"),
            "--model-name", str(local_snapshot(base, r["model"], r["revision"])),
            "--pooling", a["pooling"], "--torch-dtype", a["torch_dtype"],
            "--dataset-name", "json",
            "--data-files", str(corpus_dir(base, spec.corpus, smoke) / "tevatron.jsonl"),
            "--task-prefix", a["task_prefix"], "--max-length", str(a["max_length"]),
            "--snippet-max-tokens", str(a["snippet_max_tokens"])]
    if a["normalize"]:
        args.append("--normalize")
    if a["reasoning_aware"]:
        args.append("--reasoning-aware")
    return args


def _write_tsv(path: Path, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for e in examples:
            w.writerow([e["id"], e["question"].replace("\t", " ")])


def _runs(official: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(official.glob("run_*.json")):
        qid = read_json(p).get("query_id")
        if qid is not None:
            out[str(qid)] = p
    return out


def run(spec, server: VLLMServer, examples: list[dict], smoke: bool) -> Path:
    base = spec.base
    official = spec.run_dir / "official"
    official.mkdir(parents=True, exist_ok=True)
    tsv = official / "queries.tsv"
    _write_tsv(tsv, examples)
    settings = official / "entry_settings.json"
    write_json(settings, {"repo": str(checkout_path(base, spec.method["official_repo"]["name"])),
                          "max_turns": spec.profile["max_assistant_turns"],
                          "temperature": spec.profile["temperature"],
                          "top_p": spec.profile["top_p"],
                          "presence_penalty": spec.profile["presence_penalty"],
                          "request_timeout_s": spec.method["agent"]["request_timeout_s"],
                          "max_retries": spec.method["agent"]["max_retries"],
                          "sidecar_dir": str(official / "sidecars")})
    cmd = [ensure_agentir_env(base), str(repo_root() / "scaleseek" / "agentir_entry.py"),
           str(settings), "--"] + official_args(spec, server, tsv, official, smoke)
    tmp = Path(os.environ.get("SLURM_TMPDIR") or spec.run_dir)
    env = dict(os.environ, HF_DATASETS_CACHE=str(tmp / "agentir_hf_datasets"),
               CUDA_VISIBLE_DEVICES=server.gpus[0],
               OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    env.pop("VIRTUAL_ENV", None)
    log("agentir official client")
    with open(spec.run_dir / "harness.log", "ab") as logf:
        rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
    if not server.healthy():
        raise ServerDied("vLLM died during the AgentIR run")
    if rc != 0:
        raise PipelineError(f"AgentIR client exited {rc}; see {spec.run_dir / 'harness.log'}")
    return official


def convert(spec, official: Path, examples: list[dict]) -> list[dict]:
    """Official oss_client run_*.json -> unified records.  status: completed | incomplete
    (the official loop ran out of iterations)."""
    runs = _runs(official)
    out = []
    for ex in examples:
        p = runs.get(ex["id"])
        if p is None:
            continue
        run = read_json(p)
        side_p = official / "sidecars" / f"{ex['id']}.json"
        side = read_json(side_p) if side_p.exists() else {}
        calls = side.get("calls") or []
        result = run.get("result") or []
        texts = [r["output"] for r in result if r.get("type") == "output_text" and r.get("output")]
        final = texts[-1] if texts else None
        answer = exact_answer(final)
        rec = {"id": ex["id"], "question": ex["question"], "golden_answers": ex["golden_answers"],
               "prediction": None, "final_response": None, "finish_reason": None,
               "n_turns": sum(1 for c in calls if c.get("kind") != "turn_budget"),
               "n_retrievals": int((run.get("tool_call_counts") or {}).get("search", 0)),
               "errors": {"parse": 0,
                          "tool": max(0, int((run.get("tool_call_counts_raw") or {}).get("search", 0))
                                  - int((run.get("tool_call_counts") or {}).get("search", 0))),
                          "api": sum(1 for c in calls if not c.get("ok") and c.get("kind") != "turn_budget")},
               "error": None, "wall_time_s": side.get("seconds"),
               "trajectory": {"official_run": run, "calls": calls}}
        if run.get("status") == "completed" and final:
            rec["final_response"] = final
            rec["prediction"] = answer
            if answer:
                rec["finish_reason"] = "answer"
            else:
                rec["finish_reason"] = "no_answer"
                rec["errors"]["parse"] = 1
                rec["error"] = "final response has no `Exact Answer:` line"
        else:
            rec["finish_reason"] = "max_turns" if run.get("status") == "incomplete" else "exception"
        out.append(rec)
    return out
