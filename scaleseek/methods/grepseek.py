"""GrepSeek: the official harness `python -m inference.run`, unmodified, as a subprocess.

All settings are command-line flags of the official harness (see configs/methods/
grepseek.yaml).  We give it the subset as its generation-mode `--input` file and our own
vLLM server, then convert its predictions.jsonl into the unified record format.  The
harness's own EM/F1 is kept in each record (`official_em`, `official_f1`) as a
cross-check of our scorer.  The harness makes one attempt per call; an `api_error` row
is a failure.  If the server died we abort (ServerDied), and a resumed run continues
from the harness's own `--resume`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..corpus import corpus_dir
from ..hub import _parallel_copy
from ..llm import ServerDied, VLLMServer
from ..official import checkout_path
from ..utils import PipelineError, iter_jsonl, log, write_jsonl


def stage_corpus(spec, smoke: bool) -> Path:
    """The harness greps its corpus directory constantly: inside a job, read a node-local
    copy (same bytes) instead of Lustre."""
    src = corpus_dir(spec.base, spec.corpus, smoke) / "grepseek"
    if not os.environ.get("SLURM_TMPDIR"):
        return src
    dst = Path(os.environ["SLURM_TMPDIR"]) / f"grepseek_{spec.corpus}{'_mini' if smoke else ''}"
    if not (dst / "wiki_corpus.jsonl").exists():
        dst.mkdir(parents=True, exist_ok=True)
        log(f"staging {src / 'wiki_corpus.jsonl'} -> {dst}")
        _parallel_copy(src / "wiki_corpus.jsonl", dst / "wiki_corpus.jsonl")
    return dst


def harness_command(spec, server: VLLMServer, model_path: Path, input_file: Path,
                    corpus: Path, out_dir: Path) -> list[str]:
    h = spec.method["harness"]
    return [sys.executable, "-m", "inference.run",
            "--base_url", server.base_url + "/v1",
            "--model", spec.method["server"]["served_model_name"],
            "--tokenizer", str(model_path),
            "--input", str(input_file),
            "--corpus_dir", str(corpus),
            "--max_assistant_turns", str(spec.profile["max_assistant_turns"]),
            "--max_tokens_per_turn", str(h["max_tokens_per_turn"]),
            "--tool_max_tokens", str(h["tool_max_tokens"][spec.profile_name]),
            "--tool_timeout", str(h["tool_timeout"]),
            "--temperature", str(spec.profile["temperature"]),
            "--top_p", str(spec.profile["top_p"]),
            "--engine_mode", h["engine_mode"],
            "--parallel", str(h["parallel"]),
            "--out_dir", str(out_dir),
            "--resume"]


def run(spec, server: VLLMServer, model_path: Path, examples: list[dict], smoke: bool) -> Path:
    repo = checkout_path(spec.base, spec.method["official_repo"]["name"])
    official = spec.run_dir / "official"
    official.mkdir(parents=True, exist_ok=True)
    input_file = official / "subset.jsonl"
    write_jsonl(input_file, [{"id": e["id"], "question": e["question"],
                              "golden_answers": e["golden_answers"]} for e in examples])
    cmd = harness_command(spec, server, model_path, input_file, stage_corpus(spec, smoke), official)
    log(f"grepseek harness: {' '.join(cmd)}")
    with open(spec.run_dir / "harness.log", "ab") as logf:
        rc = subprocess.run(cmd, cwd=repo, env=dict(os.environ, PYTHONPATH=str(repo)),
                            stdout=logf, stderr=subprocess.STDOUT).returncode
    if not server.healthy():
        raise ServerDied("vLLM died during the GrepSeek harness run")
    if rc != 0:
        raise PipelineError(f"GrepSeek harness exited {rc}; see {spec.run_dir / 'harness.log'}")
    return official / "subset" / "predictions.jsonl"


def convert(spec, pred: Path, examples: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in iter_jsonl(pred)}
    out = []
    for ex in examples:
        r = by_id.get(ex["id"])
        if r is None:
            continue
        tool_turns = [t for t in r.get("turns", []) if t.get("role") == "tool"]
        parse_errs = sum(1 for t in tool_turns if t.get("synthetic"))
        tool_errs = sum(1 for t in tool_turns if not t.get("synthetic") and (
            t["tool_result"]["timed_out"] or t["tool_result"]["exit_code"] not in (0, 1)))
        pred_text = r.get("prediction")
        out.append({
            "id": ex["id"], "question": ex["question"], "golden_answers": ex["golden_answers"],
            "prediction": pred_text if pred_text else None,
            "finish_reason": r["finish_reason"],
            "n_turns": r["n_assistant_turns"], "n_retrievals": r["n_tool_calls"],
            "errors": {"parse": parse_errs, "tool": tool_errs,
                       "api": 1 if r["finish_reason"] == "api_error" else 0},
            "error": r.get("error"), "wall_time_s": r.get("total_time_s"),
            "official_em": r.get("em"), "official_f1": r.get("f1"),
            "trajectory": {"turns": r.get("turns"), "messages": r.get("messages"),
                           "llm_time_s": r.get("llm_time_s"), "tool_time_s": r.get("tool_time_s")},
        })
    return out
