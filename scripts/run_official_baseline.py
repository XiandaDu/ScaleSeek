#!/usr/bin/env python3
"""Commit-gated launcher for baseline methods that must use official harnesses.

Arguments after ``--`` are forwarded verbatim as an argv list (never through a
shell). The launcher records the exact command and refuses a mismatched checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOS = yaml.safe_load((ROOT / "configs" / "official_repos.yaml").read_text())["repos"]
BASELINES = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text())

ENTRYPOINTS = {
    "dci": ("dci_agent_lite", ["uv", "run", "python", "scripts/bcplus_eval/run_bcplus_eval.py"]),
    "dr_dci": ("dr_dci", ["uv", "run", "python", "scripts/bcplus_eval/run_bcplus_eval.py"]),
    "rise": ("rise", ["uv", "run", "python", "scripts/run_rise.py"]),
    "agentir": ("agentir", ["uv", "run", "python", "evaluation/search_agent/oss_client.py"]),
    "grepseek": ("grepseek", [sys.executable, "-m", "inference.run"]),
}

SOURCE_FILES = {
    "dci": ["prompts/system_prompt.txt", "scripts/bcplus_eval/run_bcplus_eval.py"],
    "dr_dci": ["prompts/system_prompt.txt", "scripts/bcplus_eval/run_bcplus_eval.py"],
    "rise": ["scripts/run_rise.py", "src/rise/protocol.py", "src/rise/rise_agent.py"],
    "agentir": ["evaluation/search_agent/oss_client.py", "searcher/prompts.py",
                "searcher/searchers/faiss_searcher.py"],
    "grepseek": ["inference/agent.py", "inference/tools.py", "inference/run.py"],
}


def _value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _validate(method: str, forwarded: list[str], full_eval: bool) -> None:
    if full_eval and any(x in forwarded for x in ("--limit", "-n", "--offset", "--mini-dev")):
        raise ValueError("full evaluation forbids limit/offset/mini-dev arguments")
    required_model = ("alireza7/GrepSeek-Qwen3.5-9B-GRPO" if method == "grepseek"
                      else "Qwen/Qwen3.5-9B")
    if full_eval and _value(forwarded, "--model") != required_model:
        raise ValueError(f"full evaluation requires --model {required_model}")
    if method in {"dci", "dr_dci"}:
        if _value(forwarded, "--max-turns") != "300":
            raise ValueError(f"{method} requires --max-turns 300")
        if _value(forwarded, "--runtime-context-level") != "level3":
            raise ValueError(f"{method} requires --runtime-context-level level3")
    if method == "dr_dci":
        required = {"--pull-min-top-k": "300", "--pull-max-top-k": "600",
                    "--pull-max-queries": "10",
                    "--pull-preview-limit": "20",
                    "--pull-materialization-mode": "root_flat_disclosed"}
        for flag, expected in required.items():
            if _value(forwarded, flag) != expected:
                raise ValueError(f"dr_dci requires {flag} {expected}")
    if method == "rise":
        required = {"--bm25-k": "1000", "--bm25-top-n-preview": "10",
                    "--max-turns": "100", "--bash-truncate-chars": "4000",
                    "--read-default-limit": "2000"}
        for flag, expected in required.items():
            if _value(forwarded, flag) != expected:
                raise ValueError(f"RISE requires {flag} {expected}")
        if "--structured-docs" not in forwarded:
            raise ValueError("RISE main method requires --structured-docs")
    if method == "agentir":
        for flag in ("--reasoning-aware", "--normalize"):
            if flag not in forwarded:
                raise ValueError(f"AgentIR requires {flag}")
        if _value(forwarded, "--k") != "5" or _value(forwarded, "--max-iterations") != "100":
            raise ValueError("AgentIR requires --k 5 --max-iterations 100")
    if method == "grepseek":
        required = {"--max_assistant_turns": "6", "--max_tokens_per_turn": "0",
                    "--tool_max_tokens": "2048", "--temperature": "0.6",
                    "--top_p": "1.0"}
        for flag, expected in required.items():
            if _value(forwarded, flag) != expected:
                raise ValueError(f"GrepSeek requires {flag} {expected}")
        if _value(forwarded, "--tokenizer") != "alireza7/GrepSeek-Qwen3.5-9B-GRPO":
            raise ValueError("GrepSeek requires its checkpoint tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=ENTRYPOINTS)
    parser.add_argument("--repo-root", type=Path, required=True,
                        help="Root created by bootstrap_official_repos.py")
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--model-revision",
                        help="Immutable revision of the served generator checkpoint")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    repo_key, entrypoint = ENTRYPOINTS[args.method]
    repo = args.repo_root / repo_key
    expected = REPOS[repo_key]["commit"]
    if not (repo / ".git").exists():
        sys.exit(f"Official repo not found: {repo}")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != expected:
        sys.exit(f"{repo_key} commit mismatch: {head}; expected {expected}")
    try:
        _validate(args.method, forwarded, args.full_eval)
    except ValueError as exc:
        sys.exit(str(exc))
    generator_spec = BASELINES["methods"][args.method].get("generator")
    expected_model_revision = (generator_spec["revision"] if isinstance(generator_spec, dict)
                               else BASELINES["global"]["generator"]["revision"])
    if args.full_eval and args.model_revision != expected_model_revision:
        sys.exit(f"--model-revision must be {expected_model_revision}")
    if args.full_eval and args.method == "agentir":
        expected_retriever_revision = BASELINES["methods"]["agentir"]["retriever"]["revision"]
        model_path = Path(_value(forwarded, "--model-name") or "")
        if model_path.name != expected_retriever_revision:
            sys.exit("AgentIR --model-name must be a local HF snapshot directory named by the frozen revision")
    command = entrypoint + forwarded
    source_hasher = hashlib.sha256()
    for relative in SOURCE_FILES[args.method]:
        source_hasher.update(relative.encode() + b"\0")
        source_hasher.update((repo / relative).read_bytes())
    env_overrides = ({"DCI_BASH_DEFAULT_TIMEOUT_SECONDS": "30"}
                     if args.method in {"dci", "dr_dci"} else {})
    record = {"method": args.method, "repo": str(repo), "harness_commit": head,
              "argv": command, "full_eval": args.full_eval,
              "generator_revision": expected_model_revision,
              "harness_source_sha256": source_hasher.hexdigest(),
              "harness_source_files": SOURCE_FILES[args.method],
              "env_overrides": env_overrides,
              "created_at": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if not args.dry_run:
        child_env = {**os.environ, **env_overrides}
        completed = subprocess.run(command, cwd=repo, env=child_env)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
