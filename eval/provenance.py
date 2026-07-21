"""Runtime provenance helpers for result rows."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config import load_baselines, load_official_repos

ROOT = Path(__file__).resolve().parents[1]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_hash(method: str) -> str:
    if method in {"direct", "rag", "scaleseek"}:
        from . import prompts
        name = method if method != "scaleseek" else "scaleseek_prompt"
        return sha256_text(prompts.load(name))
    if method == "search_r1":
        from . import search_r1_agent
        return sha256_text(search_r1_agent._USER_PROMPT_TEMPLATE)
    if method == "search_o1":
        from . import search_o1_agent
        pieces = [search_o1_agent._singleqa_instruction(5),
                  search_o1_agent._multiqa_instruction(10),
                  search_o1_agent._task_instruction_openqa("{question}"),
                  search_o1_agent._reasonchain_instruction(
                      "{reasoning}", "{query}", "{documents}")]
        return sha256_text("\n".join(pieces))
    raise KeyError(method)


def generator_revision(method: str, *, search_r1_checkpoint: str | None = None) -> str:
    cfg = load_baselines()
    spec = cfg["methods"][method]
    if method == "search_r1":
        for model in spec["generators"]:
            if model["repo_id"] == search_r1_checkpoint:
                return model["revision"]
        raise ValueError("Search-R1 checkpoint must be one of the frozen 7B/14B v0.3 models")
    if isinstance(spec.get("generator"), dict):
        return spec["generator"]["revision"]
    return cfg["global"]["generator"]["revision"]


def harness_metadata(method: str) -> dict:
    method_files = {
        "direct": ["eval/agent.py"],
        "rag": ["eval/agent.py", "eval/retrievers.py"],
        "scaleseek": ["eval/agent.py"],
        "search_r1": ["eval/search_r1_agent.py"],
        "search_o1": ["eval/search_o1_agent.py"],
    }
    hasher = hashlib.sha256()
    for relative in method_files[method]:
        hasher.update(relative.encode() + b"\0")
        hasher.update((ROOT / relative).read_bytes())
    method_cfg = load_baselines()["methods"][method]
    source_repo = method_cfg.get("source_repo")
    commit = (load_official_repos()["repos"][source_repo]["commit"]
              if source_repo else None)
    return {"harness_source_sha256": hasher.hexdigest(),
            "harness_source_files": method_files[method],
            "upstream_harness_commit": commit}
