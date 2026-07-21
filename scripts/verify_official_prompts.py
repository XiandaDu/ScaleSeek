#!/usr/bin/env python3
"""Byte-diff local protocol prompts against frozen official checkouts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _verify_commit(repo: Path, expected: str) -> None:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != expected:
        raise SystemExit(f"{repo}: {head} != {expected}")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _search_r1_prompt(path: Path, question: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt"
                                                 for t in node.targets):
            expression = ast.Expression(node.value)
            return eval(compile(expression, str(path), "eval"), {}, {"question": question})
    raise RuntimeError("official Search-R1 prompt assignment not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repos = yaml.safe_load((ROOT / "configs/official_repos.yaml").read_text())["repos"]
    for key in ("search_r1", "search_o1", "grepseek"):
        _verify_commit(args.repo_root / key, repos[key]["commit"])

    from eval import search_r1_agent as local_r1
    from eval import search_o1_agent as local_o1
    q = "Snapshot question?"
    official = _search_r1_prompt(args.repo_root / "search_r1/infer.py", q)
    assert local_r1._USER_PROMPT_TEMPLATE.format(question=q) == official

    o1 = _load(args.repo_root / "search_o1/scripts/prompts.py", "official_search_o1_prompts")
    assert local_o1._singleqa_instruction(5) == o1.get_singleqa_search_o1_instruction(5)
    assert local_o1._multiqa_instruction(10) == o1.get_multiqa_search_o1_instruction(10)
    assert local_o1._task_instruction_openqa(q) == o1.get_task_instruction_openqa(q)
    assert local_o1._reasonchain_instruction("R", "Q", "D") == \
        o1.get_webpage_to_reasonchain_instruction("R", "Q", "D")

    sys.path.insert(0, str(args.repo_root / "grepseek"))
    from inference.agent import build_system_prompt
    baselines = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text())
    expected_hash = baselines["methods"]["grepseek"]["official_prompt_sha256"]
    assert hashlib.sha256(build_system_prompt().encode()).hexdigest() == expected_hash
    print("official prompt diff: OK")


if __name__ == "__main__":
    main()
