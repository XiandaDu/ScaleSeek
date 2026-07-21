#!/usr/bin/env python3
"""Clone official harnesses and verify their frozen commits without resetting them."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "official_repos.yaml"


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Dedicated directory for frozen official repositories")
    parser.add_argument("--repo", action="append", default=[],
                        help="Only bootstrap this config key (repeatable)")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())["repos"]
    selected = args.repo or list(cfg)
    unknown = set(selected) - set(cfg)
    if unknown:
        sys.exit(f"Unknown official repos: {sorted(unknown)}")
    args.root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        spec = cfg[name]
        target = args.root / name
        if not target.exists():
            subprocess.run(["git", "clone", spec["url"], str(target)], check=True)
        if not (target / ".git").exists():
            sys.exit(f"Refusing non-git target: {target}")
        subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"]],
                       check=True)
        head = git_output(target, "rev-parse", "HEAD")
        if head != spec["commit"]:
            dirty = git_output(target, "status", "--porcelain")
            if dirty:
                sys.exit(f"{target} has local changes and is at {head}; expected {spec['commit']}")
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]],
                           check=True)
        print(f"{name}: {git_output(target, 'rev-parse', 'HEAD')}")


if __name__ == "__main__":
    main()
