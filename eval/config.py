"""Load and fingerprint the frozen Phase-1 baseline configuration."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINES_PATH = ROOT / "configs" / "baselines.yaml"
OFFICIAL_REPOS_PATH = ROOT / "configs" / "official_repos.yaml"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_id(value: Any) -> str:
    """Stable SHA256 identifier for a resolved configuration fragment."""
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_baselines() -> dict:
    data = yaml.safe_load(BASELINES_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported configs/baselines.yaml schema_version")
    required = {"direct", "rag", "search_r1", "search_o1", "grepseek", "dci",
                "dr_dci", "rise", "agentir", "scaleseek"}
    missing = required - set(data.get("methods", {}))
    if missing:
        raise ValueError(f"Missing baseline configurations: {sorted(missing)}")
    return data


@lru_cache(maxsize=1)
def load_official_repos() -> dict:
    data = yaml.safe_load(OFFICIAL_REPOS_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported configs/official_repos.yaml schema_version")
    return data


def resolved_method(name: str) -> dict:
    """Return method settings plus inherited global settings and its fingerprint."""
    cfg = load_baselines()
    if name not in cfg["methods"]:
        raise KeyError(f"Unknown method: {name}")
    resolved = {"global": cfg["global"], "method": cfg["methods"][name]}
    return {**resolved, "method_config_id": config_id(resolved)}
