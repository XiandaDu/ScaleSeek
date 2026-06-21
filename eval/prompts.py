"""Prompt template loader.

All prompts live in prompts/<name>.txt at the repo root.
Loading is cached so repeated calls are free.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Load prompts/<name>.txt, stripping leading/trailing whitespace."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()
