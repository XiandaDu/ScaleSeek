"""Byte diff of every copied official prompt against the pinned checkout."""
from pathlib import Path

import pytest

from scaleseek.config import load_all_methods, load_base
from scaleseek.official import verify_method


def test_official_byte_diffs():
    base = load_base()
    if not (Path(base["paths"]["repos"]) / "Search-R1").exists():
        pytest.skip("official checkouts not fetched")
    for name, m in load_all_methods(base).items():
        rec = verify_method(base, m)
        for slot, v in rec["byte_diff"].items():
            assert v == "project-defined" or v.startswith("identical"), (name, slot, v)
