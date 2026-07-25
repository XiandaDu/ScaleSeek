"""Tests for the ScaleSeek workspace-efficiency reward term."""
from __future__ import annotations

import pytest

from train.reward import scaleseek_reward


def _score(ws: int, calls: int, pred: str = "Paris", **kw) -> float:
    sol = f"I searched the corpus.</think>\n<answer>\n{pred}\n</answer>"
    out = scaleseek_reward(
        "d", sol, {"golden_answers": ["Paris"]},
        {"response_token_count": 100, "workspace_size": ws, "n_bm25_calls": calls, **kw.pop("extra", {})},
        enable_length_decay=False, enable_workspace_penalty=True,
        workspace_target=5, max_workspace_size=20, max_bm25_calls=3,
        workspace_coef=0.2, bm25_call_coef=0.1, **kw,
    )
    return out["score"]


def test_no_penalty_at_or_below_target():
    assert _score(5, 1) == pytest.approx(1.0)   # at target, single retrieval
    assert _score(3, 1) == pytest.approx(1.0)   # below target


def test_large_workspace_is_penalized():
    assert _score(20, 1) == pytest.approx(1.0 - 0.2)          # full workspace penalty
    assert _score(20, 3) == pytest.approx(1.0 - 0.2 - 0.1)    # + redundant-retrieval penalty
    assert _score(5, 1) > _score(20, 1)                       # tighter workspace scores higher


def test_penalty_is_monotonic_and_bounded():
    scores = [_score(w, 1) for w in (5, 10, 15, 20, 40)]
    assert scores == sorted(scores, reverse=True)             # non-increasing in workspace size
    assert _score(9999, 9999) >= 1.0 - 0.2 - 0.1 - 1e-9       # bounded below by total coef


def test_penalty_off_in_validation():
    assert _score(20, 3, extra={"is_validation": True}) == pytest.approx(1.0)


def test_wrong_answer_still_ranks_below_correct():
    assert _score(5, 1, pred="Berlin") < _score(20, 1, pred="Paris")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
