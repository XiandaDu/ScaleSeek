"""Tests for the ScaleSeek workspace-efficiency reward term."""
from __future__ import annotations

import pytest

from train.reward import scaleseek_reward


def _raw(solution_str: str, **extra) -> dict:
    """Full reward dict for a verbatim rollout string (no fixture scaffolding)."""
    return scaleseek_reward(
        "d", solution_str, {"golden_answers": ["Paris"]},
        {"response_token_count": 100, "workspace_size": 5, "n_bm25_calls": 1, **extra},
        enable_length_decay=False, enable_workspace_penalty=False,
    )


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


# --- Regression: the format gate must not assume thinking is on ------------
# 08bd315 froze scaleseek to enable_thinking=false. Rollouts then carry no
# <think> tags, the gate unconditionally prepended an opener, saw 1 open /
# 0 close, and zeroed every reward — GRPO got an all-zero group, hence zero
# advantage and zero gradient. The old fixture below hid this by embedding a
# bare </think>, so all five tests passed against a dead training signal.

def test_nothink_rollout_scores_like_a_thinking_rollout():
    """enable_thinking=false is a frozen production mode; it must still train."""
    nothink = _raw("<answer>Paris</answer>")
    thinking = _raw("<think>\nI recall Paris.\n</think>\n<answer>Paris</answer>")
    template = _raw("I recall Paris.</think>\n<answer>Paris</answer>")  # verl: opener in prompt
    for name, sol in (("nothink", nothink), ("thinking", thinking), ("template", template)):
        assert sol["reward/format_pass"] == 1.0, f"{name} failed the format gate"
        assert sol["reward/em"] == 1.0, f"{name} lost the EM"
        assert sol["score"] > 0.0, f"{name} scored {sol['score']} on a correct answer"


def test_answer_rehearsed_inside_thinking_is_not_the_commitment():
    """Reward must score the same string evaluation scores (shared visible_text)."""
    from eval.agent import parse_assistant
    turn = "<think>\nmaybe <answer>Berlin</answer>?\n</think>\n<answer>Paris</answer>"
    assert parse_assistant(turn).answer == "Paris"          # eval side
    assert _raw(turn)["reward/em"] == 1.0                   # RL side agrees


def test_multi_turn_trajectory_takes_the_final_answer():
    """A rollout is tool-call turns then an answer turn; the last answer wins."""
    traj = ('<think>\nsearch first\n</think>\n'
            '<tool_call>\n{"name": "bm25_retrieve", "arguments": {"query": "capital"}}\n</tool_call>\n'
            '<think>\nfound it\n</think>\n<answer>Paris</answer>')
    out = _raw(traj)
    assert out["reward/format_pass"] == 1.0
    assert out["reward/em"] == 1.0


def test_no_answer_block_still_fails():
    """The gate must stay a gate: a rollout that never answers gets nothing."""
    out = _raw('<think>\nI have <answer>Paris</answer> in mind\n</think>\nstill thinking')
    assert out["reward/failed_rollout"] == 1.0
    assert out["score"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
