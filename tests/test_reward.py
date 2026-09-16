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


# --- v2 reward: correctness-gated penalties + retrieval-process credit ----------
# reports/rl_continuation_audit_2026-09-04.md: in the additive form, an all-wrong
# GRPO group still has non-zero advantages coming purely from the penalties,
# which rewards "fewer calls" regardless of correctness.

def _v2(pred: str, ws: int, calls: int, ret: float, **kw) -> float:
    sol = f"searched.</think>\n<answer>\n{pred}\n</answer>"
    return scaleseek_reward(
        "d", sol, {"golden_answers": ["Paris"]},
        {"response_token_count": 100, "workspace_size": ws, "n_bm25_calls": calls,
         "ret_coverage": ret, **kw.pop("extra", {})},
        enable_length_decay=False, enable_workspace_penalty=True,
        workspace_target=10, max_workspace_size=30, max_bm25_calls=5,
        workspace_coef=0.2, bm25_call_coef=0.1,
        penalty_mode="gated", retrieval_coef=0.3, **kw,
    )["score"]


def test_gated_wrong_answer_never_negative_and_call_count_irrelevant():
    assert _v2("Berlin", 30, 5, 0.0) == pytest.approx(0.0)
    assert _v2("Berlin", 5, 1, 0.0) == pytest.approx(0.0)


def test_gated_retrieval_credit_ranks_evidence_found_above_not_found():
    assert _v2("Berlin", 5, 2, 1.0) > _v2("Berlin", 5, 1, 0.0)
    assert _v2("Berlin", 5, 1, 1.0) == pytest.approx(0.3)
    # a second call costs 0.1 * 1/4 of the credit, never more than the credit itself
    assert _v2("Berlin", 5, 2, 1.0) == pytest.approx(0.3 * (1 - 0.025))


def test_gated_correct_answer_dominates_credit():
    assert _v2("Paris", 5, 1, 0.0) > _v2("Berlin", 5, 1, 1.0)
    assert _v2("Paris", 5, 1, 1.0) == pytest.approx(1.3)


def test_gated_penalty_scales_multiplicatively():
    full = _v2("Paris", 10, 1, 1.0)
    big = _v2("Paris", 30, 5, 1.0)
    assert big == pytest.approx(full * (1 - 0.2 - 0.1))


def test_gated_validation_is_plain_em():
    assert _v2("Paris", 30, 5, 1.0, extra={"is_validation": True}) == pytest.approx(1.0)
    assert _v2("Berlin", 5, 1, 1.0, extra={"is_validation": True}) == pytest.approx(0.0)


def test_additive_mode_unchanged_for_reproducibility():
    sol = "x</think>\n<answer>\nBerlin\n</answer>"
    out = scaleseek_reward("d", sol, {"golden_answers": ["Paris"]},
                           {"response_token_count": 100, "workspace_size": 5, "n_bm25_calls": 2},
                           enable_length_decay=False, enable_workspace_penalty=True,
                           workspace_target=5, max_workspace_size=20, max_bm25_calls=3,
                           workspace_coef=0.2, bm25_call_coef=0.1)
    assert out["score"] == pytest.approx(-0.05)


# --- v3 "simple" reward: EM + a little format + a grep-found-the-answer bonus ----------
# No penalties and no gating: the v2 shape zeroed whole rollouts on a format slip and taxed
# every extra retrieval, so most GRPO groups were flat.

def _v3(pred: str, *, fmt=True, grep=0.0, grep_pred=0.0, source="gold", **extra) -> float:
    sol = (f"reasoning</think>\n<answer>\n{pred}\n</answer>" if fmt
           else f"reasoning</think>\n{pred}")
    return scaleseek_reward(
        "d", sol, {"golden_answers": ["Paris"]},
        {"response_token_count": 100, "workspace_size": 40, "n_bm25_calls": 7,
         "grep_hit": grep, "grep_hit_prediction": grep_pred, **extra},
        reward_shape="simple", grep_hit_source=source,
    )["score"]


def test_simple_correct_answer_is_one_plus_format():
    assert _v3("Paris") == pytest.approx(1.1)


def test_simple_ignores_workspace_size_and_call_count():
    assert _v3("Paris", workspace_size=3, n_bm25_calls=1) == pytest.approx(_v3("Paris"))


def test_simple_wrong_answer_that_found_the_evidence_beats_one_that_did_not():
    assert _v3("Berlin", grep=1.0) == pytest.approx(0.4)
    assert _v3("Berlin") == pytest.approx(0.1)
    assert _v3("Berlin", grep=1.0) > _v3("Berlin")


def test_simple_correct_and_grepped_is_the_maximum():
    assert _v3("Paris", grep=1.0) == pytest.approx(1.4)


def test_simple_no_answer_scores_zero_even_with_a_grep_hit():
    # otherwise "grep forever, never commit" is the cheapest positive score
    assert _v3("Paris", fmt=False, grep=1.0) == pytest.approx(0.0)


def test_simple_prediction_source_scores_groundedness_not_correctness():
    assert _v3("Berlin", grep=0.0, grep_pred=1.0, source="prediction") == pytest.approx(0.4)
    assert _v3("Berlin", grep=1.0, grep_pred=0.0, source="prediction") == pytest.approx(0.1)


def test_simple_validation_is_plain_em():
    assert _v3("Paris", grep=1.0, is_validation=True) == pytest.approx(1.0)
    assert _v3("Berlin", grep=1.0, is_validation=True) == pytest.approx(0.0)


# --- v3.1 "f1_gated": reward = format * (token_f1 + 0.3 * (correct AND grep hit)) ------

def _v31(pred: str, *, fmt=True, grep=0.0, gold=("Alexander I of Russia",), **extra) -> float:
    sol = (f"reasoning</think>\n<answer>\n{pred}\n</answer>" if fmt
           else f"reasoning</think>\n{pred}")
    return scaleseek_reward(
        "d", sol, {"golden_answers": list(gold)},
        {"response_token_count": 100, "workspace_size": 40, "n_bm25_calls": 7,
         "grep_hit": grep, **extra},
        reward_shape="f1_gated",
    )["score"]


def test_f1_gated_partial_credit_for_a_near_miss():
    s = _v31("Alexander I")
    assert 0.0 < s < 1.0 and s == pytest.approx(2 * 1.0 * (2 / 4) / (1 + 2 / 4))


def test_f1_gated_exact_answer_is_one_and_with_grep_is_one_point_three():
    assert _v31("Alexander I of Russia") == pytest.approx(1.0)
    assert _v31("Alexander I of Russia", grep=1.0) == pytest.approx(1.3)


def test_f1_gated_grep_bonus_needs_the_answer_to_be_exactly_right():
    # a near-miss that grepped the gold gets F1 only, no 0.3
    assert _v31("Alexander I", grep=1.0) == pytest.approx(_v31("Alexander I"))


def test_f1_gated_format_is_a_gate_not_a_bonus():
    assert _v31("Alexander I of Russia", fmt=False, grep=1.0) == pytest.approx(0.0)


def test_f1_gated_takes_the_best_gold():
    assert _v31("Pound", gold=("pound sterling", "pound")) == pytest.approx(1.0)


def test_f1_gated_validation_is_plain_em():
    assert _v31("Alexander I", grep=1.0, is_validation=True) == pytest.approx(0.0)
    assert _v31("Alexander I of Russia", is_validation=True) == pytest.approx(1.0)
