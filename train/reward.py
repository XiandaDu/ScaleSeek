"""Reward function for ScaleSeek GRPO training.

Reward composition (mirrors GrepSeek's pattern; details TBD):

    score = FORMAT_GATE × (BASE − α·p(L) − β·workspace_penalty)

    BASE: EM or F1 against golden_answers  [TBD: which metric, or mix]
    p(L): length penalty — penalizes long token trajectories
    workspace_penalty: [TBD] penalize oversized or low-density workspaces

Components that are TBD:
    - SFT warm-start vs. cold-start regime (affects format gate strictness)
    - Workspace-efficiency reward (relevant-doc density, workspace size)
    - Curriculum: harder examples / larger corpus as training progresses
    - BM25 parameter bonus (reward good k1/b choices) — or leave to EM signal

Current implementation: EM × f(L) length decay (direct port of GrepSeek's design)
plus the workspace-efficiency penalty, which `train/config/grpo_trainer.yaml`
enables (`enable_workspace_penalty: true`). The function default stays False so
that importing this module without the ScaleSeek config reproduces the plain
GrepSeek reward.
"""
from __future__ import annotations

import math
import re
from typing import Any

from eval.agent import final_answer, visible_text
from eval.metrics import normalize, exact_match, f1 as compute_f1


# ---------------------------------------------------------------------------
# Length penalty (same shapes as GrepSeek)
# ---------------------------------------------------------------------------

def _length_penalty(L: int, *, decay_type: str, a: float, L_max: int) -> float:
    """p(L) ∈ [0, a], p(0)=0, p(L_max)=a."""
    t = min(L, L_max) / max(L_max, 1)
    if decay_type == "linear":
        return a * t
    if decay_type == "quadratic":
        return a * t ** 2
    if decay_type == "cosine":
        return a * 0.5 * (1.0 - math.cos(math.pi * t))
    if decay_type == "exponential":
        return a * (math.exp(t) - 1.0) / (math.e - 1.0)
    raise ValueError(f"Unknown decay_type: {decay_type!r}")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_prediction(text: str) -> str:
    """Extract the final committed answer from trajectory text.

    Delegates to `eval.agent.final_answer` so the reward optimises exactly the
    string evaluation would score. Previously this searched the *raw* text, so an
    <answer> rehearsed inside <think> could win — the same defect that broke the
    evaluation parser (fixed there in 19198ee, missed here).

    No last-line fallback: the format gate already zeroes any rollout without an
    <answer> outside thinking, so a fabricated prediction could never raise the
    score — it could only log a bogus reward/em (e.g. trailing prose that happens
    to contain the gold) and mislabel a non-answer as a scored rollout. Evaluation
    treats the same rollout as a parse_error; the two now agree.
    """
    return final_answer(text) or ""


# ---------------------------------------------------------------------------
# Format gate
# ---------------------------------------------------------------------------

_FORMAT_TAGS = ("think", "tool_call", "answer")
_SPLIT_RE = re.compile(r"(</?(?:think|tool_call|answer)>)")
_TAG_RE = re.compile(r"</?(?:think|tool_call|answer)>")


def _normalize_think(text: str) -> str:
    """Re-attach the opening <think> that the chat template emitted into the prompt.

    verl's `solution_str` is the generated portion only. In thinking mode the Qwen3
    template has already written `<think>` into the prompt, so the rollout starts
    mid-block and closes with a bare `</think>`. Re-attach the opener *only when the
    rollout actually closed a think block* — with `enable_thinking=false` there are no
    think tags at all, and unconditionally prepending one made the gate see 1 open /
    0 close, fail every rollout, and hand GRPO an all-zero reward group (zero
    advantage, zero gradient, silent no-op).
    """
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        return "<think>\n" + text
    return text


def _check_format(text: str) -> tuple[bool, str]:
    """Lightweight structural check.

    A valid ScaleSeek trajectory ends in <answer>...</answer> and has balanced tags.
    Thinking is optional: a rollout generated with `enable_thinking=false` carries no
    <think> tags and is still well-formed. Does NOT enforce strict ordering (that can
    be tightened once SFT format is locked).

    TODO: tighten after SFT cold-start data format is finalized.
    """
    text = _normalize_think(text)
    for tag in _FORMAT_TAGS:
        n_open = len(re.findall(rf"<{tag}>", text))
        n_close = len(re.findall(rf"</{tag}>", text))
        if n_open != n_close:
            return False, f"unbalanced <{tag}>"
    if not _ANSWER_RE.search(visible_text(text)):
        return False, "no <answer> block outside thinking"
    return True, "ok"


# ---------------------------------------------------------------------------
# Workspace efficiency reward  [TBD]
# ---------------------------------------------------------------------------

def _workspace_penalty(
    workspace_size: int,
    n_bm25_calls: int,
    *,
    workspace_target: int,
    max_workspace_size: int,
    max_bm25_calls: int,
    workspace_coef: float,
    bm25_call_coef: float,
) -> float:
    """Workspace-efficiency penalty in [0, workspace_coef + bm25_call_coef].

    Rewards getting the answer into a SMALL bounded workspace with few retrievals.
    Combined with the EM base (which already requires retrieving the answer), this
    is the signal that pushes the policy to tune top_k / k1 / b so the gold lands in
    a tight workspace instead of brute-forcing a large one — the reward supplies the
    "did retrieval work efficiently" signal the model cannot observe on its own.

    - No penalty while workspace_size <= workspace_target; ramps linearly to
      workspace_coef at max_workspace_size.
    - No penalty for the first bm25_retrieve; ramps to bm25_call_coef at
      max_bm25_calls (discourages redundant re-retrieval).
    """
    span_ws = max(1, max_workspace_size - workspace_target)
    ws_over = min(1.0, max(0, workspace_size - workspace_target) / span_ws)
    span_calls = max(1, max_bm25_calls - 1)
    call_over = min(1.0, max(0, n_bm25_calls - 1) / span_calls)
    return workspace_coef * ws_over + bm25_call_coef * call_over


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def scaleseek_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    # Length penalty
    decay_type: str = "linear",
    a: float = 0.5,
    L_max: int = 12288,
    enable_length_decay: bool = True,
    # Reward signal
    reward_metric: str = "em",        # "em" | "f1"  [TBD: may mix both]
    failed_rollout_reward: float = 0.0,
    # Workspace-efficiency penalty
    enable_workspace_penalty: bool = False,
    workspace_target: int = 5,
    max_workspace_size: int = 50,
    max_bm25_calls: int = 3,
    workspace_coef: float = 0.2,
    bm25_call_coef: float = 0.1,
    # How penalties combine with the base signal:
    #   "additive": score = base - p(L) - workspace_penalty   (original GrepSeek
    #               port; can go NEGATIVE for a wrong answer, so in a GRPO group
    #               where every rollout is wrong the only surviving signal is
    #               "fewer calls / smaller workspace" — the mechanism behind the
    #               step-190 collapse to one default top-3 retrieval, see
    #               reports/rl_continuation_audit_2026-09-04.md)
    #   "gated":    score = (base + retrieval_coef * ret_coverage) * (1 - p(L) -
    #               workspace_penalty), floored at 0. A wrong answer with no
    #               evidence is 0 regardless of how cheaply it was reached.
    # v3 (2026-09-16) replaces the whole penalty machinery with a flat sum; see
    # `reward_shape` below. "penalty" keeps the v1/v2 behaviour for reproducing the audit.
    reward_shape: str = "penalty",      # "penalty" | "simple" | "f1_gated"
    em_coef: float = 1.0,               # simple: a correct answer is worth exactly this
    format_coef: float = 0.1,           # simple: well-formed trajectory ending in <answer>
    grep_hit_coef: float = 0.3,         # simple: a grep surfaced a passage containing the answer
    grep_hit_source: str = "gold",      # "gold": grep showed a GOLD answer (retrieval succeeded)
                                        # "prediction": grep showed the rollout's OWN answer (grounding)
    penalty_mode: str = "additive",
    # Retrieval-process credit: fraction of the question's supporting documents
    # (eval/evidence.py; computed by ScaleSeekAgentLoop over every passage the
    # rollout ever retrieved) — dense-ish signal in all-wrong groups that points
    # at "find the evidence", not "call fewer tools". 0 disables it.
    retrieval_coef: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute ScaleSeek reward for one rollout.

    Reward pipeline:
      1. invalid_tool_request  → 0
      2. empty / no answer     → failed_rollout_reward
      3. format gate fails     → 0
      4. score = BASE − p(L) − workspace_penalty
         where BASE = EM or F1, p(L) is length decay, workspace_penalty is TBD.

    Returns a dict with "score" (training signal) + logging metrics.
    """
    L = int(extra_info.get("response_token_count", 0))
    hit_max_context = bool(extra_info.get("hit_max_context", False))
    invalid_tool = bool(extra_info.get("invalid_tool_request", False))
    n_bm25_calls = int(extra_info.get("n_bm25_calls", 0))
    workspace_size = int(extra_info.get("workspace_size", 0))
    is_validation = bool(extra_info.get("is_validation", False))

    grep_hit = float(extra_info.get("grep_hit", 0.0) or 0.0)
    grep_hit_broad = float(extra_info.get("grep_hit_broad", 0.0) or 0.0)
    grep_hit_prediction = float(extra_info.get("grep_hit_prediction", 0.0) or 0.0)
    ret_coverage = float(extra_info.get("ret_coverage", 0.0) or 0.0)
    gold_in_ws = float(extra_info.get("gold_in_ws", 0.0) or 0.0)
    n_grep_calls = int(extra_info.get("n_grep_calls", 0) or 0)
    n_read_calls = int(extra_info.get("n_read_calls", 0) or 0)
    n_seen_docs = int(extra_info.get("n_seen_docs", 0) or 0)

    if is_validation:
        # validation reports plain EM, comparable across reward variants
        enable_length_decay = False
        enable_workspace_penalty = False
        retrieval_coef = 0.0

    golden_answers: list[str] = ground_truth.get("golden_answers", [])

    def _base_metrics() -> dict:
        return {
            "reward/em": 0.0,
            "reward/f1": 0.0,
            "reward/base": 0.0,
            "reward/length_decay": 0.0,
            "reward/workspace_penalty": 0.0,
            "reward/format_pass": 0.0,
            "reward/failed_rollout": 0.0,
            "reward/ret_coverage": ret_coverage,
            "reward/grep_hit": grep_hit,
            "reward/grep_hit_broad": grep_hit_broad,
            "reward/grep_hit_prediction": grep_hit_prediction,
            "reward/grep_credit": 0.0,
            "reward/grep_reward": 0.0,
            "reward/gold_in_ws": gold_in_ws,
            "reward/retrieval_credit": 0.0,
            "reward/hit_max_context": float(hit_max_context),
            "reward/invalid_tool_request": float(invalid_tool),
            "metrics/response_token_count": L,
            "metrics/n_bm25_calls": n_bm25_calls,
            "metrics/workspace_size": workspace_size,
            "metrics/n_grep_calls": n_grep_calls,
            "metrics/n_read_calls": n_read_calls,
            "metrics/n_seen_docs": n_seen_docs,
            "metrics/L_max": L_max,
            "metrics/a": a,
        }

    # ── v3.1 "f1_gated" shape (GrepSeek-style multiplicative format gate) ─────
    #   reward = format_valid * (token_f1(pred, gold) + 0.3 * grep_reward)
    #   grep_reward = 1 only when the answer is EXACTLY right AND a grep surfaced the gold,
    #                 i.e. the bonus pays for "you located the evidence you answered from",
    #                 never for grepping around a wrong answer.
    # Differences from "simple": partial credit via token F1 instead of binary EM (a near-miss
    # like "Alexander I" for "Alexander I of Russia" is no longer worth the same as a wrong
    # answer), format is a gate rather than a 0.1 bonus, and the grep term is conditioned on
    # correctness.
    if reward_shape == "f1_gated":
        prediction = _extract_prediction(solution_str)
        format_pass, _ = _check_format(solution_str)
        em = 1.0 if (prediction and golden_answers and exact_match(prediction, golden_answers)) else 0.0
        f1_val = compute_f1(prediction, golden_answers) if (prediction and golden_answers) else 0.0
        g = grep_hit_prediction if grep_hit_source == "prediction" else grep_hit
        grep_reward = float(em > 0.0 and g > 0.0)
        grep_credit = grep_hit_coef * grep_reward
        score = float(format_pass) * (f1_val + grep_credit)
        if is_validation:
            score = em      # validation reports plain EM, comparable across reward versions
        m = _base_metrics()
        m.update({"reward/em": em, "reward/f1": f1_val, "reward/base": f1_val,
                  "reward/format_pass": float(format_pass),
                  "reward/failed_rollout": float(not prediction),
                  "reward/grep_reward": grep_reward, "reward/grep_credit": grep_credit})
        return {"score": score, "reward/total": score, **m}

    # ── v3 "simple" shape ────────────────────────────────────────────────────
    # score = 1.0*EM + 0.1*format + 0.2*grep-found-the-answer. No penalties, no
    # multiplicative gating, nothing can be negative. The v2 shape zeroed a rollout for a
    # malformed tool call or an unbalanced tag and taxed every extra retrieval, which left
    # most GRPO groups flat and (in v1) made "call fewer tools" the only surviving gradient.
    # Here a wrong answer that still located the evidence keeps a positive, learnable score.
    if reward_shape == "simple":
        prediction = _extract_prediction(solution_str)
        format_pass, _ = _check_format(solution_str)
        em = 1.0 if (prediction and golden_answers and exact_match(prediction, golden_answers)) else 0.0
        f1_val = compute_f1(prediction, golden_answers) if (prediction and golden_answers) else 0.0
        g = grep_hit_prediction if grep_hit_source == "prediction" else grep_hit
        # The grep bonus requires the rollout to have actually answered: otherwise
        # "grep forever, never commit" would be the cheapest positive score on the board.
        grep_credit = grep_hit_coef * g * float(format_pass)
        score = em_coef * em + format_coef * float(format_pass) + grep_credit
        if is_validation:
            score = em      # validation reports plain EM, comparable across reward versions
        m = _base_metrics()
        m.update({"reward/em": em, "reward/f1": f1_val, "reward/base": em,
                  "reward/format_pass": float(format_pass),
                  "reward/failed_rollout": float(not prediction),
                  "reward/grep_credit": grep_credit})
        return {"score": score, "reward/total": score, **m}

    # 1. Invalid tool request.
    if invalid_tool:
        return {"score": 0.0, "reward/total": 0.0, **_base_metrics()}

    # 2. Failed rollout (empty or no final answer).
    prediction = _extract_prediction(solution_str)
    is_failed = not prediction
    if is_failed:
        m = _base_metrics()
        m["reward/failed_rollout"] = 1.0
        return {"score": failed_rollout_reward, "reward/total": failed_rollout_reward, **m}

    # 3. Format gate (thinking-optional; see _normalize_think).
    format_pass, _ = _check_format(solution_str)

    # 4. Compute EM / F1.
    em = 1.0 if (golden_answers and exact_match(prediction, golden_answers)) else 0.0
    f1_val = compute_f1(prediction, golden_answers) if golden_answers else 0.0

    _metric = (reward_metric or "em").lower().strip()
    base = em if _metric == "em" else f1_val

    # 5. Length penalty.
    pen_L = _length_penalty(L, decay_type=decay_type, a=a, L_max=L_max) if enable_length_decay else 0.0

    # 6. Workspace-efficiency penalty.
    pen_ws = (
        _workspace_penalty(workspace_size, n_bm25_calls,
                           workspace_target=workspace_target,
                           max_workspace_size=max_workspace_size,
                           max_bm25_calls=max_bm25_calls,
                           workspace_coef=workspace_coef,
                           bm25_call_coef=bm25_call_coef)
        if enable_workspace_penalty else 0.0
    )

    credit = retrieval_coef * ret_coverage
    if penalty_mode == "gated":
        raw_score = (base + credit) * max(0.0, 1.0 - pen_L - pen_ws)
    elif penalty_mode == "additive":
        raw_score = base + credit - pen_L - pen_ws
    else:
        raise ValueError(f"Unknown penalty_mode: {penalty_mode!r}")
    score = raw_score if format_pass else 0.0

    m = _base_metrics()
    m.update({
        "reward/em": em,
        "reward/f1": f1_val,
        "reward/base": base,
        "reward/length_decay": pen_L,
        "reward/workspace_penalty": pen_ws,
        "reward/format_pass": float(format_pass),
        "reward/retrieval_credit": credit,
    })
    return {"score": score, "reward/total": score, **m}
