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

    if is_validation:
        enable_length_decay = False
        enable_workspace_penalty = False

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
            "reward/hit_max_context": float(hit_max_context),
            "reward/invalid_tool_request": float(invalid_tool),
            "metrics/response_token_count": L,
            "metrics/n_bm25_calls": n_bm25_calls,
            "metrics/workspace_size": workspace_size,
            "metrics/L_max": L_max,
            "metrics/a": a,
        }

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

    raw_score = base - pen_L - pen_ws
    score = raw_score if format_pass else 0.0

    m = _base_metrics()
    m.update({
        "reward/em": em,
        "reward/f1": f1_val,
        "reward/base": base,
        "reward/length_decay": pen_L,
        "reward/workspace_penalty": pen_ws,
        "reward/format_pass": float(format_pass),
    })
    return {"score": score, "reward/total": score, **m}
