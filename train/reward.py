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

Current implementation: EM × f(L) length decay (direct port of GrepSeek's design).
Workspace stats are logged but not yet included in the training signal.
"""
from __future__ import annotations

import math
import re
from typing import Any

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
    """Extract final answer from trajectory text."""
    blocks = _ANSWER_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    # Fallback: strip tags, take last non-empty line.
    clean = _TOOL_CALL_RE.sub("", text)
    clean = _THINK_RE.sub("", clean)
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Format gate
# ---------------------------------------------------------------------------

_FORMAT_TAGS = ("think", "tool_call", "answer")
_SPLIT_RE = re.compile(r"(</?(?:think|tool_call|answer)>)")
_TAG_RE = re.compile(r"</?(?:think|tool_call|answer)>")


def _check_format(text: str) -> tuple[bool, str]:
    """Lightweight structural check.

    A valid ScaleSeek trajectory ends in <answer>...</answer> and has balanced tags.
    Does NOT enforce strict ordering (that can be tightened once SFT format is locked).

    TODO: tighten after SFT cold-start data format is finalized.
    """
    for tag in _FORMAT_TAGS:
        n_open = len(re.findall(rf"<{tag}>", text))
        n_close = len(re.findall(rf"</{tag}>", text))
        if n_open != n_close:
            return False, f"unbalanced <{tag}>"
    if not _ANSWER_RE.search(text):
        return False, "no <answer> block"
    return True, "ok"


# ---------------------------------------------------------------------------
# Workspace efficiency reward  [TBD]
# ---------------------------------------------------------------------------

def _workspace_penalty(
    workspace_size: int,
    n_bm25_calls: int,
    *,
    max_workspace_size: int,   # TODO: tune
    max_bm25_calls: int,       # TODO: tune
) -> float:
    """Placeholder for workspace-efficiency penalty term.

    TODO: design this properly once we understand what workspace sizes/call counts
    the prompt baseline produces. Candidates:
      - penalize workspace_size > threshold (discourage over-retrieval)
      - penalize n_bm25_calls > 1 unless the second call used 'merge'
      - reward small workspaces that still yield correct answers

    For now returns 0.0 (no effect on training signal).
    """
    return 0.0  # TBD


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
    # Workspace penalty  [TBD]
    enable_workspace_penalty: bool = False,
    max_workspace_size: int = 50,
    max_bm25_calls: int = 3,
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

    # 3. Format gate.
    # Qwen3 chat template prepends <think>\n to the generated content; verl's
    # solution_str is the generated portion only (missing the opening <think>).
    fmt_input = solution_str
    if not fmt_input.lstrip().startswith("<think>"):
        fmt_input = "<think>\n" + fmt_input
    format_pass, _ = _check_format(fmt_input)

    # 4. Compute EM / F1.
    em = 1.0 if (golden_answers and exact_match(prediction, golden_answers)) else 0.0
    f1_val = compute_f1(prediction, golden_answers) if golden_answers else 0.0

    _metric = (reward_metric or "em").lower().strip()
    base = em if _metric == "em" else f1_val

    # 5. Length penalty.
    pen_L = _length_penalty(L, decay_type=decay_type, a=a, L_max=L_max) if enable_length_decay else 0.0

    # 6. Workspace penalty  [TBD].
    pen_ws = (
        _workspace_penalty(workspace_size, n_bm25_calls,
                           max_workspace_size=max_workspace_size,
                           max_bm25_calls=max_bm25_calls)
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
