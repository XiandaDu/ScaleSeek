"""verl custom reward: format-gated EM (scaleseek.agent.reward) from the outcome fields
the agent loop leaves in extra_info.  Everything after "score" is logged as a metric."""

from __future__ import annotations

from typing import Any

from scaleseek import agent
from scaleseek.metrics import exact_match


def compute_score(data_source: str, solution_str: str, ground_truth: list[str], extra_info: dict[str, Any],
                  draft_credit: float, **kwargs: Any) -> dict[str, Any]:
    golds = list(ground_truth)
    answer, draft, valid = extra_info["answer"], extra_info["draft"], extra_info["valid"]
    pred = answer if answer is not None else draft
    return {"score": agent.reward(answer, draft, golds, valid, draft_credit),
            "em": float(exact_match(pred, golds)) if pred else 0.0,
            "valid": float(valid), "answered": float(answer is not None),
            "draft_only": float(answer is None and draft is not None),
            "n_calls": extra_info["n_calls"], "n_find": extra_info["n_find"], "n_search": extra_info["n_search"],
            "n_invalid": extra_info.get("n_invalid", 0), "n_cut": extra_info.get("n_cut", 0),
            "response_tokens": extra_info["response_tokens"]}
