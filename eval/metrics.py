"""Answer-quality metrics for ScaleSeek evaluation.

All functions follow the NQ / TriviaQA convention:
  - normalize before comparison (lower, strip punct, collapse whitespace)
  - EM = 1 if any gold matches prediction exactly after normalization
  - F1 = max token-overlap F1 over all gold answers
"""
from __future__ import annotations

import re
import string
from typing import Optional, Sequence


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    # strip leading articles
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    pred = normalize(prediction)
    return float(any(normalize(g) == pred for g in gold_answers))


def token_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold_answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1(prediction: str, gold_answers: Sequence[str]) -> float:
    return max(token_f1(prediction, g) for g in gold_answers) if gold_answers else 0.0


def score_example(prediction: Optional[str], gold_answers: Sequence[str]) -> dict:
    if prediction is None:
        return {"em": 0.0, "f1": 0.0}
    return {"em": exact_match(prediction, gold_answers), "f1": f1(prediction, gold_answers)}


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    return {
        "em": sum(s["em"] for s in scores) / len(scores),
        "f1": sum(s["f1"] for s in scores) / len(scores),
        "n": len(scores),
    }


# --- Retrieval metrics ---

def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = set(retrieved_ids[:k]) & set(gold_ids)
    return len(hits) / len(gold_ids)
