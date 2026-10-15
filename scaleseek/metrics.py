"""Scoring (BASELINES.md §7): EM / token F1 with 95% intervals, run statistics, McNemar.

One scorer for every method (SQuAD normalisation, max over gold answers; the same
definition as Search-R1's qa_em and GrepSeek's inference/scoring.py).  Only a record that
ENDED with an answer (finish_reason "answer") has a scored answer -- the same rule for every
method, also for the BCP judge; anything else scores 0.  Before anything is scored, the
run's id set must EQUAL the subset manifest's (formal runs) -- a partial run cannot produce
a number.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

from .utils import PipelineError


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def final_answer(rec: dict) -> str | None:
    """The answer that is scored and judged: the prediction of a record that ended with an
    answer, else None (an answer seen mid-trajectory before max_turns / a full context does
    not count)."""
    return rec["prediction"] if rec["finish_reason"] == "answer" else None


def exact_match(pred: str | None, golds: list[str]) -> bool:
    if not pred:
        return False
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def _f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str | None, golds: list[str]) -> float:
    if not pred:
        return 0.0
    p = normalize_answer(pred)
    return max(_f1(p, normalize_answer(g)) for g in golds)


def mean_ci(values: list[float], z: float) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {"mean": m, "ci95": z * math.sqrt(var / n), "n": n}


def proportion_ci(k: int, n: int, z: float) -> dict:
    """Normal-approximation interval (BASELINES.md: n=500 -> about +-4.4 points)."""
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    p = k / n
    return {"mean": p, "ci95": z * math.sqrt(p * (1 - p) / n), "n": n}


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Paired exact (binomial) McNemar test on the same question ids."""
    if set(a) != set(b):
        raise PipelineError("McNemar needs the same question ids on both sides")
    only_a = sum(1 for q in a if a[q] and not b[q])
    only_b = sum(1 for q in a if b[q] and not a[q])
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def check_ids(records: list[dict], expected_ids: list[str], partial_ok: bool) -> None:
    got = [r["id"] for r in records]
    dup = len(got) - len(set(got))
    if dup:
        raise PipelineError(f"{dup} duplicate ids in the run output")
    missing = set(expected_ids) - set(got)
    extra = set(got) - set(expected_ids)
    if extra:
        raise PipelineError(f"{len(extra)} ids not in the subset manifest, e.g. {sorted(extra)[:5]}")
    if missing and not partial_ok:
        raise PipelineError(f"{len(missing)} subset ids have no output, e.g. {sorted(missing)[:5]}; "
                            "resume the run before scoring")


def run_metrics(records: list[dict], z: float, wall_time_s: float) -> dict:
    n = len(records)
    em = [exact_match(final_answer(r), r["golden_answers"]) for r in records]
    f1 = [f1_score(final_answer(r), r["golden_answers"]) for r in records]
    answered = sum(1 for r in records if final_answer(r))
    return {
        "n": n,
        "em": proportion_ci(sum(em), n, z),
        "f1": mean_ci(f1, z),
        "answered_rate": answered / n if n else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
        "errors": {k: sum(r["errors"][k] for r in records) for k in ("parse", "tool", "api")},
        "rows_with_errors": {k: sum(1 for r in records if r["errors"][k]) for k in ("parse", "tool", "api")},
        "avg_turns": sum(r["n_turns"] for r in records) / n if n else None,
        "avg_retrievals": sum(r["n_retrievals"] for r in records) / n if n else None,
        "wall_time_s": wall_time_s,
        "per_question_em": {r["id"]: bool(e) for r, e in zip(records, em)},
    }
