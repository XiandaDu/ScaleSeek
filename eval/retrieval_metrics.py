"""Retrieval / workspace / DCI-process metrics for ScaleSeek eval.

Answer-quality metrics (EM/F1) live in eval/metrics.py. This module implements
the *retrieval-side* metrics defined by the DCI / DR-DCI / Pi-Serini papers, plus
the helpers that reconstruct, from a saved eval record, which corpus documents an
agent actually surfaced.

Definitions (with sources):

  Workspace recall  — DR-DCI (arxiv 2606.14885, §4.1):
      Gold R@W = |W_T ∩ G(q)| / |G(q)|
      Qrel R@W = |W_T ∩ R(q)| / |R(q)|
      W_T = final materialized workspace, G = gold docs, R = qrel/evidence docs.

  Coverage — DCI paper (arxiv 2605.05242, Eq. 1), M = surfaced gold docs:
      coverage_any  = 1[|M| >= 1]
      coverage_mean = |M| / |D*|            (== gold recall)
      coverage_all  = 1[|M| == |D*|]

  Localization — DCI paper (Eqs. 2-5), fixed-width c_seg char segments:
      nu(x)          = max(1, ceil(x / c_seg))
      psi(a; b)      = max(1 - log(a)/log(b), 0)   for 1<=a<=b, b>1 ;  psi(a;1)=1
      seg-score(d,d*)= psi( nu(len_snippet) ; nu(|d*|) )
      s(d*)          = max over aligned snippets of seg-score
      localization   = mean over surfaced gold docs of s(d*)

Keys are compared in a single space per dataset: corpus doc-ids for BrowseComp-Plus
(native qrels), or article titles for wiki-18 QA (HotpotQA / 2Wiki gold labels are
title-level; map workspace doc-ids -> titles before calling these functions).
"""
from __future__ import annotations

import json
import math
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Workspace recall (DR-DCI) and coverage (DCI)
# ---------------------------------------------------------------------------

def recall_over(found: Iterable, gold: Iterable) -> Optional[float]:
    """|found ∩ gold| / |gold|. Returns None when gold is empty (undefined)."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return len(set(found) & gold_set) / len(gold_set)


def gold_recall(workspace_keys: Iterable, gold_keys: Iterable) -> Optional[float]:
    """Gold R@W (DR-DCI) == coverage_mean (DCI)."""
    return recall_over(workspace_keys, gold_keys)


def qrel_recall(workspace_keys: Iterable, qrel_keys: Iterable) -> Optional[float]:
    """Qrel R@W (DR-DCI): recall over the evidence/qrel document set."""
    return recall_over(workspace_keys, qrel_keys)


def coverage(surfaced_keys: Iterable, gold_keys: Iterable) -> dict:
    """DCI Eq. 1: coverage_any / coverage_mean / coverage_all over gold docs."""
    gold_set = set(gold_keys)
    if not gold_set:
        return {"any": None, "mean": None, "all": None}
    m = len(set(surfaced_keys) & gold_set)
    return {
        "any": float(m >= 1),
        "mean": m / len(gold_set),
        "all": float(m == len(gold_set)),
    }


# ---------------------------------------------------------------------------
# Localization (DCI paper, Eqs. 2-5)
# ---------------------------------------------------------------------------

def _nu(x: float, c_seg: int) -> int:
    return max(1, math.ceil(x / c_seg))


def _psi(a: float, b: float) -> float:
    """psi(a;b) = max(1 - log a / log b, 0) for b>1; psi(a;1)=1 (Eq. 2)."""
    if b <= 1:
        return 1.0
    a = min(max(a, 1.0), b)  # snippet segments never exceed the document's
    return max(1.0 - math.log(a) / math.log(b), 0.0)


def seg_score(snippet_len_chars: int, gold_len_chars: int, c_seg: int) -> float:
    """DCI Eq. 3: how localized a snippet is within its gold document."""
    return _psi(_nu(snippet_len_chars, c_seg), _nu(gold_len_chars, c_seg))


def localization(
    aligned_snippets: dict[str, list[int]],
    gold_len_chars: dict[str, int],
    c_seg: int = 500,
) -> Optional[float]:
    """DCI Eqs. 4-5.

    Args:
        aligned_snippets: {gold_doc_key: [snippet_char_len, ...]} — the snippet
            lengths exposed for each *surfaced* gold document along the trajectory.
        gold_len_chars: {gold_doc_key: full_document_char_len}.
        c_seg: fixed segment width in characters (DCI-Agent-Lite §A.3 default).

    Returns trajectory-level localization, or None if no gold doc was surfaced.
    """
    if not aligned_snippets:
        return None
    scores = []
    for key, lens in aligned_snippets.items():
        glen = gold_len_chars.get(key)
        if glen is None or not lens:
            continue
        scores.append(max(seg_score(sl, glen, c_seg) for sl in lens))
    if not scores:
        return None
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Surfaced-document reconstruction from a saved eval record
# ---------------------------------------------------------------------------

def workspace_doc_ids(record: dict) -> list[str]:
    """Final workspace W_T doc-ids for workspace agents (scaleseek / dr_dci /
    bm25_rag / agentir). Falls back to the union of per-pull retrieved ids."""
    ids = record.get("workspace_doc_ids")
    if ids:
        return list(ids)
    seen: list[str] = []
    for call in record.get("bm25_calls", []) or []:
        for d in call.get("doc_ids", []) or []:
            if d not in seen:
                seen.append(d)
    return seen


def _ids_from_corpus_lines(lines: Iterable[str]) -> set[str]:
    """Extract corpus doc-ids from raw corpus JSON lines ({"id":..,"contents":..})."""
    out: set[str] = set()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln[0] != "{":
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict) and "id" in obj:
            out.add(str(obj["id"]))
    return out


def surfaced_doc_ids_from_grep(record: dict) -> set[str]:
    """Doc-ids surfaced by a raw-corpus grep agent (dci / grepseek).

    Parses each saved tool turn: uses `information_lines` when present (GrepSeek),
    else splits `stdout` (DCI). Each surfaced corpus line is a JSON object whose
    "id" is the doc-id.
    """
    surfaced: set[str] = set()
    for turn in record.get("turns", []) or []:
        if turn.get("role") != "tool" or turn.get("synthetic"):
            continue
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except Exception:
            continue
        lines = payload.get("information_lines")
        if not lines:
            stdout = payload.get("stdout") or ""
            lines = stdout.split("\n")
        surfaced |= _ids_from_corpus_lines(lines)
    return surfaced


def surfaced_doc_ids(record: dict, agent: str) -> set[str]:
    """Dispatch: which corpus doc-ids did this agent surface for this example."""
    if agent in ("dci", "grepseek"):
        return surfaced_doc_ids_from_grep(record)
    # workspace agents: everything materialized counts as surfaced
    return set(workspace_doc_ids(record))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_optional(values: Sequence[Optional[float]]) -> dict:
    """Mean over the examples where the metric is defined (non-None)."""
    defined = [v for v in values if v is not None]
    return {
        "mean": (sum(defined) / len(defined)) if defined else None,
        "n_defined": len(defined),
        "n_total": len(values),
    }
