"""Retrieval-process evidence: which documents *should* have been retrieved.

FlashRAG's train rows carry per-question supporting evidence that the canonical
{id, question, golden_answers} schema drops:

  hotpotqa / 2wikimultihopqa   metadata.supporting_facts.title  -> Wikipedia titles
  musique                      metadata.question_decomposition  -> per-hop support
                               paragraph titles and per-hop sub-answers
  nq                           nothing (single-hop; fall back to "gold string in
                               a retrieved passage")

wiki-18 passages start with their article title on the first line, quoted, so a
title-level match against every passage the policy ever pulled into its
workspace is a cheap, exact process signal ("did retrieval find the evidence"),
independent of whether the final answer was right. It is the same title-level
Gold R@W notion scripts/compute_metrics.py reports, reused here as an RL reward
term (train/reward.py, retrieval_coef).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .metrics import normalize

MULTI_HOP_SOURCES = {"hotpotqa", "2wikimultihopqa", "musique"}


def extract_evidence(name: str, row: dict) -> dict:
    """Pull support titles / sub-answers out of a raw FlashRAG row (empty if none)."""
    md = row.get("metadata") or {}
    if not isinstance(md, dict):
        return {}
    ev: dict[str, Any] = {}
    if name in ("hotpotqa", "2wikimultihopqa"):
        sf = md.get("supporting_facts") or {}
        titles = sf.get("title") if isinstance(sf, dict) else None
        if titles:
            ev["support_titles"] = list(dict.fromkeys(str(t) for t in titles if t))
    elif name == "musique":
        titles, subs = [], []
        for hop in md.get("question_decomposition") or []:
            sp = hop.get("support_paragraph") or {}
            if sp.get("title"):
                titles.append(str(sp["title"]))
            if hop.get("answer"):
                subs.append(str(hop["answer"]))
        if titles:
            ev["support_titles"] = list(dict.fromkeys(titles))
        if subs:
            ev["sub_answers"] = subs
    return ev


_QUOTED = re.compile(r'^\s*"+(.*?)"+\s*$')


def doc_title(text: str) -> str:
    """First line of a wiki-18 passage, unquoted."""
    first = (text or "").split("\n", 1)[0]
    m = _QUOTED.match(first)
    return (m.group(1) if m else first).strip()


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower().strip().strip('"').strip()
    return re.sub(r"\s+", " ", t)


def coverage(evidence: dict, golden_answers: list[str], docs: list[dict]) -> dict:
    """Score how much of the evidence the retrieved passages (`docs`, each with
    "text") cover. Returns:
        ret_coverage      fraction of support titles found (title equality on the
                          first line, or normalized title contained in a passage);
                          for questions without support titles = gold_in_ws
        gold_in_ws        1.0 if any gold answer string occurs in any passage
        sub_answers_found fraction of musique sub-answers found (0 if none given)
        n_support / n_support_found
    """
    texts = [d.get("text", "") or "" for d in docs]
    titles = {norm_title(doc_title(t)) for t in texts}
    norm_texts = [normalize(t) for t in texts]

    def in_texts(s: str) -> bool:
        ns = normalize(s)
        return bool(ns) and any(ns in nt for nt in norm_texts)

    gold_in_ws = float(any(in_texts(g) for g in (golden_answers or [])))
    support = list((evidence or {}).get("support_titles") or [])
    found = 0
    for t in support:
        if norm_title(t) in titles or in_texts(t):
            found += 1
    subs = list((evidence or {}).get("sub_answers") or [])
    sub_found = (sum(1 for s in subs if in_texts(s)) / len(subs)) if subs else 0.0
    ret = (found / len(support)) if support else gold_in_ws
    return {"ret_coverage": ret, "gold_in_ws": gold_in_ws, "sub_answers_found": sub_found,
            "n_support": len(support), "n_support_found": found}


# ---------------------------------------------------------------------------
# Phrase containment (v3 grep reward)
# ---------------------------------------------------------------------------

# A grep pattern with fewer than this many alphanumeric characters is treated as a
# catch-all (".", ".*", "\\w", "a|e") and earns no grep credit: matching every passage
# is not "locating the evidence", it is disabling the filter.
GREP_MIN_PATTERN_CHARS = 3


def contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` occurs in `text` as a whole-token run, after normalize().

    Plain substring matching is wrong for the short answers this benchmark is full of:
    gold "no" is a substring of "nothing", "1946" of "19468". Token-run matching keeps
    multi-word answers ("Los Angeles County") exact while refusing those false hits.
    """
    t = normalize(text).split()
    ph = normalize(phrase).split()
    if not ph or len(ph) > len(t):
        return False
    return any(t[i:i + len(ph)] == ph for i in range(len(t) - len(ph) + 1))


def any_phrase_in(text: str, phrases) -> bool:
    return any(contains_phrase(text, p) for p in (phrases or []) if str(p).strip())


def pattern_is_substantive(pattern: str) -> bool:
    return len(re.sub(r"[^a-z0-9]", "", (pattern or "").lower())) >= GREP_MIN_PATTERN_CHARS
