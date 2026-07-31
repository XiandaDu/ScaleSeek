"""BM25 config race for cold-start SFT data generation.

Design (2026-07-30, user-specified): instead of asking the teacher to *reason
out* k1/b — which reports/param_policy_findings.md showed teachers never do and
students don't learn from — we race a fixed grid of named configs on the same
mentor query and let the **outcome** pick the trajectory:

    1. mentor writes the BM25 query (unchanged pipeline)
    2. every config retrieves with that query           (cheap, no teacher calls)
    3. the most promising configs each continue a full
       trajectory (grep + follow-up bm25)               (teacher-priced, capped)
    4. trajectories are scored on evidence support      (see score_trajectory)
    5. best becomes the SFT positive
    6. the rest become preference/failure data

This mirrors what the GrepSeek paper's public pipeline explicitly dropped
("single trajectory per question, no multi-strategy sampling") and adopts s3's
Gain-Beyond-RAG idea: a config's score is only interesting *relative to the
default config's trajectory*, so we always race the default as the baseline arm.

Scoring note: in this pipeline the final <answer> is pinned to gold, so EM of
the answer is NOT a discriminator (it is 1 by construction). What discriminates
configs is whether the *workspace they build* actually supports the gold answer,
per hop. "Answer correctness" from the user's spec therefore maps to
`answer_supported`; "evidence coverage" maps to per-hop workspace support.

k1/b only. top_k is deliberately NOT part of the race grid: the report showed
top_k is already learnable from query shape, and racing it would conflate the
two mechanisms (and top_k=1000-style wide pulls fight the bounded-workspace
thesis in TARGET.md).
"""
from __future__ import annotations

from typing import Any, Optional

# Named (k1, b) grid: 3 levels each, as specified ("k1 小中大, b 小中大").
# `rationale` must stay inference-reachable (query-shape-plausible, no oracle
# facts) — the student can repeat it, and per the report the *when-to-deviate*
# signal is RL's job, not SFT's.
BM25_CONFIGS: list[dict[str, Any]] = [
    {"name": "default",      "k1": 1.2, "b": 0.75,
     "rationale": "Standard balance of term saturation and length normalization."},
    {"name": "k1_low",       "k1": 0.5, "b": 0.75,
     "rationale": "Damp repeated-term influence so no single term dominates."},
    {"name": "k1_high",      "k1": 2.0, "b": 0.75,
     "rationale": "Let repeated exact mentions weigh more for this specific query."},
    {"name": "b_low",        "k1": 1.2, "b": 0.25,
     "rationale": "Relax length normalization so long comprehensive passages are not penalized."},
    {"name": "b_high",       "k1": 1.2, "b": 1.0,
     "rationale": "Strong length normalization to favor short focused passages."},
    {"name": "k1low_blow",   "k1": 0.5, "b": 0.25,
     "rationale": "Broad recall: damp term repetition and length penalties together."},
    {"name": "k1high_bhigh", "k1": 2.0, "b": 1.0,
     "rationale": "High precision: reward exact repeated terms in short passages."},
    {"name": "k1low_bhigh",  "k1": 0.5, "b": 1.0,
     "rationale": "Short passages without letting term repetition dominate."},
    {"name": "k1high_blow",  "k1": 2.0, "b": 0.25,
     "rationale": "Long passages where repeated exact mentions matter."},
    # Long-document arms, evidence-backed (2026-07-30 correction): the teens-k1
    # recollection DID verify — against Pi-Serini (arXiv:2605.10848), not RISE.
    # Pi-Serini §4.2 runs BCP with tuned k1=25, b=1; its §6 grid search puts the
    # optimum near (k1=16, b=1.0) and Anserini's default (0.9, 0.4) in a
    # low-performing region, worth +18.0% answer accuracy. Mechanism: BCP's
    # median doc is ~2k tokens (p90 ~14k) — slow TF saturation plus FULL length
    # normalization is what long-document evidence search wants. Note the
    # optimum pairs high k1 with HIGH b (an earlier (12, 0.25) arm here had the
    # b direction backwards). On uniform ~102-word wiki-18 chunks these arms are
    # inert-to-harmful (measured: k1=12 dropped gold from rank 1 to 8-9), so
    # mechanical prefiltering correctly benches them there. They are dormant
    # capability: relevant only if the experiment plan later points ScaleSeek
    # training/eval at a long-document corpus (e.g. BCP) — that scheduling is
    # the plan's call, not this module's. ("Phase 3" is the baseline-evaluation
    # track's roadmap term in TASK.md; it does not govern this training track.)
    # (RISE, for contrast, uses bm25s defaults k1=1.5, b=0.75.)
    {"name": "k1_xhigh_bhigh",  "k1": 16.0, "b": 1.0,
     "rationale": "Very long documents: repeated mentions keep counting and full length normalization keeps short focused hits competitive."},
    {"name": "k1_xxhigh_bhigh", "k1": 25.0, "b": 1.0,
     "rationale": "Near-raw term frequency with full length normalization, for corpora of book-length pages."},
]

DEFAULT_CONFIG = BM25_CONFIGS[0]


def _contains_any(text: str, forms: list[str]) -> bool:
    t = text.lower()
    return any(f.lower() in t for f in forms if str(f).strip())


def mechanical_hits(retriever, query: str, forms: list[str], top_k: int = 10,
                    configs: Optional[list[dict]] = None) -> dict[str, bool]:
    """Cheap oracle prefilter: which configs surface any gold form in top_k?

    Retrieval-only (no teacher calls). Used to (a) rescue backward hops whose
    default-param retrieval missed the evidence, and (b) rank configs before
    spending teacher budget on forward continuations.
    """
    out: dict[str, bool] = {}
    for c in configs or BM25_CONFIGS:
        try:
            hits = retriever.retrieve(query, top_k=top_k, k1=c["k1"], b=c["b"])
            text = " ".join(h.get("text") or h.get("contents") or "" for h in hits)
            out[c["name"]] = _contains_any(text, forms)
        except Exception:
            out[c["name"]] = False
    return out


def select_race_configs(retriever, hops, race_width: int) -> list[dict]:
    """Rank configs by how many hops they mechanically cover; return the top
    `race_width` arms, always including `default` as the baseline arm."""
    scores: dict[str, int] = {c["name"]: 0 for c in BM25_CONFIGS}
    for hop in hops:
        trace0 = hop.trace[0] if hop.trace else None
        if not trace0 or trace0.get("name") != "bm25_retrieve":
            continue
        query = str(trace0["arguments"].get("query", ""))
        forms = [hop.expected] + list(hop.forms)
        for name, hit in mechanical_hits(retriever, query, forms).items():
            scores[name] += int(hit)
    ranked = sorted(BM25_CONFIGS, key=lambda c: (-scores[c["name"]], c["name"]))
    arms = [c for c in ranked if c["name"] != "default"][: max(race_width - 1, 1)]
    return [DEFAULT_CONFIG] + arms


def score_trajectory(meta: dict) -> float:
    """Score one finished trajectory from its meta (see _forward_pass).

    2.0 * answer_supported  — the workspace backs the final answer (dominant)
    + hop_coverage          — fraction of hops whose gold form is in the workspace
    - 0.02 * workspace_size — mild efficiency pressure, mirroring the RL
                              workspace-efficiency reward term
    """
    return (2.0 * float(bool(meta.get("answer_supported")))
            + float(meta.get("hop_coverage") or 0.0)
            - 0.02 * float(meta.get("final_workspace_size") or 0))
