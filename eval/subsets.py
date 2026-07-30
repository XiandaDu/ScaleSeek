"""Deterministic evaluation subsets for the call-budget-bound harnesses.

Why this module exists
----------------------
DCI, DR-DCI and RISE spend 300 / 300 / 100 model calls **per question**. On the
complete 14,267-row PopQA split the measured dci-agent-lite rate (n=500 in 26.9 h
at concurrency 2) extrapolates to ~32 days for a single row of the matrix -- not
a tuning problem, an arithmetic one. TASK.md therefore admits one narrow
exception (see "评测规模上限"): these three methods evaluate at most
``MAX_EVAL_N`` questions per dataset.

Cap, not fraction
-----------------
The rule is an absolute cap, not a ratio. A ratio breaks on the small datasets:
one tenth of Bamboogle is 13 questions (95% CI on EM about +-27 points, i.e. no
measurement at all) and one tenth of BrowseComp-Plus is 83. Under the cap those
two run **complete** -- they are cheap precisely because they are small -- while
the six large datasets are bounded:

    popqa 14,267 -> 1,500        nq        3,610 -> 1,500
    triviaqa 11,313 -> 1,500     musique   2,417 -> 1,500
    2wiki  9,322 -> 1,500        bcp         830 -> 830 (full)
    hotpotqa 7,405 -> 1,500      bamboogle   125 -> 125 (full)

Reproducible, not merely random
-------------------------------
The subset is an unbiased sample, but it is drawn by a **pure function of the
example ids** rather than an RNG:

    rank = SHA256(f"{dataset}:{id}")   ->  sort by digest  ->  take the first k

SHA256 is a good pseudorandom function of the id, so this is statistically a
uniform random sample; it simply also happens to be reconstructible. That
distinction is the whole point. The previous generation of results used "a fixed
random 1500 whose seed is unknowable" (reports/baselines.md), which is exactly
why none of those numbers can be regenerated. Concretely this gives:

* reproducible by anyone holding the dataset, with no seed and no state file;
* independent of the order rows happen to sit in on disk, so a re-download or a
  re-conversion yields the identical subset;
* verifiable after the fact -- `subset_manifest` fingerprints the selected id set
  so a results file can be checked against it (see
  `scripts/compute_metrics.py --expect-ids`).

Every consumer must record the manifest next to the metrics, and every table that
shows a capped row must mark it. A 1,500-question row and a 14,267-question row
are not interchangeable: at n=1,500 the 95% CI on EM is about +-2.5 points, so a
two-point gap against a full-split method is noise.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

# The one sanctioned cap. Kept as a constant so a stray `--cap 300` cannot
# quietly invent a third evaluation scope.
MAX_EVAL_N = 1500


def _rank_key(dataset: str, example_id: Any) -> str:
    return hashlib.sha256(f"{dataset}:{example_id}".encode()).hexdigest()


def subset_size(total: int, cap: int = MAX_EVAL_N) -> int:
    """How many questions this dataset contributes: `min(total, cap)`."""
    return max(0, min(total, cap))


def is_capped(total: int, cap: int = MAX_EVAL_N) -> bool:
    """True when the cap actually binds (i.e. the row is not the full split)."""
    return total > cap


def select(examples: Sequence[dict], *, dataset: str,
           cap: int = MAX_EVAL_N) -> list[dict]:
    """At most `cap` examples, chosen deterministically, in dataset order.

    Returns every example unchanged when the dataset is already at or below the
    cap -- small datasets run complete. Selection is by hash rank; the returned
    rows keep their original relative order so a diff against the full file stays
    readable.
    """
    ids = [str(ex["id"]) for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids: the subset would not be well defined")
    if len(examples) <= cap:
        return list(examples)
    chosen = set(sorted(ids, key=lambda i: _rank_key(dataset, i))[:cap])
    return [ex for ex in examples if str(ex["id"]) in chosen]


def subset_manifest(subset: Iterable[dict], *, dataset: str, total: int,
                    cap: int = MAX_EVAL_N) -> dict:
    """Fingerprint of a selected subset, for storage beside the results."""
    ids = sorted(str(ex["id"]) for ex in subset)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    capped = is_capped(total, cap)
    return {
        "dataset": dataset,
        # A dataset at or below the cap really is the full split; saying so keeps
        # a results table from marking Bamboogle as a subset row when all 125 ran.
        "eval_scope": f"capped_{cap}" if capped else "full_split",
        "capped": capped,
        "cap": cap,
        "selection": ("sha256(f'{dataset}:{id}') rank order, first cap"
                      if capped else "no selection applied (total <= cap)"),
        "source_total": total,
        "subset_n": len(ids),
        "subset_ids_sha256": digest,
        "reproduce": ("scripts/make_dcilite_datasets.py --datasets <ds> "
                      f"--cap {cap}"),
    }


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify(result_ids: Iterable[str], manifest: dict) -> tuple[bool, str]:
    """Do these result ids match the subset the manifest describes?"""
    got = sorted({str(i) for i in result_ids})
    digest = hashlib.sha256("\n".join(got).encode()).hexdigest()
    want = manifest.get("subset_ids_sha256")
    if digest == want:
        return True, f"id set matches {manifest.get('eval_scope')} ({len(got)} ids)"
    return False, (
        f"id-set mismatch: results have {len(got)} ids (sha256 {digest[:16]}…), "
        f"manifest declares {manifest.get('subset_n')} ids "
        f"(sha256 {str(want)[:16]}…)")
