"""Deterministic evaluation subsets for the call-budget-bound harnesses.

Why this module exists
----------------------
DCI, DR-DCI and RISE spend 300 / 300 / 100 model calls **per question**. On the
complete 14,267-row PopQA split, the measured dci-agent-lite rate (n=500 in
26.9 h at concurrency 2) extrapolates to ~32 days for a single row of the
matrix — not a tuning problem, an arithmetic one. TASK.md therefore admits one
narrow exception (see "1/10 规模例外"): these three methods evaluate a fixed
one-tenth of the split.

The exception is safe only if the subset is *reproducible*. The previous
generation of results used "a fixed random 1500 whose seed is unknowable"
(reports/baselines.md), which is why none of those numbers can be regenerated.
So the subset here is defined by a **pure function of the example ids**, not by
an RNG:

    rank = SHA256(f"{dataset}:{id}")   ->  sort by digest  ->  take the first k

This has the properties an RNG-based sample lacks:

* reproducible by anyone holding the dataset, with no seed and no state file;
* independent of the order rows happen to sit in on disk, so a re-download or a
  re-conversion yields the identical subset;
* uniform, since SHA256 is a good pseudorandom function of the id;
* verifiable after the fact — `subset_manifest` fingerprints the selected id set
  so a results file can be checked against it (see
  `scripts/compute_metrics.py --expect-ids`).

Every consumer must record the manifest next to the metrics, and every table
that shows a subset row must mark it. A 1/10 row and a full-split row are not
interchangeable: at n=1,427 the 95% CI on EM is about ±2.6 points, so a
two-point gap against a full-split method is noise.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

# The one sanctioned denominator. Kept as a constant so a stray `--fraction 3`
# cannot quietly invent a third evaluation scope.
DECILE = 10


def _rank_key(dataset: str, example_id: Any) -> str:
    return hashlib.sha256(f"{dataset}:{example_id}".encode()).hexdigest()


def subset_size(total: int, denominator: int = DECILE) -> int:
    """Size of the subset: ceil(total / denominator), at least 1."""
    if total <= 0:
        return 0
    return max(1, -(-total // denominator))


def select(examples: Sequence[dict], *, dataset: str,
           denominator: int = DECILE) -> list[dict]:
    """The deterministic 1/denominator subset of `examples`, in dataset order.

    Selection is by hash rank; the returned rows keep their original relative
    order so a diff against the full file stays readable.
    """
    ids = [str(ex["id"]) for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids: the subset would not be well defined")
    k = subset_size(len(examples), denominator)
    chosen = set(sorted(ids, key=lambda i: _rank_key(dataset, i))[:k])
    return [ex for ex in examples if str(ex["id"]) in chosen]


def subset_manifest(subset: Iterable[dict], *, dataset: str, total: int,
                    denominator: int = DECILE) -> dict:
    """Fingerprint of a selected subset, for storage beside the results."""
    ids = sorted(str(ex["id"]) for ex in subset)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return {
        "dataset": dataset,
        "eval_scope": f"decile_1_of_{denominator}",
        "selection": "sha256(f'{dataset}:{id}') rank order, first ceil(N/d)",
        "source_total": total,
        "subset_n": len(ids),
        "subset_ids_sha256": digest,
        "reproduce": (
            "python -c \"from eval.subsets import select; ...\" — or simply "
            "scripts/make_dcilite_datasets.py --datasets <ds> --fraction "
            f"{denominator}"),
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
