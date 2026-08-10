#!/usr/bin/env python3
"""Same-query stability acceptance probe against the full-corpus indexes.

For each backend: issue the same queries several times and require identical
ranked (doc_id, score) lists every time; require every backend to expose the
same corpus document count. This is the server-side half of the Phase-1
"three full indexes, same query, stable results" acceptance item.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.datasets import load_dataset  # noqa: E402
from eval.retrievers import build_retriever  # noqa: E402


def ranked(hits: list[dict]) -> list[tuple[str, float]]:
    return [(str(h["doc_id"]), round(float(h["score"]), 5)) for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="bm25,e5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    # The probe checks same-query determinism of the indexes; any fixed query
    # set serves. Default to the phase's active dataset ($DS) so acceptance on
    # a fresh cluster does not require assets of a dataset it will not run.
    parser.add_argument("--dataset", default=os.environ.get("DS", "popqa"))
    args = parser.parse_args()

    rows = load_dataset(args.dataset, limit=args.num_queries)
    queries = [row["question"] for row in rows]
    report: dict = {"queries": len(queries), "repeats": args.repeats,
                    "top_k": args.top_k, "backends": {}}
    counts = {}
    failures = []
    for backend in args.backends.split(","):
        backend = backend.strip()
        retriever = build_retriever(backend, device=args.device)
        counts[backend] = retriever.num_docs
        unstable = 0
        sample = None
        for query in queries:
            baseline = ranked(retriever.retrieve(query, top_k=args.top_k))
            for _ in range(args.repeats - 1):
                again = ranked(retriever.retrieve(query, top_k=args.top_k))
                if again != baseline:
                    unstable += 1
                    break
            if sample is None:
                sample = {"query": query, "hits": baseline}
        report["backends"][backend] = {
            "num_docs": retriever.num_docs,
            "unstable_queries": unstable,
            "sample": sample,
            "metadata": retriever.metadata,
        }
        if unstable:
            failures.append(f"{backend}: {unstable} unstable queries")
        del retriever

    if len(set(counts.values())) > 1:
        failures.append(f"document count mismatch across backends: {counts}")
    report["doc_counts"] = counts
    report["ok"] = not failures
    report["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("doc_counts", "ok", "failures")}, indent=2))
    if failures:
        raise SystemExit("stability probe FAILED: " + "; ".join(failures))


if __name__ == "__main__":
    main()
