#!/usr/bin/env python3
"""BM25 reachability of a question's supporting documents.

For each question, retrieve top-k with the question text as the query and score
eval.evidence.coverage against its support titles (or gold string for nq).
A question whose evidence is unreachable this way is a poor SFT seed: the
backward tutor has to find every hop's passage through BM25, and strict
verification skips the example when it cannot (2026-09-04 v2 run: 78% skipped).

    python scripts/bm25_reachability.py --in pool.jsonl --out pool_reach.jsonl --top-k 50 --threads 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.bm25_retriever import BM25Retriever  # noqa: E402
from eval.evidence import coverage  # noqa: E402

_tls = threading.local()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"))
    a = ap.parse_args()

    def ret():
        r = getattr(_tls, "r", None)
        if r is None:
            r = BM25Retriever(index_dir=a.index_dir); _tls.r = r
        return r

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]

    def work(q):
        hits = ret().retrieve(q["question"], top_k=a.top_k)
        c = coverage(q.get("evidence") or {}, q.get("golden_answers", []), hits)
        return {"id": q["id"], "source": q.get("source"), "reachable_coverage": c["ret_coverage"],
                "gold_in_top_k": c["gold_in_ws"], "n_support": c["n_support"]}

    with ThreadPoolExecutor(a.threads) as ex, open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(work, rows, chunksize=64)):
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 5000 == 0:
                print(f"[reach] {i+1}/{len(rows)}", flush=True)
    print(f"[reach] wrote {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
