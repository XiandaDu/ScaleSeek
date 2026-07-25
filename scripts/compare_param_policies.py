#!/usr/bin/env python3
"""Compare BM25 parameter policies on retrieval, holding the query fixed.

For each question we use the question itself as the BM25 query, so the ONLY thing
that varies is how each policy sets top_k/k1/b:

  omit   : runtime defaults (top_k=3, k1=1.2, b=0.75) — what the no-bias teacher yields
  heuristic : query-feature rule (train.sft.coldstart._param_policy)
  search : grid-search the index, keep params that best rank the target passage
           (train.sft.coldstart._search_params)

It reports, per policy: recall (target pulled into the bounded workspace), mean
workspace size (top_k), and mean rank of the target in the full ranking. This
isolates the retrieval value of each policy before any SFT/RL — meaningful only on
a parameter-sensitive corpus (build one with make_smoke_corpus.py --hard N).

    python scripts/compare_param_policies.py --index-dir .smoke_hard/bm25_index \
        --questions .smoke_hard/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.bm25_retriever import BM25Retriever
from train.sft.coldstart import _param_policy, _search_params, _gold_rank


def _policies(retriever, query, targets):
    heur, _ = _param_policy(query, 0)
    srch, _ = _search_params(retriever, query, targets, 0)
    return {
        "omit":      {"top_k": 3, "k1": 1.2, "b": 0.75},   # execute_tool defaults
        "heuristic": {k: heur[k] for k in ("top_k", "k1", "b")},
        "search":    {"top_k": srch.get("top_k", 3), "k1": srch.get("k1", 1.2), "b": srch.get("b", 0.75)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ["BM25_INDEX_DIR"] = args.index_dir
    r = BM25Retriever(index_dir=args.index_dir)
    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    if args.limit:
        qs = qs[: args.limit]

    names = ["omit", "heuristic", "search"]
    agg = {n: {"hit": 0, "ws": 0, "rank_sum": 0, "rank_n": 0} for n in names}

    print(f"{'qid':<11} | " + " | ".join(f"{n:>26}" for n in names))
    print(f"{'':<11} | " + " | ".join(f"{'top_k k1  b   rank hit':>26}" for n in names))
    print("-" * 98)
    for ex in qs:
        q = ex["question"]
        targets = ex.get("golden_answers", [])
        pol = _policies(r, q, targets)
        row = []
        for n in names:
            p = pol[n]
            hits = r.retrieve(q, top_k=max(p["top_k"], 20), k1=p["k1"], b=p["b"])
            rank = _gold_rank(hits, targets)                 # rank in full ranking
            hit = rank is not None and rank <= p["top_k"]    # inside the bounded workspace?
            agg[n]["hit"] += int(hit)
            agg[n]["ws"] += p["top_k"]
            if rank:
                agg[n]["rank_sum"] += rank; agg[n]["rank_n"] += 1
            row.append(f"{p['top_k']:>4} {p['k1']:>3} {p['b']:>4} {str(rank):>4} {'Y' if hit else '.':>3}")
        print(f"{ex['id']:<11} | " + " | ".join(f"{c:>26}" for c in row))

    n = len(qs)
    print("\n=== aggregate over", n, "questions ===")
    print(f"{'policy':<12}{'recall(gold in workspace)':<27}{'mean workspace':<16}{'mean gold rank'}")
    for nm in names:
        a = agg[nm]
        recall = a["hit"] / max(n, 1)
        mws = a["ws"] / max(n, 1)
        mrank = a["rank_sum"] / max(a["rank_n"], 1)
        print(f"{nm:<12}{recall:<27.2f}{mws:<16.1f}{mrank:.1f}")


if __name__ == "__main__":
    main()
