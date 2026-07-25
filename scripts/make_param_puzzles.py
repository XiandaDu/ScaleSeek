#!/usr/bin/env python3
"""Build a synthetic BM25 *parameter-puzzle* testbed.

Each puzzle uses UNIQUE rare tokens (so BM25 idf is uniform and there is no
cross-puzzle interference) and is constructed so that ONE knob decides retrieval:

  k1-puzzle : distractors densely repeat a SUBSET (2 of 3) of the query terms.
              Under default k1 their high term-frequency buries the gold; LOWERING
              k1 saturates that repetition so the gold's full-term coverage wins.
  b-puzzle  : the gold is padded long with unique filler; distractors are short and
              cover a subset of terms. Under default b the gold's length penalty
              buries it; LOWERING b lets its full-term coverage win.

Every candidate is verified against a real Lucene index by grid-searching (k1,b):
kept only if the DEFAULT setting buries the gold (rank > keep_default_rank) AND some
non-default setting rescues it into a small top_k (rank <= keep_rescued_rank). The
verified optimum is recorded as the puzzle's answer key, and the difficulty knobs
are swept so the optimal (k1,b) VARIES across puzzles — a real query->param mapping,
not one constant.

Outputs under --out-dir:
  corpus/wiki_corpus.jsonl   Pyserini JsonCollection of every kept puzzle's docs
  questions.jsonl            {id, question, golden_answers, split}  (train / heldout)
  answer_key.json            per-puzzle verified optimal params + rescue knob
  bm25_index/                Lucene index over the kept corpus

SMOKE/testbed only — synthetic, never a result table.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.bm25_retriever import BM25Retriever
from train.sft.coldstart import _gold_rank, _K1_GRID, _B_GRID

_TOPK_LADDER = (3, 5, 10, 20)


def _tok(i: int, tag: str) -> str:
    return f"zqx{i:02d}{tag}"


def _candidate(i: int, kind: str, knob: int):
    """Return (question, answer, gold_passage, distractor_passages, meta).

    Queries are token-only so no natural words leak matches onto the gold. Only the
    b-lever is physically isolable in BM25: a long gold that fully covers the query
    terms is buried by short distractors that cover a SUBSET, and only LOWERING b
    (removing the gold's length penalty) lets its extra term rescue it. `knob` is the
    number of unique filler tokens padding the gold — more padding needs a lower b.
    (k1 cannot be isolated: a term unique to the gold has high idf and wins at every
    k1, so those candidates never pass verification and are pruned.)"""
    a, b, c = _tok(i, "alpha"), _tok(i, "beta"), _tok(i, "gamma")
    ans = _tok(i, "answer")
    question = f"{a} {b} {c}"                 # token-only query
    subset = f"{a} {b}"                       # distractors cover 2 of the 3 terms
    if kind == "easy":                        # control: short gold, default is optimal
        gold = f"{a} {b} {c} {ans}"
        dists = [f"{subset}" for _ in range(4)]
    else:                                     # b-puzzle: long gold buried by short subset docs
        pad = " ".join(_tok(i, f"f{j}") for j in range(knob))
        gold = f"{a} {b} {c} {ans} {pad}"
        dists = [f"{subset}" for _ in range(5)]
    meta = {"kind": kind, "knob": knob, "terms": [a, b, c], "answer": ans, "question": question}
    return question, ans, gold, dists, meta


def _write_corpus(docs: list[dict], out_dir: Path) -> Path:
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    f = corpus_dir / "wiki_corpus.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return corpus_dir


def _build_index(corpus_dir: Path, index_dir: Path) -> None:
    cmd = [sys.executable, str(_REPO / "scripts" / "build_bm25_index.py"),
           "--corpus-dir", str(corpus_dir), "--index-dir", str(index_dir), "--threads", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"index build failed:\n{r.stderr[-800:]}")


def _best_params(retriever, query, ans):
    """Return (default_rank, best_rank, best_k1, best_b) over the grid."""
    top = max(_TOPK_LADDER)
    default_rank = _gold_rank(retriever.retrieve(query, top_k=top, k1=1.2, b=0.75), [ans])
    best = None
    for k1 in _K1_GRID:
        for b in _B_GRID:
            rank = _gold_rank(retriever.retrieve(query, top_k=top, k1=k1, b=b), [ans])
            if rank is None:
                continue
            cand = (rank, abs(k1 - 1.2) + abs(b - 0.75), k1, b)
            if best is None or cand < best:
                best = cand
    return default_rank, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".puzzles")
    ap.add_argument("--n-keep", type=int, default=16, help="target number of verified puzzles")
    ap.add_argument("--heldout", type=int, default=4, help="how many kept puzzles go to the heldout split")
    ap.add_argument("--keep-default-rank", type=int, default=3,
                    help="default params must bury the gold deeper than this")
    ap.add_argument("--keep-rescued-rank", type=int, default=3,
                    help="some non-default params must rescue the gold to <= this")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    # sweep pad length finely so the optimal b varies across b-puzzles; add a few
    # "easy" controls where default params are already optimal.
    candidates = []
    i = 0
    for knob in range(3, 40):                 # b-puzzles: fine pad sweep
        candidates.append((i, "b", knob)); i += 1
    for _ in range(6):                        # easy controls
        candidates.append((i, "easy", 0)); i += 1

    # one shared candidate corpus (unique tokens -> no interference)
    docs, meta_by_id = [], {}
    for (ci, kind, knob) in candidates:
        q, ans, gold, dists, meta = _candidate(ci, kind, knob)
        pid = f"pz_{ci:02d}"
        meta_by_id[pid] = (meta, gold, dists)
        docs.append({"id": f"{pid}_gold", "contents": gold})
        for j, d in enumerate(dists):
            docs.append({"id": f"{pid}_d{j}", "contents": d})

    cand_dir = out_dir / "_cand"
    corpus_dir = _write_corpus(docs, cand_dir)
    _build_index(corpus_dir, cand_dir / "bm25_index")
    retr = BM25Retriever(index_dir=str(cand_dir / "bm25_index"))

    kept, easy = [], []
    for pid, (meta, gold, dists) in meta_by_id.items():
        default_rank, best = _best_params(retr, meta["question"], meta["answer"])
        if best is None:
            continue
        best_rank, _, k1, b = best
        if meta["kind"] == "easy":
            # control: gold already retrievable with default params -> teach default
            if default_rank is not None and default_rank <= args.keep_rescued_rank:
                easy.append({
                    "id": pid, "question": meta["question"], "golden_answers": [meta["answer"]],
                    "kind": "easy", "default_rank": default_rank,
                    "target": {"k1": 1.2, "b": 0.75, "top_k": 3, "rescued_rank": default_rank},
                })
            continue
        buried = default_rank is None or default_rank > args.keep_default_rank
        rescued = best_rank <= args.keep_rescued_rank
        deviates = (k1, b) != (1.2, 0.75)
        if buried and rescued and deviates:
            top_k = next((t for t in _TOPK_LADDER if t >= best_rank), max(_TOPK_LADDER))
            kept.append({
                "id": pid, "question": meta["question"], "golden_answers": [meta["answer"]],
                "kind": meta["kind"], "default_rank": default_rank,
                "target": {"k1": round(k1, 2), "b": round(b, 2), "top_k": top_k, "rescued_rank": best_rank},
            })

    # diversify the b-puzzles across their optimal b, then add easy controls
    n_easy = min(len(easy), 6)
    n_b = max(args.n_keep - n_easy, 1)
    kept.sort(key=lambda r: (r["target"]["b"], r["target"]["top_k"]))
    if len(kept) > n_b:
        step = len(kept) / n_b
        kept = [kept[int(x * step)] for x in range(n_b)]
    kept += easy[:n_easy]
    print(f"[puzzles] verified {len(kept)} puzzles ({len(kept)-n_easy} b-lever + "
          f"{n_easy} easy) out of {len(candidates)} candidates")
    if not kept:
        sys.exit("no decisive puzzles found — loosen --keep-* thresholds or widen knobs")

    # stratified split: hold out both b-lever and easy puzzles so train teaches the
    # CONDITIONAL (retry+lower-b only when the first retrieval fails) and heldout
    # tests it on fresh tokens for BOTH regimes.
    kept_ids = {r["id"] for r in kept}
    buried = [r for r in kept if r["kind"] != "easy"]
    easy_k = [r for r in kept if r["kind"] == "easy"]
    for r in kept:
        r["split"] = "train"
    h = max(args.heldout // 2, 1)
    for r in buried[-h:] + easy_k[-h:]:
        r["split"] = "heldout"

    # final corpus = only kept puzzles' docs
    final_docs = [d for d in docs if d["id"].split("_")[0] + "_" + d["id"].split("_")[1] in kept_ids]
    fcorpus = _write_corpus(final_docs, out_dir)
    _build_index(fcorpus, out_dir / "bm25_index")

    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({"id": r["id"], "question": r["question"],
                                 "golden_answers": r["golden_answers"], "split": r["split"]}) + "\n")
    (out_dir / "answer_key.json").write_text(json.dumps(kept, indent=2) + "\n")

    n_train = sum(1 for r in kept if r["split"] == "train")
    import collections
    kinds = collections.Counter(r["kind"] for r in kept)
    k1v = collections.Counter(r["target"]["k1"] for r in kept)
    bv = collections.Counter(r["target"]["b"] for r in kept)
    print(f"[puzzles] {n_train} train / {len(kept)-n_train} heldout | kinds={dict(kinds)}")
    print(f"[puzzles] optimal k1 spread={dict(k1v)}  optimal b spread={dict(bv)}")
    print(f"[puzzles] -> {out_dir}/questions.jsonl, answer_key.json, bm25_index/")


if __name__ == "__main__":
    main()
