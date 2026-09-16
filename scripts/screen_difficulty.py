#!/usr/bin/env python3
"""Offline difficulty screening for GRPO: sample the current policy n times per
question through the real ScaleSeek loop (eval.agent.run_agent + BM25) against a
vLLM OpenAI endpoint, and record per-question pass rate, tool usage and whether
the supporting documents are even reachable by BM25.

GRPO only learns from questions whose rollouts disagree (all-equal rewards give
zero advantage): the 2026-09-04 audit found 75% of groups degenerate. Use the
stats with `scripts/prepare_rl_data.py --screen` to keep the learnable band.

    python scripts/screen_difficulty.py --questions $RL_DATA_DIR/screen_pool.jsonl \
        --out $RL_DATA_DIR/screen_stats.jsonl --base-url http://127.0.0.1:8002/v1 \
        --model agent --n 4 --concurrency 48

Resumable: finished (id, sample) pairs are appended to <out>.samples.jsonl and
skipped on restart; the per-question stats file is rebuilt at the end.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.agent import run_agent  # noqa: E402
from eval.bm25_retriever import BM25Retriever  # noqa: E402
from eval.evidence import MULTI_HOP_SOURCES, coverage  # noqa: E402
from eval.metrics import exact_match  # noqa: E402

_tls = threading.local()


def _retriever(index_dir: str) -> BM25Retriever:
    # LuceneSearcher.set_bm25 mutates searcher state, so BM25Retriever serialises
    # retrieve() behind a lock; one searcher per thread keeps 48 workers from
    # queueing on a single 100 ms Lucene call.
    r = getattr(_tls, "r", None)
    if r is None:
        r = BM25Retriever(index_dir=index_dir)
        _tls.r = r
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="agent")
    ap.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096, help="per-turn completion budget")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--reach-top-k", type=int, default=50,
                    help="BM25(question) depth used to decide whether the support docs are reachable at all")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip sampling; rebuild <out> from the samples collected so far (needs no LLM endpoint)")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="aggregate-only: keep questions with at least this many finished samples")
    args = ap.parse_args()
    if not args.index_dir:
        sys.exit("Set --index-dir or $BM25_INDEX_DIR")
    os.environ["BM25_INDEX_DIR"] = args.index_dir

    qs = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    if args.limit:
        qs = qs[:args.limit]
    by_id = {str(q["id"]): q for q in qs}
    samples_path = Path(args.out + ".samples.jsonl")
    if args.aggregate_only:
        _aggregate(args, by_id, samples_path, min_samples=args.min_samples)
        return

    from openai import OpenAI
    from transformers import AutoTokenizer
    client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=1800, max_retries=2)
    tok = AutoTokenizer.from_pretrained(os.environ.get("LLM_TOKENIZER", "Qwen/Qwen3.5-9B"), trust_remote_code=True)

    done: set[tuple[str, int]] = set()
    if samples_path.exists():
        for l in open(samples_path, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["id"], int(r["sample"])))
            except Exception:
                pass
    jobs = [(q, i) for q in qs for i in range(args.n) if (str(q["id"]), i) not in done]
    print(f"[screen] {len(qs)} questions x {args.n} samples; {len(done)} done, {len(jobs)} to run; "
          f"concurrency={args.concurrency}", flush=True)

    lock = threading.Lock()
    fout = open(samples_path, "a", encoding="utf-8")
    t0 = time.time(); n_done = 0

    def work(q, i):
        ret = _retriever(args.index_dir)
        rec = run_agent(q, client=client, model=args.model, retriever=ret, max_turns=args.max_turns,
                        max_tokens=args.max_tokens, temperature=args.temperature, top_p=1.0,
                        tokenizer=tok, enable_thinking=True)
        golds = q.get("golden_answers", [])
        pred = rec.prediction or ""
        em = 1.0 if (golds and exact_match(pred, golds)) else 0.0
        cov = coverage(q.get("evidence") or {}, golds, [{"text": x} for x in _seen_texts(rec)])
        return {"id": str(q["id"]), "sample": i, "em": em, "pred": pred, "finish": rec.finish_reason,
                "n_tool_calls": rec.n_tool_calls, "n_bm25_calls": rec.n_bm25_calls,
                "ret_coverage": cov["ret_coverage"], "gold_in_ws": cov["gold_in_ws"],
                "llm_s": round(rec.llm_time_s, 1)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, q, i) for q, i in jobs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # one bad sample must not kill a 6h screen
                print(f"[screen] sample failed: {type(e).__name__}: {e}", flush=True); continue
            with lock:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush(); n_done += 1
                if n_done % 100 == 0:
                    el = time.time() - t0
                    print(f"[screen] {n_done}/{len(jobs)}  {el/60:.1f} min  {n_done/el*3600:.0f} samples/h", flush=True)
    fout.close()
    _aggregate(args, by_id, samples_path, min_samples=1)


def _aggregate(args, by_id: dict, samples_path: Path, *, min_samples: int) -> None:
    """Per-question stats + BM25 reachability of the support docs."""
    agg: dict[str, list[dict]] = collections.defaultdict(list)
    for l in open(samples_path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r["id"] in by_id:
            agg[r["id"]].append(r)
    ret = _retriever(args.index_dir)
    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for qid, rs in agg.items():
            if len(rs) < min_samples:
                continue
            n_written += 1
            q = by_id[qid]; ev = q.get("evidence") or {}
            hits = ret.retrieve(q["question"], top_k=args.reach_top_k)
            reach = coverage(ev, q.get("golden_answers", []), hits)
            f.write(json.dumps({
                "id": qid, "source": q.get("source", qid.split("_")[0]),
                "n": len(rs), "pass_rate": sum(r["em"] for r in rs) / len(rs),
                "mean_calls": sum(r["n_tool_calls"] for r in rs) / len(rs),
                "mean_ret_coverage": sum(r["ret_coverage"] for r in rs) / len(rs),
                "reachable_coverage": reach["ret_coverage"],
                "multi_hop": q.get("source") in MULTI_HOP_SOURCES,
            }, ensure_ascii=False) + "\n")
    print(f"[screen] wrote {n_written} question stats (>= {min_samples} samples each; "
          f"{len(agg)} questions had any) -> {args.out}", flush=True)


def _seen_texts(rec) -> list[str]:
    """Passage texts from every tool response in an AgentRecord.

    run_agent records tool turns as {"role": "tool", "content": <JSON payload>};
    bm25/grep payloads carry "results": ["[doc_id=.. score=..]\n<passage>", ...]
    and read_doc carries "text". Passages omitted by the response budget are not
    visible here, so this is a lower bound on what the workspace held — the
    same bound the policy itself observed.
    """
    out = []
    for t in rec.turns:
        if not (isinstance(t, dict) and t.get("role") == "tool"):
            continue
        try:
            res = json.loads(t.get("content") or "")
        except Exception:
            continue
        if not isinstance(res, dict):
            continue
        for x in res.get("results", []) or []:
            if isinstance(x, str):
                out.append(x.split("\n", 1)[1] if x.startswith("[doc_id=") and "\n" in x else x)
        if isinstance(res.get("text"), str):
            out.append(res["text"])
    return out


if __name__ == "__main__":
    main()
