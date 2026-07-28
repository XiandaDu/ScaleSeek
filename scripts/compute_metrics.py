#!/usr/bin/env python3
"""Offline metric computation for ScaleSeek eval runs.

Re-scores a saved results/*.jsonl WITHOUT invoking any model or GPU: answer
quality (EM/F1), workspace/retrieval recall (Gold R@W / coverage), and latency.
Auto-detects which metrics the dataset supports from the labels on disk.

Examples
--------
    source setup_env.sh

    # 1) Build the one-time corpus title index (heavy; ~21M passages)
    python scripts/compute_metrics.py --build-title-index \
        --corpus-path $CORPUS_FILE --title-index-db $DATA/corpus_title_index.db

    # 2) Score a run (dataset+agent inferred from the filename)
    python scripts/compute_metrics.py --results results/popqa_grepseek.jsonl

    # 3) Score a wiki-18 run with title-level Gold R@W
    python scripts/compute_metrics.py --results results/hotpotqa_scaleseek.jsonl \
        --title-index-db $DATA/corpus_title_index.db

    # 4) BrowseComp-Plus (native doc-id qrels; done in the BCP step)
    python scripts/compute_metrics.py --results results/bcp_dr_dci.jsonl \
        --bcp-qrels $DATASETS/browsecomp_plus/qrels.json \
        --bcp-doclen $DATASETS/browsecomp_plus/doclen.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import score_example, aggregate                     # noqa: E402
from eval import retrieval_metrics as rm                               # noqa: E402
from eval import qrels as Q                                            # noqa: E402
from eval import subsets                                               # noqa: E402

_DATASETS = [
    "nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa",
    "musique", "bamboogle", "bright", "browsecomp", "bcp", "browsecomp_plus",
]
_AGENTS = [
    "scaleseek", "rag", "direct", "dci", "agentir",
    "search_r1", "grepseek", "dr_dci", "search_o1", "rise",
    # Read-only compatibility for historical files, never accepted by run_eval.
    "bm25_rag", "agentir_rag",
]


def _infer(stem: str, known: list[str]) -> str | None:
    hits = [k for k in known if stem == k or stem.startswith(k + "_") or ("_" + k) in stem]
    return max(hits, key=len) if hits else None


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bootstrap_ci(values: list[float], n_boot: int = 1000,
                  seed: int = 0) -> dict:
    """Percentile bootstrap 95% CI for the mean of a per-example metric."""
    n = len(values)
    if n == 0:
        return {"lo": 0.0, "hi": 0.0, "n_boot": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return {"lo": means[int(0.025 * n_boot)],
            "hi": means[min(int(0.975 * n_boot), n_boot - 1)],
            "n_boot": n_boot}


def _latency_report(rows: list[dict]) -> dict:
    n = len(rows) or 1
    def col(k): return [r.get(k) or 0.0 for r in rows]
    total = col("total_time_s")
    fr = [r.get("finish_reason", "") for r in rows]
    # Every finish_reason, not just the four we happened to name. The fixed list
    # let 376/14267 search_o1 rows (no_answer + max_turns) hide behind
    # "parse_error_rate": 0.0.
    breakdown = {}
    for f in fr:
        breakdown[f or "(none)"] = breakdown.get(f or "(none)", 0) + 1
    # Concurrency is a property of the run, not of a row; if rows disagree the
    # file was merged from runs at different settings and latency is meaningless.
    concs = sorted({r.get("concurrency") for r in rows if r.get("concurrency")})
    seeds = sorted({r.get("decode_seed") for r in rows if r.get("decode_seed") is not None})
    return {
        "sec_per_query": sum(total) / n,
        "llm_time_s_mean": sum(col("llm_time_s")) / n,
        "tool_time_s_mean": sum(col("tool_time_s")) / n,
        "n_tool_calls_mean": sum(col("n_tool_calls")) / n,
        "n_bm25_calls_mean": sum(col("n_bm25_calls")) / n,
        "max_turns_rate": sum(f == "max_turns" for f in fr) / n,
        "api_error_rate": sum(f == "api_error" for f in fr) / n,
        "parse_error_rate": sum(f == "parse_error" for f in fr) / n,
        "no_answer_rate": sum(f == "no_answer" for f in fr) / n,
        "answer_rate": sum(f == "answer" for f in fr) / n,
        "finish_reason_counts": breakdown,
        "concurrency": concs[0] if len(concs) == 1 else (concs or None),
        "decode_seed": seeds[0] if len(seeds) == 1 else (seeds or None),
        "latency_comparable": len(concs) == 1 and concs[0] == 1,
        "latency_note": (
            "sec_per_query/llm_time_s/tool_time_s are wall-clock under "
            f"concurrency={concs[0] if len(concs) == 1 else concs}; they include "
            "queueing and are NOT per-query latency unless concurrency == 1."
            if concs else
            "concurrency not recorded (run predates 2026-07-28); latency columns "
            "are wall-clock under an unknown batch load and are not comparable."),
    }


def _title_gold_recall(rows, dataset, agent, title_db):
    """Title-level Gold R@W / coverage for wiki-18 QA (HotpotQA / 2Wiki)."""
    gold_map = Q.load_gold_titles(dataset)
    if not gold_map:
        return None, "no gold-title labels found for this dataset"
    idx = Q.CorpusTitleIndex(title_db)
    grec, cov_any, cov_all = [], [], []
    for r in rows:
        raw_id = Q.strip_dataset_prefix(str(r.get("id", "")), dataset)
        gold_titles = gold_map.get(raw_id, [])
        if not gold_titles:
            grec.append(None); cov_any.append(None); cov_all.append(None)
            continue
        keys = rm.surfaced_doc_ids(r, agent)
        hit = sum(1 for t in gold_titles if idx.docids_for_title(t) & keys)
        grec.append(hit / len(gold_titles))
        cov_any.append(float(hit >= 1))
        cov_all.append(float(hit == len(gold_titles)))
    return {
        "gold_recall@W": rm.aggregate_optional(grec),
        "coverage_any": rm.aggregate_optional(cov_any),
        "coverage_all": rm.aggregate_optional(cov_all),
        "granularity": "title",
    }, None


def _bcp_recall(rows, agent, qrels_path, doclen_path, c_seg):
    """Doc-id Gold/Qrel R@W + coverage + localization for BrowseComp-Plus."""
    qrels = json.load(open(qrels_path, encoding="utf-8"))
    doclen = json.load(open(doclen_path, encoding="utf-8")) if doclen_path else {}
    grec, qrec, cany, cmean, call, loc = [], [], [], [], [], []
    for r in rows:
        rid = str(r.get("id", ""))
        q = qrels.get(rid) or {}
        gold = set(map(str, q.get("gold", [])))
        evid = set(map(str, q.get("evidence", [])))
        keys = rm.surfaced_doc_ids(r, agent)
        grec.append(rm.gold_recall(keys, gold))
        qrec.append(rm.qrel_recall(keys, evid))
        cov = rm.coverage(keys, gold)
        cany.append(cov["any"]); cmean.append(cov["mean"]); call.append(cov["all"])
        # localization: snippet lengths per surfaced gold doc (best-effort from turns)
        if doclen:
            snips = _bcp_snippets(r, gold)
            glen = {d: doclen.get(d, 0) for d in snips}
            loc.append(rm.localization(snips, glen, c_seg=c_seg))
    out = {
        "gold_recall@W": rm.aggregate_optional(grec),
        "qrel_recall@W": rm.aggregate_optional(qrec),
        "coverage_any": rm.aggregate_optional(cany),
        "coverage_mean": rm.aggregate_optional(cmean),
        "coverage_all": rm.aggregate_optional(call),
        "granularity": "doc_id",
    }
    if doclen:
        out["localization"] = rm.aggregate_optional(loc)
    return out


def _bcp_snippets(record: dict, gold: set[str]) -> dict[str, list[int]]:
    """Per gold doc, the char-lengths of snippets the agent exposed for it.
    Parses saved turns' tool payloads for {"id","contents"} corpus lines."""
    snips: dict[str, list[int]] = {}
    for turn in record.get("turns", []) or []:
        if turn.get("role") != "tool":
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
            lines = (payload.get("stdout") or "").split("\n")
        for ln in lines:
            ln = ln.strip()
            if not ln or ln[0] != "{":
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            did = str(obj.get("id", ""))
            if did in gold:
                snips.setdefault(did, []).append(len(obj.get("contents", "")))
    return snips


def main():
    ap = argparse.ArgumentParser(description="Offline ScaleSeek metric computation")
    ap.add_argument("--results", help="results/*.jsonl to score")
    ap.add_argument("--dataset", choices=_DATASETS, default=None)
    ap.add_argument("--agent", choices=_AGENTS, default=None)
    ap.add_argument("--title-index-db", default=os.environ.get("TITLE_INDEX_DB"))
    ap.add_argument("--bcp-qrels", default=None)
    ap.add_argument("--bcp-doclen", default=None)
    ap.add_argument("--c-seg", type=int, default=500,
                    help="Localization segment width (DCI-Agent-Lite §A.3).")
    ap.add_argument("--out", default=None, help="Write the JSON report here too.")
    ap.add_argument("--expect-ids", default=None, metavar="MANIFEST.json",
                    help="Subset manifest (eval/subsets.subset_manifest) that this "
                         "run must match exactly. Required for the 1/10-scope "
                         "harnesses so a truncated or drifted run cannot be "
                         "reported as if it were the sanctioned subset.")
    # index-build mode
    ap.add_argument("--build-title-index", action="store_true")
    ap.add_argument("--corpus-path", default=os.environ.get("CORPUS_FILE"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.build_title_index:
        if not args.corpus_path or not args.title_index_db:
            sys.exit("--build-title-index needs --corpus-path and --title-index-db")
        Q.build_corpus_title_index(args.corpus_path, args.title_index_db, limit=args.limit)
        return

    if not args.results:
        sys.exit("--results is required (or use --build-title-index)")
    path = Path(args.results)
    stem = path.stem
    dataset = args.dataset or _infer(stem, _DATASETS)
    agent = args.agent or _infer(stem, _AGENTS) or "unknown"
    if dataset in ("bcp", "browsecomp_plus"):
        dataset = "browsecomp_plus"
    rows = _load_rows(path)

    report: dict = {
        "file": str(path), "dataset": dataset, "agent": agent, "n": len(rows),
    }

    # --- evaluation scope -------------------------------------------------
    # Every row must be able to say whether it covers the complete split or the
    # sanctioned 1/10 subset. Without this a results table can silently put a
    # 1,427-question DCI row next to a 14,267-question RAG row.
    if args.expect_ids:
        manifest = subsets.load_manifest(args.expect_ids)
        ok, msg = subsets.verify((r.get("id") for r in rows), manifest)
        if not ok:
            sys.exit(f"SCOPE-FAIL {path}: {msg}")
        print(f"[scope] {msg}")
        report["eval_scope"] = manifest["eval_scope"]
        report["eval_scope_manifest"] = {
            k: manifest[k] for k in
            ("subset_n", "source_total", "subset_ids_sha256", "selection")
        }
    else:
        report["eval_scope"] = "full_split"

    # --- answer quality ---
    scores = [score_example(r.get("prediction"), r.get("gold_answers", [])) for r in rows]
    report["answer_quality"] = aggregate(scores)
    # Bootstrap CIs so a method-vs-method gap can be read against its own noise.
    # NB this covers example-level variance only; at temperature > 0 (search_r1,
    # search_o1) decoding variance needs repeated seeds and is NOT in this band.
    report["answer_quality"]["em_ci95"] = _bootstrap_ci([s["em"] for s in scores])
    report["answer_quality"]["f1_ci95"] = _bootstrap_ci([s["f1"] for s in scores])

    # --- latency ---
    report["latency"] = _latency_report(rows)

    # --- retrieval metrics (auto-detect) ---
    retrieval = None
    note = None
    if dataset == "browsecomp_plus" and args.bcp_qrels:
        retrieval = _bcp_recall(rows, agent, args.bcp_qrels, args.bcp_doclen, args.c_seg)
    elif Q.supports_title_qrels(dataset):
        if not args.title_index_db:
            note = ("dataset supports title-level Gold R@W but --title-index-db "
                    "was not provided (build it first).")
        else:
            retrieval, note = _title_gold_recall(rows, dataset, agent, args.title_index_db)
    else:
        note = f"dataset {dataset!r} has answer-only labels -> EM/F1 + latency only."
    if retrieval is not None:
        report["retrieval"] = retrieval
    if note:
        report["retrieval_note"] = note

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
