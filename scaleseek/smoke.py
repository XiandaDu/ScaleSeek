"""Smoke test (BASELINES.md §5.5): every method, real models, a few questions, small
corpora built from the real ones -- so that the formal runs need no code changes.

smoke-prepare: for each corpus of `smoke.datasets`, a mini corpus = first `head_docs`
documents + documents containing the smoke questions' gold answers (Wiki-18, via rg) or
their evidence/gold documents (BCP); then the SAME index builders as the formal pipeline
(bm25, e5, qwen3_emb_4b, AgentIR-4B official encoder) and the retriever check.

smoke: every run of the matrix on the first `n_questions` subset questions, the BCP judge,
and a smoke report.  Results live under paths.smoke and are never reported.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from .config import enumerate_runs, resolve_run
from .corpus import DocStore, corpus_dir, grep_rows, make_mini_corpus
from .data import dataset_dir, load_subset
from .utils import log, now_iso, read_json, write_json


def _smoke_questions(base: dict, dataset: str) -> list[dict]:
    return load_subset(base, dataset)[0][: base["smoke"]["n_questions"]]


def smoke_prepare(base: dict, stage: str) -> None:
    """stage "cpu": mini corpora + bm25;  "gpu" (or "all"): also the dense and AgentIR
    indexes and the retriever check.  Every step skips work whose manifest matches."""
    from .retrieval import agentir, bm25, dense
    from .retrieval.check import check_retrievers

    corpora = sorted({base["datasets"][d]["corpus"] for d in base["smoke"]["datasets"]})
    for corpus in corpora:
        rows: set[int] = set()
        full = corpus_dir(base, corpus)
        store = DocStore(full)
        for d in base["smoke"]["datasets"]:
            if base["datasets"][d]["corpus"] != corpus:
                continue
            qs = _smoke_questions(base, d)
            extra_p = dataset_dir(base, d) / "extra.json"
            if extra_p.exists():          # BCP: evidence + gold documents
                extra = read_json(extra_p)
                for q in qs:
                    for docid in extra[q["id"]]["evidence_docs"] + extra[q["id"]]["gold_docs"]:
                        rows.add(store.row_of(docid))
            else:                         # Wiki: passages that contain a gold answer
                needles = [g for q in qs for g in q["golden_answers"]]
                rows.update(grep_rows(full / "corpus.jsonl", needles, base["smoke"]["gold_hits_per_answer"]))
        make_mini_corpus(base, corpus, base["smoke"]["head_docs"][corpus], sorted(rows))
        n = base["smoke"]["encode_shards"]
        bm25.build(base, corpus, smoke=True)
        if stage == "cpu":
            continue
        for r in ("e5", "qwen3_emb_4b"):
            for s in range(n):
                dense.encode_shard(base, r, corpus, s, n, smoke=True, device="cuda:0")
            dense.build_hnsw(base, r, corpus, n, smoke=True, threads=8)
        for s in range(n):
            agentir.encode_shard(base, corpus, s, n, smoke=True)
        agentir.finalize(base, corpus, n, smoke=True)
        check_retrievers(base, corpus, smoke=True)    # the same check as the formal pipeline


def smoke_all(base: dict, only: str) -> int:
    from .judge import judge_runs
    from .report import dataset_report
    from .runner import eval_run

    want = None if only == "all" else set(only.split(","))
    summary = {"created": now_iso(), "runs": {}}
    out = Path(base["paths"]["smoke"]) / "SMOKE_SUMMARY.json"
    failed = 0
    for d in base["smoke"]["datasets"]:
        for method, retr in enumerate_runs(base, d):
            name = method + (f"__{retr}" if retr else "")
            # --only accepts run names, "dataset/run" keys and "dataset/" (whole dataset)
            if want and not ({name, f"{d}/{name}", f"{d}/"} & want):
                continue
            key = f"{d}/{name}"
            try:
                spec = resolve_run(base, method, d, retr, "smoke")
                res = eval_run(spec, base["smoke"]["n_questions"])
                recs = [json.loads(l) for l in open(spec.run_dir / "trajectories.jsonl")]
                summary["runs"][key] = {
                    "status": "ok" if not res["missing"] else "missing records",
                    "records": len(recs),
                    "finish": [r["finish_reason"] for r in recs],
                    "turns": [r["n_turns"] for r in recs],
                    "retrievals": [r["n_retrievals"] for r in recs],
                    "errors": [r["errors"] for r in recs],
                    "predictions": [r["prediction"] for r in recs],
                    "gold": [r["golden_answers"][:3] for r in recs]}
                failed += bool(res["missing"])
            except Exception as exc:
                failed += 1
                summary["runs"][key] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()[-3000:]}
                log(f"SMOKE FAILED {key}: {exc}")
            write_json(out, summary)
        if base["datasets"][d]["profile"] == "bcp":
            root = Path(base["paths"]["smoke"]) / d
            dirs = sorted(p for p in root.iterdir() if (p / "trajectories.jsonl").exists()) if root.exists() else []
            try:
                judge_runs(base, dirs, "bcp_judge")
                summary["runs"][f"{d}/judge"] = {"status": "ok", "runs": [p.name for p in dirs]}
            except Exception as exc:
                failed += 1
                summary["runs"][f"{d}/judge"] = {"status": "FAILED", "error": str(exc),
                                                 "traceback": traceback.format_exc()[-3000:]}
        try:
            dataset_report(base, d, Path(base["paths"]["smoke"]), smoke=True, judge="bcp_judge")
        except Exception as exc:
            failed += 1
            summary["runs"][f"{d}/report"] = {"status": "FAILED", "error": str(exc),
                                              "traceback": traceback.format_exc()[-3000:]}
        write_json(out, summary)
    log(f"smoke done: {failed} failure(s); summary {out}")
    return 1 if failed else 0
