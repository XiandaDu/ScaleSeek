#!/usr/bin/env python
"""ScaleSeek baselines -- the single entry point (protocol: BASELINES.md).

    source setup_env.sh
    python run.py check                                    # configs, prompt hashes, byte diffs
    python run.py plan --method rag --dataset popqa --retriever bm25

Pipeline (see README.md for which step runs where):
    prefetch            LOGIN node: pinned models, datasets, corpora, official repos
    prepare             CPU job: dataset preflight + fixed-seed subsets, corpora + manifests
    index-bm25          CPU job: Lucene index
    encode / build-hnsw GPU array / large-memory CPU job: e5 and qwen3_emb_4b
    encode-agentir / finalize-agentir   GPU array: the official AgentIR-4B index
    check-retrievers    doc-id alignment + repeat-determinism of all retrievers
    eval                GPU job: one run = (method, dataset[, retriever])
    judge               GPU job: BrowseComp-Plus judge over finished BCP runs
    score               metrics.json per run + <dataset>/REPORT.md
    smoke-prepare/smoke one GPU: mini corpora + indexes, every method on a few questions

Every parameter is read from configs/base.yaml + configs/methods/<method>.yaml; this file
only routes.  `--smoke` switches paths, corpora and indexes to the smoke area; smoke
numbers are never reported.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scaleseek.config import (ConfigError, enumerate_runs, load_all_methods, load_base,
                              load_method, method_names, prompt_fingerprint, resolve_constant,
                              resolve_run)
from scaleseek.utils import PipelineError, log


def _shard(text: str) -> tuple[int, int]:
    k, n = (int(x) for x in text.split("/"))
    return k, n


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prefetch", help="download pinned assets (login node)")
    s.add_argument("--only", choices=["all", "repos", "models", "datasets"], default="all")

    s = sub.add_parser("prepare", help="dataset preflight + subsets, corpora (CPU job)")
    s.add_argument("--datasets", default="all")
    s.add_argument("--corpora", default="all")

    sub.add_parser("check", help="validate every config, prompt hash and byte diff")
    sub.add_parser("prompts", help="print the sha256 of every prompt slot (to freeze them)")

    s = sub.add_parser("plan", help="print what a run would do")
    s.add_argument("--method", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--retriever")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("matrix", help="list the runs of one dataset (one per line)")
    s.add_argument("--dataset", required=True)

    for name in ("index-bm25", "check-retrievers"):
        s = sub.add_parser(name)
        s.add_argument("--corpus", required=True)
        s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("encode", help="encode one corpus shard (e5 / qwen3_emb_4b)")
    s.add_argument("--retriever", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--shard", type=_shard, required=True, metavar="K/N")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("build-hnsw")
    s.add_argument("--retriever", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--threads", type=int, required=True)
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("encode-agentir", help="official tevatron encoding of one shard")
    s.add_argument("--corpus", required=True)
    s.add_argument("--shard", type=_shard, required=True, metavar="K/N")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("finalize-agentir")
    s.add_argument("--corpus", required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("serve-retriever", help="(internal) the retrieval service of a run")
    s.add_argument("--retriever", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--device", required=True)
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("eval", help="one run (GPU job)")
    s.add_argument("--method", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--retriever")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("judge", help="BrowseComp-Plus judge over finished runs")
    s.add_argument("--dataset", required=True)
    s.add_argument("--judge", default="bcp_judge", help="judge config (configs/methods/<judge>.yaml)")
    s.add_argument("--runs", default="all", help="comma-separated run names, or all")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("score", help="metrics.json per run + REPORT.md")
    s.add_argument("--dataset", required=True)
    s.add_argument("--judge", default="bcp_judge", help="whose verdicts score BCP runs")
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("smoke-prepare", help="mini corpora + all mini indexes (one GPU)")
    s.add_argument("--stage", choices=["cpu", "gpu", "all"], default="all")
    s = sub.add_parser("smoke", help="every method on a few questions (one GPU)")
    s.add_argument("--only", default="all", help="comma-separated run names, or all")
    return p.parse_args(argv)


# ============================================================================= commands
def cmd_check(base: dict) -> None:
    from scaleseek.config import load_prompts
    from scaleseek.official import verify_method

    methods = load_all_methods(base)
    for name, m in methods.items():
        load_prompts(m)
        rec = verify_method(base, m)
        log(f"ok  {name:15s} " + ", ".join(f"{k}: {v.split(':')[0]}" for k, v in rec["byte_diff"].items()))
    log(f"all {len(methods)} method configs valid; prompts frozen; official byte diffs identical")


def cmd_prompts(base: dict) -> None:
    for name in method_names():
        m = load_method(name, base)
        for slot, ref in m.get("prompts", {}).items():
            fp = prompt_fingerprint(resolve_constant(ref["constant"]))
            flag = "" if fp == ref["sha256"] else "   <-- yaml differs"
            print(f"{name:15s} {slot:22s} {fp}{flag}")


def cmd_plan(base: dict, a) -> None:
    spec = resolve_run(base, a.method, a.dataset, a.retriever, "smoke" if a.smoke else "formal")
    from scaleseek.hub import snapshot_dir
    from scaleseek.llm import VLLMServer
    model = snapshot_dir(base, spec.method["generator"]["model"], spec.method["generator"]["revision"])
    server = VLLMServer(model, spec.method["server"], spec.profile["max_model_len"], spec.run_dir / "vllm.log", ["0"])
    print(json.dumps({"run": spec.run_name, "run_dir": str(spec.run_dir), "config_id": spec.config_id,
                      "profile": spec.profile, "dataset": spec.dataset_cfg,
                      "retriever": base["retrievers"].get(spec.retriever) if spec.retriever else None,
                      "server_command": server.command()}, indent=2, ensure_ascii=False))


def cmd_prepare(base: dict, a) -> None:
    from scaleseek.corpus import prepare_corpus
    from scaleseek.data import make_subset, prepare_dataset

    names = (list(base["datasets"]) if a.datasets == "all" else
             [] if a.datasets == "none" else a.datasets.split(","))
    for n in names:
        prepare_dataset(base, n)
        make_subset(base, n)
    if a.corpora != "none":
        for c in (list(base["corpora"]) if a.corpora == "all" else a.corpora.split(",")):
            prepare_corpus(base, c)


def _bcp_run_dirs(base: dict, dataset: str, runs: str, smoke: bool) -> list[Path]:
    root = Path(base["paths"]["smoke" if smoke else "runs"]) / dataset
    dirs = sorted(p for p in root.iterdir() if (p / "trajectories.jsonl").exists())
    if runs != "all":
        want = set(runs.split(","))
        dirs = [d for d in dirs if d.name in want]
    return dirs


def _sigterm(signum, frame):
    # Slurm sends SIGTERM at the time limit (SIGKILL follows after a grace period): turn it
    # into a normal exit so `finally` blocks stop the servers and log the session end.
    raise SystemExit(128 + signum)


def main(argv=None) -> int:
    a = parse_args(argv)
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        base = load_base()
        if a.cmd == "prefetch":
            from scaleseek.hub import prefetch
            prefetch(base, a.only)
        elif a.cmd == "prepare":
            cmd_prepare(base, a)
        elif a.cmd == "check":
            cmd_check(base)
        elif a.cmd == "prompts":
            cmd_prompts(base)
        elif a.cmd == "plan":
            cmd_plan(base, a)
        elif a.cmd == "matrix":
            for m, r in enumerate_runs(base, a.dataset):
                print(m, r or "-")
        elif a.cmd == "index-bm25":
            from scaleseek.retrieval import bm25
            bm25.build(base, a.corpus, a.smoke)
        elif a.cmd == "encode":
            from scaleseek.retrieval import dense
            dense.encode_shard(base, a.retriever, a.corpus, a.shard[0], a.shard[1], a.smoke, a.device)
        elif a.cmd == "build-hnsw":
            from scaleseek.retrieval import dense
            dense.build_hnsw(base, a.retriever, a.corpus, a.shards, a.smoke, a.threads)
        elif a.cmd == "encode-agentir":
            from scaleseek.retrieval import agentir
            agentir.encode_shard(base, a.corpus, a.shard[0], a.shard[1], a.smoke)
        elif a.cmd == "finalize-agentir":
            from scaleseek.retrieval import agentir
            agentir.finalize(base, a.corpus, a.shards, a.smoke)
        elif a.cmd == "check-retrievers":
            from scaleseek.retrieval.check import check_retrievers
            check_retrievers(base, a.corpus, a.smoke)
        elif a.cmd == "serve-retriever":
            from scaleseek.retrieval.service import serve
            serve(base, a.retriever, a.corpus, a.smoke, a.port, a.device)
        elif a.cmd == "eval":
            from scaleseek.runner import eval_run
            spec = resolve_run(base, a.method, a.dataset, a.retriever, "smoke" if a.smoke else "formal")
            res = eval_run(spec, base["smoke"]["n_questions"] if a.smoke else None)
            print(json.dumps({k: v for k, v in res.items() if k != "missing"} | {"n_missing": len(res["missing"])}))
            return 0 if not res["missing"] else 3
        elif a.cmd == "judge":
            from scaleseek.judge import judge_runs
            judge_runs(base, _bcp_run_dirs(base, a.dataset, a.runs, a.smoke), a.judge)
        elif a.cmd == "score":
            from scaleseek.report import dataset_report
            root = Path(base["paths"]["smoke" if a.smoke else "runs"])
            print(dataset_report(base, a.dataset, root, a.smoke, a.judge))
        elif a.cmd == "smoke-prepare":
            from scaleseek.smoke import smoke_prepare
            smoke_prepare(base, a.stage)
        elif a.cmd == "smoke":
            from scaleseek.smoke import smoke_all
            return smoke_all(base, a.only)
        return 0
    except (PipelineError, ConfigError) as exc:
        log(f"STOP: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
