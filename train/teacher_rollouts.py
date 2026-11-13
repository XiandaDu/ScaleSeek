"""Teacher trajectories for the SFT warm start (GrepSeek also distils Qwen3.5-27B).

Qwen3.5-27B runs OUR agent exactly as the eval harness does (scaleseek/methods/scaleseek.py:
same prompt, tools, budgets, decoding, e5 behind `search`) on NQ + HotpotQA TRAIN questions
-- HotpotQA medium/hard only, the RL dev hold-out excluded -- several samples per question.
Same tokenizer and chat template as Qwen3.5-9B, so a trajectory's text is the student's
format as is.  train/build_sft.py keeps the correct, well-formed ones.

    python train/teacher_rollouts.py <out_dir> --exclude <rl dev.parquet> --nq 2000 --hotpotqa 4000 --samples 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleseek.config import load_base, resolve_run
from scaleseek.hub import local_snapshot, snapshot_dir
from scaleseek.llm import LLMClient, VLLMServer
from scaleseek.methods.base import run_examples
from scaleseek.methods.scaleseek import ScaleSeek
from scaleseek.retrieval.service import RetrievalClient
from scaleseek.runner import RetrievalProcess, job_gpus
from scaleseek.utils import git_state, now_iso, repo_root, write_json

TEACHER = {"model": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654"}
SEED = 20261105


def questions(base: dict, n: dict[str, int], exclude: Path, take: int | None = None) -> list[dict]:
    d = base["datasets"]["nq"]
    snap = snapshot_dir(base, d["hf_repo"], d["revision"], repo_type="dataset")
    held = {e["id"] for e in pq.read_table(exclude).column("extra_info").to_pylist()}
    rng = random.Random(SEED)
    out = []
    for source, k in n.items():
        rows = []
        for line in open(snap / source / "train.jsonl"):
            r = json.loads(line)
            qid, golds = f"{source}_{r['id']}", [g for g in r["golden_answers"] if str(g).strip()]
            if golds and qid not in held and r.get("metadata", {}).get("level") != "easy":
                rows.append({"id": qid, "question": r["question"], "golden_answers": golds})
        out += rng.sample(rows, k)[:take]                 # same rng draws for any take
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--exclude", type=Path, required=True, help="the RL dev parquet (held out)")
    p.add_argument("--nq", type=int, default=2000)
    p.add_argument("--hotpotqa", type=int, default=4000)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--take", type=int, help="only the first N of each source's questions (a comparison subset)")
    p.add_argument("--model", default=TEACHER["model"], help="another generator on the same questions")
    p.add_argument("--revision", default=TEACHER["revision"])
    p.add_argument("--tp", type=int, default=4)
    a = p.parse_args()
    base = load_base()
    spec = resolve_run(base, "scaleseek_prompt", "hotpotqa", "e5", "formal")
    generator = {"model": a.model, "revision": a.revision}
    server_cfg = spec.method["server"] | {"served_model_name": "generator", "tensor_parallel_size": a.tp,
                                          "max_num_seqs": 128}
    client_cfg = spec.method["client"] | {"concurrency": 128}
    spec = dataclasses.replace(spec, run_dir=a.out_dir, method=spec.method | {"generator": generator,
                                                                              "server": server_cfg})
    a.out_dir.mkdir(parents=True, exist_ok=True)
    qs = questions(base, {"nq": a.nq, "hotpotqa": a.hotpotqa}, a.exclude, a.take)
    examples = [{**q, "id": f"{q['id']}#{j}"} for q in qs for j in range(a.samples)]
    write_json(a.out_dir / "config.json", {
        "created": now_iso(), "harness_commit": git_state(repo_root())["commit"], "teacher": generator, "take": a.take,
        "server": server_cfg, "profile": spec.profile, "retriever": "e5", "seed": SEED,
        "questions": {"nq": a.nq, "hotpotqa": a.hotpotqa}, "samples": a.samples, "exclude": str(a.exclude)})

    path = local_snapshot(base, a.model, a.revision)
    gpus = job_gpus()
    server = VLLMServer(path, server_cfg, spec.profile["max_model_len"], a.out_dir / "vllm.log", gpus)
    retr = RetrievalProcess(spec, False, gpus[0])
    exact = RetrievalProcess(spec, False, "", retriever="exact")
    try:
        for proc in (retr, exact):
            proc.wait(base["runtime"]["retriever_start_timeout_s"])
        server.start(base["runtime"]["server_start_timeout_s"])
        timeout = client_cfg["request_timeout_s"]
        ctx = SimpleNamespace(spec=spec, llm=LLMClient(server, client_cfg),
                              retrieval=RetrievalClient(retr.url, timeout),
                              exact=RetrievalClient(exact.url, timeout),
                              tokenizer=AutoTokenizer.from_pretrained(str(path)))
        print(run_examples(ScaleSeek(ctx), examples, a.out_dir / "trajectories.jsonl", client_cfg["concurrency"]))
    finally:
        server.stop()
        retr.stop()
        exact.stop()


if __name__ == "__main__":
    main()
