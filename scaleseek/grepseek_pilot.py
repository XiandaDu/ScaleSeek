"""The validation questions, and GrepSeek / GrepSeek+ on them.

Validation questions: 500 NQ dev + 500 HotpotQA dev questions, the HotpotQA ones outside
the test subset (which is drawn from the same dev file).  NQ and HotpotQA are the two
datasets every trained method here trains on (their train splits); no validation question
is a training question or a test question (reports/scripts/leak_check.py), so variants
and checkpoints are chosen without touching a test question.  train/rollouts.py --dev
runs our own agent on the same questions.

    python run.py grepseek-pilot --name <run> [--variants budget,dedup,fold,relax,title] [--n 500]

The official GrepSeek checkpoint and harness, with the GrepSeek+ variants (grepseek_plus.py)
that are named.  Output: <root>/pilots/grepseek/<run>/trajectories.jsonl, in the unified
record format.  `--presence-penalty 0`: the paper's decoding (no presence penalty), to
measure what our shared 2.0 costs GrepSeek.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from .config import resolve_run
from .data import load_subset
from .hub import local_snapshot, snapshot_dir
from .llm import VLLMServer
from .methods import grepseek
from .runner import job_gpus
from .utils import git_state, now_iso, repo_root, write_json, write_jsonl

SEED = 20261114


def questions(base: dict, n: int) -> list[dict]:
    d = base["datasets"]["nq"]
    snap = snapshot_dir(base, d["hf_repo"], d["revision"], repo_type="dataset")
    test = {e["id"] for e in load_subset(base, "hotpotqa")[0]}
    rng = random.Random(SEED)
    out = []
    for source, skip in (("nq", set()), ("hotpotqa", test)):
        rows = [json.loads(line) for line in open(snap / source / "dev.jsonl")]
        rows = [r for r in rows if r["id"] not in skip and any(str(g).strip() for g in r["golden_answers"])]
        out += [{"id": f"{source}_{r['id']}", "question": r["question"], "golden_answers": r["golden_answers"]}
                for r in rng.sample(rows, n)]
    return out


def pilot(base: dict, name: str, variants: list[str], n: int, presence_penalty: float | None) -> None:
    spec = resolve_run(base, "grepseek", "hotpotqa", None, "formal")
    out = Path(base["paths"]["root"]) / "pilots" / "grepseek" / name
    out.mkdir(parents=True, exist_ok=True)
    profile = spec.profile | ({} if presence_penalty is None else {"presence_penalty": presence_penalty})
    spec = dataclasses.replace(spec, run_dir=out, profile=profile,
                               method=spec.method | {"harness": spec.method["harness"] | {"variants": variants}})
    examples = questions(base, n)
    write_json(out / "config.json", {
        "created": now_iso(), "harness_commit": git_state(repo_root())["commit"], "variants": variants,
        "n_per_source": n, "seed": SEED, "generator": spec.method["generator"], "profile": spec.profile})
    path = local_snapshot(base, spec.method["generator"]["model"], spec.method["generator"]["revision"])
    server = VLLMServer(path, spec.method["server"], spec.profile["max_model_len"], out / "vllm.log", job_gpus())
    try:
        server.start(base["runtime"]["server_start_timeout_s"])
        pred = grepseek.run(spec, server, path, examples, False)
        write_jsonl(out / "trajectories.jsonl", grepseek.convert(spec, pred, examples))
    finally:
        server.stop()
