"""Zero-training pilots of GrepSeek+ (grepseek_plus.py): the official GrepSeek checkpoint and
harness with some variants switched on, on VALIDATION questions -- NQ dev and HotpotQA dev
questions outside the test subset.  These are GrepSeek's two training datasets; none of
the questions is in its training data (the train splits) or in any reported subset, so
variants are chosen without touching a test question.

    python run.py grepseek-pilot --name <run> [--variants budget,dedup] [--cap 2048] [--n 500]

Output: <root>/pilots/grepseek/<run>/trajectories.jsonl, in the unified record format.
`--cap` is the harness's own per-turn generation cap (0 = none, as the baseline runs).
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


def pilot(base: dict, name: str, variants: list[str], cap: int, n: int) -> None:
    spec = resolve_run(base, "grepseek", "hotpotqa", None, "formal")
    out = Path(base["paths"]["root"]) / "pilots" / "grepseek" / name
    harness = spec.method["harness"] | {"variants": variants, "max_tokens_per_turn": cap}
    spec = dataclasses.replace(spec, run_dir=out, method=spec.method | {"harness": harness})
    out.mkdir(parents=True, exist_ok=True)
    examples = questions(base, n)
    write_json(out / "config.json", {
        "created": now_iso(), "harness_commit": git_state(repo_root())["commit"], "variants": variants,
        "max_tokens_per_turn": cap, "n_per_source": n, "seed": SEED, "generator": spec.method["generator"],
        "profile": spec.profile})
    g = spec.method["generator"]
    path = local_snapshot(base, g["model"], g["revision"])
    server = VLLMServer(path, spec.method["server"], spec.profile["max_model_len"], out / "vllm.log", job_gpus())
    try:
        server.start(base["runtime"]["server_start_timeout_s"])
        pred = grepseek.run(spec, server, path, examples, False)
        write_jsonl(out / "trajectories.jsonl", grepseek.convert(spec, pred, examples))
    finally:
        server.stop()
