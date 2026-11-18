"""Zero-training pilots of GrepSeek+ (grepseek_plus.py): the official GrepSeek checkpoint and
harness with some variants switched on, on VALIDATION questions -- NQ dev and HotpotQA dev
questions outside the test subset.  These are GrepSeek's two training datasets; none of
the questions is in its training data (the train splits) or in any reported subset, so
variants are chosen without touching a test question.

    python run.py grepseek-pilot --name <run> [--variants budget,dedup] [--cap 2048] [--n 500]

Output: <root>/pilots/grepseek/<run>/trajectories.jsonl, in the unified record format.
`--cap` is the harness's own per-turn generation cap (0 = none, as the baseline runs).

`--replay a,b --turns N`: a second stage.  Every question starts from the tool calls and
outputs of the earlier pilot runs a, b (interleaved, repeated lines dropped, the model's
reasoning left out) and has N turns from there; with N = 1 it can only answer, i.e. it
re-reads the pooled evidence in a clean context.
"""

from __future__ import annotations

import dataclasses
import json
import random
from itertools import zip_longest
from pathlib import Path

from transformers import AutoTokenizer

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


REPLAY_TOKENS = 11000      # of a 16K window: leaves room for the prompt and the answering turn
REREAD = ("These are all the search results. Give your final answer now in an <answer> block: "
          "the best answer supported by what you have read.")


def replay(root: Path, runs: list[str], tokenizer) -> dict[str, list[dict]]:
    """{question id: messages} -- the (tool call, tool output) pairs of earlier pilot runs."""
    calls: dict[str, list[list]] = {}
    for run in runs:
        for r in map(json.loads, open(root / run / "trajectories.jsonl")):
            turns = r["trajectory"]["turns"] or []
            pairs = [(a["parse"]["tool_call_json"], json.loads(t["content"])) for a, t in zip(turns, turns[1:])
                     if a["role"] == "assistant" and t["role"] == "tool" and not t.get("synthetic")]
            calls.setdefault(r["id"], []).append(pairs)
    out = {}
    for qid, per_run in calls.items():
        messages, seen, used = [], set(), 0
        for call, payload in (p for group in zip_longest(*per_run) for p in group if p):
            lines = [ln for ln in payload["stdout"].split("\n") if ln.strip() and not ln.startswith("[... output truncated")]
            new = [ln for ln in lines if ln not in seen]
            if lines and not new:
                continue                        # nothing the earlier calls did not show
            seen.update(new)
            fields = {k: payload[k] for k in ("command", "stdout", "stderr", "exit_code", "timed_out", "truncated")}
            fields["stdout"] = "".join(ln + "\n" for ln in new)
            pair = [{"role": "assistant", "content": f"<tool_call>\n{call}\n</tool_call>"},
                    {"role": "tool", "content": json.dumps(fields, ensure_ascii=False)}]
            used += sum(len(tokenizer(m["content"], add_special_tokens=False)["input_ids"]) for m in pair)
            if used > REPLAY_TOKENS:
                break
            messages += pair
        if messages:
            last = json.loads(messages[-1]["content"])
            messages[-1]["content"] = json.dumps(last | {"notice": REREAD}, ensure_ascii=False)
        out[qid] = messages
    return out


def pilot(base: dict, name: str, variants: list[str], cap: int, n: int, sources: list[str], turns: int | None) -> None:
    spec = resolve_run(base, "grepseek", "hotpotqa", None, "formal")
    root = Path(base["paths"]["root"]) / "pilots" / "grepseek"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    g = spec.method["generator"]
    path = local_snapshot(base, g["model"], g["revision"])
    harness = spec.method["harness"] | {"variants": variants, "max_tokens_per_turn": cap}
    if sources:
        write_json(out / "inject.json", replay(root, sources, AutoTokenizer.from_pretrained(str(path))))
        harness["inject"] = str(out / "inject.json")
    profile = spec.profile | ({"max_assistant_turns": turns} if turns else {})
    spec = dataclasses.replace(spec, run_dir=out, profile=profile, method=spec.method | {"harness": harness})
    examples = questions(base, n)
    write_json(out / "config.json", {
        "created": now_iso(), "harness_commit": git_state(repo_root())["commit"], "variants": variants,
        "max_tokens_per_turn": cap, "n_per_source": n, "seed": SEED, "generator": spec.method["generator"],
        "profile": spec.profile, "replay": sources})
    server = VLLMServer(path, spec.method["server"], spec.profile["max_model_len"], out / "vllm.log", job_gpus())
    try:
        server.start(base["runtime"]["server_start_timeout_s"])
        pred = grepseek.run(spec, server, path, examples, False)
        write_jsonl(out / "trajectories.jsonl", grepseek.convert(spec, pred, examples))
    finally:
        server.stop()
