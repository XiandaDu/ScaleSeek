#!/usr/bin/env python3
"""Deterministic FEEDBACK-DRIVEN cold-start trajectories for the puzzle testbed.

The generalizable behaviour we want the student to learn is NOT "these tokens ->
this b" (unique tokens carry no such signal) but "my first retrieval didn't return
the answer -> apply my policy's fix". So every trajectory is two-stage:

  step 1: bm25_retrieve(query)          # default params (top_k=3)
  if the answer is now in the workspace -> grep + answer          (easy puzzles)
  else, by policy:
    teacher/omit : no retry -> answer anyway (honest failure baseline)
    heuristic    : retry widening top_k -> answer                 (big workspace)
    search       : retry lowering b to the verified optimum -> answer  (tight workspace)

The trigger (a first retrieval without the answer) is shared across puzzles, so the
learned fix can generalize to held-out puzzles with fresh tokens.

    python scripts/make_puzzle_trajectories.py --puzzles-dir .puzzles \
        --policy search --out .puzzles/traj_search.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.agent import Workspace, execute_tool
from eval.bm25_retriever import BM25Retriever
from prompts.scaleseek_prompt import PROMPT as SYS
from train.sft.coldstart import _fmt_tool_turn, _fmt_answer_turn

_TOK = 512


def _answer_in_ws(ws: Workspace, ans: str) -> bool:
    a = ans.lower()
    return any(a in d.get("text", "").lower() for d in ws.docs)


def build_traj(puzzle: dict, key: dict, policy: str, retriever) -> dict:
    q = puzzle["question"]
    ans = puzzle["golden_answers"][0]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}"}]
    mask = [0, 0]
    ws = Workspace()

    def _call(tc, think):
        res = execute_tool(tc["name"], tc["arguments"], ws, retriever, max_response_tokens=_TOK)
        msgs.append({"role": "assistant", "content": _fmt_tool_turn(think, tc)}); mask.append(1)
        msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)}); mask.append(0)

    # step 1 — default retrieval
    _call({"name": "bm25_retrieve", "arguments": {"query": q, "mode": "replace"}},
          "I'll search for this record with the given terms.")
    found = _answer_in_ws(ws, ans)
    n_tune = 0

    if not found and policy != "teacher":
        if policy == "search":
            t = key["target"]
            tc = {"name": "bm25_retrieve",
                  "arguments": {"query": q, "b": t["b"], "top_k": t["top_k"], "mode": "replace"}}
            think = ("The entry isn't in these results. The passage that holds it is long and is being "
                     "penalised by length normalisation, so I'll lower b to surface it.")
        else:  # heuristic — widen the net
            tc = {"name": "bm25_retrieve", "arguments": {"query": q, "top_k": 10, "mode": "replace"}}
            think = "The entry isn't in these results; I'll widen the search to pull it in."
        _call(tc, think)
        found = _answer_in_ws(ws, ans)
        n_tune = 1

    if found:
        _call({"name": "grep_workspace", "arguments": {"pattern": ans, "case_insensitive": True}},
              "The passage holding the entry is in the workspace; I'll locate it.")

    msgs.append({"role": "assistant", "content": _fmt_answer_turn(
        "The workspace identifies the registered entry.", ans)}); mask.append(1)

    return {"id": puzzle["id"], "question": q, "golden_answers": [ans],
            "messages": msgs, "loss_mask": mask, "status": "ok",
            "meta": {"policy": policy, "found": found, "n_tune": n_tune, "workspace_size": ws.size}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles-dir", default=".puzzles")
    ap.add_argument("--policy", choices=["teacher", "heuristic", "search"], required=True)
    ap.add_argument("--split", default="train", help="which split to build (train/heldout/all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pd = Path(args.puzzles_dir)
    os.environ["BM25_INDEX_DIR"] = str(pd / "bm25_index")
    retriever = BM25Retriever(index_dir=str(pd / "bm25_index"))
    key_by_id = {r["id"]: r for r in json.load(open(pd / "answer_key.json"))}
    puzzles = [json.loads(l) for l in open(pd / "questions.jsonl") if l.strip()]
    if args.split != "all":
        puzzles = [p for p in puzzles if p.get("split") == args.split]

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p in puzzles:
            traj = build_traj(p, key_by_id[p["id"]], args.policy, retriever)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            n += 1
    print(f"[puzzle_traj] policy={args.policy} split={args.split}: wrote {n} -> {args.out}")


if __name__ == "__main__":
    main()
