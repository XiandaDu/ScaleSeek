#!/usr/bin/env python3
"""Gate probe: does Qwen3.5 greedy decoding close its thinking within budget?

2026-07-22 incident: p2_direct ran 12h with 100% parse_error because at
temperature=0 the model thought past max_tokens=8192 on every question and the
reasoning parser returns neither content nor reasoning_content on truncation.
TASK.md freezes temperature=0 for direct/rag and mandates a budget large enough
for thinking + answer, so before burning a full run we probe N real questions.

Exit 0  -> >=80% of probes produced an <answer> block: safe to run the frozen
           config at this budget.
Exit 3  -> gate failed. The script then re-probes two alternative modes
           (enable_thinking=false + temp 0, and the model's recommended
           sampling) and prints a comparison table as evidence for a human
           decision. Nothing under results/ is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.agent import parse_assistant  # noqa: E402  (exact production parser)
from eval import prompts  # noqa: E402  (exact production prompt loader)


def probe_one(client, model, question, mode, max_tokens, thinking_budget):
    messages = [
        {"role": "system", "content": prompts.load("direct")},
        {"role": "user", "content": f"Question: {question}"},
    ]
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if mode == "greedy_think":
        kwargs.update(temperature=0.0, top_p=1.0)
    elif mode == "greedy_nothink":
        kwargs.update(temperature=0.0, top_p=1.0,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    elif mode == "sampling_think":
        # Qwen thinking-mode recommended sampling (model card): the vendor's
        # documented alternative if greedy decoding cannot close its thinking.
        kwargs.update(temperature=0.6, top_p=0.95, extra_body={"top_k": 20})
    elif mode == "budget_think":
        # Production candidate: no output cap (vLLM bounds at context length,
        # as the official GrepSeek/DCI harnesses do) + vLLM-native forced
        # thinking closure. Server must run VLLM_USE_V2_MODEL_RUNNER=0.
        kwargs.update(temperature=0.0, top_p=1.0, max_tokens=None,
                      extra_body={"thinking_token_budget": thinking_budget})
    else:
        raise ValueError(mode)
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content or ""
    reasoning = (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None) or "")
    # Mirror eval.agent._chat_completion's reconstruction exactly.
    if reasoning.strip():
        text = f"<think>\n{reasoning.strip()}\n</think>\n{content}"
    else:
        text = content
        if text and "</think>" in text and "<think>" not in text:
            text = "<think>\n" + text
    answer = parse_assistant(text or "").answer
    return {
        "finish_reason": choice.finish_reason,
        "completion_tokens": resp.usage.completion_tokens,
        "answer": answer,
        "content_head": (content or "")[:160],
        "reasoning_tail": (reasoning or "")[-120:],
    }


def run_mode(client, model, examples, mode, max_tokens, thinking_budget):
    def one(ex):
        try:
            r = probe_one(client, model, ex["question"], mode, max_tokens,
                          thinking_budget)
        except Exception as e:  # a dead server must fail the gate, not hang it
            r = {"finish_reason": f"error:{e}", "completion_tokens": 0,
                 "answer": None, "content_head": "", "reasoning_tail": ""}
        r["id"] = ex["id"]
        r["golds"] = ex["golden_answers"][:3]
        return r
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, examples))
    answered = sum(1 for r in rows if r["answer"])
    toks = sorted(r["completion_tokens"] for r in rows)
    print(f"\n== mode {mode} (max_tokens={max_tokens}, "
          f"thinking_budget={thinking_budget}) ==")
    for r in rows:
        print(f"  {r['id']:16s} finish={str(r['finish_reason']):10s} "
              f"tokens={r['completion_tokens']:6d} answer={str(r['answer'])[:40]!r} "
              f"golds={r['golds']}")
        if not r["answer"]:  # show what came out instead of an <answer> block
            print(f"      content_head={r['content_head']!r}")
            print(f"      reasoning_tail={r['reasoning_tail']!r}")
    print(f"  -> answered {answered}/{len(rows)}, completion_tokens "
          f"median={toks[len(toks)//2]} max={toks[-1]}")
    return answered, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--popqa", type=Path, required=True,
                    help="normalized dataset jsonl (id/question/golden_answers)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=28000)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--gate-mode", default="greedy_think",
                    choices=["greedy_think", "greedy_nothink",
                             "sampling_think", "budget_think"])
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="default: the frozen value from configs/baselines.yaml "
                         "(direct entry), so the gate always tests production")
    args = ap.parse_args()
    if args.thinking_budget is None:
        from eval.config import resolved_method
        args.thinking_budget = resolved_method("direct")["method"]["thinking_token_budget"]

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    examples = []
    with args.popqa.open() as fh:
        for line in fh:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= args.n:
                break

    answered, n = run_mode(client, args.model, examples, args.gate_mode,
                           args.max_tokens, args.thinking_budget)
    if answered / n >= args.threshold:
        print(f"GATE-PASS: mode {args.gate_mode} answered {answered}/{n}; "
              f"config is runnable.")
        return
    print(f"\nGATE-FAIL: only {answered}/{n} answered under gate mode "
          f"{args.gate_mode}. Collecting evidence from the other modes:")
    for mode in ("greedy_think", "greedy_nothink", "sampling_think",
                 "budget_think"):
        if mode != args.gate_mode:
            run_mode(client, args.model, examples, mode, args.max_tokens,
                     args.thinking_budget)
    print(f"\nGATE-FAIL summary: gate mode {args.gate_mode} cannot produce "
          "answers; a config decision is needed. See the mode tables above.")
    sys.exit(3)


if __name__ == "__main__":
    main()
