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


def probe_one(client, model, question, mode, max_tokens, thinking_budget,
              agent="direct"):
    # The gate must exercise the prompt of the agent it gates. It used to hardcode
    # `direct` for every agent, so the scaleseek gate never loaded the tool schema,
    # never produced a <tool_call>, and could not have observed the tool-call
    # behaviour its failure was attributed to.
    messages = [
        {"role": "system", "content": prompts.load(agent)},
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
    parsed = parse_assistant(text or "")
    return {
        "finish_reason": choice.finish_reason,
        "completion_tokens": resp.usage.completion_tokens,
        # A tool-using agent commits by calling a tool; treat that as a pass too,
        # otherwise the gate scores a perfectly healthy first turn as a failure.
        "answer": parsed.answer,
        "tool_name": parsed.tool_name,
        "actionable": bool(parsed.answer or parsed.tool_name),
        "parse_error": parsed.error,
        "content_head": (content or "")[:160],
        "reasoning_tail": (reasoning or "")[-120:],
    }


def run_mode(client, model, examples, mode, max_tokens, thinking_budget,
             agent="direct"):
    def one(ex):
        try:
            r = probe_one(client, model, ex["question"], mode, max_tokens,
                          thinking_budget, agent=agent)
        except Exception as e:  # a dead server must fail the gate, not hang it
            r = {"finish_reason": f"error:{e}", "completion_tokens": 0,
                 "answer": None, "tool_name": None, "actionable": False,
                 "parse_error": str(e), "content_head": "", "reasoning_tail": ""}
        r["id"] = ex["id"]
        r["golds"] = ex["golden_answers"][:3]
        return r
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, examples))
    # Pass = the turn produced a parseable commitment (an answer, or a tool call
    # for tool-using agents). Scoring only <answer> made every healthy scaleseek
    # first turn look like a failure.
    ok = sum(1 for r in rows if r["actionable"])
    n_tool = sum(1 for r in rows if r["tool_name"])
    toks = sorted(r["completion_tokens"] for r in rows)
    print(f"\n== mode {mode} on agent={agent} (max_tokens={max_tokens}, "
          f"thinking_budget={thinking_budget}) ==")
    for r in rows:
        commit = r["tool_name"] and f"tool:{r['tool_name']}" or str(r["answer"])[:40]
        print(f"  {r['id']:16s} finish={str(r['finish_reason']):10s} "
              f"tokens={r['completion_tokens']:6d} commit={commit!r} "
              f"golds={r['golds']}")
        if not r["actionable"]:  # show what came out instead
            print(f"      parse_error={r['parse_error']!r}")
            print(f"      content_head={r['content_head']!r}")
            print(f"      reasoning_tail={r['reasoning_tail']!r}")
    print(f"  -> actionable {ok}/{len(rows)} (of which {n_tool} tool_call), "
          f"completion_tokens median={toks[len(toks)//2]} max={toks[-1]}")
    return ok, len(rows)


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
                         "for --agent, so the gate always tests production")
    ap.add_argument("--agent", default="direct",
                    choices=["direct", "rag", "scaleseek"],
                    help="which agent's system prompt to probe (must match the "
                         "agent being gated; the gate hardcoded 'direct' before "
                         "2026-07-28, so it never exercised scaleseek's tools)")
    args = ap.parse_args()
    if args.thinking_budget is None:
        from eval.config import resolved_method
        budget_source = "direct" if args.agent == "scaleseek" else args.agent
        args.thinking_budget = resolved_method(budget_source)["method"].get(
            "thinking_token_budget", 4096)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    examples = []
    with args.popqa.open() as fh:
        for line in fh:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= args.n:
                break

    ok, n = run_mode(client, args.model, examples, args.gate_mode,
                     args.max_tokens, args.thinking_budget, agent=args.agent)
    if ok / n >= args.threshold:
        print(f"GATE-PASS: mode {args.gate_mode} on agent {args.agent} produced "
              f"{ok}/{n} actionable turns; config is runnable.")
        return
    print(f"\nGATE-FAIL: only {ok}/{n} actionable under gate mode "
          f"{args.gate_mode} on agent {args.agent}. "
          f"Collecting evidence from the other modes:")
    scores = {args.gate_mode: (ok, n)}
    for mode in ("greedy_think", "greedy_nothink", "sampling_think",
                 "budget_think"):
        if mode != args.gate_mode:
            scores[mode] = run_mode(client, args.model, examples, mode,
                                    args.max_tokens, args.thinking_budget,
                                    agent=args.agent)
    print(f"\nGATE-FAIL summary for agent={args.agent} "
          f"(threshold {args.threshold:.0%}):")
    for mode, (o, tot) in scores.items():
        verdict = "would PASS" if o / tot >= args.threshold else "fails"
        print(f"  {mode:16s} {o:2d}/{tot}  {verdict}")
    print("A config decision is needed. Before concluding that a mode cannot "
          "work, check that this run postdates the parser fix in 19198ee — the "
          "07-23/07-24 gate tables scored 0/16 for modes that were in fact "
          "emitting valid answers.")
    sys.exit(3)


if __name__ == "__main__":
    main()
