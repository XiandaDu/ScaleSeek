"""Prompt-based DCI agent — grep on full corpus, no BM25 first stage.

Baseline for 'Beyond Semantic Similarity: Rethinking Retrieval for Agentic
Search via Direct Corpus Interaction' (arxiv:2605.05242). The agent issues
shell grep/rg commands directly against the full 21M-passage corpus with no
BM25 pre-filtering step. Uses the same Qwen3-4B vLLM as ScaleSeek, with a
prompt that includes search strategy hints (prompt-only, not trained).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from .shell_tool import run_shell
from .agent import AgentRecord, _chat_completion, clean_answer
from . import prompts

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _parse(text: str):
    """Return (command, answer, error)."""
    if not text:
        return None, None, "empty response"
    tm = _TOOL_CALL_RE.search(text)
    am = _ANSWER_RE.search(text)
    if tm and am:
        first = "tool" if tm.start() < am.start() else "answer"
    elif tm:
        first = "tool"
    elif am:
        first = "answer"
    else:
        return None, None, "no <tool_call> or <answer> found"
    if first == "answer":
        return None, clean_answer(am.group(1)), None
    try:
        obj = json.loads(tm.group(1).strip())
    except Exception as e:
        return None, None, f"JSON parse error: {e}"
    if obj.get("name") != "shell":
        return None, None, f"unknown tool: {obj.get('name')!r}"
    cmd = (obj.get("arguments") or {}).get("command", "").strip()
    if not cmd:
        return None, None, "missing command"
    return cmd, None, None


def run_dci(
    example: dict,
    *,
    client: Any,
    model: str,
    corpus_path: str,
    max_turns: int = 8,
    max_tokens: Optional[int] = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tool_timeout: float = 30.0,
    tool_max_chars: int = 20000,
) -> AgentRecord:
    """Run prompt-based DCI (raw corpus grep) on one example.

    tool_max_chars=20000 matches the DCI paper's L3 per-tool-result truncation cap
    (arxiv 2605.05242, Table 1), replacing the earlier 8000-char default.
    """
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))

    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    messages: list = [
        {"role": "system", "content": prompts.load("dci_prompt")},
        {"role": "user", "content": f"Question: {question}"},
    ]
    consecutive_errors = 0

    for _ in range(max_turns):
        t_llm = time.perf_counter()
        text, err = _chat_completion(
            client, model=model, messages=messages,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        record.llm_time_s += time.perf_counter() - t_llm

        if err:
            record.finish_reason = "api_error"
            record.error = err
            break

        record.n_turns += 1
        messages.append({"role": "assistant", "content": text})
        cmd, answer, parse_err = _parse(text)

        record.turns.append({
            "role": "assistant", "content": text,
            "parse": {"command": cmd, "answer": answer, "error": parse_err},
        })

        if answer is not None:
            record.prediction = answer
            record.finish_reason = "answer"
            break

        if parse_err:
            consecutive_errors += 1
            if consecutive_errors >= 2:
                record.finish_reason = "parse_error"
                record.error = parse_err
                break
            feedback = json.dumps({"stderr": f"format error: {parse_err}", "exit_code": -2,
                                   "stdout": "", "timed_out": False})
            messages.append({"role": "tool", "content": feedback})
            record.turns.append({"role": "tool", "content": feedback, "synthetic": True})
            continue

        consecutive_errors = 0
        t_tool = time.perf_counter()
        result = run_shell(cmd, corpus_path=corpus_path, timeout=tool_timeout,
                           max_chars=tool_max_chars)
        record.tool_time_s += time.perf_counter() - t_tool
        record.n_tool_calls += 1

        payload = result.to_payload()
        messages.append({"role": "tool", "content": payload})
        record.turns.append({"role": "tool", "content": payload})

    else:
        record.finish_reason = "max_turns"

    record.messages = messages
    record.total_time_s = time.perf_counter() - t_start
    return record
