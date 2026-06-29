"""GrepSeek model eval wrapper.

GrepSeek (arxiv:2605.29307) is a trained search agent that issues shell
grep/rg commands against the full 21M-passage Wikipedia corpus. This module
wraps a GrepSeek checkpoint served via a separate vLLM instance.

The system prompt and tool format are copied from GrepSeek's
inference/agent.py so that the model sees exactly the format it was trained on.
Tool responses use the same JSON shape as GrepSeek's SFT trajectory tool turns.

Usage:
    # Start GrepSeek vLLM server on port 8002, then:
    python -m eval.run_eval --dataset hotpotqa --agent grepseek \\
        --grepseek-port 8002 --corpus-path /data/.../wiki_corpus.jsonl \\
        --output results/hotpotqa_grepseek.jsonl
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from .shell_tool import run_shell
from .agent import AgentRecord, _chat_completion

# System prompt reproduced from GrepSeek inference/agent.py so the model sees
# its exact training format without a dependency on the grepseek package.
_CORPUS_DESCRIPTION = """Corpus file: corpus.jsonl

Treat the corpus as a large text file with one Wikipedia passage per line. The file has 21 million lines, so always cap output (use `| head -n 3` for narrow searches, up to `| head -n 8` when you need to scan more chunks of the same article).

Useful command patterns:
- Substring search:
    rg -F "distinctive phrase" corpus.jsonl | head -n 3
- AND-narrow with a second grep:
    rg -F "phrase1" corpus.jsonl | rg -i -F "phrase2" | head -n 3
- Count first, then narrow if too many results:
    rg -F "pattern" corpus.jsonl | wc -l
- Useful flags: -F (fixed string, no regex), -i (case-insensitive), -w (whole word).

Allowed shell tools: rg, grep, find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr.
You MAY pipe with |. You MAY NOT use redirection (>, <), command chaining (; && ||), or command substitution ($(...), `...`)."""

_SYSTEM_PROMPT = (
    "You are a research agent that searches a Wikipedia corpus to answer multi-hop questions "
    "by issuing shell commands.\n\n"
    + _CORPUS_DESCRIPTION
    + """

For every step, write a short paragraph of reasoning in plain prose (2-5 sentences) — what you have learned from prior tool outputs, what's still missing, what you'll search for next. Then output exactly one of:

<tool_call>
{"name": "shell", "arguments": {"command": "your single-pipeline shell command, no newlines"}}
</tool_call>

OR (only when you have enough information to answer):

<answer>
the final answer (concise — typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""
)

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
        return None, am.group(1).strip(), None
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


def run_grepseek(
    example: dict,
    *,
    client: Any,
    model: str,
    corpus_path: str,
    max_turns: int = 6,
    max_tokens: Optional[int] = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tool_timeout: float = 60.0,
) -> AgentRecord:
    """Run GrepSeek checkpoint on one example.

    Conversation format matches GrepSeek's SFT training format:
        system: SYSTEM_PROMPT
        user:   "Query: {question}"   ← note "Query:", not "Question:"
        assistant <-> tool turns ...
        assistant: <answer>...</answer>
    """
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))

    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    messages: list = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {question}"},
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
            # GrepSeek's SFT tool turn format on validation failure
            err_payload = json.dumps({
                "command": "", "stdout": "",
                "stderr": f"format error: {parse_err}",
                "exit_code": -2, "timed_out": False,
                "truncated": False, "information_lines": [],
            })
            messages.append({"role": "tool", "content": err_payload})
            record.turns.append({"role": "tool", "content": err_payload, "synthetic": True})
            continue

        consecutive_errors = 0
        t_tool = time.perf_counter()
        result = run_shell(cmd, corpus_path=corpus_path, timeout=tool_timeout)
        record.tool_time_s += time.perf_counter() - t_tool
        record.n_tool_calls += 1

        # Wrap in GrepSeek's tool turn format so the model sees what it was trained on
        payload = json.dumps({
            "command": result.command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": False,
            "information_lines": [ln for ln in result.stdout.split("\n") if ln.strip()],
        }, ensure_ascii=False)
        messages.append({"role": "tool", "content": payload})
        record.turns.append({"role": "tool", "content": payload})

    else:
        record.finish_reason = "max_turns"

    record.messages = messages
    record.total_time_s = time.perf_counter() - t_start
    return record
