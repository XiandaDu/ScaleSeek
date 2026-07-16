"""Search-R1 eval wrapper.

Search-R1 (PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo) uses:

    User prompt (via Qwen2.5 chat template):
        "Answer the given question. You must conduct reasoning inside <think>
        and </think>... call a search engine by <search> query </search> ...
        results between <information> and </information>...
        provide the answer inside <answer> and </answer>."

    Multi-turn loop:
        model generates → stops at </search>
        → we inject <information>BM25 results</information>
        → continue until <answer>...</answer> or EOS

The model is served via vLLM's /v1/completions (raw text, not chat) on a
separate port (default 8001) so it can run alongside the main Qwen3-4B server.

Ref: https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from .bm25_retriever import BM25Retriever
from .agent import AgentRecord

_SEARCH_RE = re.compile(r"<search>(.*?)(?:</search>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_USER_PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {question}"
)


def _apply_qwen25_template(user_content: str) -> str:
    """Hardcoded Qwen2.5 chat template — avoids downloading tokenizer at eval time."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _make_search_r1_client(host: str, port: int):
    from openai import OpenAI
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")


def _format_hits(hits: list[dict], max_chars: int = 1500) -> str:
    parts = []
    for i, h in enumerate(hits):
        text = h.get("text", "")[:max_chars]
        parts.append(f"Doc {i + 1}: {text}")
    return "\n".join(parts)


def run_search_r1(
    example: dict,
    *,
    client: Any,                  # openai client → Search-R1 vLLM server (port 8001)
    model: str = "search_r1",
    retriever: BM25Retriever,
    max_turns: int = 8,
    max_tokens: int = 1024,
    bm25_top_k: int = 3,
    temperature: float = 0.0,
) -> AgentRecord:
    """Run Search-R1 on one example using vLLM /v1/completions."""
    ex_id = str(example.get("id", ""))
    question = example.get("question", "").strip()
    if question and question[-1] != "?":
        question += "?"
    golds = list(example.get("golden_answers", []))

    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    user_content = _USER_PROMPT_TEMPLATE.format(question=question)
    # Raw text prompt (completions API, not chat); vLLM handles this correctly
    prompt_text = _apply_qwen25_template(user_content)

    for _ in range(max_turns):
        t_llm = time.perf_counter()
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt_text,
                max_tokens=max_tokens,
                stop=["</search>"],
                temperature=temperature,
                top_p=1.0,
            )
        except Exception as e:
            record.finish_reason = "api_error"
            record.error = str(e)
            break

        record.llm_time_s += time.perf_counter() - t_llm
        generated = resp.choices[0].text
        finish_reason = resp.choices[0].finish_reason
        record.n_turns += 1

        prompt_text += generated

        # EOS / length — check for final answer
        if finish_reason != "stop":
            matches = _ANSWER_RE.findall(prompt_text)
            if matches:
                record.prediction = matches[-1].strip()
                record.finish_reason = "answer"
            else:
                record.finish_reason = "max_tokens" if finish_reason == "length" else "no_answer"
            break

        # Stopped on </search> — extract query and retrieve
        sm = _SEARCH_RE.search(generated)
        if not sm:
            record.finish_reason = "parse_error"
            break

        query = sm.group(1).strip()
        record.n_tool_calls += 1
        record.n_bm25_calls += 1

        t_tool = time.perf_counter()
        hits = retriever.retrieve(query, top_k=bm25_top_k)
        record.tool_time_s += time.perf_counter() - t_tool
        record.final_workspace_size = max(record.final_workspace_size, len(hits))
        # record retrieval trace so Gold/Qrel R@W is computable (was missing ->
        # workspace_doc_ids stayed empty -> GoldR@W read as 0)
        doc_ids = [h["doc_id"] for h in hits]
        record.workspace_doc_ids = list(dict.fromkeys(record.workspace_doc_ids + doc_ids))
        record.bm25_calls.append({
            "query": query, "k1": None, "b": None, "top_k": bm25_top_k,
            "mode": "merge", "doc_ids": doc_ids,
        })

        results_text = _format_hits(hits)
        # vLLM stop excludes the stop string — re-add </search> then inject
        prompt_text += f"</search>\n<information>{results_text}</information>\n\n"
    else:
        record.finish_reason = "max_turns"
        matches = _ANSWER_RE.findall(prompt_text)
        if matches:
            record.prediction = matches[-1].strip()
            record.finish_reason = "answer"

    record.total_time_s = time.perf_counter() - t_start
    return record
