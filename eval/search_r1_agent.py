"""Search-R1 eval wrapper.

Search-R1 uses one of the frozen PeterJinGo 7B/14B v0.3 GRPO checkpoints and:

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
separate port (default 8001) so it can run alongside the Qwen3.5-9B server.

Ref: https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from .retrievers import Retriever
from .agent import AgentRecord

_SEARCH_RE = re.compile(r"<search>(.*?)(?:</search>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_USER_PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)


def _apply_checkpoint_template(user_content: str, tokenizer: Any) -> str:
    """Use the selected Search-R1 checkpoint's own chat template."""
    if tokenizer is None:
        raise ValueError("Search-R1 requires its checkpoint tokenizer")
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True, tokenize=False)
    return user_content


def _make_search_r1_client(host: str, port: int):
    from openai import OpenAI
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")


def _format_hits(hits: list[dict]) -> str:
    rendered = ""
    for i, h in enumerate(hits):
        content = h.get("text", "")
        title, _, text = content.partition("\n")
        rendered += f"Doc {i + 1}(Title: {title}) {text}\n"
    return rendered


def run_search_r1(
    example: dict,
    *,
    client: Any,                  # openai client → Search-R1 vLLM server (port 8001)
    model: str = "search_r1",
    retriever: Retriever,
    tokenizer: Any,
    action_budget: int = 4,
    max_tokens: int = 1024,
    retrieval_top_k: int = 3,
    temperature: float = 0.7,
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
    prompt_text = _apply_checkpoint_template(user_content, tokenizer)

    for _ in range(action_budget):
        t_llm = time.perf_counter()
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt_text,
                max_tokens=max_tokens,
                stop=["</search>", " </search>", "</search>\n", " </search>\n",
                      "</search>\n\n", " </search>\n\n"],
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
        stop_reason = getattr(resp.choices[0], "stop_reason", None)
        record.n_turns += 1

        matches = _ANSWER_RE.findall(generated)
        if matches:
            record.prediction = matches[-1].strip()
            record.finish_reason = "answer"
            break

        stopped_on_search = bool(_SEARCH_RE.search(generated)) and finish_reason == "stop"
        if stop_reason is not None:
            stopped_on_search = "</search>" in str(stop_reason)
        if not stopped_on_search:
            record.finish_reason = "max_tokens" if finish_reason == "length" else "no_answer"
            break

        # Stopped on </search> — extract query and retrieve
        sm = _SEARCH_RE.search(generated)
        if not sm:
            record.finish_reason = "parse_error"
            break

        query = sm.group(1).strip()
        record.n_tool_calls += 1
        record.n_bm25_calls += int(retriever.__class__.__name__ == "FixedBM25Retriever")

        t_tool = time.perf_counter()
        hits = retriever.retrieve(query, top_k=retrieval_top_k)
        record.tool_time_s += time.perf_counter() - t_tool
        record.final_workspace_size = max(record.final_workspace_size, len(hits))
        # record retrieval trace so Gold/Qrel R@W is computable (was missing ->
        # workspace_doc_ids stayed empty -> GoldR@W read as 0)
        doc_ids = [h["doc_id"] for h in hits]
        record.workspace_doc_ids = list(dict.fromkeys(record.workspace_doc_ids + doc_ids))
        record.bm25_calls.append({
            "query": query, "k1": getattr(retriever, "k1", None),
            "b": getattr(retriever, "b", None), "top_k": retrieval_top_k,
            "mode": "merge", "doc_ids": doc_ids,
        })

        results_text = _format_hits(hits)
        # Official local generation includes the stop sequence; OpenAI-compatible
        # servers usually exclude it. Normalize to exactly one closing marker.
        search_output = generated if generated.rstrip().endswith("</search>") \
            else generated + "</search>"
        prompt_text += f"\n\n{search_output}<information>{results_text}</information>\n\n"
    else:
        record.finish_reason = "action_budget"
        matches = _ANSWER_RE.findall(prompt_text)
        if matches:
            record.prediction = matches[-1].strip()
            record.finish_reason = "answer"

    record.total_time_s = time.perf_counter() - t_start
    return record
