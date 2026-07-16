"""IRCoT baseline (Interleaving Retrieval with Chain-of-Thought, Trivedi et al.
ACL 2023, arxiv 2212.10509) — prompt-based, no training.

Core mechanism (the paper's contribution is the *interleaving*, not the few-shot
demos): alternate between a **reason step** and a **retrieve step**:

    1. retrieve top-k for the QUESTION -> seed the doc collection
    2. loop:
       a. reason: generate the NEXT single CoT sentence, conditioned on the
          question + ALL collected docs + the CoT so far
       b. if that sentence states the answer ("answer is X") -> extract, stop
       c. else use that sentence as the next query -> retrieve top-k -> merge
          new docs into the collection
    3. stop at max_steps; final answer from the last "answer is" or a reader call

Faithfulness note: the official IRCoT (github.com/StonyBrookNLP/ircot) uses
dataset-specific few-shot demonstrations. We use a zero-shot instruction form of
the SAME interleaving procedure on our Qwen3-4B — matching the mechanism (which is
what GrepSeek Table 1's IRCoT row also re-implements on its own backbone), with
the few-shot demos as the one documented divergence. Retriever = our BM25/E5 index
over wiki-18 (same index every retrieval baseline uses), top-k per --bm25-top-k.
"""
from __future__ import annotations

import re
import time
from typing import Any

from .bm25_retriever import BM25Retriever
from .agent import AgentRecord, _chat_completion

_ANSWER_RE = re.compile(r"\banswer is[:\s]+(.+)", re.IGNORECASE)


def _format_docs(hits: list[dict], max_chars: int = 1200) -> str:
    """IRCoT context blocks: 'Wikipedia Title: T\\n{text}'. wiki-18 passages carry
    the title as the first line of `text`, so we surface it as the block header."""
    blocks = []
    for i, h in enumerate(hits):
        txt = (h.get("text", "") or "").strip()
        first_nl = txt.find("\n")
        title = txt[:first_nl].strip().strip('"') if first_nl > 0 else f"doc {i+1}"
        body = txt[first_nl + 1:] if first_nl > 0 else txt
        blocks.append(f"Wikipedia Title: {title}\n{body[:max_chars]}")
    return "\n\n".join(blocks)


def _reason_prompt(question: str, docs_block: str, cot_so_far: str) -> str:
    return (
        "You answer a question by reasoning step by step, using the retrieved "
        "Wikipedia passages below. Write exactly ONE next reasoning sentence that "
        "follows from the passages and the reasoning so far. When you can conclude, "
        "write a sentence of the form \"So the answer is: <answer>.\" and nothing else.\n\n"
        f"{docs_block}\n\n"
        f"Question: {question}\n"
        f"Reasoning so far: {cot_so_far.strip() or '(none yet)'}\n\n"
        "Next reasoning sentence:"
    )


def _first_sentence(text: str) -> str:
    """Take the model's next-sentence output as one CoT step. Prefer the content
    after </think>; if the model spent all its budget thinking and emitted no
    post-think content (Qwen3 thinking mode), fall back to the last sentence of
    the thinking itself so the step is never silently empty."""
    after = text.split("</think>", 1)[1] if "</think>" in text else text
    after = after.strip()
    if not after:
        think = text
        if "<think>" in text and "</think>" in text:
            think = text.split("<think>", 1)[1].split("</think>", 1)[0]
        elif "<think>" in text:
            think = text.split("<think>", 1)[1]
        sents = re.findall(r"[^.!?\n]+[.!?]", think)
        after = (sents[-1] if sents else (think.strip().split("\n")[-1:] or [""])[0]).strip()
    after = after.split("\n")[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", after)
    return (m.group(1) if m else after).strip()


def run_ircot(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    max_turns: int = 6,
    max_tokens: int = 512,
    bm25_top_k: int = 3,
    temperature: float = 0.0,
) -> AgentRecord:
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    # 1. seed retrieval with the question
    collected: list[dict] = []
    seen_ids: set = set()

    def _merge(hits: list[dict], query: str) -> None:
        ids = [h["doc_id"] for h in hits]
        for h in hits:
            if h["doc_id"] not in seen_ids:
                seen_ids.add(h["doc_id"])
                collected.append(h)
        record.n_tool_calls += 1
        record.n_bm25_calls += 1
        record.workspace_doc_ids = list(dict.fromkeys(record.workspace_doc_ids + ids))
        record.bm25_calls.append({
            "query": query, "k1": None, "b": None, "top_k": bm25_top_k,
            "mode": "merge", "doc_ids": ids,
        })

    t_tool = time.perf_counter()
    _merge(retriever.retrieve(question, top_k=bm25_top_k), question)
    record.tool_time_s += time.perf_counter() - t_tool

    cot = ""
    for _ in range(max_turns):
        docs_block = _format_docs(collected)
        t_llm = time.perf_counter()
        text, err = _chat_completion(
            client, model=model,
            messages=[{"role": "user", "content": _reason_prompt(question, docs_block, cot)}],
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        record.llm_time_s += time.perf_counter() - t_llm
        if err:
            record.finish_reason = "api_error"
            record.error = err
            break

        record.n_turns += 1
        sentence = _first_sentence(text or "")
        cot = (cot + " " + sentence).strip()
        record.turns.append({"role": "assistant", "content": sentence})

        m = _ANSWER_RE.search(sentence)
        if m:
            ans = m.group(1).strip().rstrip(".").strip()
            record.prediction = ans or None
            record.finish_reason = "answer" if record.prediction else "no_answer"
            break

        # otherwise: use this sentence as the next query
        if not sentence:
            record.finish_reason = "no_answer"
            break
        t_tool = time.perf_counter()
        _merge(retriever.retrieve(sentence, top_k=bm25_top_k), sentence)
        record.tool_time_s += time.perf_counter() - t_tool
    else:
        # exhausted steps without an explicit "answer is": one reader call.
        t_llm = time.perf_counter()
        text, err = _chat_completion(
            client, model=model,
            messages=[{"role": "user", "content": (
                f"{_format_docs(collected)}\n\nQuestion: {question}\n"
                f"Reasoning: {cot}\n\nGive only the final short answer:")}],
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        record.llm_time_s += time.perf_counter() - t_llm
        if not err and text:
            record.prediction = _first_sentence(text).rstrip(".").strip() or None
        record.finish_reason = "answer" if record.prediction else "max_turns"

    record.final_workspace_size = len(record.workspace_doc_ids)
    record.total_time_s = time.perf_counter() - t_start
    return record
