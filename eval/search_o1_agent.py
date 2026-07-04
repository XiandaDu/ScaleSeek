"""Search-O1 baseline (prompt-based, no training).

Search-O1 (Li et al. 2025, "Search-o1: Agentic Search-Enhanced Large Reasoning
Models") interleaves free-form reasoning with retrieval: when the model needs a
fact it emits <|begin_search_query|> ... <|end_search_query|>; the harness
retrieves passages, runs a **Reason-in-Documents** condensation step (a separate
LLM call that distills the retrieved pages into "**Final Information**"), and
injects the result as <|begin_search_result|> ... <|end_search_result|>. The
model continues until it emits the final \\boxed{answer}.

Prompts and special tokens are reproduced verbatim from the official repo
(github.com/sunnynexus/Search-o1, scripts/prompts.py). Base model + condenser =
our main Qwen3-4B; retriever = our BM25 over wiki-18. We drive it through the
chat API with a stop on <|end_search_query|> (a harness detail; the prompt,
tokens, and condensation prompt are the paper's).
"""
from __future__ import annotations

import re
import time
from typing import Any

from .bm25_retriever import BM25Retriever
from .agent import AgentRecord, _chat_completion

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

# --- Verbatim prompts from sunnynexus/Search-o1 scripts/prompts.py ------------

def _multiqa_instruction(max_search_limit: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        f"- To perform a search: write {BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}.\n"
        f"Then, the system will search and analyze relevant web pages, then provide you with "
        f"helpful information in the format {BEGIN_SEARCH_RESULT} ...search results... {END_SEARCH_RESULT}.\n\n"
        "You can repeat the search process multiple times if necessary. The maximum number of "
        f"search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning."
    )


def _task_instruction_openqa(question: str) -> str:
    return (
        "Please answer the following question. You should think step by step to solve it.\n\n"
        "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question}\n\n"
    )


def _reasonchain_instruction(prev_reasoning: str, search_query: str, document: str) -> str:
    return (
        "**Task Instruction:**\n\n"
        "You are tasked with reading and analyzing web pages based on the following inputs: "
        "**Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. "
        "Your objective is to extract relevant and helpful information for **Current Search Query** "
        "from the **Searched Web Pages** and seamlessly integrate this information into the "
        "**Previous Reasoning Steps** to continue reasoning for the original question.\n\n"
        "**Guidelines:**\n\n"
        "1. **Analyze the Searched Web Pages:**\n"
        "- Carefully review the content of each searched web page.\n"
        "- Identify factual information that is relevant to the **Current Search Query** and can "
        "aid in the reasoning process for the original question.\n\n"
        "2. **Extract Relevant Information:**\n"
        "- Select the information from the Searched Web Pages that directly contributes to "
        "advancing the **Previous Reasoning Steps**.\n"
        "- Ensure that the extracted information is accurate and relevant.\n\n"
        "3. **Output Format:**\n"
        "- **If the web pages provide helpful information for current search query:** Present the "
        "information beginning with `**Final Information**` as shown below.\n"
        "**Final Information**\n\n[Helpful information]\n\n"
        "- **If the web pages do not provide any helpful information for current search query:** "
        "Output the following text.\n\n**Final Information**\n\nNo helpful information found.\n\n"
        "**Inputs:**\n"
        f"- **Previous Reasoning Steps:**\n{prev_reasoning}\n\n"
        f"- **Current Search Query:**\n{search_query}\n\n"
        f"- **Searched Web Pages:**\n{document}\n\n"
        f'Now you should analyze each web page and find helpful information based on the current '
        f'search query "{search_query}" and previous reasoning steps.\n'
    )


_QUERY_RE = re.compile(re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY), re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)


def _extract_answer(text: str) -> str | None:
    m = list(_BOXED_RE.finditer(text))
    if m:
        return m[-1].group(1).strip()
    return None


def _format_docs(hits: list[dict], max_chars: int = 1500) -> str:
    return "\n\n".join(
        f"Web Page {i+1}:\n{h.get('text', '')[:max_chars]}" for i, h in enumerate(hits)
    )


def run_search_o1(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    max_turns: int = 8,
    max_tokens: int = 2048,
    bm25_top_k: int = 5,
    temperature: float = 0.0,
    max_search_limit: int = 5,
) -> AgentRecord:
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    messages = [
        {"role": "system", "content": _multiqa_instruction(max_search_limit)},
        {"role": "user", "content": _task_instruction_openqa(question)},
    ]
    reasoning_so_far = ""
    n_searches = 0

    for _ in range(max_turns):
        t_llm = time.perf_counter()
        text, err = _chat_completion(
            client, model=model, messages=messages,
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
            stop=[END_SEARCH_QUERY],
        )
        record.llm_time_s += time.perf_counter() - t_llm
        if err:
            record.finish_reason = "api_error"
            record.error = err
            break

        record.n_turns += 1
        qm = _QUERY_RE.search((text or "") + END_SEARCH_QUERY)
        has_query = qm is not None and n_searches < max_search_limit

        assistant_text = text or ""
        if has_query:
            assistant_text = assistant_text + END_SEARCH_QUERY  # re-add the stripped stop
        messages.append({"role": "assistant", "content": assistant_text})
        reasoning_so_far += "\n" + (text or "")
        record.turns.append({"role": "assistant", "content": assistant_text})

        answer = _extract_answer(assistant_text)
        if answer is not None and not has_query:
            record.prediction = answer
            record.finish_reason = "answer"
            break

        if not has_query:
            # No search, no boxed answer -> stop (EOS-like)
            record.prediction = _extract_answer(assistant_text)
            record.finish_reason = "answer" if record.prediction else "no_answer"
            break

        # --- retrieve + Reason-in-Documents condensation ---
        query = qm.group(1).strip()
        n_searches += 1
        record.n_tool_calls += 1
        record.n_bm25_calls += 1
        t_tool = time.perf_counter()
        hits = retriever.retrieve(query, top_k=bm25_top_k)
        record.tool_time_s += time.perf_counter() - t_tool
        record.workspace_doc_ids = list(dict.fromkeys(
            record.workspace_doc_ids + [h["doc_id"] for h in hits]))
        record.bm25_calls.append({
            "query": query, "k1": None, "b": None, "top_k": bm25_top_k,
            "mode": "merge", "doc_ids": [h["doc_id"] for h in hits],
        })

        t_llm = time.perf_counter()
        condensed, cerr = _chat_completion(
            client, model=model,
            messages=[{"role": "user", "content": _reasonchain_instruction(
                reasoning_so_far, query, _format_docs(hits))}],
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        record.llm_time_s += time.perf_counter() - t_llm
        info = "No helpful information found."
        if not cerr and condensed:
            idx = condensed.rfind("**Final Information**")
            info = condensed[idx + len("**Final Information**"):].strip() if idx >= 0 else condensed.strip()

        result_block = f"\n\n{BEGIN_SEARCH_RESULT}\n{info}\n{END_SEARCH_RESULT}\n\n"
        messages.append({"role": "user", "content": result_block})
        record.turns.append({"role": "tool", "content": result_block, "query": query})
    else:
        record.finish_reason = "max_turns"

    record.final_workspace_size = len(record.workspace_doc_ids)
    record.total_time_s = time.perf_counter() - t_start
    return record
