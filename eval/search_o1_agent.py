"""Search-O1 baseline (prompt-based, no training) — official inline-continuation loop.

Search-o1 (Li et al. 2025, "Search-o1: Agentic Search-Enhanced Large Reasoning
Models") interleaves ONE continuous reasoning stream with retrieval:

    model generates ... <|begin_search_query|> q <|end_search_query|>   (stop here)
      -> harness retrieves top-k passages for q from the FULL corpus index
      -> Reason-in-Documents: a separate LLM call condenses the pages
      -> harness appends <|begin_search_result|> ... <|end_search_result|>
         to the SAME generation stream and resumes generation
    ... until EOS; final answer = last \\boxed{...}.

This mirrors the official implementation (github.com/sunnynexus/Search-o1,
scripts/run_search_o1.py): raw completions with a stop token and inline injection —
NOT a chat-turn loop (a chat loop breaks the reasoning chain and is unfaithful).
Prompts/special tokens are verbatim from the official scripts/prompts.py.

Role in our comparison (per the DCI / GrepSeek / DR-DCI / Pi-Serini / s3 papers):
Search-o1 is a *retriever-mediated* agentic-search baseline — its search action
queries an index over the full corpus and sees only top-k results — contrasted
against DCI-style agents that grep the raw corpus. The papers back it with a local
corpus retriever (E5/BM25) instead of Bing; we use our BM25 index over wiki-18
(same index every other retrieval baseline uses; swap in the dense index when built).
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

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
        "Once you have all the information you need, continue your reasoning.\n\n"
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


def _clean_latex(s: str) -> str:
    # unwrap \text{...}/\mathrm{...} etc, drop escaped spaces and leftover commands
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit|mathsf)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\ ", " ").replace("\\,", " ").replace("\\;", " ").replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _extract_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} with balanced-brace matching (handles nested
    \\text{}), then strip LaTeX. Returns None if no boxed answer."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    j = text.find("{", idx)
    if j < 0:
        return None
    depth, inner = 0, None
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                inner = text[j + 1:k]
                break
    if inner is None:
        inner = text[j + 1:]
    cleaned = _clean_latex(inner)
    return cleaned or None


def _format_docs(hits: list[dict], max_chars: int = 1500) -> str:
    return "\n\n".join(
        f"Web Page {i+1}:\n{h.get('text', '')[:max_chars]}" for i, h in enumerate(hits)
    )


def _apply_chat_template(user_content: str) -> str:
    """Hardcoded ChatML (Qwen3) — completions API needs the raw prompt text.
    The model opens its own <think> block after 'assistant\\n'."""
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _completion(client: Any, *, model: str, prompt: str, max_tokens: int,
                temperature: float, stop: list) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (text, stop_reason_or_finish, error). Clamps max_tokens on context overflow."""
    def _create(mt):
        return client.completions.create(
            model=model, prompt=prompt, max_tokens=mt,
            temperature=temperature, top_p=1.0, stop=stop,
        )
    try:
        resp = _create(max_tokens)
    except Exception as e:
        m = re.search(r"(\d+) input tokens.*?context length is only\s*(\d+)|maximum context length is (\d+) tokens.*?(\d+) input tokens",
                      str(e), re.DOTALL)
        room = None
        m2 = re.search(r"context length is (?:only )?(\d+).*?(\d+) input tokens", str(e), re.DOTALL) \
             or re.search(r"(\d+) input tokens.*?context length (?:is )?(?:only )?(\d+)", str(e), re.DOTALL)
        if m2:
            a, b = int(m2.group(1)), int(m2.group(2))
            ctx, n_in = max(a, b), min(a, b)
            room = ctx - n_in - 8
        if room is not None and room > 16:
            try:
                resp = _create(room)
            except Exception as e2:
                return None, None, f"API error (after clamp): {e2}"
        else:
            return None, None, f"API error: {e}"
    if not resp.choices:
        return None, None, "API returned no choices"
    ch = resp.choices[0]
    stop_reason = getattr(ch, "stop_reason", None)
    return ch.text or "", (stop_reason if stop_reason else ch.finish_reason), None


def run_search_o1(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    max_turns: int = 10,
    max_tokens: int = 2048,
    bm25_top_k: int = 5,
    temperature: float = 0.0,
    max_search_limit: int = 5,
) -> AgentRecord:
    """Official Search-o1 loop: continuous completions stream + inline injection."""
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    user_content = _multiqa_instruction(max_search_limit) + _task_instruction_openqa(question)
    prompt_text = _apply_chat_template(user_content)
    generated_total = ""
    n_searches = 0

    for _ in range(max_turns):
        t_llm = time.perf_counter()
        text, stop_or_finish, err = _completion(
            client, model=model, prompt=prompt_text,
            max_tokens=max_tokens, temperature=temperature,
            stop=[END_SEARCH_QUERY],
        )
        record.llm_time_s += time.perf_counter() - t_llm
        if err:
            record.finish_reason = "api_error"
            record.error = err
            break

        record.n_turns += 1
        generated_total += text
        record.turns.append({"role": "assistant", "content": text, "stop": str(stop_or_finish)})

        # Did we stop on the search-query stop string?
        stopped_on_query = (stop_or_finish == END_SEARCH_QUERY)
        if not stopped_on_query:
            # Heuristic fallback: an unclosed <|begin_search_query|> at the tail
            tail = text[-400:]
            stopped_on_query = (BEGIN_SEARCH_QUERY in tail
                                and END_SEARCH_QUERY not in tail.split(BEGIN_SEARCH_QUERY)[-1])

        if not stopped_on_query:
            # Natural EOS or length: extract the final boxed answer and stop.
            record.prediction = _extract_answer(generated_total)
            record.finish_reason = "answer" if record.prediction else "no_answer"
            break

        # --- stopped at <|end_search_query|>: extract the query ---
        qm = _QUERY_RE.findall(text + END_SEARCH_QUERY)
        if not qm:
            record.prediction = _extract_answer(generated_total)
            record.finish_reason = "answer" if record.prediction else "no_answer"
            break
        query = qm[-1].strip()
        prompt_text += text + END_SEARCH_QUERY + "\n\n"
        generated_total += END_SEARCH_QUERY + "\n\n"

        if n_searches >= max_search_limit:
            # Official behavior when the search budget is exhausted.
            info = ("The maximum search limit is exceeded. "
                    "You are not allowed to search.")
        else:
            n_searches += 1
            record.n_tool_calls += 1
            record.n_bm25_calls += 1
            t_tool = time.perf_counter()
            hits = retriever.retrieve(query, top_k=bm25_top_k)
            record.tool_time_s += time.perf_counter() - t_tool
            doc_ids = [h["doc_id"] for h in hits]
            record.workspace_doc_ids = list(dict.fromkeys(record.workspace_doc_ids + doc_ids))
            record.bm25_calls.append({
                "query": query, "k1": None, "b": None, "top_k": bm25_top_k,
                "mode": "merge", "doc_ids": doc_ids,
            })
            # Reason-in-Documents condensation (official; separate LLM call).
            t_llm = time.perf_counter()
            condensed, cerr = _chat_completion(
                client, model=model,
                messages=[{"role": "user", "content": _reasonchain_instruction(
                    generated_total[-10000:], query, _format_docs(hits))}],
                temperature=temperature, top_p=1.0, max_tokens=max_tokens,
            )
            record.llm_time_s += time.perf_counter() - t_llm
            info = "No helpful information found."
            if not cerr and condensed:
                idx = condensed.rfind("**Final Information**")
                info = (condensed[idx + len("**Final Information**"):].strip()
                        if idx >= 0 else condensed.strip())

        result_block = f"{BEGIN_SEARCH_RESULT}\n{info}\n{END_SEARCH_RESULT}\n\n"
        prompt_text += result_block
        generated_total += result_block
        record.turns.append({"role": "search_result", "content": result_block, "query": query})
    else:
        record.finish_reason = "max_turns"
        record.prediction = _extract_answer(generated_total)
        if record.prediction:
            record.finish_reason = "answer"

    record.final_workspace_size = len(record.workspace_doc_ids)
    record.total_time_s = time.perf_counter() - t_start
    return record
