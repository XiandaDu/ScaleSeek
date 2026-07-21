"""Search-O1 baseline (prompt-based, no training) — official inline-continuation loop.

Search-o1 (Li et al. 2025, "Search-o1: Agentic Search-Enhanced Large Reasoning
Models") interleaves ONE continuous reasoning stream with retrieval:

    model generates ... <|begin_search_query|> q <|end_search_query|>   (stop here)
      -> harness retrieves top-k passages for q from the FULL corpus index
      -> Reason-in-Documents: a separate LLM call condenses the pages
      -> harness appends <|begin_search_result|> ... <|end_search_result|>
         to the SAME generation stream and resumes generation
    ... until EOS; final answer = last \\boxed{...}.

This mirrors the pinned official implementation (github.com/RUC-NLPIR/Search-o1,
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

from .retrievers import Retriever
from .agent import AgentRecord

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

# --- Verbatim prompts from sunnynexus/Search-o1 scripts/prompts.py ------------

def _singleqa_instruction(max_search_limit: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        f"- To perform a search: write {BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}.\n"
        f"Then, the system will search and analyze relevant web pages, then provide you with "
        f"helpful information in the format {BEGIN_SEARCH_RESULT} ...search results... {END_SEARCH_RESULT}.\n\n"
        "You can repeat the search process multiple times if necessary. The maximum number of "
        f"search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        f"{BEGIN_SEARCH_QUERY}first Nobel Prize in Physics winner{END_SEARCH_QUERY}\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        f"- Use {BEGIN_SEARCH_QUERY} to request a web search and end with {END_SEARCH_QUERY}.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def _multiqa_instruction(max_search_limit: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        f"- To perform a search: write {BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with "
        f"helpful information in the format {BEGIN_SEARCH_RESULT} ...search results... {END_SEARCH_RESULT}.\n\n"
        "You can repeat the search process multiple times if necessary. The maximum number of "
        f"search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        f"{BEGIN_SEARCH_QUERY}Alice David Lara Croft voice{END_SEARCH_QUERY}\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        f"{BEGIN_SEARCH_QUERY}video game developed by Alice David Lara Croft{END_SEARCH_QUERY}\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        f"- Use {BEGIN_SEARCH_QUERY} to request a web search and end with {END_SEARCH_QUERY}.\n"
        "- When done searching, continue your reasoning.\n\n"
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
        f"- **Previous Reasoning Steps:**  \n{prev_reasoning}\n\n"
        f"- **Current Search Query:**  \n{search_query}\n\n"
        f"- **Searched Web Pages:**  \n{document}\n\n"
        f'Now you should analyze each web page and find helpful information based on the current '
        f'search query "{search_query}" and previous reasoning steps.\n'
    )


_QUERY_RE = re.compile(re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY), re.DOTALL)

# Lenient marker match for when the model malforms the <|...|> delimiters
# (e.g. "|<end_search_query>|", "<| end_search_query |>"): tolerate any mix of
# <, |, > around the begin/end keywords so a real query still triggers retrieval
# instead of degrading into a hallucinated result block.
_LENIENT_QUERY_RE = re.compile(
    r"[<|]{1,2}\s*begin_search_query\s*[|>]{1,2}(.*?)[<|]{1,2}\s*end_search_query\s*[|>]{0,2}",
    re.DOTALL | re.IGNORECASE)


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


def _format_docs(hits: list[dict]) -> str:
    return "\n\n".join(
        f"Web Page {i+1}:\n{h.get('text', '')}" for i, h in enumerate(hits)
    )


def _truncate_previous_reasoning(reasoning: str) -> str:
    """Verbatim Search-O1 step-selection logic before Reason-in-Documents."""
    steps = reasoning.replace("\n\n", "\n").split("\n")
    rendered = "".join(f"Step {i + 1}: {step}\n\n" for i, step in enumerate(steps))
    parts = rendered.split("\n\n")
    if len(parts) <= 5:
        return "\n\n".join(parts).strip("\n")
    kept = ""
    for i, step in enumerate(parts):
        if i == 0 or i >= len(parts) - 4 or BEGIN_SEARCH_QUERY in step \
                or BEGIN_SEARCH_RESULT in step:
            kept += step + "\n\n"
        elif not kept.endswith("\n\n...\n\n"):
            kept += "...\n\n"
    return kept.strip("\n")


def _apply_chat_template(user_content: str, tokenizer: Any) -> str:
    if tokenizer is None or not tokenizer.chat_template:
        raise ValueError("Search-O1 requires the Qwen3.5-9B checkpoint tokenizer")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True, tokenize=False)


def _completion(client: Any, *, model: str, prompt: str, max_tokens: int,
                temperature: float, top_p: float, sampling_top_k: int,
                stop: list) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (text, stop_reason_or_finish, error). Clamps max_tokens on context overflow."""
    def _create(mt):
        return client.completions.create(
            model=model, prompt=prompt, max_tokens=mt,
            temperature=temperature, top_p=top_p, stop=stop,
            extra_body={"top_k": sampling_top_k},
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


def _reason_in_documents(client: Any, *, model: str, prompt: str,
                         max_tokens: int) -> tuple[str | None, str | None]:
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.7, top_p=0.8,
            extra_body={"top_k": 20, "repetition_penalty": 1.05})
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not response.choices:
        return None, "API returned no choices"
    message = response.choices[0].message
    content = message.content or ""
    reasoning = (getattr(message, "reasoning_content", None)
                 or getattr(message, "reasoning", None) or "")
    return ((f"<think>\n{reasoning.strip()}\n</think>\n{content}"
             if reasoning.strip() else content), None)


def run_search_o1(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: Retriever,
    dataset_name: str,
    tokenizer: Any,
    max_turns: int = 15,
    max_tokens: int = 2048,
    retrieval_top_k: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.8,
    sampling_top_k: int = 20,
    max_search_limit: int | None = None,
) -> AgentRecord:
    """Official Search-o1 loop: continuous completions stream + inline injection."""
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    multi_hop = dataset_name in {"hotpotqa", "2wikimultihopqa", "musique", "bamboogle"}
    max_search_limit = max_search_limit or (10 if multi_hop else 5)
    instruction = _multiqa_instruction if multi_hop else _singleqa_instruction
    user_content = instruction(max_search_limit) + _task_instruction_openqa(question)
    prompt_text = _apply_chat_template(user_content, tokenizer)
    generated_total = ""
    n_searches = 0
    executed_search_queries: set[str] = set()

    for _ in range(max_turns):
        t_llm = time.perf_counter()
        text, stop_or_finish, err = _completion(
            client, model=model, prompt=prompt_text,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            sampling_top_k=sampling_top_k,
            stop=[END_SEARCH_QUERY],
        )
        record.llm_time_s += time.perf_counter() - t_llm
        if err:
            record.finish_reason = "api_error"
            record.error = err
            break

        record.n_turns += 1

        # Did vLLM stop on the exact query stop string?
        stopped_on_query = (stop_or_finish == END_SEARCH_QUERY)

        # Extract the search query. Strict markers first; if vLLM did NOT stop on
        # the token, fall back to a LENIENT match — the 4B frequently malforms the
        # <|...|> delimiters (e.g. "|<end_search_query>|"), which used to slip past
        # the stop string and let the model hallucinate its own result block
        # (~24% of the no-retrieval cases). We rebuild clean markers and drop the
        # hallucinated tail so the real retriever runs.
        query = None
        cut_text = text
        if stopped_on_query:
            qm = _QUERY_RE.findall(text + END_SEARCH_QUERY)
            if qm:
                query = qm[-1].strip()
                if not text.rstrip().endswith(END_SEARCH_QUERY):
                    cut_text = text.rstrip() + END_SEARCH_QUERY
        else:
            last = None
            for last in _LENIENT_QUERY_RE.finditer(text):
                pass
            if last and last.group(1).strip():
                query = last.group(1).strip()
                cut_text = (text[:last.start()].rstrip() + "\n"
                            + BEGIN_SEARCH_QUERY + query + END_SEARCH_QUERY)

        record.turns.append({"role": "assistant", "content": cut_text, "stop": str(stop_or_finish)})

        if not query:
            # No search intent at all: natural EOS -> take the boxed answer.
            generated_total += cut_text
            record.prediction = _extract_answer(generated_total)
            record.finish_reason = "answer" if record.prediction else "no_answer"
            break

        prompt_text += cut_text + "\n\n"
        generated_total += cut_text + "\n\n"

        if query in executed_search_queries:
            info = "You have searched this query. Please refer to previous results."
        elif n_searches >= max_search_limit:
            # Official behavior when the search budget is exhausted.
            info = ("The maximum search limit is exceeded. "
                    "You are not allowed to search.")
        else:
            n_searches += 1
            executed_search_queries.add(query)
            record.n_tool_calls += 1
            record.n_bm25_calls += int(retriever.__class__.__name__ == "FixedBM25Retriever")
            t_tool = time.perf_counter()
            hits = retriever.retrieve(query, top_k=retrieval_top_k)
            record.tool_time_s += time.perf_counter() - t_tool
            doc_ids = [h["doc_id"] for h in hits]
            record.workspace_doc_ids = list(dict.fromkeys(record.workspace_doc_ids + doc_ids))
            record.bm25_calls.append({
                "query": query, "k1": getattr(retriever, "k1", None),
                "b": getattr(retriever, "b", None), "top_k": retrieval_top_k,
                "mode": "merge", "doc_ids": doc_ids,
            })
            # Reason-in-Documents condensation (official; separate LLM call).
            t_llm = time.perf_counter()
            condensed, cerr = _reason_in_documents(
                client, model=model, max_tokens=max_tokens,
                prompt=_reasonchain_instruction(
                    _truncate_previous_reasoning(generated_total), query, _format_docs(hits)))
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
