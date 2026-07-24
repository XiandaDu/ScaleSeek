"""Prompt-based ScaleSeek agent (temporary; deprecated once RL training lands).

Architecture mirrors GrepSeek's agent loop but adds a BM25 retrieval tool that
constructs a bounded workspace before DCI-style search.

Tools available to the agent:
    bm25_retrieve(query, top_k, k1, b, mode)   — fetch passages into workspace
    grep_workspace(pattern, case_insensitive)   — regex search within workspace
    read_doc(doc_id)                            — read a specific workspace doc

Workspace = list of passages retrieved from the BM25 index. The agent controls:
    - when to retrieve (BM25 vs. already having enough context)
    - query string, top_k (1–50), k1 (0.5–3.0), b (0.0–1.0)
    - merge (add new docs to workspace) vs. replace (start fresh)
    - when to stop and answer

The agent loop is plain ChatML, matching the format the RL agent will be
trained on later.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from .bm25_retriever import BM25Retriever
from . import prompts


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# Default token budget for a single tool response (the rendered bm25_retrieve /
# grep_workspace passage list, or one read_doc). Passages are atomic ~100-word DPR
# chunks (mean ~170 tokens, max ~1.1k), so we do NOT truncate individual passages.
# Instead we bound the *whole* response by token count and drop trailing
# (lower-ranked) passages — they stay in the workspace, reachable via
# grep_workspace / read_doc. This mirrors GrepSeek's tokenizer-based stdout cap so
# eval and RL training truncate tool outputs identically.
DEFAULT_MAX_RESPONSE_TOKENS = 2048


def _fmt_doc(doc: dict) -> str:
    """Render one passage whole — no per-passage truncation (see _budget_docs)."""
    return f'[doc_id={doc["doc_id"]} score={doc["score"]:.2f}]\n{doc.get("text", "")}'


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Token count via the model tokenizer; falls back to a ~4 char/token estimate."""
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_tokens(text: str, *, tokenizer: Any, max_tokens: Optional[int]) -> str:
    """Truncate a single string to a token budget (used by read_doc)."""
    if not max_tokens:
        return text
    if tokenizer is None:
        return text if len(text) <= max_tokens * 4 else text[: max_tokens * 4] + " [...]"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens]) + " [...]"


def _budget_docs(
    docs: list[dict], *, tokenizer: Any, max_tokens: Optional[int]
) -> tuple[list[str], int]:
    """Render passages whole, in rank order, until the token budget is reached.

    Returns (rendered_strings, n_total). Always includes at least the first passage
    so the response is never empty; passages that don't fit are dropped (they remain
    in the workspace and are reachable via grep_workspace / read_doc).
    """
    rendered: list[str] = []
    used = 0
    for d in docs:
        s = _fmt_doc(d)
        n = _count_tokens(s, tokenizer)
        if max_tokens and rendered and used + n > max_tokens:
            break
        rendered.append(s)
        used += n
    return rendered, len(docs)


@dataclass
class Workspace:
    docs: list[dict] = field(default_factory=list)  # {doc_id, score, text}

    def merge(self, hits: list[dict]) -> None:
        existing_ids = {d["doc_id"] for d in self.docs}
        for h in hits:
            if h["doc_id"] not in existing_ids:
                self.docs.append(h)
                existing_ids.add(h["doc_id"])

    def replace(self, hits: list[dict]) -> None:
        self.docs = list(hits)

    def grep(self, pattern: str, case_insensitive: bool = True,
             max_results: int = 20) -> list[dict]:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), flags)
        results = []
        for doc in self.docs:
            if rx.search(doc["text"]):
                results.append(doc)
                if len(results) >= max_results:
                    break
        return results

    def get_doc(self, doc_id: str) -> Optional[dict]:
        for d in self.docs:
            if d["doc_id"] == doc_id:
                return d
        return None

    @property
    def size(self) -> int:
        return len(self.docs)


def execute_tool(
    name: str,
    arguments: dict,
    workspace: Workspace,
    retriever: BM25Retriever,
    *,
    tokenizer: Any = None,
    max_response_tokens: Optional[int] = DEFAULT_MAX_RESPONSE_TOKENS,
) -> dict:
    """Run a tool and return a structured result dict.

    Tool responses are bounded to `max_response_tokens` tokens (counted with
    `tokenizer`, or a ~4 char/token estimate when it is None) by dropping trailing
    passages — never by cutting a passage mid-text. Pass max_response_tokens=None
    to disable the budget.
    """
    if name == "bm25_retrieve":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"error": "bm25_retrieve requires a non-empty query"}
        top_k = int(arguments.get("top_k", 3))
        top_k = max(1, min(top_k, 50))
        k1 = float(arguments.get("k1", 1.2))
        b = float(arguments.get("b", 0.75))
        mode = str(arguments.get("mode", "replace")).lower()

        hits = retriever.retrieve(query, top_k=top_k, k1=k1, b=b)
        if mode == "merge":
            workspace.merge(hits)
        else:
            workspace.replace(hits)

        rendered, n_total = _budget_docs(hits, tokenizer=tokenizer, max_tokens=max_response_tokens)
        result = {
            "tool": "bm25_retrieve",
            "query": query,
            "top_k": top_k,
            "k1": k1,
            "b": b,
            "mode": mode,
            "new_hits": len(hits),
            "shown": len(rendered),
            "workspace_size": workspace.size,
            "doc_ids": [h["doc_id"] for h in hits],
            "results": rendered,
        }
        if len(rendered) < n_total:
            result["note"] = (
                f"{n_total - len(rendered)} of {n_total} retrieved passages omitted to fit "
                "the response budget; they are in the workspace — narrow with grep_workspace "
                "or open one with read_doc."
            )
        return result

    elif name == "grep_workspace":
        if workspace.size == 0:
            return {"error": "workspace is empty; call bm25_retrieve first"}
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return {"error": "grep_workspace requires a non-empty pattern"}
        ci = bool(arguments.get("case_insensitive", True))
        matches = workspace.grep(pattern, case_insensitive=ci)
        rendered, n_total = _budget_docs(matches, tokenizer=tokenizer, max_tokens=max_response_tokens)
        result = {
            "tool": "grep_workspace",
            "pattern": pattern,
            "workspace_size": workspace.size,
            "matched": len(matches),
            "shown": len(rendered),
            "results": rendered,
        }
        if len(rendered) < n_total:
            result["note"] = (
                f"{n_total - len(rendered)} of {n_total} matches omitted to fit the response "
                "budget; refine the pattern or open a specific doc_id with read_doc."
            )
        return result

    elif name == "read_doc":
        doc_id = str(arguments.get("doc_id", "")).strip()
        doc = workspace.get_doc(doc_id)
        if doc is None:
            ids = [d["doc_id"] for d in workspace.docs[:5]]
            return {
                "error": f"doc_id {doc_id!r} not in workspace",
                "available_ids_sample": ids,
            }
        full_text = _truncate_to_tokens(
            doc["text"], tokenizer=tokenizer, max_tokens=max_response_tokens
        )
        return {"tool": "read_doc", "doc_id": doc_id, "text": full_text}

    else:
        return {"error": f"unknown tool: {name!r}. Use bm25_retrieve, grep_workspace, or read_doc"}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    stop: Optional[list] = None,
    extra_body: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return (text, error). Handles vLLM reasoning_content reconstruction."""
    _extra = {"stop": stop} if stop else {}
    if extra_body:
        _extra["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **_extra,
        )
    except Exception as e:
        # context-length overflow retry
        m = re.search(
            r"passed\s+(\d+)\s+input tokens.*?context length is only\s+(\d+)",
            str(e), re.DOTALL,
        )
        if m:
            n_in, ctx = int(m.group(1)), int(m.group(2))
            room = ctx - n_in - 8
            if room <= 0:
                return None, f"context overflow: {n_in} input tokens >= {ctx}"
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, top_p=top_p, max_tokens=room,
                    **_extra,
                )
            except Exception as e2:
                return None, f"API error: {e2}"
        else:
            return None, f"API error: {e}"

    if not resp.choices:
        return None, "API returned no choices"
    choice = resp.choices[0]
    content = choice.message.content or ""
    reasoning = (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None) or "")
    if reasoning.strip():
        text = f"<think>\n{reasoning.strip()}\n</think>\n{content}"
    else:
        text = content
        # No reasoning parser on the server: the chat template injects the opening
        # "<think>" into the prompt, so vLLM's content carries the reasoning and its
        # "</think>" but NOT the opening tag. Restore it so recorded/fed-back
        # assistant turns are well-formed <think>...</think> — a trained model
        # (GrepSeek) degrades when prior turns don't match its training format.
        if text and "</think>" in text and "<think>" not in text:
            text = "<think>\n" + text
    return text, None


# ---------------------------------------------------------------------------
# Parse assistant turn
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    raw: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    answer: Optional[str] = None
    error: Optional[str] = None


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_answer(s: str) -> str:
    """Strip reasoning leakage from an extracted answer.

    Reasoning models sometimes emit <think>...</think> or stray answer/think tags
    inside what our regex captures as the answer (observed on DCI: predictions like
    'tags.\\n</think>\\n<answer>\\n...'). Remove think blocks, keep only text after
    the last </think>, and drop any leftover tag tokens.
    """
    if not s:
        return s
    s = _THINK_BLOCK_RE.sub("", s)
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        s = s.replace(tag, "")
    return s.strip()


def parse_assistant(text: str) -> ParseResult:
    if not text:
        return ParseResult(raw="", error="empty response")

    # Only the text outside <think> blocks is actionable. Thinking models
    # quote the format instruction verbatim (a literal empty
    # "<answer></answer>") and rehearse candidate blocks while reasoning, so
    # the first tag match in the full text is usually not the commitment
    # (2026-07-24: budget-forced direct emitted a valid <answer> in content
    # on 16/16 probes, yet the instruction quote inside thinking matched
    # first and every prediction came out empty).
    visible = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in visible:
        visible = visible.rsplit("</think>", 1)[-1]

    tool_m = _TOOL_CALL_RE.search(visible)
    ans_m = _ANSWER_RE.search(visible)

    if tool_m and ans_m:
        first = "tool" if tool_m.start() < ans_m.start() else "answer"
    elif tool_m:
        first = "tool"
    elif ans_m:
        first = "answer"
    else:
        return ParseResult(raw=text, error="no <tool_call> or <answer> block found")

    if first == "answer":
        return ParseResult(raw=text, answer=clean_answer(ans_m.group(1)))

    tc_text = tool_m.group(1).strip()
    try:
        obj = json.loads(tc_text)
    except Exception as e:
        return ParseResult(raw=text, error=f"tool_call JSON parse error: {e}")

    name = obj.get("name")
    if not name:
        return ParseResult(raw=text, error="tool_call missing 'name'")
    args = obj.get("arguments") or {}
    if not isinstance(args, dict):
        return ParseResult(raw=text, error="tool_call 'arguments' must be a dict")

    return ParseResult(raw=text, tool_name=name, tool_args=args)


# ---------------------------------------------------------------------------
# Agent record
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    id: str
    question: str
    gold_answers: list

    prediction: Optional[str] = None
    finish_reason: str = ""
    n_turns: int = 0
    n_tool_calls: int = 0
    n_bm25_calls: int = 0
    final_workspace_size: int = 0
    error: Optional[str] = None

    llm_time_s: float = 0.0
    tool_time_s: float = 0.0
    total_time_s: float = 0.0

    turns: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    # Structured retrieval trace (for workspace-level Gold/Qrel R@W and future
    # training data). bm25_calls: one entry per retrieval action with the query,
    # BM25 params, and the doc-ids it returned. workspace_doc_ids: the final
    # materialized workspace W_T (deduped, in insertion order).
    bm25_calls: list = field(default_factory=list)
    workspace_doc_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answers": self.gold_answers,
            "prediction": self.prediction,
            "finish_reason": self.finish_reason,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "n_bm25_calls": self.n_bm25_calls,
            "final_workspace_size": self.final_workspace_size,
            "error": self.error,
            "llm_time_s": self.llm_time_s,
            "tool_time_s": self.tool_time_s,
            "total_time_s": self.total_time_s,
            "bm25_calls": self.bm25_calls,
            "workspace_doc_ids": self.workspace_doc_ids,
            "turns": self.turns,
        }


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    max_turns: int = 8,
    max_tokens: Optional[int] = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tokenizer: Any = None,
    max_tool_response_tokens: Optional[int] = DEFAULT_MAX_RESPONSE_TOKENS,
) -> AgentRecord:
    """Run ScaleSeek prompt agent on one example.

    Conversation format:
        system: SYSTEM_PROMPT
        user:   "Question: {question}"
        assistant <-> tool turns ...
        assistant: <answer>...</answer>
    """
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))

    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    workspace = Workspace()

    t_start = time.perf_counter()
    messages: list = [
        # SCALESEEK_PROMPT selects an alternate registered prompt for ablations
        # (e.g. scaleseek_prompt_withparams — adds numeric parameter hints; the
        # default scaleseek_prompt explains the knobs without anchoring values).
        {"role": "system", "content": prompts.load(
            os.environ.get("SCALESEEK_PROMPT", "scaleseek_prompt"))},
        {"role": "user", "content": f"Question: {question}"},
    ]

    consecutive_parse_errors = 0

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
        parsed = parse_assistant(text)

        record.turns.append({
            "role": "assistant", "content": text,
            "parse": {
                "tool_name": parsed.tool_name,
                "answer": parsed.answer,
                "error": parsed.error,
            },
        })

        if parsed.answer is not None:
            record.prediction = parsed.answer
            record.finish_reason = "answer"
            break

        if parsed.error:
            consecutive_parse_errors += 1
            if consecutive_parse_errors >= 2:
                record.finish_reason = "parse_error"
                record.error = parsed.error
                break
            feedback = json.dumps({"error": f"format error: {parsed.error}"})
            messages.append({"role": "tool", "content": feedback})
            record.turns.append({"role": "tool", "content": feedback, "synthetic": True})
            continue

        consecutive_parse_errors = 0

        t_tool = time.perf_counter()
        result = execute_tool(
            parsed.tool_name, parsed.tool_args, workspace, retriever,
            tokenizer=tokenizer, max_response_tokens=max_tool_response_tokens,
        )
        record.tool_time_s += time.perf_counter() - t_tool

        record.n_tool_calls += 1
        if parsed.tool_name == "bm25_retrieve":
            record.n_bm25_calls += 1
            record.bm25_calls.append({
                "query": result.get("query"),
                "k1": result.get("k1"),
                "b": result.get("b"),
                "top_k": result.get("top_k"),
                "mode": result.get("mode"),
                "doc_ids": result.get("doc_ids", []),
            })

        payload = json.dumps(result, ensure_ascii=False)
        messages.append({"role": "tool", "content": payload})
        record.turns.append({"role": "tool", "content": payload})

    else:
        record.finish_reason = "max_turns"

    record.final_workspace_size = workspace.size
    record.workspace_doc_ids = [d["doc_id"] for d in workspace.docs]
    record.messages = messages
    record.total_time_s = time.perf_counter() - t_start
    return record


# ---------------------------------------------------------------------------
# Direct-answer baseline (no retrieval)
# ---------------------------------------------------------------------------

def run_direct(
    example: dict,
    *,
    client: Any,
    model: str,
    max_tokens: Optional[int] = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    thinking_token_budget: Optional[int] = None,
) -> AgentRecord:
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()
    messages = [
        {"role": "system", "content": prompts.load("direct")},
        {"role": "user", "content": f"Question: {question}"},
    ]
    text, err = _chat_completion(
        client, model=model, messages=messages,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        extra_body=({"thinking_token_budget": thinking_token_budget}
                    if thinking_token_budget else None),
    )
    record.llm_time_s = time.perf_counter() - t_start
    record.total_time_s = record.llm_time_s
    if err:
        record.finish_reason = "api_error"
        record.error = err
        return record
    parsed = parse_assistant(text or "")
    record.prediction = parsed.answer
    record.finish_reason = "answer" if parsed.answer else "parse_error"
    if not (text or "").strip():
        # vLLM returns neither content nor reasoning_content when generation
        # hits max_tokens inside thinking; tag it so the error taxonomy can
        # tell truncation apart from a malformed answer.
        record.error = "empty completion (max_tokens likely exhausted inside thinking)"
    record.n_turns = 1
    return record


# ---------------------------------------------------------------------------
# BM25-RAG baseline (single retrieve then answer)
# ---------------------------------------------------------------------------

def run_rag(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    top_k: int = 3,
    max_tokens: Optional[int] = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    thinking_token_budget: Optional[int] = None,
) -> AgentRecord:
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    t_tool = time.perf_counter()
    hits = retriever.retrieve(question, top_k=top_k)
    record.tool_time_s = time.perf_counter() - t_tool
    record.n_tool_calls = 1
    record.n_bm25_calls = int(retriever.__class__.__name__ == "FixedBM25Retriever")
    record.final_workspace_size = len(hits)
    doc_ids = [h["doc_id"] for h in hits]
    record.workspace_doc_ids = doc_ids
    record.bm25_calls = [{
        "query": question, "k1": getattr(retriever, "k1", None),
        "b": getattr(retriever, "b", None), "top_k": top_k,
        "mode": "replace", "doc_ids": doc_ids,
    }]

    passages_text = "\n\n".join(
        f"[Passage {i+1}]\n{h['text']}" for i, h in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": prompts.load("rag")},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages_text}"},
    ]

    t_llm = time.perf_counter()
    text, err = _chat_completion(
        client, model=model, messages=messages,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        extra_body=({"thinking_token_budget": thinking_token_budget}
                    if thinking_token_budget else None),
    )
    record.llm_time_s = time.perf_counter() - t_llm
    record.total_time_s = time.perf_counter() - t_start

    if err:
        record.finish_reason = "api_error"
        record.error = err
        return record

    parsed = parse_assistant(text or "")
    record.prediction = parsed.answer
    record.finish_reason = "answer" if parsed.answer else "parse_error"
    if not (text or "").strip():
        record.error = "empty completion (max_tokens likely exhausted inside thinking)"
    record.n_turns = 1
    return record
