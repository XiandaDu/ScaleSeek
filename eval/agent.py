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
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from .bm25_retriever import BM25Retriever


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are ScaleSeek, a research agent that answers questions by adaptively \
searching a large Wikipedia corpus.

You work in two stages:
1. BM25 RETRIEVAL — fetch a small workspace of relevant passages from the corpus.
2. WORKSPACE SEARCH — find specific details by searching within that workspace.

## Tools

### bm25_retrieve
Retrieve passages from the Wikipedia corpus into your workspace.
Parameters:
  query  (str)   — search query
  top_k  (int)   — number of passages to retrieve, 1-50 (default 5)
  k1     (float) — BM25 term-frequency saturation, 0.5-3.0 (default 1.5)
  b      (float) — BM25 length normalization, 0.0-1.0 (default 0.75)
  mode   (str)   — "replace" to start a new workspace, "merge" to add to existing

### grep_workspace
Search for a regex pattern within all passages currently in your workspace.
Parameters:
  pattern           (str)  — regex or literal string to search for
  case_insensitive  (bool) — default true

### read_doc
Read the full text of a specific passage by its doc_id.
Parameters:
  doc_id  (str) — the doc_id from a bm25_retrieve or grep_workspace result

## Output format

For every turn, write 1-3 sentences of reasoning (what you know, what is missing, \
what you will do), then exactly ONE of:

<tool_call>
{"name": "bm25_retrieve", "arguments": {"query": "...", "top_k": 5, "k1": 1.5, "b": 0.75, "mode": "replace"}}
</tool_call>

<tool_call>
{"name": "grep_workspace", "arguments": {"pattern": "...", "case_insensitive": true}}
</tool_call>

<tool_call>
{"name": "read_doc", "arguments": {"doc_id": "..."}}
</tool_call>

<answer>
concise answer (noun phrase, name, date, yes/no)
</answer>

Rules:
- Start with bm25_retrieve to build your workspace before searching.
- Use grep_workspace to find exact facts inside retrieved passages.
- Use read_doc to read a full passage when grep results are truncated.
- Re-issue bm25_retrieve with a different query (mode=merge) if the workspace \
  is missing key information.
- Answer when you have enough information. Keep answers concise.\
"""


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

MAX_TEXT_CHARS = 2000   # per passage in tool response


def _fmt_doc(doc: dict, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = doc.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + " [...]"
    return f'[doc_id={doc["doc_id"]} score={doc["score"]:.2f}]\n{text}'


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
) -> dict:
    """Run a tool and return a structured result dict."""
    if name == "bm25_retrieve":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"error": "bm25_retrieve requires a non-empty query"}
        top_k = int(arguments.get("top_k", 5))
        top_k = max(1, min(top_k, 50))
        k1 = float(arguments.get("k1", 1.5))
        b = float(arguments.get("b", 0.75))
        mode = str(arguments.get("mode", "replace")).lower()

        hits = retriever.retrieve(query, top_k=top_k, k1=k1, b=b)
        if mode == "merge":
            workspace.merge(hits)
        else:
            workspace.replace(hits)

        return {
            "tool": "bm25_retrieve",
            "query": query,
            "top_k": top_k,
            "mode": mode,
            "new_hits": len(hits),
            "workspace_size": workspace.size,
            "results": [_fmt_doc(h) for h in hits],
        }

    elif name == "grep_workspace":
        if workspace.size == 0:
            return {"error": "workspace is empty; call bm25_retrieve first"}
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return {"error": "grep_workspace requires a non-empty pattern"}
        ci = bool(arguments.get("case_insensitive", True))
        matches = workspace.grep(pattern, case_insensitive=ci)
        return {
            "tool": "grep_workspace",
            "pattern": pattern,
            "workspace_size": workspace.size,
            "matched": len(matches),
            "results": [_fmt_doc(m) for m in matches],
        }

    elif name == "read_doc":
        doc_id = str(arguments.get("doc_id", "")).strip()
        doc = workspace.get_doc(doc_id)
        if doc is None:
            ids = [d["doc_id"] for d in workspace.docs[:5]]
            return {
                "error": f"doc_id {doc_id!r} not in workspace",
                "available_ids_sample": ids,
            }
        full_text = doc["text"][:8000]
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
) -> tuple[Optional[str], Optional[str]]:
    """Return (text, error). Handles vLLM reasoning_content reconstruction."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
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


def parse_assistant(text: str) -> ParseResult:
    if not text:
        return ParseResult(raw="", error="empty response")

    tool_m = _TOOL_CALL_RE.search(text)
    ans_m = _ANSWER_RE.search(text)

    if tool_m and ans_m:
        first = "tool" if tool_m.start() < ans_m.start() else "answer"
    elif tool_m:
        first = "tool"
    elif ans_m:
        first = "answer"
    else:
        return ParseResult(raw=text, error="no <tool_call> or <answer> block found")

    if first == "answer":
        return ParseResult(raw=text, answer=ans_m.group(1).strip())

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
        {"role": "system", "content": SYSTEM_PROMPT},
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
        result = execute_tool(parsed.tool_name, parsed.tool_args, workspace, retriever)
        record.tool_time_s += time.perf_counter() - t_tool

        record.n_tool_calls += 1
        if parsed.tool_name == "bm25_retrieve":
            record.n_bm25_calls += 1

        payload = json.dumps(result, ensure_ascii=False)
        messages.append({"role": "tool", "content": payload})
        record.turns.append({"role": "tool", "content": payload})

    else:
        record.finish_reason = "max_turns"

    record.final_workspace_size = workspace.size
    record.messages = messages
    record.total_time_s = time.perf_counter() - t_start
    return record


# ---------------------------------------------------------------------------
# Direct-answer baseline (no retrieval)
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = """\
You are a knowledgeable assistant. Answer the question as concisely as possible \
(noun phrase, name, date, or yes/no). Output only:

<answer>
your answer here
</answer>
"""


def run_direct(
    example: dict,
    *,
    client: Any,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> AgentRecord:
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]
    text, err = _chat_completion(
        client, model=model, messages=messages,
        temperature=temperature, top_p=1.0, max_tokens=max_tokens,
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
    record.n_turns = 1
    return record


# ---------------------------------------------------------------------------
# BM25-RAG baseline (single retrieve then answer)
# ---------------------------------------------------------------------------

BM25_RAG_SYSTEM = """\
You are a knowledgeable assistant. You will be given a question and several \
Wikipedia passages. Use the passages to answer the question as concisely as \
possible. Output only:

<answer>
your answer here
</answer>
"""


def run_bm25_rag(
    example: dict,
    *,
    client: Any,
    model: str,
    retriever: BM25Retriever,
    top_k: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
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
    record.n_bm25_calls = 1
    record.final_workspace_size = len(hits)

    passages_text = "\n\n".join(
        f"[Passage {i+1}]\n{h['text'][:1500]}" for i, h in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": BM25_RAG_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages_text}"},
    ]

    t_llm = time.perf_counter()
    text, err = _chat_completion(
        client, model=model, messages=messages,
        temperature=temperature, top_p=1.0, max_tokens=max_tokens,
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
    record.n_turns = 1
    return record
