"""The ScaleSeek agent: prompt, tools, tool-response rendering and answer rules.

Shared, byte for byte, by RL training (train/, verl) and evaluation (methods/scaleseek.py),
so it imports only the standard library and scaleseek.metrics.

Two tools over the same Wikipedia corpus (Qwen3.5 native tool calling):
  find    exact phrase search (retrieval/exact.py): every hit contains the phrase verbatim,
          case/accents ignored; ranked, title matches first; zero-hit calls report the
          count of each phrase alone, so an empty AND still says which phrase is rare.
  search  dense semantic search (e5 / Qwen3-Embedding-4B).
Every tool call carries `current_answer`; if the budget runs out before an <answer>, the
last non-"unknown" current_answer is the prediction (anytime answering).
"""

from __future__ import annotations

import re

from .metrics import exact_match

MAX_TOOL_CALLS = 5          # + 1 answering turn = 6 assistant turns (GrepSeek's budget)
MAX_TURN_TOKENS = 2048      # generation cap per assistant turn (thinking + call)
CONTEXT_TOKENS = 16384      # prompt + whole trajectory (GrepSeek's budget)
FIND_TOP_K = 5
SEARCH_TOP_K = 5
UNKNOWN = "unknown"

SYSTEM_PROMPT = """You answer questions by searching a Wikipedia corpus (December 2018 dump, about 21 million passages of about 100 words each).

You have two tools:
- find: exact search. Every passage it returns contains your phrase word for word (case, accents and punctuation are ignored). Use it for names, titles, dates and other distinctive wording. Passages whose title contains the phrase come first. With two phrases, both must appear; if no passage has both, you get the count for each phrase alone.
- search: semantic search. Use it when you do not know the exact wording; it returns the passages closest in meaning to your query.

Every tool call also takes current_answer: your best answer so far given everything you have read (a short answer), or "unknown". If you run out of tool calls, your last current_answer becomes your final answer.

You can make at most 5 tool calls. As soon as you know the answer, reply with <answer>your answer</answer> instead of calling a tool.

The answer must be as short as possible: a single name, entity, number, date or phrase, with no explanation. Give a date at the precision the question asks for. Answer yes/no questions with yes or no."""

TOOLS = [
    {"type": "function", "function": {
        "name": "find",
        "description": "Exact phrase search over the corpus. Returns the number of passages containing the phrase(s) and the top 5, passages whose title contains the phrase first.",
        "parameters": {"type": "object", "properties": {
            "phrase": {"type": "string", "description": "Words that must appear exactly, in this order."},
            "phrase2": {"type": "string", "description": "Optional second phrase that must also appear in the same passage."},
            "hint": {"type": "string", "description": "Optional extra words that only rank the results."},
            "current_answer": {"type": "string", "description": 'Your best answer so far, or "unknown".'}},
            "required": ["phrase", "current_answer"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Semantic search over the corpus. Returns the 5 passages closest in meaning to the query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What you are looking for, in natural language."},
            "current_answer": {"type": "string", "description": 'Your best answer so far, or "unknown".'}},
            "required": ["query", "current_answer"]}}},
]


def user_message(question: str) -> str:
    return f"Question: {question.strip()}"


# ----------------------------------------------------------------------------- tool calls
class ToolCallError(ValueError):
    """A malformed call; its message is shown to the agent as the tool response."""


def find_query(args: dict) -> str:
    """The exact.py query string: quoted phrases, then hint words."""
    phrases = [str(args.get(k) or "").replace('"', " ").strip() for k in ("phrase", "phrase2")]
    phrases = [p for p in phrases if p]
    if not phrases:
        raise ToolCallError("find needs a non-empty phrase")
    return " ".join(f'"{p}"' for p in phrases) + " " + str(args.get("hint") or "").replace('"', " ")


def search_query(args: dict) -> str:
    q = str(args.get("query") or "").strip()
    if not q:
        raise ToolCallError("search needs a non-empty query")
    return q


def _passages(hits: list[dict]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        title, _, body = h["contents"].partition("\n")
        out.append(f"[{i}] {title.strip().strip('\"')}\n{body.strip()}")
    return "\n\n".join(out)


def render_find(args: dict, res: dict) -> str:
    phrases = [p for p in (args.get("phrase"), args.get("phrase2")) if p and str(p).strip()]
    quoted = " and ".join(f'"{str(p).strip()}"' for p in phrases)
    if res["count"] == 0:
        if res["phrase_counts"]:
            alone = "; ".join(f'"{p}" alone: {n:,}' for p, n in zip(phrases, res["phrase_counts"]))
            return f"No passage contains both {quoted}. {alone} passages."
        return f"No passage contains {quoted}."
    shown = len(res["hits"])
    return f"{res['count']:,} passages contain {quoted} (showing {shown}, title matches first).\n\n" + _passages(res["hits"])


def render_search(args: dict, hits: list[dict]) -> str:
    return f"Top {len(hits)} passages for \"{str(args.get('query')).strip()}\":\n\n" + _passages(hits)


# ----------------------------------------------------------------------------- answers
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """The last <answer> outside the thinking block."""
    text = text.split("</think>")[-1]
    found = _ANSWER.findall(text)
    return found[-1].strip() if found else None


def draft(calls_args: list[dict]) -> str | None:
    """The last informative current_answer of the trajectory's tool calls."""
    for args in reversed(calls_args):
        a = str(args.get("current_answer") or "").strip()
        if a and a.lower() != UNKNOWN:
            return a
    return None


def reward(answer: str | None, last_draft: str | None, golds: list[str], valid: bool,
           draft_credit: float) -> float:
    """Format-gated EM; a prediction that is only a draft earns `draft_credit` of it, so
    answering explicitly when sure always pays more than running out the budget."""
    if not valid:
        return 0.0
    if answer is not None:
        return float(exact_match(answer, golds))
    return draft_credit * float(exact_match(last_draft, golds)) if last_draft else 0.0
