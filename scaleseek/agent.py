"""The ScaleSeek agent: prompt, tools, tool-response rendering and answer rules.

Shared, byte for byte, by RL training (train/, verl) and evaluation (methods/scaleseek.py),
so it imports only the standard library and scaleseek.metrics.

Three tools (Qwen3.5 native tool calling); every turn is tool calls only, at most
MAX_PARALLEL_CALLS of find/search, or a lone `answer`:
  find    exact phrase search (retrieval/exact.py): every hit contains the phrase verbatim,
          case/accents ignored; ranked, title matches first; zero-hit calls report the
          count of each phrase alone, so an empty AND still says which phrase is rare.
  search  dense semantic search (e5 / Qwen3-Embedding-4B).
  answer  ends the episode with the final answer (never executed).  A tool, not an
          <answer> tag: the untrained model reaches for a tool to answer, and under the
          shared presence_penalty it could not close a tag it had just opened.
find/search carry `current_answer`; if the budget runs out before `answer`, the last
non-"unknown" current_answer is the prediction (anytime answering).
"""

from __future__ import annotations

import json
import re

from .metrics import exact_match

SEARCH_TURNS = 5            # turns with find/search calls; + 1 answering turn = 6 assistant turns (GrepSeek's budget)
MAX_PARALLEL_CALLS = 2      # find/search calls run per turn (both sides of a comparison at once)
MAX_TURN_TOKENS = 2048      # generation cap per assistant turn (thinking + call)
CONTEXT_TOKENS = 16384      # prompt + whole trajectory (GrepSeek's budget)
FIND_TOP_K = 5
SEARCH_TOP_K = 5
UNKNOWN = "unknown"

SYSTEM_PROMPT = """You answer questions by searching a Wikipedia corpus (December 2018 dump, about 21 million passages of about 100 words each).

You have three tools:
- find: exact search. Every passage it returns contains your phrase word for word (case, accents and punctuation are ignored). Keep the phrase short, usually 1 to 4 words that you expect verbatim in the text, such as a name, a title or a date, never a question or a description. Passages whose title contains the phrase come first. With two phrases, both must appear; if no passage has both, you get the count for each phrase alone.
- search: semantic search. Use it when you do not know the exact wording; it returns the passages closest in meaning to your query.
- answer: give your final answer and stop.

find and search also take current_answer: your best answer so far given everything you have read (a short answer), or "unknown". If you run out of searches, your last current_answer becomes your final answer.

Each turn, call find or search once, or twice at the same time when you need two independent lookups (for example both sides of a comparison); you have at most 5 such turns. As soon as you know the answer, call answer, alone in its turn.

The answer must be as short as possible: a single name, entity, number, date or phrase, with no explanation. Give a date at the precision the question asks for. Answer yes/no questions with yes or no."""

TOOLS = [
    {"type": "function", "function": {
        "name": "find",
        "description": "Exact phrase search over the corpus. Returns the number of passages containing the phrase(s) and the top 5, passages whose title contains the phrase first.",
        "parameters": {"type": "object", "properties": {
            "phrase": {"type": "string", "description": "A short phrase (usually 1-4 words, e.g. a name or a title) that must appear exactly, in this order."},
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
    {"type": "function", "function": {
        "name": "answer",
        "description": "Give the final answer and stop.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "The answer, as short as possible: a name, entity, number, date or yes/no."}},
            "required": ["answer"]}}},
]
ANSWER_TOOL = "answer"


def user_message(question: str) -> str:
    return f"Question: {question.strip()}"


# ----------------------------------------------------------------------------- tool calls
class ToolCallError(ValueError):
    """A malformed call; its message is shown to the agent as the tool response."""


def call_error(name: str) -> str | None:
    """Why a call in a find/search turn cannot run (None: it can)."""
    if name == ANSWER_TOOL:
        return "answer must be called alone in its turn"
    if name not in ("find", "search"):
        return f"unknown tool {name!r}"
    return None


_CALL = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
_FUNC = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_PARAM = re.compile(r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Qwen3.5 XML tool calls -> [(name, {param: str})]: verl's qwen3_coder parser
    (verl/experimental/agent_loop/tool_parser.py) for our all-string parameters, so
    evaluation reads a turn exactly as training did."""
    if "<tool_call>" not in text:
        return []
    out = []
    for call in [a or b for a, b in _CALL.findall(text)]:
        for fn in [a or b for a, b in _FUNC.findall(call)]:
            name, _, params = fn.partition(">")
            args = {}
            for p in [a or b for a, b in _PARAM.findall(params)]:
                key, _, value = p.partition(">")
                value = value[1:] if value.startswith("\n") else value
                value = value[:-1] if value.endswith("\n") else value
                args[key] = None if value.lower() == "null" else value
            out.append((name, args))
    return out


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
        title = title.strip().strip('"')
        out.append(f"[{i}] {title}\n{body.strip()}")
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


# ----------------------------------------------------------------------------- protocol variants
# Under test in zero-training pilots (train/teacher_rollouts.py --variants); none of them is
# part of the v1 protocol above, whose prompts stay frozen.
#   wide    TURN_PASSAGES passages per turn, split between the turn's calls (GrepSeek shows
#           up to 2,048 tool-output tokens per turn; 10 passages are ~1,600)
#   relax   a find with no exact match falls back to passages containing all its words
#   budget  each tool turn ends with the number of search turns left; the last asks for answer
VARIANTS = {"wide", "relax", "budget"}
TURN_PASSAGES = 10


def top_k(variants: set[str], n_calls: int) -> int:
    return TURN_PASSAGES // n_calls if "wide" in variants else SEARCH_TOP_K


def tools(variants: set[str]) -> list[dict]:
    if "wide" not in variants:
        return TOOLS
    share = f"{TURN_PASSAGES} ({TURN_PASSAGES // 2} each when two calls share a turn)"
    return json.loads(json.dumps(TOOLS).replace("the top 5", f"the top {share}")
                      .replace("the 5 passages", f"the {share} passages"))


def render_relaxed(args: dict, res: dict) -> str:
    """render_find, plus the fallback hits when there was no exact match."""
    text = render_find(args, res)
    if res["count"] or not res["relaxed_count"]:
        return text
    return (f"{text} {res['relaxed_count']:,} passages contain all of its words in any order "
            f"(showing {len(res['hits'])}, title matches first).\n\n" + _passages(res["hits"]))


def budget_note(turns_left: int) -> str:
    if turns_left:
        return f"\n\n(You have {turns_left} search turn{'s' if turns_left > 1 else ''} left.)"
    return "\n\n(You have used all your search turns. Call answer now with your best answer.)"


# ----------------------------------------------------------------------------- answers
def final_answer(turn_text: str) -> str | None:
    """The answer of a turn whose (first) call is `answer`, else None."""
    calls = parse_tool_calls(turn_text)
    if calls and calls[0][0] == ANSWER_TOOL:
        return str(calls[0][1].get("answer") or "").strip() or None
    return None


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
