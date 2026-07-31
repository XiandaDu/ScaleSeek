"""GrepSeek-style cold-start trajectory generation for ScaleSeek SFT.

Given a QA example ``{id, question, golden_answers}`` and a *teacher* model, this
module synthesizes a verified multi-turn agent trajectory in the exact ScaleSeek
format (``<think>…</think>`` + one ``<tool_call>``/``<answer>`` per turn), grounded
in real BM25 retrievals against the (smoke or full) corpus.

Pipeline (uses the prompt constants in ``prompts/sft_prompts.py`` verbatim):

  Backward pass — Tutor role, knows the gold answer
    1. DECOMPOSE           question -> ordered sub-questions (last answer = gold)
    2. for hop i = n-1 … 0:
         BACKWARD_TOOL      propose a bm25_retrieve(+grep) trace for sub_q[i]
         (execute)          run it against the real retriever
         JUDGE              does the retrieval confirm expected[i]?  refine if not
         BRIDGE_EXTRACT     read the passage -> bridge entity = expected[i-1]
       (ANSWER-LEAK rule: neither a bm25 query nor a grep pattern may contain
        expected[i] — the student cannot form either at inference time.)

  Forward pass — Planner + Tutor, respects the information frontier
    3. for each verified tool call, in question order:
         PLANNER            draft reasoning from (question, history) only
         (execute)          run the *verified* target call -> real tool output
         TUTOR_EDIT         rewrite reasoning to lead to that call w/o leaking
    4. FINAL_ANSWER         synthesize; the emitted <answer> is pinned to gold
    5. QUALITY_JUDGE        drop trajectories that leak future facts / are unsupported

The result is a list of chat messages plus a per-message loss mask (assistant
turns only) that ``train/sft_dataset.py`` turns into an SFT example.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from eval.agent import Workspace, execute_tool
from eval.metrics import normalize
from prompts import sft_prompts as P
from train.sft import bm25_race as R
from prompts.scaleseek_prompt import PROMPT as SCALESEEK_SYSTEM

logger = logging.getLogger(__name__)

_VALID_TOOLS = {"bm25_retrieve", "grep_workspace", "read_doc"}

# BM25 grids for the search-then-teach mentor. TWO grids, because parameter
# sensitivity is a corpus-length regime (measured on wiki-18 + Pi-Serini's
# Table 3 on BCP — same mechanism, two ends):
#   short — uniform passage corpora (wiki-18: 102±5 words). High k1 is harmful
#           there (k1=12 dropped gold from rank 1 to 8-9), so the grid stays in
#           the classical range; probing 16/25 would be dead retrievals the
#           tie-break can never select.
#   long  — document corpora (BCP: median ~2k tokens, p90 ~14k). Pi-Serini's
#           grid-search optimum is near (k1=16, b=1.0), runs use (25, 1);
#           default-range grids cannot reach that region at all.
# The caller picks per corpus (see generate_sft_data --param-grid auto).
_K1_GRID = (0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5)
_B_GRID = (0.3, 0.5, 0.75, 0.9)
_K1_GRID_LONG = (0.9, 1.2, 2.0, 4.0, 8.0, 16.0, 25.0)
_B_GRID_LONG = (0.4, 0.75, 1.0)
_TOPK_LADDER = (3, 5, 10, 20)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ColdStartConfig:
    max_refine: int = 2               # backward-discovery retries per hop
    top_k_default: int = 5
    tool_response_tokens: int = 512   # per-response budget in forward execution
    # Default True: an unverified hop means retrieval never confirmed the expected
    # answer, yet the forward pass still pins <answer> to gold — i.e. the trajectory
    # asserts a fact its own workspace does not support, which is exactly the
    # hallucination behaviour SFT would teach. Set False only for pipeline smokes.
    strict: bool = True               # skip examples whose hops fail to verify
    # False: a grep pattern may not contain the expected answer either. The old
    # behaviour (grep may leak) put teacher-only knowledge into the student's
    # imitation target; see _call_leaks_answer.
    grep_may_leak_answer: bool = False
    run_quality_judge: bool = True
    teacher_max_tokens: int = 768
    preview_chars: int = 700
    # BM25 parameter policy for the forward trajectory:
    #   "heuristic" — assign k1/b/top_k/mode from query features (a rule).
    #   "search"    — grid-search BM25 params against the real index and teach the
    #                 setting that ranks the target passage best (empirically grounded).
    #   "teacher"   — keep exactly what the teacher emitted (often omitted -> runtime
    #                 defaults; no adaptation shown).
    #   "race"      — race a fixed (k1,b) grid (train/sft/bm25_race.py): every
    #                 config retrieves, the most promising arms each continue a
    #                 full trajectory, the best-scoring one is the SFT positive
    #                 and the rest become preference data. Also rescues backward
    #                 hops whose default-param retrieval missed the evidence.
    # heuristic/search/race annotate the turn's reasoning with the rationale.
    param_policy: str = "heuristic"
    # race mode only:
    race_width: int = 4          # forward arms incl. the default baseline arm
    race_judge_budget: int = 3   # max extra judge calls per backward rescue
    # search policy: (k1, b) grid, selected per corpus-length regime
    # (generate_sft_data --param-grid). Defaults = short-passage grid (wiki-18).
    k1_grid: tuple = _K1_GRID
    b_grid: tuple = _B_GRID
    # Replay each hop's real failed first attempt before the verified call, so
    # trajectories demonstrate observe-failure -> reformulate (-> replace when
    # the workspace holds only the junk). See _forward_pass.
    inject_failures: bool = True


# ---------------------------------------------------------------------------
# Extraction helpers (robust to markdown fences / stray prose)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _extract_json_array(text: str) -> Optional[list]:
    text = _strip_fences(text)
    i = text.find("[")
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    text = _strip_fences(text)
    i = text.find("{")
    if i == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _split_reasoning_and_toolcall(text: str) -> tuple[str, Optional[dict]]:
    """Planner output = free reasoning then one <tool_call> block."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        # some drafts wrap reasoning in <think>…</think>
        think = _extract_tag(text, "think")
        return (think or text).strip(), None
    reasoning = text[: m.start()].strip()
    think = _extract_tag(reasoning, "think")
    if think:
        reasoning = think
    tc = None
    try:
        obj = json.loads(m.group(1).strip())
        if isinstance(obj, dict) and obj.get("name"):
            tc = obj
    except json.JSONDecodeError:
        pass
    return reasoning, tc


# ---------------------------------------------------------------------------
# Tool-trace validation
# ---------------------------------------------------------------------------

def _normalize_tool_call(tc: dict, cfg: ColdStartConfig) -> Optional[dict]:
    if not isinstance(tc, dict):
        return None
    name = tc.get("name")
    args = tc.get("arguments") or {}
    if name not in _VALID_TOOLS or not isinstance(args, dict):
        return None
    if name == "bm25_retrieve":
        if not str(args.get("query", "")).strip():
            return None
        # No parameter bias: preserve exactly what the teacher chose (or omitted)
        # for top_k/k1/b/mode. execute_tool applies its own defaults at run time for
        # any omitted knob, so the recorded assistant turn carries no anchored value.
    elif name == "grep_workspace":
        if not str(args.get("pattern", "")).strip():
            return None
        args.setdefault("case_insensitive", True)
    elif name == "read_doc":
        if not str(args.get("doc_id", "")).strip():
            return None
    return {"name": name, "arguments": args}


def _query_leaks_answer(query: str, forbidden: list[str]) -> bool:
    q = normalize(query)
    for f in forbidden:
        nf = normalize(f)
        if nf and nf in q:
            return True
    return False


def _call_leaks_answer(tc: dict, forbidden: list[str], cfg: "ColdStartConfig") -> bool:
    """Would this tool call require knowing the answer the student is looking for?

    The backward tracer knows the gold, so it happily writes
    `grep_workspace(pattern="Tatsumi")`. That call is replayed verbatim into the
    forward trajectory, where it teaches the student to grep for a string it has no
    way to produce at inference — the same unattainable-behaviour problem as quoting
    the gold's retrieval rank. Excluding leaky greps costs some hop-verification
    yield (the teacher gets `max_refine` more attempts) and buys trajectories the
    student can actually reproduce.
    """
    if tc["name"] == "bm25_retrieve":
        return _query_leaks_answer(str(tc["arguments"].get("query", "")), forbidden)
    if tc["name"] == "grep_workspace" and not cfg.grep_may_leak_answer:
        return _query_leaks_answer(str(tc["arguments"].get("pattern", "")), forbidden)
    return False


def _workspace_supports(workspace, golds: list[str]) -> bool:
    """Does any passage currently in the workspace contain a gold answer string?

    Guards the gold-pinned <answer> turn: if nothing in the workspace supports it,
    the trajectory would demonstrate asserting an unretrieved fact.
    """
    norm_golds = [normalize(g) for g in golds if str(g).strip()]
    if not norm_golds:
        return False
    for doc in workspace.docs:
        text = normalize(doc.get("text", ""))
        if any(ng and ng in text for ng in norm_golds):
            return True
    return False


# ---------------------------------------------------------------------------
# Teacher-call wrappers (one per prompt-suite phase)
# ---------------------------------------------------------------------------

def _decompose(teacher, question: str, answer: str, cfg: ColdStartConfig) -> list[str]:
    out = teacher.complete(
        [{"role": "user", "content": P.DECOMPOSE_PROMPT.format(question=question, answer=answer)}],
        max_tokens=cfg.teacher_max_tokens, temperature=0.0, enable_thinking=False,
    )
    arr = _extract_json_array(out) or []
    subs = [str(x.get("sub_question", "")).strip() for x in arr if isinstance(x, dict)]
    subs = [s for s in subs if s]
    return subs or [question]


def _judge(teacher, sub_q: str, expected: str, forms: list[str], tool_output: str,
           cfg: ColdStartConfig) -> tuple[bool, str]:
    out = teacher.complete(
        [{"role": "user", "content": P.JUDGE_PROMPT.format(
            sub_question=sub_q, expected_answer=expected,
            acceptable_forms=", ".join(forms) or "(none)", tool_output=tool_output)}],
        max_tokens=200, temperature=0.0, enable_thinking=False,
    )
    obj = _extract_json_object(out) or {}
    verdict = str(obj.get("verdict", "")).strip().upper() == "YES"
    return verdict, str(obj.get("reasoning", ""))


def _bridge_extract(teacher, question: str, sub_prev: str, sub_next: str,
                    expected_next: str, doc_next: str, cfg: ColdStartConfig) -> dict:
    out = teacher.complete(
        [{"role": "user", "content": P.BRIDGE_EXTRACT_PROMPT.format(
            question=question, sub_q_prev=sub_prev, sub_q_next=sub_next,
            expected_next=expected_next, doc_next=doc_next)}],
        max_tokens=300, temperature=0.0, enable_thinking=False,
    )
    obj = _extract_json_object(out) or {}
    return {
        "bridge_entity": (obj.get("bridge_entity") or "").strip() if obj.get("bridge_entity") else None,
        "aliases": [str(a) for a in (obj.get("aliases") or [])],
    }


def _planner_draft(teacher, question: str, history: str, cfg: ColdStartConfig) -> tuple[str, Optional[dict]]:
    sys = P.PLANNER_SYSTEM.format(corpus_description=P.CORPUS_DESCRIPTION)
    out = teacher.complete(
        [{"role": "system", "content": sys},
         {"role": "user", "content": P.PLANNER_USER.format(question=question, history=history or "(no steps yet)")}],
        max_tokens=cfg.teacher_max_tokens, temperature=0.0, enable_thinking=False,
    )
    return _split_reasoning_and_toolcall(out)


def _tutor_edit(teacher, question: str, history: str, draft_think: str,
                draft_tc: Optional[dict], target_tc: dict, preview: str,
                cfg: ColdStartConfig) -> str:
    out = teacher.complete(
        [{"role": "user", "content": P.TUTOR_EDIT_PROMPT.format(
            question=question, history=history or "(no steps yet)",
            draft_think=draft_think or "(none)",
            draft_tool_call=json.dumps(draft_tc) if draft_tc else "(none)",
            target_tool_call=json.dumps(target_tc),
            target_output_preview=preview)}],
        max_tokens=cfg.teacher_max_tokens, temperature=0.0, enable_thinking=False,
    )
    edited = _extract_tag(out, "edited_reasoning")
    return (edited or draft_think or "I will search the corpus for the relevant passage.").strip()


def _final_reasoning(teacher, question: str, history: str, cfg: ColdStartConfig) -> str:
    sys = P.PLANNER_SYSTEM.format(corpus_description=P.CORPUS_DESCRIPTION)
    out = teacher.complete(
        [{"role": "system", "content": sys},
         {"role": "user", "content": P.FINAL_ANSWER_USER.format(question=question, history=history)}],
        max_tokens=400, temperature=0.0, enable_thinking=False,
    )
    # keep only the reasoning; the <answer> we emit is pinned to gold downstream
    think = _extract_tag(out, "think")
    if think:
        return think.strip()
    ans_m = re.search(r"<answer>", out)
    reasoning = out[: ans_m.start()] if ans_m else out
    return reasoning.strip() or "The retrieved passages contain the answer to the question."


def _quality_pass(teacher, question: str, trajectory_text: str) -> tuple[bool, str]:
    out = teacher.complete(
        [{"role": "user", "content": P.QUALITY_JUDGE_PROMPT.format(
            question=question, trajectory_text=trajectory_text)}],
        max_tokens=200, temperature=0.0, enable_thinking=False,
    )
    obj = _extract_json_object(out) or {}
    verdict = str(obj.get("verdict", "")).strip().upper() == "PASS"
    return verdict, str(obj.get("reasoning", ""))


# ---------------------------------------------------------------------------
# Trajectory result
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    id: str
    question: str
    golden_answers: list[str]
    messages: list[dict] = field(default_factory=list)  # {role, content}
    loss_mask: list[int] = field(default_factory=list)  # 1 on assistant turns
    status: str = "ok"           # ok | skipped | quality_fail | unsupported_answer
    meta: dict = field(default_factory=dict)
    # race mode: losing arms, kept for preference/failure data. Each entry is a
    # full {config, score, messages, loss_mask, meta} record. Never enters SFT —
    # load_ok_trajectories keys on status of the *main* record only.
    siblings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "question": self.question,
            "golden_answers": self.golden_answers,
            "messages": self.messages, "loss_mask": self.loss_mask,
            "status": self.status, "meta": self.meta,
        }
        if self.siblings:
            d["siblings"] = self.siblings
        return d


_STRUCT_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<answer>", "</answer>")


def _sanitize_reasoning(text: str) -> str:
    """Strip structural tags from a reasoning string before wrapping it in <think>.

    A teacher sometimes emits a premature <answer>…</answer> or a stray <tool_call>
    inside its reasoning. Left in place, it would nest an action tag inside <think>,
    breaking the one-action-per-turn contract and leaking the final answer into an
    intermediate step. We cut the reasoning at the first structural tag.
    """
    if not text:
        return ""
    cut = len(text)
    for tag in _STRUCT_TAGS:
        i = text.find(tag)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _fmt_tool_turn(reasoning: str, tool_call: dict) -> str:
    return (f"<think>\n{_sanitize_reasoning(reasoning)}\n</think>\n"
            f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>")


def _fmt_answer_turn(reasoning: str, answer: str) -> str:
    return f"<think>\n{_sanitize_reasoning(reasoning)}\n</think>\n<answer>\n{answer}\n</answer>"


def _history_append(history: str, assistant: str, tool_result: Optional[str]) -> str:
    block = f"ASSISTANT:\n{assistant}\n"
    if tool_result is not None:
        block += f"TOOL:\n{tool_result}\n"
    return history + block + "\n"


def _param_policy(query: str, bm25_idx: int) -> tuple[dict, str]:
    """Heuristic cold-start BM25 parameter policy: query features -> parameters.

    A sensible starting policy the student imitates; RL later optimizes it. It ties
    the knobs to a generalizable feature (query length / specificity) rather than a
    fixed constant:
      - long descriptive query -> lower k1, higher b, wider top_k (don't let long
        generic articles crowd out the specific passage)
      - short focused query    -> higher k1, lower b (let exact repeated mentions win)
      - medium                 -> moderate settings
    The first retrieval replaces the workspace; later ones merge to keep prior evidence.
    Returns (params, rationale) — rationale is appended to the turn's reasoning so the
    action stays consistent with the reasoning.
    """
    n = len(query.split())
    if n >= 6:
        params = {"top_k": 10, "k1": 1.2, "b": 0.9}
        why = ("This is a longer, descriptive query, so I'll lower k1 and raise b so long "
               "generic articles don't crowd out the specific passage, and widen top_k a little.")
    elif n <= 3:
        params = {"top_k": 5, "k1": 2.0, "b": 0.5}
        why = ("This is a short, focused query on a distinctive term, so I'll raise k1 so its "
               "repeated exact mentions dominate the match and lower b to favor short, dense passages.")
    else:
        params = {"top_k": 5, "k1": 1.5, "b": 0.75}
        why = ""
    params["mode"] = "replace" if bm25_idx == 0 else "merge"
    if bm25_idx > 0:
        why = (why + " " if why else "") + "I'll merge these into the workspace to keep the earlier evidence."
    return params, why.strip()


# Search-then-teach mentor: try BM25 settings, keep the one that actually ranks the
# target passage best, and teach THOSE parameters (empirically grounded, not a rule).
# Grids (_K1_GRID/_B_GRID and the _LONG variants) are defined at the top of the
# module, before ColdStartConfig, whose fields default to them.


def _gold_rank(hits: list[dict], targets: list[str]) -> Optional[int]:
    """1-based rank of the first retrieved passage that contains a target string."""
    norm_targets = [normalize(t) for t in targets if t]
    for i, h in enumerate(hits, 1):
        t = normalize(h.get("text", ""))
        if any(nt and nt in t for nt in norm_targets):
            return i
    return None


def _search_params(retriever, query: str, targets: list[str], bm25_idx: int,
                   k1_grid: tuple = _K1_GRID, b_grid: tuple = _B_GRID) -> tuple[dict, str]:
    """Grid-search (k1, b); pick the setting that ranks the target passage highest,
    then the smallest top_k that includes it. Returns (params, rationale).
    Falls back to explicit default params if no setting surfaces the target."""
    top_probe = max(_TOPK_LADDER)
    default_rank = _gold_rank(retriever.retrieve(query, top_k=top_probe, k1=1.2, b=0.75), targets)
    best = None  # (rank, closeness_to_default, k1, b)
    for k1 in k1_grid:
        for b in b_grid:
            rank = _gold_rank(retriever.retrieve(query, top_k=top_probe, k1=k1, b=b), targets)
            if rank is None:
                continue
            cand = (rank, abs(k1 - 1.2) + abs(b - 0.75), k1, b)
            if best is None or cand < best:
                best = cand
    mode = "replace" if bm25_idx == 0 else "merge"
    if best is None:
        # No grid point surfaces the target in top-20 (possible when the judge
        # verified semantically but no target form substring-matches). Emit the
        # DEFAULT params rather than a bare {mode}: "no signal -> stay default"
        # is the honest action, and a paramless call would teach the student
        # that the knobs are optional (and previously tripped the zero-tolerance
        # param gate on a single edge case at the end of a 24h run).
        return {"top_k": 5, "k1": 1.2, "b": 0.75, "mode": mode}, ""
    rank, _, k1, b = best
    top_k = next((t for t in _TOPK_LADDER if t >= rank), top_probe)
    params = {"top_k": top_k, "k1": round(k1, 2), "b": round(b, 2), "mode": mode}
    # The rationale must stay *inference-reachable*: the student never knows the
    # gold's rank, so a sentence quoting default_rank/rank teaches it to assert an
    # observation it cannot make (and, being a fixed template, to memorise the
    # phrasing rather than the policy). Justify the knobs by the query's own
    # shape — which the student can see — and let the reward supply the
    # did-it-work signal at RL time.
    if default_rank and default_rank <= 3 and rank >= default_rank:
        why = ""  # default already retrieves it well; no need to justify a tweak
    elif params["b"] < 0.75:
        why = (f"This looks like it turns on a detail that a longer passage would bury, "
               f"so I'll lower b to {params['b']} to stop favouring short passages and "
               f"widen top_k to {top_k}.")
    elif params["b"] > 0.75:
        why = (f"The query is broad enough that sprawling articles would match it loosely, "
               f"so I'll raise b to {params['b']} to penalise length and widen top_k to {top_k}.")
    elif params["k1"] > 1.2:
        why = (f"The distinguishing term here should dominate the match, so I'll raise "
               f"k1 to {params['k1']} and set top_k={top_k}.")
    elif params["k1"] < 1.2:
        why = (f"One mention of the key term is as good as several here, so I'll lower "
               f"k1 to {params['k1']} and set top_k={top_k}.")
    else:
        why = f"I'll widen top_k to {top_k} to pull more candidates into the workspace."
    if bm25_idx > 0:
        why = (why + " " if why else "") + "I'll merge it with the earlier evidence."
    return params, why.strip()


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

@dataclass
class Hop:
    sub_question: str
    expected: str
    forms: list[str]
    trace: list[dict] = field(default_factory=list)
    best_docs: list[dict] = field(default_factory=list)
    verified: bool = False
    # race mode: (k1,b) that rescued this hop's verification when the default
    # parameters missed the evidence; None if default worked or no rescue ran.
    rescue_params: Optional[dict] = None
    # failure->recovery injection: the first FAILED backward attempt's trace,
    # kept when a later refined attempt verified. This is the raw material for
    # teaching observable failure handling (2026-07-30): strict success-filtering
    # + outcome-grounded params collapsed every knob to a constant, because the
    # when-to-adapt signal lives precisely in the attempts that filtering drops.
    failed_trace: Optional[list] = None


def _run_trace_isolated(trace: list[dict], retriever, tokenizer, cfg: ColdStartConfig) -> tuple[str, list[dict]]:
    """Execute a candidate trace on a fresh workspace; return (rendered_output, docs)."""
    ws = Workspace()
    outputs = []
    for tc in trace:
        res = execute_tool(tc["name"], tc["arguments"], ws, retriever,
                           tokenizer=tokenizer, max_response_tokens=cfg.tool_response_tokens)
        outputs.append(json.dumps(res, ensure_ascii=False))
    return "\n".join(outputs), list(ws.docs)


def _discover_hop(teacher, hop: Hop, downstream_docs: str, retriever, tokenizer,
                  cfg: ColdStartConfig) -> Hop:
    forbidden = [hop.expected] + hop.forms
    sys_msg = {"role": "system", "content": P.BACKWARD_TOOL_SYSTEM.format(
        corpus_description=P.CORPUS_DESCRIPTION)}
    user = P.BACKWARD_TOOL_USER_INITIAL.format(
        sub_question=hop.sub_question, expected_answer=hop.expected,
        forbidden_forms=", ".join(forbidden), downstream_docs=downstream_docs or "(none)")

    prior_attempts: list[str] = []
    last_trace: list[dict] = []
    last_output = ""
    last_judge = ""
    _first_failed: Optional[list] = None   # attempt-0 trace, kept if a refine succeeds

    for attempt in range(cfg.max_refine + 1):
        if attempt == 0:
            msgs = [sys_msg, {"role": "user", "content": user}]
        else:
            refine = P.BACKWARD_TOOL_USER_REFINE.format(
                sub_question=hop.sub_question, expected_answer=hop.expected,
                forbidden_forms=", ".join(forbidden),
                prior_attempts="\n".join(prior_attempts) or "(none)",
                last_tool_trace=json.dumps(last_trace), last_output=last_output[:1200],
                judge_reasoning=last_judge)
            msgs = [sys_msg, {"role": "user", "content": refine}]

        out = teacher.complete(msgs, max_tokens=cfg.teacher_max_tokens,
                               temperature=0.0, enable_thinking=False)
        raw_trace = _extract_json_array(_extract_tag(out, "tool_trace") or out) or []
        trace = [t for t in (_normalize_tool_call(x, cfg) for x in raw_trace) if t]
        # enforce ANSWER-LEAK rule (see cfg.grep_may_leak_answer)
        trace = [t for t in trace if not _call_leaks_answer(t, forbidden, cfg)]
        if not trace or trace[0]["name"] != "bm25_retrieve":
            prior_attempts.append(f"attempt {attempt}: no valid bm25_retrieve produced")
            continue

        output, docs = _run_trace_isolated(trace, retriever, tokenizer, cfg)
        ok, reason = _judge(teacher, hop.sub_question, hop.expected, hop.forms, output, cfg)
        last_trace, last_output, last_judge = trace, output, reason
        prior_attempts.append(f"attempt {attempt}: query={trace[0]['arguments']['query']!r} -> {'YES' if ok else 'NO'}")
        if ok:
            hop.trace, hop.best_docs, hop.verified = trace, docs, True
            if attempt > 0 and _first_failed is not None:
                hop.failed_trace = _first_failed
            return hop
        if attempt == 0 and trace:
            _first_failed = trace

        # Race-mode rescue: the teacher never varies k1/b on refine (it only
        # rewrites the query — see reports/param_policy_findings.md), so a hop
        # whose evidence is rank-buried under default params burns every refine
        # attempt on the wrong knob. Before the next query rewrite, replay the
        # SAME trace under the config grid; judge only mechanically-promising
        # configs (gold form visible in output), bounded by race_judge_budget.
        # This is where the 72% strict-verification failure rate gets attacked.
        if cfg.param_policy == "race":
            judged = 0
            for config in R.BM25_CONFIGS[1:]:  # skip default: just failed above
                if judged >= cfg.race_judge_budget:
                    break
                cand = [dict(t, arguments=dict(t["arguments"])) for t in trace]
                for t in cand:
                    if t["name"] == "bm25_retrieve":
                        t["arguments"].update({"k1": config["k1"], "b": config["b"]})
                c_out, c_docs = _run_trace_isolated(cand, retriever, tokenizer, cfg)
                if not R._contains_any(c_out, [hop.expected] + hop.forms):
                    continue
                judged += 1
                c_ok, c_reason = _judge(teacher, hop.sub_question, hop.expected,
                                        hop.forms, c_out, cfg)
                prior_attempts.append(
                    f"attempt {attempt} [rescue {config['name']}]: -> {'YES' if c_ok else 'NO'}")
                if c_ok:
                    hop.trace, hop.best_docs, hop.verified = cand, c_docs, True
                    hop.rescue_params = {"k1": config["k1"], "b": config["b"],
                                         "name": config["name"]}
                    return hop

    # Not verified within budget: keep the last candidate for lenient mode.
    hop.trace, hop.best_docs, hop.verified = last_trace, [], False
    return hop


def _backward_pass(teacher, question: str, golds: list[str], retriever, tokenizer,
                   cfg: ColdStartConfig) -> Optional[list[Hop]]:
    subs = _decompose(teacher, question, golds[0], cfg)
    n = len(subs)
    hops = [Hop(sub_question=s, expected="", forms=[]) for s in subs]
    hops[-1].expected = golds[0]
    hops[-1].forms = [g for g in golds[1:]]

    downstream = ""
    for i in range(n - 1, -1, -1):
        hop = _discover_hop(teacher, hops[i], downstream, retriever, tokenizer, cfg)
        if not hop.verified and cfg.strict:
            logger.info("hop %d unverified (strict) -> skip example", i)
            return None
        # bridge back to the previous hop's expected answer
        if i > 0:
            doc_text = hop.best_docs[0]["text"] if hop.best_docs else ""
            bridge = _bridge_extract(teacher, question, hops[i - 1].sub_question,
                                     hops[i].sub_question, hops[i].expected, doc_text, cfg)
            if bridge["bridge_entity"]:
                hops[i - 1].expected = bridge["bridge_entity"]
                hops[i - 1].forms = bridge["aliases"]
            elif cfg.strict:
                logger.info("bridge extraction failed at hop %d (strict) -> skip", i)
                return None
            else:
                hops[i - 1].expected = hops[i - 1].expected or hops[i - 1].sub_question
        if hop.best_docs:
            downstream = (downstream + "\n" + hop.best_docs[0]["text"])[:1500]
    return hops


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward_pass(teacher, question: str, golds: list[str], hops: list[Hop],
                  retriever, tokenizer, cfg: ColdStartConfig,
                  force_config: Optional[dict] = None) -> Trajectory:
    """Assemble the forward trajectory.

    force_config (race mode): a BM25_CONFIGS entry applied to every
    bm25_retrieve call in this arm, overriding the per-call param policy. The
    turn's reasoning gets the config's inference-reachable rationale — never
    oracle facts like the gold's rank (see the 2026-07-28 leak fix note above
    _search_params).
    """
    traj = Trajectory(id="", question=question, golden_answers=golds)
    messages: list[dict] = [
        {"role": "system", "content": SCALESEEK_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]
    mask = [0, 0]
    workspace = Workspace()
    history = ""
    n_tool_calls = 0
    bm25_idx = 0
    n_failure_turns = 0

    for hop in hops:
        # ── failure -> recovery injection ────────────────────────────────────
        # Replay the hop's REAL failed first attempt (backward judge rejected it,
        # a refined query then verified), so the trajectory demonstrates the one
        # adaptation signal that is actually observable to the student: results
        # that do not contain what was needed -> reformulate. Without this,
        # strict success-filtering + outcome-grounded params collapse every knob
        # to a constant (2026-07-30 probe: top_k=3 on 13/13 calls) and the
        # discarded failures are precisely where when-to-adapt lives. Also the
        # only content-driven `replace` demonstration in the data: when the
        # workspace holds nothing but the junk of a failed first search, the
        # recovery call REPLACES instead of merging.
        failed_tc = None
        if cfg.inject_failures and hop.failed_trace:
            cand = hop.failed_trace[0]
            if cand.get("name") == "bm25_retrieve":
                failed_tc = {"name": cand["name"], "arguments": dict(cand["arguments"])}
        if failed_tc is not None:
            draft_think, _ = _planner_draft(teacher, question, history, cfg)
            was_first = bm25_idx == 0
            failed_tc["arguments"].setdefault("top_k", cfg.top_k_default)
            failed_tc["arguments"].setdefault("k1", 1.2)
            failed_tc["arguments"].setdefault("b", 0.75)
            failed_tc["arguments"]["mode"] = "replace" if was_first else "merge"
            result = execute_tool(failed_tc["name"], failed_tc["arguments"], workspace,
                                  retriever, tokenizer=tokenizer,
                                  max_response_tokens=cfg.tool_response_tokens)
            result_json = json.dumps(result, ensure_ascii=False)
            reasoning = _tutor_edit(teacher, question, history, draft_think, None,
                                    failed_tc, result_json[: cfg.preview_chars], cfg)
            assistant = _fmt_tool_turn(reasoning, failed_tc)
            messages.append({"role": "assistant", "content": assistant}); mask.append(1)
            messages.append({"role": "tool", "content": result_json}); mask.append(0)
            history = _history_append(history, assistant, result_json)
            n_tool_calls += 1
            bm25_idx += 1
            n_failure_turns += 1

        for tc_i, target_tc in enumerate(hop.trace):
            draft_think, draft_tc = _planner_draft(teacher, question, history, cfg)
            # Apply the cold-start parameter policy so the trajectory demonstrates
            # adaptive k1/b/top_k/mode control instead of a constant or all-omitted set.
            param_note = ""
            if target_tc["name"] == "bm25_retrieve" and force_config is not None:
                target_tc["arguments"].update(
                    {"k1": force_config["k1"], "b": force_config["b"]})
                param_note = force_config["rationale"]
                bm25_idx += 1
            elif cfg.param_policy in ("heuristic", "search") and target_tc["name"] == "bm25_retrieve":
                query = str(target_tc["arguments"].get("query", ""))
                if cfg.param_policy == "heuristic":
                    params, param_note = _param_policy(query, bm25_idx)
                else:
                    targets = [hop.expected] + list(hop.forms)
                    params, param_note = _search_params(retriever, query, targets, bm25_idx,
                                                        k1_grid=cfg.k1_grid, b_grid=cfg.b_grid)
                target_tc["arguments"].update(params)
                bm25_idx += 1
            # Recovery semantics for the verified call right after an injected
            # failure. The note references only what is observable in history.
            if failed_tc is not None and tc_i == 0 and target_tc["name"] == "bm25_retrieve":
                if was_first:
                    target_tc["arguments"]["mode"] = "replace"
                    extra = ("The results so far don't contain what I'm looking for, "
                             "so I'll replace the workspace with a reformulated search.")
                else:
                    extra = ("That search didn't surface what I need; I'll reformulate "
                             "and merge a better query's results.")
                param_note = f"{param_note} {extra}".strip() if param_note else extra
            result = execute_tool(target_tc["name"], target_tc["arguments"], workspace,
                                  retriever, tokenizer=tokenizer,
                                  max_response_tokens=cfg.tool_response_tokens)
            result_json = json.dumps(result, ensure_ascii=False)
            preview = result_json[: cfg.preview_chars]
            reasoning = _tutor_edit(teacher, question, history, draft_think, draft_tc,
                                    target_tc, preview, cfg)
            if param_note:
                reasoning = (_sanitize_reasoning(reasoning) + " " + param_note).strip()
            assistant = _fmt_tool_turn(reasoning, target_tc)
            messages.append({"role": "assistant", "content": assistant}); mask.append(1)
            messages.append({"role": "tool", "content": result_json}); mask.append(0)
            history = _history_append(history, assistant, result_json)
            n_tool_calls += 1

    # Final answer turn — reasoning synthesized by the planner, <answer> pinned to gold.
    # Pinning is only legitimate when the workspace actually supports the answer;
    # otherwise the trajectory teaches "answer from nowhere". This is a programmatic
    # check on the real workspace, deliberately NOT delegated to the quality judge —
    # a same-family LLM judge is the wrong instrument here (reports/baselines.md
    # §10j measured 11-13% self-contradictory verdicts on this model class).
    answer_supported = _workspace_supports(workspace, golds)
    # Per-hop evidence coverage: fraction of hops whose expected answer (any
    # form) survives into the FINAL workspace. This — not answer EM, which is
    # pinned to gold by construction — is what discriminates race arms.
    n_covered = sum(
        1 for h in hops if _workspace_supports(workspace, [h.expected] + list(h.forms)))
    final_reason = _final_reasoning(teacher, question, history, cfg)
    answer_turn = _fmt_answer_turn(final_reason, golds[0])
    messages.append({"role": "assistant", "content": answer_turn}); mask.append(1)

    traj.messages = messages
    traj.loss_mask = mask
    traj.meta = {
        "n_hops": len(hops),
        "n_tool_calls": n_tool_calls,
        "all_hops_verified": all(h.verified for h in hops),
        "final_workspace_size": workspace.size,
        "answer_supported": answer_supported,
        "hop_coverage": n_covered / max(len(hops), 1),
        "n_failure_turns": n_failure_turns,
    }
    if force_config is not None:
        traj.meta["race_config"] = force_config["name"]
    if not answer_supported:
        traj.status = "unsupported_answer"
    return traj


def _trajectory_text(messages: list[dict]) -> str:
    """Flatten a trajectory for the QUALITY_JUDGE prompt."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        out.append(f"[{m['role'].upper()}]\n{m['content']}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_trajectory(example: dict, teacher, retriever, *, tokenizer=None,
                     cfg: Optional[ColdStartConfig] = None) -> Trajectory:
    """Build one cold-start trajectory. Never raises; returns status in the result."""
    cfg = cfg or ColdStartConfig()
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = [g for g in example.get("golden_answers", []) if str(g).strip()]
    if not golds:
        return Trajectory(id=ex_id, question=question, golden_answers=[], status="skipped",
                          meta={"reason": "no gold answer"})

    try:
        hops = _backward_pass(teacher, question, golds, retriever, tokenizer, cfg)
    except Exception as e:  # teacher / parsing blowups shouldn't kill the run
        logger.warning("backward pass failed for %s: %s", ex_id, e)
        return Trajectory(id=ex_id, question=question, golden_answers=golds,
                          status="skipped", meta={"reason": f"backward_error: {e}"})
    if hops is None:
        return Trajectory(id=ex_id, question=question, golden_answers=golds,
                          status="skipped", meta={"reason": "strict verification failed"})

    if cfg.param_policy == "race":
        # User-specified config race (2026-07-30): the backward pass ran ONCE
        # (shared across arms — decompose/verify is the expensive teacher work);
        # only the forward assembly forks per config.
        arms = R.select_race_configs(retriever, hops, cfg.race_width)
        # A hop rescued by a specific config is direct evidence that config
        # matters for this question — force its arm into the race.
        for h in hops:
            if h.rescue_params and not any(a["name"] == h.rescue_params["name"] for a in arms):
                arms.append(next(c for c in R.BM25_CONFIGS
                                 if c["name"] == h.rescue_params["name"]))
        scored: list[tuple[float, Trajectory, dict]] = []
        for config in arms:
            t = _forward_pass(teacher, question, golds, hops, retriever, tokenizer,
                              cfg, force_config=config)
            scored.append((R.score_trajectory(t.meta), t, config))
        # Winner: highest score; ties break toward default (teach deviation
        # only when it demonstrably pays — s3's gain-over-baseline logic),
        # then toward the smaller workspace.
        scored.sort(key=lambda s: (-s[0],
                                   0 if s[2]["name"] == "default" else 1,
                                   s[1].meta.get("final_workspace_size", 0)))
        best_score, traj, best_cfg = scored[0]
        default_score = next((s for s, _, c in scored if c["name"] == "default"), None)
        traj.meta["race"] = {
            "winner": best_cfg["name"], "score": round(best_score, 4),
            "gain_over_default": (round(best_score - default_score, 4)
                                  if default_score is not None else None),
            "arms": [{"config": c["name"], "score": round(s, 4)} for s, _, c in scored],
            # Which hops only verified under a non-default config. Without this
            # the rescue mechanism is invisible in the output (2026-07-30 probe:
            # had to guess whether 36->35 skips was rescue or variance).
            "rescues": [h.rescue_params["name"] for h in hops if h.rescue_params],
        }
        # Losers become preference/failure data. Full messages are kept only
        # when the winner is decisively better (margin >= 0.5 == one hop of
        # coverage) — near-ties are noise, not preference signal.
        for s, t, c in scored[1:]:
            rec = {"config": c["name"], "score": round(s, 4), "meta": t.meta}
            if best_score - s >= 0.5:
                rec["messages"] = t.messages
                rec["loss_mask"] = t.loss_mask
            traj.siblings.append(rec)
    else:
        traj = _forward_pass(teacher, question, golds, hops, retriever, tokenizer, cfg)
    traj.id = ex_id

    if cfg.run_quality_judge:
        try:
            ok, reason = _quality_pass(teacher, question, _trajectory_text(traj.messages))
            traj.meta["quality_reason"] = reason
            if not ok and traj.status == "ok":
                # Don't mask a programmatic rejection (unsupported_answer) with the
                # weaker LLM verdict — the former is the more reliable signal.
                traj.status = "quality_fail"
        except Exception as e:
            logger.warning("quality judge failed for %s: %s", ex_id, e)
            traj.meta["quality_reason"] = f"judge_error: {e}"
    return traj
