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
       (ANSWER-LEAK rule: the bm25 query may never contain expected[i].)

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
from prompts.scaleseek_prompt import PROMPT as SCALESEEK_SYSTEM

logger = logging.getLogger(__name__)

_VALID_TOOLS = {"bm25_retrieve", "grep_workspace", "read_doc"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ColdStartConfig:
    max_refine: int = 2               # backward-discovery retries per hop
    top_k_default: int = 5
    tool_response_tokens: int = 512   # per-response budget in forward execution
    strict: bool = False              # skip examples whose hops fail to verify
    run_quality_judge: bool = True
    teacher_max_tokens: int = 768
    preview_chars: int = 700


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
        args.setdefault("top_k", cfg.top_k_default)
        args.setdefault("k1", 1.2)
        args.setdefault("b", 0.75)
        args.setdefault("mode", "replace")
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
    status: str = "ok"                                   # ok | skipped | quality_fail
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "question": self.question,
            "golden_answers": self.golden_answers,
            "messages": self.messages, "loss_mask": self.loss_mask,
            "status": self.status, "meta": self.meta,
        }


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
        # enforce ANSWER-LEAK rule on bm25 queries
        trace = [t for t in trace
                 if not (t["name"] == "bm25_retrieve"
                         and _query_leaks_answer(t["arguments"]["query"], forbidden))]
        if not trace or trace[0]["name"] != "bm25_retrieve":
            prior_attempts.append(f"attempt {attempt}: no valid bm25_retrieve produced")
            continue

        output, docs = _run_trace_isolated(trace, retriever, tokenizer, cfg)
        ok, reason = _judge(teacher, hop.sub_question, hop.expected, hop.forms, output, cfg)
        last_trace, last_output, last_judge = trace, output, reason
        prior_attempts.append(f"attempt {attempt}: query={trace[0]['arguments']['query']!r} -> {'YES' if ok else 'NO'}")
        if ok:
            hop.trace, hop.best_docs, hop.verified = trace, docs, True
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
                  retriever, tokenizer, cfg: ColdStartConfig) -> Trajectory:
    traj = Trajectory(id="", question=question, golden_answers=golds)
    messages: list[dict] = [
        {"role": "system", "content": SCALESEEK_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]
    mask = [0, 0]
    workspace = Workspace()
    history = ""
    n_tool_calls = 0

    for hop in hops:
        for target_tc in hop.trace:
            draft_think, draft_tc = _planner_draft(teacher, question, history, cfg)
            result = execute_tool(target_tc["name"], target_tc["arguments"], workspace,
                                  retriever, tokenizer=tokenizer,
                                  max_response_tokens=cfg.tool_response_tokens)
            result_json = json.dumps(result, ensure_ascii=False)
            preview = result_json[: cfg.preview_chars]
            reasoning = _tutor_edit(teacher, question, history, draft_think, draft_tc,
                                    target_tc, preview, cfg)
            assistant = _fmt_tool_turn(reasoning, target_tc)
            messages.append({"role": "assistant", "content": assistant}); mask.append(1)
            messages.append({"role": "tool", "content": result_json}); mask.append(0)
            history = _history_append(history, assistant, result_json)
            n_tool_calls += 1

    # Final answer turn — reasoning synthesized by the planner, <answer> pinned to gold.
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
    }
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

    traj = _forward_pass(teacher, question, golds, hops, retriever, tokenizer, cfg)
    traj.id = ex_id

    if cfg.run_quality_judge:
        try:
            ok, reason = _quality_pass(teacher, question, _trajectory_text(traj.messages))
            traj.meta["quality_reason"] = reason
            if not ok:
                traj.status = "quality_fail"
        except Exception as e:
            logger.warning("quality judge failed for %s: %s", ex_id, e)
            traj.meta["quality_reason"] = f"judge_error: {e}"
    return traj
