"""No-GPU contract tests for the SFT cold-start generator.

Uses a scripted FakeTeacher + FakeRetriever so the full pipeline
(decompose -> backward discover -> judge -> forward planner/tutor -> answer ->
quality judge) runs without a model, GPU, or Lucene index. Verifies the emitted
trajectory obeys the ScaleSeek format contract consumed by eval/agent and
train/reward.
"""
from __future__ import annotations

import json
import re

import pytest

from train.sft.coldstart import (
    ColdStartConfig, build_trajectory,
    _extract_json_array, _extract_json_object, _extract_tag,
    _split_reasoning_and_toolcall, _query_leaks_answer, _sanitize_reasoning,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRetriever:
    """Returns a canned passage for any query; matches the BM25Retriever API."""

    def __init__(self, doc_id="fake_1", text="Paris is the capital of France."):
        self._doc = {"doc_id": doc_id, "score": 5.0, "text": text}

    def retrieve(self, query, top_k=3, k1=1.2, b=0.75):
        return [dict(self._doc)]


def _single_hop_teacher():
    from train.sft.teacher import FakeTeacher

    def respond(messages):
        last = messages[-1]["content"]
        if "decomposing a multi-hop question" in last:
            return '[{"sub_question": "What is the capital of France?"}]'
        if "Sub-question to find supporting evidence for" in last or "propose a different bm25_retrieve" in last:
            return ('<reasoning>Use question terms, not the answer.</reasoning>\n'
                    '<tool_trace>[{"name": "bm25_retrieve", "arguments": '
                    '{"query": "capital city of France", "top_k": 5, "mode": "replace"}}]</tool_trace>')
        if "judging whether a BM25 retrieval result" in last:
            return '{"verdict": "YES", "reasoning": "passage states the capital"}'
        if "Produce the next step" in last:
            return ('I should search for the capital.\n<tool_call>\n'
                    '{"name": "bm25_retrieve", "arguments": {"query": "capital France"}}\n</tool_call>')
        if "editing a research agent's draft reasoning" in last:
            return '<edited_reasoning>I need the capital of France; I will retrieve with those terms.</edited_reasoning>'
        if "You now have enough information" in last:
            return 'The passage names the capital.\n<answer>\nParis\n</answer>'
        if "expert reviewer of multi-hop QA trajectories" in last:
            return '{"verdict": "PASS", "failing_check": null, "first_failing_turn": null, "reasoning": "coherent"}'
        return ""

    return FakeTeacher(respond)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def test_extract_json_array_with_fence():
    assert _extract_json_array('```json\n[{"sub_question": "a"}]\n```') == [{"sub_question": "a"}]


def test_extract_json_object_with_nested_and_prose():
    txt = 'sure: {"verdict": "YES", "obj": {"x": 1}, "s": "has } brace"} done'
    assert _extract_json_object(txt) == {"verdict": "YES", "obj": {"x": 1}, "s": "has } brace"}


def test_extract_tag():
    assert _extract_tag("<edited_reasoning>\nhi\n</edited_reasoning>", "edited_reasoning") == "hi"


def test_split_reasoning_and_toolcall():
    txt = 'I will search.\n<tool_call>\n{"name": "bm25_retrieve", "arguments": {"query": "q"}}\n</tool_call>'
    reasoning, tc = _split_reasoning_and_toolcall(txt)
    assert reasoning == "I will search."
    assert tc["name"] == "bm25_retrieve"


def test_query_leak_rule():
    assert _query_leaks_answer("Oberoi Group hotel", ["The Oberoi Group"])
    assert not _query_leaks_answer("Oberoi family hotel company", ["The Oberoi Group"])


def test_sanitize_reasoning_cuts_at_first_struct_tag():
    # a premature <answer> inside reasoning must not survive into the <think> block
    r = _sanitize_reasoning("I am confident now.\n<answer>\nCanberra\n</answer>")
    assert r == "I am confident now." and "<answer>" not in r
    assert _sanitize_reasoning("search now <tool_call>{}</tool_call>") == "search now"
    assert _sanitize_reasoning("plain reasoning") == "plain reasoning"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_single_hop_trajectory_format():
    ex = {"id": "t1", "question": "What is the capital of France?", "golden_answers": ["Paris"]}
    cfg = ColdStartConfig(max_refine=1)
    traj = build_trajectory(ex, _single_hop_teacher(), FakeRetriever(), tokenizer=None, cfg=cfg)

    assert traj.status == "ok", traj.meta
    roles = [m["role"] for m in traj.messages]
    assert roles[0] == "system" and roles[1] == "user"
    assert roles[-1] == "assistant"

    # loss mask: 1 exactly on assistant turns
    assert len(traj.loss_mask) == len(traj.messages)
    for m, bit in zip(traj.messages, traj.loss_mask):
        assert bit == (1 if m["role"] == "assistant" else 0)

    # at least one tool turn + a final answer turn
    assistants = [m["content"] for m in traj.messages if m["role"] == "assistant"]
    assert any("<tool_call>" in a for a in assistants)
    final = assistants[-1]
    assert "<answer>" in final and "Paris" in final

    # every assistant turn is well-formed <think>...</think> then one action block
    for a in assistants:
        assert a.count("<think>") == a.count("</think>") == 1
        assert ("<tool_call>" in a) ^ ("<answer>" in a)
        # the <think> block itself must not contain a nested action tag
        think_body = re.search(r"<think>(.*?)</think>", a, re.DOTALL).group(1)
        assert "<answer>" not in think_body and "<tool_call>" not in think_body

    # tool responses are valid JSON
    for m in traj.messages:
        if m["role"] == "tool":
            json.loads(m["content"])


def test_answer_is_pinned_to_gold_even_if_planner_differs():
    from train.sft.teacher import FakeTeacher

    def respond(messages):
        last = messages[-1]["content"]
        if "decomposing a multi-hop question" in last:
            return '[{"sub_question": "capital of France"}]'
        if "Sub-question to find" in last:
            return ('<tool_trace>[{"name":"bm25_retrieve","arguments":{"query":"France capital city"}}]</tool_trace>')
        if "judging whether" in last:
            return '{"verdict":"YES","reasoning":"ok"}'
        if "Produce the next step" in last:
            return '<tool_call>{"name":"bm25_retrieve","arguments":{"query":"x"}}</tool_call>'
        if "editing a research agent" in last:
            return '<edited_reasoning>search now</edited_reasoning>'
        if "You now have enough information" in last:
            return '<answer>WrongCity</answer>'    # planner disagrees
        if "expert reviewer" in last:
            return '{"verdict":"PASS","failing_check":null,"first_failing_turn":null,"reasoning":"ok"}'
        return ""

    ex = {"id": "t2", "question": "What is the capital of France?", "golden_answers": ["Paris"]}
    traj = build_trajectory(ex, FakeTeacher(respond), FakeRetriever(), cfg=ColdStartConfig(max_refine=0))
    final = [m["content"] for m in traj.messages if m["role"] == "assistant"][-1]
    assert "Paris" in final and "WrongCity" not in final


def test_no_gold_is_skipped():
    ex = {"id": "t3", "question": "q?", "golden_answers": []}
    traj = build_trajectory(ex, _single_hop_teacher(), FakeRetriever(), cfg=ColdStartConfig())
    assert traj.status == "skipped"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
