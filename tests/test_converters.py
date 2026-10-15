"""Official-harness outputs -> unified records (synthetic files in the official formats)."""
import json

from scaleseek.methods import agentir, grepseek

EX = [{"id": "a", "question": "Q?", "golden_answers": ["x"]},
      {"id": "b", "question": "R?", "golden_answers": ["y"]}]


def test_grepseek_convert(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "b", "prediction": "y", "finish_reason": "answer", "n_assistant_turns": 3,
         "n_tool_calls": 2, "error": None, "total_time_s": 4.0, "em": True, "f1": 1.0,
         "turns": [{"role": "tool", "tool_result": {"timed_out": False, "exit_code": 1}},
                   {"role": "tool", "synthetic": True},
                   {"role": "tool", "tool_result": {"timed_out": True, "exit_code": -1}}],
         "messages": []},
        {"id": "a", "prediction": None, "finish_reason": "max_turns", "n_assistant_turns": 6,
         "n_tool_calls": 6, "error": None, "turns": [], "messages": []},
    ]
    pred.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = grepseek.convert(None, pred, EX)
    assert [r["id"] for r in out] == ["a", "b"]                   # subset order
    b = out[1]
    assert b["prediction"] == "y" and b["n_turns"] == 3 and b["n_retrievals"] == 2
    assert b["errors"] == {"parse": 1, "tool": 1, "api": 0}       # rg exit 1 = no match, not an error
    assert out[0]["prediction"] is None and out[0]["finish_reason"] == "max_turns"


def test_agentir_convert(tmp_path):
    (tmp_path / "sidecars").mkdir()
    final = "Explanation: found it [7].\nExact Answer: x\nConfidence: 90%"
    run = {"query_id": "a", "status": "completed", "tool_call_counts": {"search": 2},
           "tool_call_counts_raw": {"search": 3},
           "result": [{"type": "reasoning", "output": "r"}, {"type": "tool_call", "output": "..."},
                      {"type": "output_text", "output": final}]}
    (tmp_path / "run_1.json").write_text(json.dumps(run))
    side = {"calls": [{"ok": True}, {"ok": False, "kind": "transient"}, {"ok": True}], "seconds": 3.5}
    (tmp_path / "sidecars" / "a.json").write_text(json.dumps(side))
    run_b = {"query_id": "b", "status": "incomplete", "result": [{"type": "reasoning", "output": "r"}]}
    (tmp_path / "run_2.json").write_text(json.dumps(run_b))
    a, b = agentir.convert(None, tmp_path, EX)
    assert a["prediction"] == "x" and a["final_response"] == final and a["finish_reason"] == "answer"
    assert a["n_turns"] == 3 and a["n_retrievals"] == 2
    assert a["errors"] == {"parse": 0, "tool": 1, "api": 1} and a["wall_time_s"] == 3.5
    assert b["prediction"] is None and b["finish_reason"] == "max_turns"


def test_agentir_one_search_per_model_call():
    from types import SimpleNamespace as NS
    from scaleseek.agentir_entry import drop_parallel_calls
    out = [NS(type="reasoning"), NS(type="function_call", call_id="1"),
           NS(type="function_call", call_id="2"), NS(type="function_call", call_id="3")]
    response = NS(output=out)
    assert drop_parallel_calls(response) == 2
    assert [o.type for o in response.output] == ["reasoning", "function_call"]
    assert response.output[1].call_id == "1"
