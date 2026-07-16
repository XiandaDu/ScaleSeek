"""Unit tests for eval/retrieval_metrics.py on tiny hand-computed fixtures.

Run:  python -m pytest tests/test_retrieval_metrics.py
  or: python tests/test_retrieval_metrics.py
No model, GPU, corpus, or network needed.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import retrieval_metrics as rm


def test_gold_and_qrel_recall():
    assert rm.gold_recall({"a", "b"}, {"a", "b", "c"}) == 2 / 3
    assert rm.qrel_recall({"a", "x"}, {"a", "b"}) == 1 / 2
    assert rm.gold_recall({"a"}, set()) is None          # undefined on empty gold


def test_coverage():
    cov = rm.coverage({"a", "z"}, {"a", "b", "c"})
    assert cov == {"any": 1.0, "mean": 1 / 3, "all": 0.0}
    cov2 = rm.coverage({"a", "b", "c"}, {"a", "b", "c"})
    assert cov2 == {"any": 1.0, "mean": 1.0, "all": 1.0}
    assert rm.coverage({"a"}, set()) == {"any": None, "mean": None, "all": None}


def test_localization():
    # c_seg=100, gold doc = 1000 chars -> nu=10 (=b). snippet 500 -> nu=5 (=a).
    # seg = 1 - log(5)/log(10) = 0.30103...
    expected = 1 - math.log(5) / math.log(10)
    loc = rm.localization({"d1": [500]}, {"d1": 1000}, c_seg=100)
    assert abs(loc - expected) < 1e-9
    # best snippet wins: a tighter 100-char snippet -> nu=1 -> seg=1.0
    loc2 = rm.localization({"d1": [500, 100]}, {"d1": 1000}, c_seg=100)
    assert abs(loc2 - 1.0) < 1e-9
    # tiny gold doc (nu(glen)=1 -> b=1) -> psi(a;1)=1.0
    assert rm.localization({"d": [50]}, {"d": 40}, c_seg=100) == 1.0
    assert rm.localization({}, {}, c_seg=100) is None


def test_surfaced_from_grep():
    # GrepSeek-style tool turn: information_lines carry corpus json lines
    turn_gs = {"role": "tool", "content": json.dumps({
        "information_lines": [
            json.dumps({"id": "11", "contents": "\"A\"\nx"}),
            json.dumps({"id": "12", "contents": "\"B\"\ny"}),
            "42",  # a stray wc -l count line -> ignored
        ]})}
    # DCI-style tool turn: stdout string with corpus json lines
    turn_dci = {"role": "tool", "content": json.dumps({
        "stdout": json.dumps({"id": "13", "contents": "\"C\"\nz"}) + "\nnot json\n",
        "information_lines": None})}
    rec = {"turns": [turn_gs, turn_dci]}
    assert rm.surfaced_doc_ids_from_grep(rec) == {"11", "12", "13"}


def test_workspace_keys_and_dispatch():
    rec = {"workspace_doc_ids": ["1", "2", "3"]}
    assert rm.workspace_doc_ids(rec) == ["1", "2", "3"]
    assert rm.surfaced_doc_ids(rec, "scaleseek") == {"1", "2", "3"}
    # fallback: reconstruct from bm25_calls when workspace_doc_ids missing
    rec2 = {"bm25_calls": [{"doc_ids": ["1", "2"]}, {"doc_ids": ["2", "9"]}]}
    assert rm.workspace_doc_ids(rec2) == ["1", "2", "9"]


def test_aggregate_optional():
    agg = rm.aggregate_optional([1.0, None, 0.0, 0.5])
    assert agg == {"mean": 0.5, "n_defined": 3, "n_total": 4}
    assert rm.aggregate_optional([None, None]) == {"mean": None, "n_defined": 0, "n_total": 2}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
