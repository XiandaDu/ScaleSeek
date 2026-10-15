import math

from scaleseek.metrics import exact_match, f1_score, mcnemar_exact, normalize_answer, proportion_ci


def test_normalization_and_em():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert exact_match("the Beatles", ["Beatles"])
    assert not exact_match(None, ["x"])
    assert not exact_match("", ["x"])


def test_f1_max_over_gold():
    assert f1_score("John Lennon", ["Lennon", "Paul"]) == 2 * 0.5 * 1 / 1.5
    assert f1_score("x", ["y"]) == 0.0


def test_ci_matches_baselines_md():
    ci = proportion_ci(250, 500, 1.96)
    assert abs(ci["ci95"] - 0.0438) < 1e-3


def test_mcnemar_exact():
    a = {str(i): i < 10 for i in range(20)}
    b = {str(i): False for i in range(20)}
    r = mcnemar_exact(a, b)
    assert r["only_a"] == 10 and r["only_b"] == 0
    assert math.isclose(r["p_value"], 2 * 0.5 ** 10)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_only_a_finished_answer_is_scored():
    from scaleseek.metrics import final_answer, run_metrics
    done = {"id": "a", "prediction": "x", "finish_reason": "answer", "golden_answers": ["x"],
            "errors": {"parse": 0, "tool": 0, "api": 0}, "n_turns": 2, "n_retrievals": 1}
    cut = dict(done, id="b", finish_reason="max_turns")          # answer seen mid-trajectory
    assert final_answer(done) == "x" and final_answer(cut) is None
    m = run_metrics([done, cut], 1.96, 0.0)
    assert m["per_question_em"] == {"a": True, "b": False} and m["answered_rate"] == 0.5
