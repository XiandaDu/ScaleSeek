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
