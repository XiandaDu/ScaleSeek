"""eval/evidence.py: support extraction and title-level coverage."""
from eval.evidence import coverage, doc_title, extract_evidence, norm_title


def _doc(title, body):
    return {"text": f'"{title}"\n{body}'}


def test_extract_hotpot_titles_dedup():
    row = {"metadata": {"supporting_facts": {"title": ["A", "B", "A"], "sent_id": [0, 0, 1]}}}
    assert extract_evidence("hotpotqa", row) == {"support_titles": ["A", "B"]}


def test_extract_musique_titles_and_sub_answers():
    row = {"metadata": {"question_decomposition": [
        {"answer": "X", "support_paragraph": {"title": "T1"}},
        {"answer": "Y", "support_paragraph": {"title": "T2"}}]}}
    assert extract_evidence("musique", row) == {"support_titles": ["T1", "T2"], "sub_answers": ["X", "Y"]}


def test_nq_has_no_evidence():
    assert extract_evidence("nq", {"id": "1"}) == {}


def test_doc_title_strips_quotes():
    assert doc_title('"Bat Out of Hell"\nBat Out of Hell is') == "Bat Out of Hell"
    assert norm_title('  "The  Thing" ') == "the thing"


def test_coverage_counts_titles_and_gold():
    docs = [_doc("Arthur's Magazine", "an American periodical"), _doc("First for Women", "a magazine")]
    c = coverage({"support_titles": ["Arthur's Magazine", "First for Women", "Radio City"]},
                 ["First for Women"], docs)
    assert c["n_support"] == 3 and c["n_support_found"] == 2
    assert abs(c["ret_coverage"] - 2 / 3) < 1e-9
    assert c["gold_in_ws"] == 1.0


def test_coverage_without_titles_falls_back_to_gold_in_text():
    docs = [_doc("Paris", "Paris is the capital of France")]
    assert coverage({}, ["France"], docs)["ret_coverage"] == 1.0
    assert coverage({}, ["Berlin"], docs)["ret_coverage"] == 0.0


def test_coverage_empty_workspace():
    c = coverage({"support_titles": ["A"]}, ["x"], [])
    assert c["ret_coverage"] == 0.0 and c["gold_in_ws"] == 0.0


# --- v3 grep reward helpers ------------------------------------------------
from eval.evidence import any_phrase_in, contains_phrase, pattern_is_substantive


def test_contains_phrase_is_token_level_not_substring():
    assert contains_phrase("There is nothing here", "nothing")
    assert not contains_phrase("There is nothing here", "no")     # 'no' inside 'nothing'
    assert contains_phrase("the answer is no", "no")
    assert not contains_phrase("born in 19468", "1946")


def test_contains_phrase_multiword_and_punctuation():
    assert contains_phrase("located in Los Angeles County, California", "Los Angeles County")
    assert not contains_phrase("Los County Angeles", "Los Angeles County")
    assert any_phrase_in("won the Nobel Prize", ["Pulitzer", "Nobel Prize"])


def test_catch_all_patterns_earn_no_grep_credit():
    for p in (".", ".*", "\\w+", "a|e", ""):
        assert not pattern_is_substantive(p)
    for p in ("Nobel", "19.6", "Los Angeles"):
        assert pattern_is_substantive(p)
