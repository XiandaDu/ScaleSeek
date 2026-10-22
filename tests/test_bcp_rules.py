"""BCP-specific rules: official answer line, 512-token display truncation, single-call cap."""
import pytest

from scaleseek.config import load_base
from scaleseek.methods.agentir import exact_answer


def test_exact_answer_line():
    text = ("Explanation: Poe edited magazines [12].\nExact Answer: editor\nConfidence: 80%")
    assert exact_answer(text) == "editor"
    assert exact_answer("**Exact Answer:** Queen Arwa University\n**Confidence:** 90%") == "Queen Arwa University"
    assert exact_answer("exact answer: a\nExact Answer: b") == "b"            # the last one
    assert exact_answer("no such line") is None and exact_answer(None) is None


def test_bcp_display_truncation_matches_official_rule():
    base = load_base()
    trunc = base["corpora"]["bcp"]["display_truncation"]
    assert trunc["max_tokens"] == 512 and base["corpora"]["wiki18"]["display_truncation"] is None
    try:
        from scaleseek.retrieval.service import display_function
        show = display_function(base, "bcp")
    except Exception as exc:  # tokenizer not downloaded on this machine
        pytest.skip(f"tokenizer unavailable: {exc}")
    from transformers import AutoTokenizer
    from scaleseek.hub import snapshot_dir
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(base, trunc["tokenizer"]["model"],
                                                         trunc["tokenizer"]["revision"])))
    long = "---\ntitle: T\n---\n" + "word " * 2000
    ids = tok.encode(long, add_special_tokens=False)
    assert show(long) == tok.decode(ids[:512], skip_special_tokens=True)   # official base.py rule
    assert show("short text") == "short text"
    assert display_function(base, "wiki18")("x" * 10000) == "x" * 10000


def test_single_call_cap_finish_reason():
    from fakes import make_ctx
    from scaleseek.config import load_method, load_prompts
    from scaleseek.methods.direct import Direct
    P = load_prompts(load_method("direct", load_base()))
    ctx = make_ctx("direct", P, [("thinking...", "length", None)], dataset="browsecomp_plus")
    rec = Direct(ctx).run_one({"id": "1", "question": "q?", "golden_answers": ["a"]})
    llm = [e for e in rec["trajectory"] if e["type"] == "llm"][0]
    assert llm["max_tokens"] == 10000 and rec["finish_reason"] == "max_tokens_cap"
