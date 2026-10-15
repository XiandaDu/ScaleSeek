"""Method loops against a scripted fake LLM: prompt construction, stops, limits."""
from fakes import make_ctx

from scaleseek.config import load_base, load_method, load_prompts


def _prompts(name):
    return load_prompts(load_method(name, load_base()))


def test_search_r1_loop_matches_infer_py():
    P = _prompts("search_r1_7b")
    script = [("<think>need info</think>\n<search> capital of France </search>", "stop", "</search>"),
              ("<think>found</think>\n<answer> Paris </answer>", "stop", None)]
    ctx = make_ctx("search_r1_7b", P, script)
    from scaleseek.methods.search_r1 import SearchR1
    rec = SearchR1(ctx).run_one({"id": "1", "question": " What is the capital of France", "golden_answers": ["Paris"]})
    assert rec["finish_reason"] == "answer" and rec["prediction"] == "Paris"
    assert rec["n_turns"] == 2 and rec["n_retrievals"] == 1
    assert ctx.retrieval.queries == [" capital of France "]       # get_query does not strip
    first, second = ctx.llm.prompts[0][0], ctx.llm.prompts[1][0]
    assert "Question: What is the capital of France?\n" in first   # strip + '?'
    expect = P["search_template"].format(
        output_text=script[0][0],
        search_results=P["passages2string"]([{"document": {"contents": h["contents"]}}
                                             for h in ctx.retrieval.search(" capital of France ", 3)]))
    assert second == first + expect
    kw = ctx.llm.prompts[0][1]
    assert kw["stop_token_ids"] == [151645, 151643] and "</search>" in kw["stop"]


def test_search_r1_stops_at_max_turns():
    P = _prompts("search_r1_7b")
    script = [("<search> q </search>", "stop", "</search>")] * 6
    ctx = make_ctx("search_r1_7b", P, script)
    from scaleseek.methods.search_r1 import SearchR1
    rec = SearchR1(ctx).run_one({"id": "1", "question": "q?", "golden_answers": ["a"]})
    assert rec["finish_reason"] == "max_turns" and rec["n_turns"] == 6 and rec["prediction"] is None


def test_direct_parses_answer_after_think():
    P = _prompts("direct")
    ctx = make_ctx("direct", P, [("hmm <answer> no </answer>\n</think>\n\n<answer> Yes </answer>", "stop", None)])
    from scaleseek.methods.direct import Direct
    rec = Direct(ctx).run_one({"id": "1", "question": "q?", "golden_answers": ["yes"]})
    assert rec["prediction"] == "Yes" and rec["finish_reason"] == "answer"
    ctx = make_ctx("direct", P, [("still thinking", "length", None)])
    rec = Direct(ctx).run_one({"id": "1", "question": "q?", "golden_answers": ["yes"]})
    assert rec["prediction"] is None and rec["finish_reason"] == "context_exhausted"


def test_rag_passes_full_passages():
    P = _prompts("rag")
    ctx = make_ctx("rag", P, [("</think><answer> x </answer>", "stop", None)])
    from scaleseek.methods.rag import RAG
    rec = RAG(ctx).run_one({"id": "1", "question": "who?", "golden_answers": ["x"]})
    prompt = ctx.llm.prompts[0][0]
    assert 'Doc 1(Title: "Title 0") Text about who? 0.\n' in prompt
    assert rec["n_retrievals"] == 1 and rec["prediction"] == "x"


def test_search_o1_limit_and_repeat_messages():
    P = _prompts("search_o1")
    BQ, EQ = P["begin_search_query"], P["end_search_query"]
    rid = ("</think>\n**Final Information**\n\nParis is the capital.", "stop", None)
    script = [(f"think {BQ}capital{EQ}", "stop", EQ), rid,
              (f"again {BQ}capital{EQ}", "stop", EQ),
              ("maybe \\boxed{Rome}</think>\n\nso \\boxed{Paris}", "stop", None)]
    ctx = make_ctx("search_o1", P, script)
    from scaleseek.methods.search_o1 import SearchO1
    rec = SearchO1(ctx).run_one({"id": "1", "question": "Capital of France?", "golden_answers": ["Paris"]})
    assert rec["prediction"] == "Paris" and rec["finish_reason"] == "answer"
    final = ctx.llm.prompts[-1][0]
    assert "<|begin_search_result|>Paris is the capital.<|end_search_result|>" in final
    assert "You have searched this query. Please refer to previous results." in final
    assert ctx.retrieval.queries == ["capital"]
    rid_prompt = ctx.llm.prompts[1][0]
    assert '**Web Page 1:**\n{\n  "id": 1,\n  "title": "\\"Title 0\\"",' in rid_prompt
    first = ctx.llm.prompts[0][0]
    assert "limited to 5." in first            # singleqa branch, limit = turns - 1


def test_search_o1_fill_is_single_pass():
    from scaleseek.methods.search_o1 import fill
    assert fill("{A}{x}{B}", A="<", x="{B}", B=">") == "<{B}>"


def test_title_rules():
    from scaleseek.corpus import split_title
    assert split_title('"Aaron"\nAaron is a prophet.', "first_line") == ('"Aaron"', "Aaron is a prophet.")
    page = "---\ntitle: Arwa University\ndate: 2002-05-13\n---\nBody text\nmore"
    assert split_title(page, "bcp_front_matter") == ("Arwa University", "Body text\nmore")
    assert split_title("plain\nbody", "bcp_front_matter") == ("plain", "body")


def test_search_o1_answer_only_after_think():
    P = _prompts("search_o1")
    ctx = make_ctx("search_o1", P, [("the answer is \\boxed{Paris}, but let me think", "stop", None)])
    from scaleseek.methods.search_o1 import SearchO1
    rec = SearchO1(ctx).run_one({"id": "1", "question": "Capital of France?", "golden_answers": ["Paris"]})
    assert rec["prediction"] is None and rec["finish_reason"] == "no_answer"   # thinking never closed
