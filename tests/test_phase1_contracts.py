from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval.config import ROOT, config_id, load_baselines, resolved_method
from eval import datasets as dataset_module
from eval import prompts as prompt_registry
from eval.datasets import FLASHRAG_DATASETS, _LOCAL_NAME_OVERRIDES, validate_complete_dataset
from eval import search_o1_agent as so1
from eval import search_r1_agent as sr1
from eval.agent import run_rag
from eval.bm25_retriever import BM25Retriever
from eval.qwen3_embedding_retriever import format_query
from eval.qwen3_embedding_retriever import last_token_pool
from eval.e5_retriever import format_query as format_e5_query, masked_mean_pool
from scripts.run_official_baseline import _validate
from eval.browsecomp_plus_judge import BCP_F_JUDGE_PROMPT
from eval.retrievers import merge_ranked_hits
from eval.run_eval import (validate_exact_id_set, validate_resume_row,
                           validate_retriever_manifest)


class FakeRetriever:
    def __init__(self, text="Title\nBody"):
        self.calls = []
        self.text = text

    def retrieve(self, query, top_k=3):
        self.calls.append((query, top_k))
        return [{"doc_id": "d1", "score": 1.0, "text": self.text}]


def _choice(text, *, finish_reason="stop", stop_reason=None):
    return SimpleNamespace(text=text, finish_reason=finish_reason, stop_reason=stop_reason)


class SequenceCompletions:
    def __init__(self, choices):
        self.choices = list(choices)
        self.prompts = []

    def create(self, **kwargs):
        self.prompts.append(kwargs)
        return SimpleNamespace(choices=[self.choices.pop(0)])


class FakeChat:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")])


class FakeTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return "USER_ONLY:" + messages[0]["content"] + ":ASSISTANT"


def test_frozen_global_contract():
    cfg = load_baselines()
    assert cfg["global"]["fixed_bm25"] == {"k1": 1.2, "b": 0.75}
    assert cfg["global"]["ordinary_retrieval_top_k"] == 3
    assert cfg["methods"]["search_r1"]["action_budget"] == 4
    assert cfg["methods"]["search_o1"]["single_hop_search_limit"] == 5
    assert cfg["methods"]["search_o1"]["multi_hop_search_limit"] == 10
    assert cfg["methods"]["agentir"]["retrieval_top_k"] == 5
    assert "ircot" not in cfg["methods"]
    assert resolved_method("rag")["method_config_id"] == config_id(
        {"global": cfg["global"], "method": cfg["methods"]["rag"]})


def test_popqa_is_canonical_complete_split_without_override(monkeypatch):
    assert FLASHRAG_DATASETS["popqa"] == ("popqa", "test")
    assert ("popqa", "test") not in _LOCAL_NAME_OVERRIDES
    assert "popqa_full" not in FLASHRAG_DATASETS
    rows = [{"id": f"popqa_{i}"} for i in range(14_267)]
    monkeypatch.setitem(dataset_module.EXPECTED_NORMALIZED_SHA256, "popqa", "")
    # Compute once, then make the pinned-hash gate explicit for the fixture.
    with pytest.raises(ValueError, match="SHA256"):
        validate_complete_dataset("popqa", rows)
    dataset_module.EXPECTED_NORMALIZED_SHA256.pop("popqa")
    manifest = validate_complete_dataset("popqa", rows)
    assert manifest["count"] == manifest["unique_ids"] == 14_267
    with pytest.raises(ValueError, match="14,267"):
        validate_complete_dataset("popqa", rows[:-1])


def test_bm25_and_qwen_embedding_defaults():
    sig = inspect.signature(BM25Retriever.retrieve)
    assert sig.parameters["top_k"].default == 3
    assert sig.parameters["k1"].default == 1.2
    assert sig.parameters["b"].default == 0.75
    assert format_query("capital of France") == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:capital of France")
    assert format_e5_query("capital of France") == "query: capital of France"


def test_formal_index_manifest_is_bound_to_frozen_backend():
    validate_retriever_manifest("bm25", {
        "backend": "bm25_lucene", "k1": 1.2, "b": 0.75})
    cfg = load_baselines()["global"]["retrievers"]["e5"]
    validate_retriever_manifest("e5", {
        "backend": "e5", "model_id": cfg["repo_id"],
        "model_revision": cfg["revision"], "pooling": "masked_mean",
        "normalize": True})
    with pytest.raises(ValueError, match="violates frozen settings"):
        validate_retriever_manifest("bm25", {
            "backend": "bm25_lucene", "k1": 0.9, "b": 0.4})


def test_shard_merge_equals_global_topk_and_deduplicates():
    shards = [
        [{"doc_id": "b", "score": 0.8}, {"doc_id": "a", "score": 0.7}],
        [{"doc_id": "c", "score": 0.9}, {"doc_id": "a", "score": 0.75}],
    ]
    merged = merge_ranked_hits(shards, 3)
    assert [(h["doc_id"], h["score"]) for h in merged] == [
        ("c", 0.9), ("b", 0.8), ("a", 0.75)]


def test_dense_pooling_contracts():
    np = pytest.importorskip("numpy")

    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        @property
        def shape(self): return self.data.shape
        @property
        def dtype(self): return self.data.dtype
        def __getitem__(self, key):
            if isinstance(key, tuple):
                key = tuple(x.data if isinstance(x, Tensor) else x for x in key)
            return Tensor(self.data[key])
        def __sub__(self, other): return Tensor(self.data - other)
        def __mul__(self, other): return Tensor(self.data * other.data)
        def __truediv__(self, other): return Tensor(self.data / other.data)
        def __eq__(self, other): return Tensor(self.data == other)
        def sum(self, dim=None): return Tensor(self.data.sum(axis=dim))
        def unsqueeze(self, dim): return Tensor(np.expand_dims(self.data, dim))
        def to(self, _dtype): return self
        def clamp(self, min): return Tensor(np.maximum(self.data, min))
        def item(self): return self.data.item()
        def flatten(self): return Tensor(self.data.flatten())
        def tolist(self): return self.data.tolist()

    hidden = Tensor([[[1.0], [3.0], [99.0]], [[4.0], [5.0], [6.0]]])
    mask = Tensor([[1, 1, 0], [1, 1, 1]])
    assert masked_mean_pool(hidden, mask).flatten().tolist() == [2.0, 5.0]
    assert last_token_pool(hidden, mask).flatten().tolist() == [3.0, 6.0]
    left_mask = Tensor([[0, 1, 1], [1, 1, 1]])
    assert last_token_pool(hidden, left_mask).flatten().tolist() == [99.0, 6.0]


def test_prompt_snapshots():
    snapshots = yaml.safe_load((ROOT / "configs/prompt_snapshots.yaml").read_text())["snapshots"]
    values = {
        "search_r1_template": sr1._USER_PROMPT_TEMPLATE,
        "search_o1_single_5": so1._singleqa_instruction(5),
        "search_o1_multi_10": so1._multiqa_instruction(10),
        "search_o1_task": so1._task_instruction_openqa("{question}"),
        "search_o1_rid": so1._reasonchain_instruction("{reasoning}", "{query}", "{documents}"),
        "browsecomp_plus_appendix_f_judge": BCP_F_JUDGE_PROMPT,
    }
    for module_name, logical_name in {
        "prompts.direct:PROMPT": "direct",
        "prompts.rag:PROMPT": "rag",
        "prompts.scaleseek_prompt:PROMPT": "scaleseek_prompt",
        "prompts.scaleseek_prompt_withparams:PROMPT": "scaleseek_prompt_withparams",
    }.items():
        values[module_name] = prompt_registry.load(logical_name)
    assert {key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in values.items()} == snapshots


def test_training_and_eval_share_canonical_scaleseek_prompt():
    from train.dataset import _build_system_prompt
    from prompts.scaleseek_prompt import PROMPT
    assert prompt_registry.load("scaleseek") == PROMPT
    assert _build_system_prompt() == PROMPT


def test_sft_prompt_suite_is_self_contained():
    from prompts import sft_prompts
    assert sft_prompts.PLANNER_USER.startswith("Question: {question}")
    assert "<answer>" in sft_prompts.FINAL_ANSWER_USER


def test_search_r1_fake_loop_uses_checkpoint_template_and_top3():
    completions = SequenceCompletions([
        _choice("<think>x</think><search>alpha", stop_reason="</search>"),
        _choice("<think>done</think><answer>A</answer>", finish_reason="eos"),
    ])
    client = SimpleNamespace(completions=completions)
    retriever = FakeRetriever()
    record = sr1.run_search_r1(
        {"id": "x", "question": "Q", "golden_answers": ["A"]},
        client=client, model="m", retriever=retriever, tokenizer=FakeTokenizer(),
        action_budget=4)
    assert record.prediction == "A"
    assert retriever.calls == [("alpha", 3)]
    assert completions.prompts[0]["temperature"] == 0.7
    assert completions.prompts[0]["prompt"].startswith("USER_ONLY:")
    assert "system" not in completions.prompts[0]["prompt"].lower()


def test_search_o1_fake_loop_selects_singlehop_and_top3():
    completions = SequenceCompletions([
        _choice(f"thinking {so1.BEGIN_SEARCH_QUERY}alpha", stop_reason=so1.END_SEARCH_QUERY),
        _choice(r"therefore \boxed{A}"),
    ])
    chat = FakeChat(["**Final Information**\nEvidence"])
    client = SimpleNamespace(completions=completions,
                             chat=SimpleNamespace(completions=chat))
    retriever = FakeRetriever()
    record = so1.run_search_o1(
        {"id": "x", "question": "Q", "golden_answers": ["A"]},
        client=client, model="m", retriever=retriever, dataset_name="popqa",
        tokenizer=FakeTokenizer())
    assert record.prediction == "A"
    assert retriever.calls == [("alpha", 3)]
    assert "first Nobel Prize" in completions.prompts[0]["prompt"]
    assert completions.prompts[0]["top_p"] == 0.8
    assert completions.prompts[0]["extra_body"] == {"top_k": 20}
    assert chat.requests[0]["temperature"] == 0.7
    assert chat.requests[0]["top_p"] == 0.8
    assert chat.requests[0]["extra_body"] == {"top_k": 20, "repetition_penalty": 1.05}


def test_rag_passes_full_passage_and_backend_neutral_call():
    long_text = "x" * 5000
    retriever = FakeRetriever(long_text)
    chat = FakeChat(["<answer>A</answer>"])
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    record = run_rag({"id": "x", "question": "Q", "golden_answers": ["A"]},
                     client=client, model="m", retriever=retriever)
    assert record.prediction == "A"
    assert retriever.calls == [("Q", 3)]
    assert long_text in chat.requests[0]["messages"][1]["content"]


def test_official_harness_invariants():
    with pytest.raises(ValueError, match="max-turns 300"):
        _validate("dci", ["--model", "Qwen/Qwen3.5-9B"], True)
    _validate("grepseek", [
        "--model", "alireza7/GrepSeek-Qwen3.5-9B-GRPO",
        "--tokenizer", "alireza7/GrepSeek-Qwen3.5-9B-GRPO",
        "--max_assistant_turns", "6", "--max_tokens_per_turn", "0",
        "--tool_max_tokens", "2048", "--temperature", "0.6", "--top_p", "1.0",
    ], True)
    with pytest.raises(ValueError, match="forbids"):
        _validate("rise", ["--limit", "10"], True)


def test_resume_and_full_id_guards():
    provenance = {"method_config_id": "m", "prompt_sha256": "p"}
    row = {**provenance, "id": "a", "dataset_manifest": {"count": 2}}
    validate_resume_row(row, provenance, {"count": 2})
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_resume_row({**row, "prompt_sha256": "old"}, provenance)
    validate_exact_id_set(["a", "b"], {"a", "b"})
    with pytest.raises(ValueError, match="exactly"):
        validate_exact_id_set(["a", "a"], {"a", "b"})


# ---------------------------------------------------------------------------
# RL <-> eval contract. These two configs describe the same agent; when they
# drift, the policy is trained under different rules than it is scored under and
# no result row is attributable. (2026-07-28: turn budget was 6 vs 8, base model
# Qwen3-8B vs the frozen Qwen3.5-9B, and the RL prompt budget was 1024 against a
# 1015-token system prompt.)
# ---------------------------------------------------------------------------

def _grpo_cfg():
    return yaml.safe_load((ROOT / "train/config/grpo_trainer.yaml").read_text())


def test_rl_and_eval_agree_on_turn_budget():
    rl = _grpo_cfg()["actor_rollout_ref"]["rollout"]["multi_turn"]["max_assistant_turns"]
    ev = resolved_method("scaleseek")["method"]["max_assistant_turns"]
    assert rl == ev, f"RL trains with {rl} assistant turns but eval scores with {ev}"


def test_rl_context_budget_is_self_consistent_and_fits_the_prompt():
    r = _grpo_cfg()["actor_rollout_ref"]["rollout"]
    assert r["max_model_len"] == r["prompt_length"] + r["response_length"]
    # The system prompt must fit with real headroom for the question + template.
    # 4 chars/token is a deliberate over-estimate of Qwen3.5 on English prose
    # (measured: 4630 chars -> 1015 tokens, i.e. 4.56), so this stays a lower
    # bound on the true budget and cannot pass a genuinely too-small setting.
    n_chars = len(prompt_registry.load("scaleseek"))
    assert r["prompt_length"] >= n_chars / 4 + 256, (
        f"prompt_length={r['prompt_length']} leaves no room for the "
        f"{n_chars}-char ScaleSeek system prompt plus a question")


def test_rl_base_model_defaults_to_the_frozen_generator():
    path = _grpo_cfg()["actor_rollout_ref"]["model"]["path"]
    frozen = load_baselines()["global"]["generator"]["repo_id"]
    assert frozen in path, (
        f"RL falls back to {path!r}, but prompt baselines run on {frozen!r}; "
        "a different base makes the comparison unattributable")


def test_scaleseek_eval_params_come_from_the_frozen_config():
    """run_eval must not hardcode scaleseek's loop/decoding budgets."""
    import inspect as _inspect
    from eval import run_eval as _re
    src = _inspect.getsource(_re.run_eval)
    head, _, _ = src.partition('elif args.agent == "rag"')
    _, _, scaleseek_block = head.partition('if args.agent == "scaleseek"')
    for literal in ("max_turns=8", "max_tokens=8192"):
        assert literal not in scaleseek_block, (
            f"{literal} is hardcoded in the scaleseek dispatch; it must be read "
            "from configs/baselines.yaml or the frozen config is decorative")


# ---------------------------------------------------------------------------
# The 1/10 scope exception (TASK.md "评测规模上限"). It is only defensible
# because the subset is a pure function of the ids -- no seed, no state file --
# so these tests pin exactly that property.
# ---------------------------------------------------------------------------

def test_capped_subset_is_deterministic_and_order_independent():
    from eval.subsets import select, subset_manifest, subset_size
    ex = [{"id": f"popqa_test_{i}"} for i in range(14267)]
    assert subset_size(14267) == 1500                     # min(N, cap)

    a = select(ex, dataset="popqa")
    shuffled = list(reversed(ex))
    b = select(shuffled, dataset="popqa")
    assert {e["id"] for e in a} == {e["id"] for e in b}, \
        "subset must not depend on the order rows sit in on disk"
    assert (subset_manifest(a, dataset="popqa", total=14267)["subset_ids_sha256"]
            == subset_manifest(b, dataset="popqa", total=14267)["subset_ids_sha256"])


def test_small_datasets_run_complete_under_the_cap():
    """A cap, not a ratio: 1/10 of Bamboogle would be 13 questions."""
    from eval.subsets import select, subset_manifest, subset_size, is_capped
    for total, name in ((125, "bamboogle"), (830, "browsecomp_plus"), (1500, "edge")):
        ex = [{"id": f"{name}_{i}"} for i in range(total)]
        sub = select(ex, dataset=name)
        assert len(sub) == total == subset_size(total), f"{name} must run complete"
        assert not is_capped(total)
        man = subset_manifest(sub, dataset=name, total=total)
        assert man["eval_scope"] == "full_split", (
            f"{name} ran every question; calling it a subset row would understate it")
    # one over the cap and it binds
    ex = [{"id": f"x_{i}"} for i in range(1501)]
    assert len(select(ex, dataset="x")) == 1500
    assert subset_manifest(select(ex, dataset="x"), dataset="x",
                           total=1501)["eval_scope"] == "capped_1500"


def test_capped_subset_is_dataset_scoped():
    """Different datasets must not select 'the same' positional rows."""
    from eval.subsets import select
    ex = [{"id": str(i)} for i in range(1000)]
    a = {e["id"] for e in select(ex, dataset="popqa", cap=100)}
    b = {e["id"] for e in select(ex, dataset="hotpotqa", cap=100)}
    assert a != b


def test_subset_verify_rejects_drift():
    from eval.subsets import select, subset_manifest, verify
    ex = [{"id": f"q{i}"} for i in range(5000)]
    sub = select(ex, dataset="popqa")
    man = subset_manifest(sub, dataset="popqa", total=5000)
    ids = [e["id"] for e in sub]
    assert verify(ids, man)[0] is True
    assert verify(ids[:-1], man)[0] is False           # truncated run
    assert verify(ids + ["extra"], man)[0] is False    # contaminated run


def test_only_the_three_call_bound_harnesses_evaluate_on_the_capped_set():
    """Guard the blast radius of the scope exception.

    Two distinct uses of the subset file must not be conflated:
      * EVALUATING on it -- scoring a matrix row at n=1,427. TASK.md sanctions
        exactly three methods, and each must scope-check with --expect-ids.
      * merely READING the question list to build an asset (p1_rise_toc selects
        which articles to section from it). That produces no result row, so it
        does not widen the exception.
    """
    evaluators = {"p2_dci.sbatch", "p2_dr_dci.sbatch", "p2_rise.sbatch",
                  # Re-normalizes/re-scores finished runs of exactly the
                  # sanctioned methods (its case arms are dci|dr_dci|rise with
                  # --expect-ids, plus full-split grepseek); it starts no new
                  # capped evaluation. Predates this registration on main —
                  # the test was not run when 418941d added the file.
                  "p2_official_recover.sbatch"}
    non_evaluating = {"p0_assets.sbatch",      # generates the subset
                      "p1_rise_toc.sbatch"}    # picks articles to section
    for f in sorted((ROOT / "sbatch").glob("*.sbatch")):
        text = f.read_text()
        if "cap1500" not in text:
            continue
        if f.name in non_evaluating:
            assert "compute_metrics.py" not in text, (
                f"{f.name} is registered as non-evaluating yet scores a run; "
                "if it now produces a matrix row it needs TASK.md sanction")
            continue
        assert f.name in evaluators, (
            f"{f.name} evaluates the capped subset but TASK.md only sanctions "
            "DCI / DR-DCI / RISE")


def test_capped_harnesses_enforce_the_id_set():
    """A subset run must be scope-checked, or the exception loses its guarantee."""
    for name in ("p2_dci.sbatch", "p2_dr_dci.sbatch", "p2_rise.sbatch"):
        text = (ROOT / "sbatch" / name).read_text()
        assert "cap1500.jsonl" in text, f"{name} should read the subset file"
        assert "--expect-ids" in text, \
            f"{name} computes metrics without --expect-ids; a truncated run " \
            "would be reported as the sanctioned subset"
