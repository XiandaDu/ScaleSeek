import copy

import pytest

from scaleseek import config as C


def test_all_configs_load():
    base = C.load_base()
    methods = C.load_all_methods(base)
    assert set(methods) == {"direct", "rag", "search_r1_7b", "search_r1_14b", "search_o1",
                            "grepseek", "agentir", "bcp_judge"}
    text = (C.config_dir() / "base.yaml").read_text()
    assert "PIN_AT_DOWNLOAD" not in text
    for p in (C.config_dir() / "methods").glob("*.yaml"):
        assert "PIN_AT_DOWNLOAD" not in p.read_text()


def test_matrix_has_15_runs_per_dataset():
    base = C.load_base()
    for d in base["datasets"]:
        assert len(C.enumerate_runs(base, d)) == 15


def test_unknown_and_missing_keys_are_errors():
    base = C.load_base()
    raw = C.load_yaml(C.config_dir() / "methods" / "rag.yaml")
    bad = copy.deepcopy(raw)
    bad["server"]["top_k"] = 20
    with pytest.raises(C.ConfigError, match="unknown key"):
        C._check(C.METHOD_SCHEMAS["rag"], bad, "rag")
    bad = copy.deepcopy(raw)
    del bad["client"]["concurrency"]
    with pytest.raises(C.ConfigError, match="missing key"):
        C._check(C.METHOD_SCHEMAS["rag"], bad, "rag")


def test_decoding_profiles_match_baselines():
    base = C.load_base()
    assert base["profiles"]["wiki"] == {"temperature": 0.6, "top_p": 1.0, "max_assistant_turns": 6,
                                        "max_model_len": 16384, "single_call_max_tokens": None}
    # BCP single-call cap: the BrowseComp-Plus official Qwen client default (decision 2026-10-14)
    assert base["profiles"]["bcp"] == {"temperature": 0.6, "top_p": 1.0, "max_assistant_turns": 16,
                                       "max_model_len": 131072, "single_call_max_tokens": 10000}
    assert base["retrievers"]["bm25"]["k1"] == 1.2 and base["retrievers"]["bm25"]["b"] == 0.75
    for r in ("e5", "qwen3_emb_4b"):
        assert base["retrievers"][r]["hnsw"] == {"M": 32, "ef_construction": 128, "ef_search": 128}


def test_every_search_method_uses_top3():
    base = C.load_base()
    for name, m in C.load_all_methods(base).items():
        if m["retrieval"]:
            assert m["retrieval"]["top_k"] == 3, name


def test_config_id_depends_on_parameters_only():
    base = C.load_base()
    a = C.resolve_run(base, "rag", "popqa", "bm25", "formal")
    b = C.resolve_run(base, "rag", "popqa", "bm25", "smoke")
    assert a.config_id == b.config_id and a.run_dir != b.run_dir
    c = C.resolve_run(base, "rag", "popqa", "e5", "formal")
    assert c.config_id != a.config_id


def test_prompts_are_frozen():
    base = C.load_base()
    for name, m in C.load_all_methods(base).items():
        C.load_prompts(m)
