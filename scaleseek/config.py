"""Configuration: strict loading, validation and run resolution.

Two files make a run: configs/base.yaml (shared) and configs/methods/<name>.yaml.  The
schemas below are EXACT: a missing key and an unknown key are both errors, so a typo can
never silently fall back to something.  Nothing in this package has a default value for
an experimental parameter -- if code needs a number, it reads it from a validated config.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .utils import PipelineError, canonical_json, repo_root, sha256_json, sha256_text


class ConfigError(PipelineError):
    """The configuration cannot produce a meaningful run."""


# ============================================================================= schema DSL
class Nullable:
    def __init__(self, inner: Any):
        self.inner = inner


class ListOf:
    def __init__(self, inner: Any, min_len: int = 0):
        self.inner, self.min_len = inner, min_len


class MapOf:
    """Arbitrary (string) keys, every value matching `inner`."""

    def __init__(self, inner: Any):
        self.inner = inner


class OneOf:
    def __init__(self, *values: Any):
        self.values = values


class Pattern:
    def __init__(self, regex: str, what: str):
        self.regex, self.what = re.compile(regex), what


TEXT = Pattern(r"(?s)^.*$", "a string (may be empty)")
SHA1_HEX = Pattern(r"^[0-9a-f]{40}$", "a 40-hex git/HF commit (never a branch name)")
SHA256_HEX = Pattern(r"^[0-9a-f]{64}$", "a sha256 hex digest")
PROFILES = ("wiki", "bcp")


def PerProfile(inner: Any) -> dict:
    return {p: inner for p in PROFILES}


ANY = object()   # validated later by a source- or kind-specific schema


def _check(node: Any, value: Any, where: str) -> None:
    if node is ANY:
        return
    if node is None:                 # the key must be present and null
        if value is not None:
            raise ConfigError(f"{where}: must be null for this method")
        return
    if isinstance(node, Nullable):
        if value is not None:
            _check(node.inner, value, where)
        return
    if isinstance(node, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        unknown = sorted(set(value) - set(node))
        missing = sorted(set(node) - set(value))
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {unknown}; valid: {sorted(node)}")
        if missing:
            raise ConfigError(f"{where}: missing key(s) {missing} (there are no defaults)")
        for k, sub in node.items():
            _check(sub, value[k], f"{where}.{k}")
        return
    if isinstance(node, MapOf):
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{where}: expected a non-empty mapping")
        for k, v in value.items():
            _check(node.inner, v, f"{where}.{k}")
        return
    if isinstance(node, ListOf):
        if not isinstance(value, list) or len(value) < node.min_len:
            raise ConfigError(f"{where}: expected a list of >= {node.min_len} items")
        for i, v in enumerate(value):
            _check(node.inner, v, f"{where}[{i}]")
        return
    if isinstance(node, OneOf):
        if value not in node.values:
            raise ConfigError(f"{where}: {value!r} is not one of {list(node.values)}")
        return
    if isinstance(node, Pattern):
        if not isinstance(value, str) or not node.regex.match(value):
            raise ConfigError(f"{where}: {value!r} must be {node.what}")
        return
    if node is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return
    if node is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return
    if node is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return
    if node is str:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
        return
    raise AssertionError(f"bad schema node at {where}: {node!r}")


# ============================================================================= base.yaml
HNSW = {"M": int, "ef_construction": int, "ef_search": int}

BASE_SCHEMA = {
    "paths": {k: str for k in ("root", "hf_home", "repos", "data", "indexes", "runs",
                               "smoke", "agentir_python")},
    "sampling": {"seed": int, "max_questions": int},
    "profiles": PerProfile({"temperature": float, "top_p": float, "presence_penalty": float,
                            "max_assistant_turns": int, "max_model_len": int,
                            "single_call_max_tokens": Nullable(int)}),
    "datasets": MapOf(ANY),          # per-source schemas below
    "corpora": MapOf(ANY),
    "retrievers": MapOf(ANY),
    "retrieval_check": {"dataset": str, "n_queries": int, "repeats": int, "top_k": int},
    "reporting": {"ci_z": float, "significance_alpha": float},
    "runtime": {"server_start_timeout_s": float, "retriever_start_timeout_s": float},
    "smoke": {"datasets": ListOf(str, 1), "n_questions": int, "head_docs": MapOf(int),
              "gold_hits_per_answer": int, "encode_shards": int},
}

DATASET_SCHEMAS = {
    "flashrag": {"source": str, "hf_repo": str, "revision": SHA1_HEX, "file": str,
                 "split": str, "expected_rows": int, "corpus": str,
                 "profile": OneOf(*PROFILES), "search_o1_instruction": OneOf("singleqa", "multiqa")},
    "browsecomp_plus": {"source": str, "hf_repo": str, "revision": SHA1_HEX, "file": str,
                        "split": str, "expected_rows": int, "corpus": str,
                        "profile": OneOf(*PROFILES),
                        "search_o1_instruction": OneOf("singleqa", "multiqa"), "canary": str},
}

DISPLAY_TRUNCATION = {"tokenizer": {"model": str, "revision": SHA1_HEX}, "max_tokens": int}

CORPUS_SCHEMAS = {
    "wiki18": {"source": str, "hf_repo": str, "revision": SHA1_HEX, "file": str,
               "expected_docs": int, "title_rule": OneOf("first_line"),
               "display_truncation": Nullable(DISPLAY_TRUNCATION)},
    "browsecomp_plus_corpus": {"source": str, "hf_repo": str, "revision": SHA1_HEX,
                               "file": str, "expected_docs": int,
                               "title_rule": OneOf("first_line", "bcp_front_matter"),
                               "display_truncation": Nullable(DISPLAY_TRUNCATION)},
}

RETRIEVER_SCHEMAS = {
    "bm25": {"kind": str, "k1": float, "b": float, "index_threads": int},
    "dense": {"kind": str, "model": str, "revision": SHA1_HEX, "query_prefix": TEXT,
              "doc_prefix": TEXT, "pooling": OneOf("mean", "last_token"),
              "normalize": bool, "query_max_length": int, "doc_max_length": int,
              "encode_dtype": OneOf("float16", "bfloat16", "float32"),
              "encode_batch_tokens": int, "hnsw": HNSW},
    "tevatron_flat": {"kind": str, "model": str, "revision": SHA1_HEX, "pooling": str,
                      "normalize": bool, "passage_max_len": int, "passage_prefix": TEXT,
                      "per_device_eval_batch_size": int, "fp16": bool},
}

# ============================================================================= methods
PROMPT_REF = {"constant": str, "sha256": str}
GENERATOR = {"model": str, "revision": SHA1_HEX}
OFFICIAL_REPO = {"name": str, "url": str, "commit": SHA1_HEX}
SERVER = {"served_model_name": str, "tensor_parallel_size": int,
          "gpu_memory_utilization": float, "max_num_seqs": int, "seed": int,
          "reasoning_parser": Nullable(str), "tool_call_parser": Nullable(str),
          "enable_auto_tool_choice": bool, "language_model_only": bool,
          "trust_remote_code": bool}
CLIENT = {"concurrency": int, "request_timeout_s": float, "max_retries": int,
          "retry_backoff_s": float}
RETRIEVAL = {"retrievers": ListOf(str, 1), "top_k": int}


def _method_schema(extra: dict, server: dict = SERVER, retrieval: Any = None) -> dict:
    schema = {"method": str, "display_name": str, "generator": GENERATOR, "server": server,
              "retrieval": retrieval}
    schema.update(extra)
    return schema


METHOD_SCHEMAS = {
    "direct": _method_schema({"client": CLIENT, "prompts": MapOf(PROMPT_REF)}),
    "rag": _method_schema({"client": CLIENT, "prompts": MapOf(PROMPT_REF)}, retrieval=RETRIEVAL),
    "search_r1": _method_schema({"official_repo": OFFICIAL_REPO, "client": CLIENT,
                                 "prompts": MapOf(PROMPT_REF)}, retrieval=RETRIEVAL),
    "search_o1": _method_schema({"official_repo": OFFICIAL_REPO, "client": CLIENT,
                                 "prompts": MapOf(PROMPT_REF),
                                 "search_o1": {"max_doc_len": int}}, retrieval=RETRIEVAL),
    "grepseek": _method_schema({"official_repo": OFFICIAL_REPO,
                                "harness": {"max_tokens_per_turn": int,
                                            "tool_max_tokens": PerProfile(int),
                                            "tool_timeout": float, "parallel": int,
                                            "engine_mode": OneOf("none"),
                                            "request_timeout_s": float, "max_retries": int}}),
    "agentir": _method_schema({"official_repo": OFFICIAL_REPO,
                               "agent": {"scaffold": OneOf("oss_client"),
                                         "task_prefix": str, "reasoning_aware": bool,
                                         "normalize": bool, "pooling": str,
                                         "torch_dtype": OneOf("float16", "bfloat16", "float32"),
                                         "max_length": int, "snippet_max_tokens": int,
                                         "snippet_tokenizer": GENERATOR, "num_threads": int,
                                         "reasoning_effort": OneOf("low", "medium", "high"),
                                         "request_timeout_s": float, "max_retries": int}},
                              retrieval=RETRIEVAL),
    "bcp_judge": _method_schema({"official_repo": OFFICIAL_REPO, "client": CLIENT,
                                 "decoding": {"temperature": float, "top_p": float,
                                              "max_new_tokens": int},
                                 "prompts": MapOf(PROMPT_REF)},
                                server={**SERVER, "max_model_len": int}),
}

# Prompt slots each method must declare (and nothing else).
REQUIRED_PROMPTS = {
    "direct": {"user"},
    "rag": {"user", "passage"},
    "search_r1": {"user", "search_template", "stop_sequences", "eos_token_ids",
                  "get_query", "passages2string"},
    "search_o1": {"singleqa", "multiqa", "task", "reason_in_documents", "extract_answer",
                  "begin_search_query", "end_search_query", "begin_search_result",
                  "end_search_result", "extract_between", "limit_message",
                  "repeat_message", "result_message", "step_line", "web_page_header"},
    "bcp_judge": {"grader", "create_judge_prompt", "parse_judge_response"},
}


# ============================================================================= loading
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, where: str) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str) and "${" in value:
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if not os.environ.get(name):
                raise ConfigError(f"{where}: environment variable {name} is not set "
                                  "(did you `source setup_env.sh`?)")
            return os.environ[name]
        return _VAR.sub(sub, value)
    return value


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_dir() -> Path:
    return repo_root() / "configs"


def load_base(path: str | Path | None = None) -> dict:
    path = Path(path) if path else config_dir() / "base.yaml"
    raw = load_yaml(path)
    _check(BASE_SCHEMA, raw, "base")
    for name, ds in raw["datasets"].items():
        _check(DATASET_SCHEMAS[ds["source"]], ds, f"base.datasets.{name}")
    for name, c in raw["corpora"].items():
        _check(CORPUS_SCHEMAS[c["source"]], c, f"base.corpora.{name}")
    for name, r in raw["retrievers"].items():
        _check(RETRIEVER_SCHEMAS[r["kind"]], r, f"base.retrievers.{name}")
    raw["paths"] = _expand(raw["paths"], "base.paths")
    return raw


def method_names() -> list[str]:
    return sorted(p.stem for p in (config_dir() / "methods").glob("*.yaml"))


def load_method(name: str, base: dict) -> dict:
    path = config_dir() / "methods" / f"{name}.yaml"
    raw = load_yaml(path)
    kind = raw["method"]
    _check(METHOD_SCHEMAS[kind], raw, f"methods/{name}")
    if "prompts" in raw:
        got, want = set(raw["prompts"]), REQUIRED_PROMPTS[kind]
        if got != want:
            raise ConfigError(f"methods/{name}.prompts: expected exactly {sorted(want)}, "
                              f"got {sorted(got)}")
    if kind == "agentir":
        # BCP documents must look the same to every agent (BASELINES.md §8): the official
        # AgentIR snippet rule has to equal the corpus display truncation.
        for cname, c in base["corpora"].items():
            t = c["display_truncation"]
            if t and (t["tokenizer"] != raw["agent"]["snippet_tokenizer"]
                      or t["max_tokens"] != raw["agent"]["snippet_max_tokens"]):
                raise ConfigError(f"methods/{name}.agent snippet settings differ from "
                                  f"base.corpora.{cname}.display_truncation")
    raw["_name"] = name
    raw["_path"] = str(path)
    return raw


def load_all_methods(base: dict) -> dict[str, dict]:
    return {n: load_method(n, base) for n in method_names()}


# ============================================================================= prompts
def resolve_constant(dotted: str) -> Any:
    module, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def prompt_fingerprint(obj: Any) -> str:
    """sha256 of a prompt object: text as UTF-8, lists as canonical JSON, functions by
    their exact source text."""
    if isinstance(obj, str):
        return sha256_text(obj)
    if isinstance(obj, (list, tuple)):
        return sha256_text(canonical_json(list(obj)))
    if callable(obj):
        return sha256_text(inspect.getsource(obj))
    raise ConfigError(f"unsupported prompt object {type(obj).__name__}")


def load_prompts(method_cfg: dict) -> dict[str, Any]:
    """Resolve every prompt slot and check it against the frozen sha256 in the yaml."""
    out, bad = {}, []
    for slot, ref in method_cfg.get("prompts", {}).items():
        obj = resolve_constant(ref["constant"])
        fp = prompt_fingerprint(obj)
        if fp != ref["sha256"]:
            bad.append(f"  {slot}: yaml {ref['sha256']} != actual {fp}  ({ref['constant']})")
        out[slot] = obj
    if bad:
        raise ConfigError(f"methods/{method_cfg['_name']}: prompt hash mismatch -- a frozen "
                          "prompt changed (or the yaml was never filled):\n" + "\n".join(bad))
    return out


# ============================================================================= run spec
@dataclasses.dataclass
class RunSpec:
    mode: str                 # "formal" | "smoke"
    dataset: str
    method_name: str          # yaml stem, e.g. "search_r1_7b"
    retriever: str | None
    base: dict
    method: dict
    profile: dict
    run_name: str
    run_dir: Path
    config_id: str

    @property
    def kind(self) -> str:
        return self.method["method"]

    @property
    def dataset_cfg(self) -> dict:
        return self.base["datasets"][self.dataset]

    @property
    def corpus(self) -> str:
        return self.dataset_cfg["corpus"]

    @property
    def profile_name(self) -> str:
        return self.dataset_cfg["profile"]


def resolve_run(base: dict, method_name: str, dataset: str, retriever: str | None,
                mode: str) -> RunSpec:
    method = load_method(method_name, base)
    allowed = method["retrieval"]["retrievers"] if method["retrieval"] else [None]
    if retriever not in allowed:
        raise ConfigError(f"{method_name}: --retriever must be one of {allowed}, got {retriever!r}")
    profile_name = base["datasets"][dataset]["profile"]
    profile = dict(base["profiles"][profile_name])
    run_name = method_name + (f"__{retriever}" if retriever else "")
    root = Path(base["paths"]["runs" if mode == "formal" else "smoke"])
    run_dir = root / dataset / run_name
    # Everything that can change an output, and nothing that cannot (paths are excluded).
    identity = {
        "method": {k: v for k, v in method.items() if not k.startswith("_")},
        "profile": profile,
        "dataset": base["datasets"][dataset],
        "corpus": base["corpora"][base["datasets"][dataset]["corpus"]],
        "retriever": base["retrievers"][retriever] if retriever else None,
        "sampling": base["sampling"],
    }
    return RunSpec(mode=mode, dataset=dataset, method_name=method_name, retriever=retriever,
                   base=base, method=method, profile=profile, run_name=run_name,
                   run_dir=run_dir, config_id=sha256_json(identity))


def enumerate_runs(base: dict, dataset: str) -> list[tuple[str, str | None]]:
    """The full matrix of (method_name, retriever) for one dataset (15 runs, §1)."""
    runs = []
    for name in method_names():
        m = load_method(name, base)
        if m["method"] == "bcp_judge":
            continue
        rets = m["retrieval"]["retrievers"] if m["retrieval"] else [None]
        runs.extend((name, r) for r in rets)
    return runs

