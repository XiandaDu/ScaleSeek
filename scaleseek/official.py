"""Official repositories: pinned checkouts and the byte diff of copied prompts/parsers.

BASELINES.md §0: a prompt taken from an official repository must be compared, byte for
byte, with the original at the pinned commit.  `verify_method` does that for every slot of
a method that has an entry in OFFICIAL_SOURCES, after checking that the checkout is at the
pinned commit and has no local modifications.  Two comparisons exist:

  function  the SOURCE TEXT of our copy equals the official definition's source text
            (nested definitions are compared after removing their indentation);
  assign    the VALUE of our constant equals the official assignment's value.  An
            f-string is compared as a str.format template ({expr} kept verbatim), and
            `'''...'''.strip()` is evaluated.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

from .config import ConfigError, resolve_constant
from .utils import PipelineError, git_state

# our constant -> (official repo name, file inside the repo, kind, official name)
OFFICIAL_SOURCES: dict[str, tuple[str, str, str, str]] = {
    # Search-R1 infer.py
    "prompts.official.search_r1.PROMPT_TEMPLATE": ("Search-R1", "infer.py", "assign", "prompt"),
    "prompts.official.search_r1.SEARCH_TEMPLATE": ("Search-R1", "infer.py", "assign", "curr_search_template"),
    "prompts.official.search_r1.STOP_SEQUENCES": ("Search-R1", "infer.py", "assign", "target_sequences"),
    "prompts.official.search_r1.EOS_TOKEN_IDS": ("Search-R1", "infer.py", "assign", "curr_eos"),
    "prompts.official.search_r1.get_query": ("Search-R1", "infer.py", "function", "get_query"),
    "prompts.official.search_r1._passages2string": ("Search-R1", "infer.py", "function", "_passages2string"),
    # Search-o1 scripts/
    "prompts.official.search_o1.get_singleqa_search_o1_instruction":
        ("Search-o1", "scripts/prompts.py", "function", "get_singleqa_search_o1_instruction"),
    "prompts.official.search_o1.get_multiqa_search_o1_instruction":
        ("Search-o1", "scripts/prompts.py", "function", "get_multiqa_search_o1_instruction"),
    "prompts.official.search_o1.get_task_instruction_openqa":
        ("Search-o1", "scripts/prompts.py", "function", "get_task_instruction_openqa"),
    "prompts.official.search_o1.get_webpage_to_reasonchain_instruction":
        ("Search-o1", "scripts/prompts.py", "function", "get_webpage_to_reasonchain_instruction"),
    "prompts.official.search_o1.extract_answer":
        ("Search-o1", "scripts/evaluate.py", "function", "extract_answer"),
    "prompts.official.search_o1.BEGIN_SEARCH_QUERY":
        ("Search-o1", "scripts/run_search_o1.py", "assign", "BEGIN_SEARCH_QUERY"),
    "prompts.official.search_o1.END_SEARCH_QUERY":
        ("Search-o1", "scripts/run_search_o1.py", "assign", "END_SEARCH_QUERY"),
    "prompts.official.search_o1.BEGIN_SEARCH_RESULT":
        ("Search-o1", "scripts/run_search_o1.py", "assign", "BEGIN_SEARCH_RESULT"),
    "prompts.official.search_o1.END_SEARCH_RESULT":
        ("Search-o1", "scripts/run_search_o1.py", "assign", "END_SEARCH_RESULT"),
    "prompts.official.search_o1.extract_between":
        ("Search-o1", "scripts/run_search_o1.py", "function", "extract_between"),
    "prompts.official.search_o1.LIMIT_MESSAGE": ("Search-o1", "scripts/run_search_o1.py", "fstring", ""),
    "prompts.official.search_o1.REPEAT_MESSAGE": ("Search-o1", "scripts/run_search_o1.py", "fstring", ""),
    "prompts.official.search_o1.RESULT_MESSAGE": ("Search-o1", "scripts/run_search_o1.py", "fstring", ""),
    "prompts.official.search_o1.STEP_LINE": ("Search-o1", "scripts/run_search_o1.py", "fstring", ""),
    "prompts.official.search_o1.WEB_PAGE_HEADER": ("Search-o1", "scripts/run_search_o1.py", "fstring", ""),
    # BrowseComp-Plus scripts_evaluation/evaluate_run.py
    "prompts.official.bcp_judge.GRADER_TEMPLATE":
        ("BrowseComp-Plus", "scripts_evaluation/evaluate_run.py", "assign", "GRADER_TEMPLATE"),
    "prompts.official.bcp_judge.create_judge_prompt":
        ("BrowseComp-Plus", "scripts_evaluation/evaluate_run.py", "function", "create_judge_prompt"),
    "prompts.official.bcp_judge.parse_judge_response":
        ("BrowseComp-Plus", "scripts_evaluation/evaluate_run.py", "function", "parse_judge_response"),
}


# ----------------------------------------------------------------------------- checkouts
def checkout_path(base: dict, name: str) -> Path:
    return Path(base["paths"]["repos"]) / name


def verify_checkout(base: dict, repo_cfg: dict) -> dict:
    """The official checkout exists, is at the pinned commit and is unmodified."""
    path = checkout_path(base, repo_cfg["name"])
    if not path.is_dir():
        raise PipelineError(f"missing official checkout {path} (run.py prefetch --only repos)")
    state = git_state(path)
    if state["commit"] != repo_cfg["commit"]:
        raise PipelineError(f"{path} is at {state['commit']}, pinned {repo_cfg['commit']}")
    if state["dirty"]:
        raise PipelineError(f"{path} has local modifications: {state['dirty_files']}")
    return state


# ----------------------------------------------------------------------------- AST helpers
def _fstring_template(node: ast.JoinedStr) -> str:
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(v.value.replace("{", "{{").replace("}", "}}"))
        elif isinstance(v, ast.FormattedValue):
            if v.conversion != -1 or v.format_spec is not None:
                raise PipelineError("f-string with conversion/format spec is not supported")
            parts.append("{" + ast.unparse(v.value) + "}")
        else:
            raise PipelineError(f"unexpected f-string part {type(v).__name__}")
    return "".join(parts)


def _value_of(node: ast.AST) -> Any:
    if isinstance(node, ast.JoinedStr):
        return _fstring_template(node)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "strip" and not node.args and isinstance(node.func.value, ast.Constant)):
        return node.func.value.value.strip()
    return ast.literal_eval(node)


def _official_assign(tree: ast.Module, name: str) -> Any:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                 for t in node.targets):
            return _value_of(node.value)
    raise PipelineError(f"no assignment to {name!r}")


def _official_function_source(text: str, tree: ast.Module, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(text, node, padded=True)
            pad = node.col_offset
            lines = seg.split("\n")
            out = []
            for line in lines:
                if pad and line[:pad].strip() == "" and len(line) >= pad:
                    line = line[pad:]
                out.append(line)
            return "\n".join(out)
    raise PipelineError(f"no function {name!r}")


def official_object(base: dict, dotted: str) -> tuple[str, Any]:
    repo, rel, kind, name = OFFICIAL_SOURCES[dotted]
    path = checkout_path(base, repo) / rel
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if kind == "function":
        return kind, _official_function_source(text, tree, name)
    if kind == "fstring":
        return kind, {_fstring_template(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)}
    return kind, _official_assign(tree, name)


def _ours(dotted: str, kind: str) -> Any:
    obj = resolve_constant(dotted)
    if kind == "function":
        return inspect.getsource(obj).rstrip("\n")
    return obj


def diff_report(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at char {i}: ours {a[max(0, i-30):i+30]!r} vs official {b[max(0, i-30):i+30]!r}"
    return f"lengths differ: ours {len(a)} vs official {len(b)}"


def verify_method(base: dict, method_cfg: dict) -> dict:
    """Check the pinned checkout and byte-diff every official prompt slot of a method.
    Returns a provenance record; raises PipelineError on the first mismatch."""
    record: dict = {"checkout": None, "byte_diff": {}}
    repo_cfg = method_cfg.get("official_repo")
    if repo_cfg:
        record["checkout"] = verify_checkout(base, repo_cfg)
    for slot, ref in method_cfg.get("prompts", {}).items():
        dotted = ref["constant"]
        if dotted not in OFFICIAL_SOURCES:
            if dotted.startswith("prompts.official."):
                raise ConfigError(f"{dotted} is under prompts/official but has no OFFICIAL_SOURCES entry")
            record["byte_diff"][slot] = "project-defined"
            continue
        repo = OFFICIAL_SOURCES[dotted][0]
        if not repo_cfg or repo_cfg["name"] != repo:
            raise ConfigError(f"{dotted} comes from {repo} but the method pins {repo_cfg}")
        kind, theirs = official_object(base, dotted)
        ours = _ours(dotted, kind)
        if kind == "function":
            theirs = theirs.rstrip("\n")
        if kind == "fstring":
            if ours not in theirs:
                raise PipelineError(f"byte diff FAILED for {slot} ({dotted}): no f-string in "
                                    f"{repo}/{OFFICIAL_SOURCES[dotted][1]} equals {ours!r}")
        elif ours != theirs:
            detail = diff_report(ours, theirs) if isinstance(ours, str) and isinstance(theirs, str) \
                else f"ours {ours!r} vs official {theirs!r}"
            raise PipelineError(f"byte diff FAILED for {slot} ({dotted}) vs "
                                f"{repo}/{OFFICIAL_SOURCES[dotted][1]}: {detail}")
        record["byte_diff"][slot] = f"identical to {repo}@{repo_cfg['commit'][:12]}:{OFFICIAL_SOURCES[dotted][1]}"
    return record
