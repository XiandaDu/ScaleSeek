"""Safe shell executor for prompt-based DCI agents (DCI and GrepSeek baselines).

Mirrors GrepSeek's tools.py safety rules:
  - Whitelist of read-only commands (rg, grep, head, ...).
  - No redirection, chaining, or substitution.
  - Rewrites the logical corpus name (corpus.jsonl) to the real filename and
    runs the command in the corpus directory, so agent prompts are portable.
  - Char-based stdout truncation (no tokenizer dependency here).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass

ALLOWED_COMMANDS = {
    "rg", "grep", "find", "sed", "awk", "head", "tail", "cat",
    "ls", "wc", "sort", "cut", "uniq", "tr",
}

_DANGEROUS = [
    (re.compile(r"\bsed\b[^|]*\s-(?:[a-zA-Z]*i|-in-place)\b"), "sed -i"),
    (re.compile(r"\bfind\b[^|]*\s-delete\b"), "find -delete"),
    (re.compile(r"\bfind\b[^|]*\s-exec(?:dir)?\b"), "find -exec"),
    (re.compile(r"\b(?:rm|mv|cp|dd|chmod|chown|tee|ln)\b"), "destructive command"),
]

_MAX_OUTPUT_CHARS = 8000

_LOGICAL_CORPUS = "corpus.jsonl"


class ToolError(Exception):
    pass


def _strip_quoted(s: str) -> str:
    s = re.sub(r"'[^']*'", "", s)
    s = re.sub(r'"[^"]*"', "", s)
    return s


def validate(cmd: str) -> None:
    if "\n" in cmd:
        raise ToolError("newlines not allowed")
    if len(cmd) > 2000:
        raise ToolError("command too long")
    for pat, label in _DANGEROUS:
        if pat.search(cmd):
            raise ToolError(f"disallowed: {label}")
    bare = _strip_quoted(cmd)
    for bad in ("`", "$(", "&&", "||", ";", ">", "<"):
        if bad in bare:
            raise ToolError(f"disallowed: {bad!r}")
    for seg in (s.strip() for s in cmd.split("|")):
        if not seg:
            raise ToolError("empty pipeline segment")
        try:
            tokens = shlex.split(seg)
        except ValueError as e:
            raise ToolError(f"unparseable segment: {e}")
        if not tokens:
            raise ToolError("empty command")
        prog = tokens[0].split("/")[-1]
        if prog not in ALLOWED_COMMANDS:
            raise ToolError(f"{prog!r} not in whitelist")


@dataclass
class ToolResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    def to_payload(self) -> str:
        return json.dumps({
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }, ensure_ascii=False)


def run_shell(
    cmd: str,
    *,
    corpus_path: str,
    timeout: float = 30.0,
    max_chars: int = _MAX_OUTPUT_CHARS,
) -> ToolResult:
    """Validate, rewrite corpus.jsonl → actual filename, execute in corpus dir."""
    try:
        validate(cmd)
    except ToolError as e:
        return ToolResult(command=cmd, stdout="", stderr=f"validation error: {e}",
                          exit_code=-2, timed_out=False)

    corpus_dir = os.path.dirname(os.path.abspath(corpus_path))
    corpus_file = os.path.basename(corpus_path)
    cmd_run = cmd.replace(_LOGICAL_CORPUS, corpus_file)

    env = {**os.environ, "LC_ALL": "C"}
    env.pop("LD_LIBRARY_PATH", None)

    try:
        proc = subprocess.run(
            cmd_run, shell=True, executable="/bin/bash",
            cwd=corpus_dir, timeout=timeout,
            capture_output=True, text=True, env=env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        stdout = f"[command timed out after {timeout}s]"
        stderr = ""
        exit_code = -1
        timed_out = True

    if len(stdout) > max_chars:
        stdout = stdout[:max_chars] + f"\n[... truncated at {max_chars} chars]"

    return ToolResult(command=cmd, stdout=stdout, stderr=stderr,
                      exit_code=exit_code, timed_out=timed_out)
