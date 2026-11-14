"""GrepSeek+: the official GrepSeek checkpoint and harness, with harness-side fixes for the
failure modes measured in reports/grepseek_failures.md.  No training, no index, no dense
retriever; same turns, context window, tool-output cap and decoding as GrepSeek.

The official loop (inference/agent.py) is not edited.  Three call sites are wrapped
(grepseek_entry.py installs them when settings["variants"] is not empty):

    client.chat.completions.create        counts the example's turns and context use
    inference.agent.run_tool              runs the command with the fallbacks below
    inference.run.run_agent_on_example    resets the per-example state

Variants, each off in the baseline:
  budget     anytime answering.  Before the last turn, or when the context window is almost
             full, the tool response says so and asks for the answer; a tool output is cut
             to what still leaves room for that answer.  (17.7% of GrepSeek's Wiki questions
             end without an answer: turns used up 11.1%, context full 6.6%.)
  countdown  with budget: every tool response also says how many tool calls are left.
  dedup      the tool response drops `information_lines`, a line-by-line copy of stdout
             (38% of the window when the context fills up).
  fold       non-ASCII characters of a fixed-string pattern are written as the corpus
             stores them (\\uXXXX: the corpus is ASCII-escaped JSON, so such patterns never
             matched); an empty result is retried case-insensitively.
  relax      an empty result whose first pattern does match is retried without the last
             filter, then without any filter (1,000 of 1,914 empty calls were emptied by
             the AND filters).
  title      matches come title first instead of in file order: the article named by the
             first pattern ("X", "X (...)"), then titles containing it, then the rest
             (36% of the calls are cut by `head`).

Whatever a fallback did is said in the response's `notice` field; `command` stays as the
model wrote it.
"""

from __future__ import annotations

import json
import re
import shlex
import threading

from inference import tools

VARIANTS = {"budget", "countdown", "dedup", "fold", "relax", "title"}
RESERVE = 1536         # tokens kept free for the answering turn
LOW_ROOM = 4096        # less room than this after a tool response: ask for the answer
COPY = {False: 2.3, True: 1.15}   # payload tokens per stdout token, with information_lines / with dedup

CFG: dict = {}                 # {"variants": set, "context": int}, set by install()
S = threading.local()          # per example (one worker thread runs one example at a time)


# ----------------------------------------------------------------------------- interception
def install(official_run, client_factory, settings: dict):
    """Patch the official modules; returns the client factory to use."""
    import inference.agent as agent

    CFG.update(variants=set(settings["variants"]), context=settings["context"])
    assert CFG["variants"] <= VARIANTS, CFG["variants"] - VARIANTS
    run_example = official_run.run_agent_on_example

    def run_agent_on_example(example, **kw):
        S.turn, S.used, S.turns = 0, 0, kw["max_assistant_turns"]
        return run_example(example, **kw)

    def make_client(*args, **kw):
        client = client_factory(*args, **kw)
        create = client.chat.completions.create

        def tracked(**request):
            resp = create(**request)
            S.turn += 1
            S.used = resp.usage.prompt_tokens + resp.usage.completion_tokens
            return resp

        client.chat.completions.create = tracked
        return client

    official_run.run_agent_on_example = run_agent_on_example
    agent.run_tool = run_tool
    return make_client


class Response:
    """What the official loop reads of a ToolResult, with our payload."""

    def __init__(self, res: tools.ToolResult, payload: str):
        self.__dict__.update(vars(res))
        self.payload = payload

    def to_payload(self) -> str:
        return self.payload


def run_tool(cmd: str, *, corpus_dir, timeout, max_tokens, tokenizer, engine=None):
    v = CFG["variants"]

    def run(c: str) -> tools.ToolResult:        # untruncated: we cut once, at the end
        return tools.run_tool(c, corpus_dir=corpus_dir, timeout=timeout, max_tokens=10 ** 9,
                              tokenizer=tokenizer, engine=engine)

    notes: list[str] = []
    res = search(run, cmd, v, notes)
    res.command = cmd
    cap = max_tokens
    if "budget" in v:                           # leave room for the answering turn
        room = CFG["context"] - S.used - RESERVE
        cap = max(128, min(max_tokens, int(room / COPY["dedup" in v])))
    res.stdout, res.truncated = tools.truncate_tokens(res.stdout, cap, tokenizer)
    if res.truncated:
        res.stdout += f"\n[... output truncated at {cap} tokens]"
    res.information_lines = [ln for ln in res.stdout.split("\n") if ln.strip()]
    fields = {k: getattr(res, k) for k in ("command", "stdout", "stderr", "exit_code", "timed_out", "truncated")}
    if "dedup" not in v:
        fields["information_lines"] = res.information_lines
    if "budget" in v:
        left = S.turns - S.turn                 # assistant turns after this tool response
        used = S.used + len(tokenizer(json.dumps(fields, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        if left <= 1:
            notes.append("This was your last tool call. Give your final answer now in an <answer> block: "
                         "the best answer supported by what you have read.")
        elif CFG["context"] - used < LOW_ROOM:
            notes.append("The context window is almost full. Give your final answer now in an <answer> block: "
                         "the best answer supported by what you have read.")
        elif "countdown" in v:
            notes.append(f"{left - 1} tool call{'s' if left > 2 else ''} left before you must answer.")
    if notes:
        fields["notice"] = " ".join(notes)
    return Response(res, json.dumps(fields, ensure_ascii=False))


# ----------------------------------------------------------------------------- the command
def stages(cmd: str) -> list[str]:
    return [s.strip() for s in cmd.split("|")]          # as the official validator splits


def flags(stage: str) -> tuple[str, set[str], list[str]] | None:
    """(program, flags, other arguments) of an rg/grep stage; None for any other stage.
    Short flags are split into letters ("-iF" -> {"i", "F"}), long ones kept ("--count")."""
    try:
        tok = shlex.split(stage)
    except ValueError:
        return None
    if not tok or tok[0] not in ("rg", "grep"):
        return None
    opts: set[str] = set()
    for t in tok[1:]:
        if t.startswith("--"):
            opts.add(t)
        elif t.startswith("-"):
            opts.update(t[1:])
    return tok[0], opts, [t for t in tok[1:] if not t.startswith("-")]


def empty(res: tools.ToolResult) -> bool:
    return not res.stdout.strip() and not res.timed_out and res.exit_code in (0, 1)


def search(run, cmd: str, v: set[str], notes: list[str]) -> tools.ToolResult:
    go = (lambda c: title_first(run, c)) if "title" in v else run
    st = stages(cmd)
    if "fold" in v:                             # fixed-string stages: non-ASCII as the corpus has it
        st = [s if not (f := flags(s)) or not f[1] & {"F", "--fixed-strings"} else
              "".join(c if ord(c) < 128 else json.dumps(c)[1:-1] for c in s) for s in st]
    res = go(" | ".join(st)) if "fold" in v else go(cmd)
    first = flags(st[0])
    if not empty(res) or first is None:
        return res
    if "fold" in v and not first[1] & {"i", "--ignore-case"}:
        st[0] = st[0].replace(first[0], first[0] + " -i", 1)
        alt = go(" | ".join(st))
        if not empty(alt):
            notes.append("No match with this exact case; these are case-insensitive matches.")
            return alt
    if "relax" in v:
        filters = [i for i in range(1, len(st)) if flags(st[i])]
        for drop in dict.fromkeys((tuple(filters[-1:]), tuple(filters))):
            if drop:
                alt = go(" | ".join(s for i, s in enumerate(st) if i not in drop))
                if not empty(alt):
                    notes.append("No line matches all the patterns; these lines match without: "
                                 + " | ".join(st[i] for i in drop))
                    return alt
    return res


# ----------------------------------------------------------------------------- title first
# A corpus line is {"id": "7", "contents": "\"Evan Morris\"\ntext..."}: the title, quoted only
# when it has a space, then a literal \n.  The regexes use no alternation: the official
# validator splits a command on every "|".
_META = set("\\.+*?()|[]{}^$#&-~")
_LINE = r'^\{"id": "[0-9]+", "contents": "(?:\\")?'           # up to the start of the title
_TEXT = r'[^"\\]*(?:\\u[0-9a-f]{4}[^"\\]*)*'                  # title text, \uXXXX escapes included


def title_plan(cmd: str):
    """For `rg|grep -F "X" corpus.jsonl [| rg/grep filters] [| head -n K]`: (the pipeline
    restricted to the article X or "X (...)", the pipeline restricted to titles containing
    X; K or None; the two restrictions as Python patterns).  None for any other command."""
    st = stages(cmd)
    first = flags(st[0])
    if first is None or first[1] - {"F", "i", "w"} or "F" not in first[1]:
        return None
    if len(first[2]) != 2 or first[2][1] != tools.CORPUS_FILENAME_LOGICAL or not first[2][0] or set(first[2][0]) & {'"', "|"}:
        return None
    k = None
    for i, s in enumerate(st[1:], 1):
        if flags(s):
            continue
        m = re.fullmatch(r"head(?: -n ?(\d+)| -(\d+))?", s)
        if not m or i != len(st) - 1:
            return None
        k = int(m.group(1) or m.group(2) or 10)
    x = "".join("\\" + c if c in _META else c for c in first[2][0])
    b = r"\b" if "w" in first[1] else ""
    regexes = [_LINE + x + r" ?[\\(]", _LINE + _TEXT + b + x + b]
    i = " -i" if "i" in first[1] else ""
    cmds = [" | ".join([f"rg{i} -e '" + r.replace("'", "'\\''") + f"' {tools.CORPUS_FILENAME_LOGICAL}"] + st[1:])
            for r in regexes]
    return cmds, k, [re.compile(r, re.IGNORECASE if i else 0) for r in regexes]


def title_first(run, cmd: str) -> tools.ToolResult:
    res = run(cmd)
    plan = title_plan(cmd)
    if plan is None or empty(res) or res.timed_out or res.exit_code not in (0, 1):
        return res
    cmds, k, patterns = plan
    lines = [ln for ln in res.stdout.split("\n") if ln]
    if k is not None and len(lines) >= k:       # cut by head: title matches may lie further down the file
        top: list[str] = []
        for c in cmds:
            if len(top) < k:
                t = run(c)
                if not t.timed_out and t.exit_code in (0, 1):
                    top += [ln for ln in t.stdout.split("\n") if ln]
        lines = list(dict.fromkeys(top + lines))[:k]
    else:                                       # every match is here: reorder in place
        lines.sort(key=lambda ln: [not p.match(ln) for p in patterns])
    res.stdout = "\n".join(lines) + "\n"
    return res
