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
  soft       ranked instead of Boolean filters: when fewer than K lines match every filter,
             the rest of the K lines are the passages of the first pattern that satisfy
             the most filters (coordination-level ranking; a generalisation of relax).
  title      matches come title first instead of in file order: the article named by the
             first pattern ("X", "X (...)"), then titles containing it, then the rest
             (36% of the calls are cut by `head`).

Whatever a fallback did is said in the response's `notice` field; `command` stays as the
model wrote it.

Under test for the next method (not part of GrepSeek+; they also wrap inference.agent.
_chat_completion):
  verify     an answer whose string is in no tool output so far is not accepted at once:
             the model is told so and asked to check it with one more search, once per
             question and only while two turns are left (answers found in a tool output are
             right 57% of the time, the others 24%).
  form       every tool response ends with a one-line reminder of the answer format (one
             short answer, no alternatives): 12% of GrepSeek+'s validation answers are wrong
             only in form (they contain the gold answer or are contained in it).
  commit     the generation that follows an "answer now" notice must be an <answer> block
             (constrained decoding of the text after the reasoning): no question ends
             without an answer because the model kept searching.
  settings["inject"]  a json file {question id: [messages]}: the example starts from these
             messages (the tool calls and outputs of earlier rollouts, replayed without the
             reasoning; grepseek_pilot.py builds them), e.g. to re-read pooled evidence.
"""

from __future__ import annotations

import json
import re
import shlex
import threading

from inference import tools

VARIANTS = {"budget", "commit", "countdown", "dedup", "fold", "form", "relax", "soft", "title", "verify"}
RESERVE = 1536         # tokens kept free for the answering turn
LOW_ROOM = 4096        # less room than this after a tool response: ask for the answer
MAX_BYTES = 20_000_000   # of one command's output read into memory (2,048 tokens are ~8 KB)
COPY = {False: 2.3, True: 1.15}   # payload tokens per stdout token, with information_lines / with dedup

VERIFY = ('Your answer "%s" does not appear in any search result so far. Check it with one more search '
          "(for example the answer together with a key term of the question), then give your final answer: "
          "keep it if the search supports it, otherwise give the answer the evidence supports.")

ANSWER = r"<answer>\n[^<\n]{1,200}\n</answer>"      # the harness's answer block (commit)
FORM = ("Answer format: one short answer only, a single name, entity, number or date (the full date when the "
        "question asks when); no alternatives, no explanation.")

CFG: dict = {}                 # {"variants": set, "context": int, "inject": dict}, set by install()
S = threading.local()          # per example (one worker thread runs one example at a time)


# ----------------------------------------------------------------------------- interception
def install(official_run, client_factory, settings: dict):
    """Patch the official modules; returns the client factory to use."""
    import inference.agent as agent

    CFG.update(variants=set(settings["variants"]), context=settings["context"],
               inject=json.load(open(settings["inject"])) if settings.get("inject") else {})
    assert CFG["variants"] <= VARIANTS, CFG["variants"] - VARIANTS
    run_example = official_run.run_agent_on_example

    def run_agent_on_example(example, **kw):
        S.turn, S.used, S.turns, S.verified, S.commit = 0, 0, kw["max_assistant_turns"], False, False
        S.inject = CFG["inject"].get(str(example["id"]))
        return run_example(example, **kw)

    def make_client(*args, **kw):
        client = client_factory(*args, **kw)
        create = client.chat.completions.create

        def tracked(**request):
            if S.commit and "commit" in CFG["variants"]:
                request["extra_body"] = {"structured_outputs": {"regex": ANSWER}}
            resp = create(**request)
            S.turn += 1
            S.used = resp.usage.prompt_tokens + resp.usage.completion_tokens
            return resp

        client.chat.completions.create = tracked
        return client

    chat = agent._chat_completion

    def _chat_completion(client, *, messages, **kw):
        if S.turn == 0 and S.inject:
            messages.extend(S.inject)
        if S.turn >= S.turns:                   # a verification already took the last generation
            return "", "stop", None
        out = chat(client, messages=messages, **kw)
        if "verify" in CFG["variants"] and out[0] and not S.verified and S.turn <= S.turns - 2:
            answer = agent.parse_assistant(out[0]).answer
            if answer is not None and not grounded(answer, messages):
                S.verified = True
                ask = {"command": "", "stdout": "", "stderr": "", "exit_code": 0, "timed_out": False,
                       "truncated": False, "notice": VERIFY % answer}
                messages += [{"role": "assistant", "content": out[0]},
                             {"role": "tool", "content": json.dumps(ask, ensure_ascii=False)}]
                out = chat(client, messages=messages, **kw)
        return out

    official_run.run_agent_on_example = run_agent_on_example
    agent.run_tool = run_tool
    if "verify" in CFG["variants"] or CFG["inject"]:
        agent._chat_completion = _chat_completion
    return make_client


def normalize(text: str) -> str:
    """The usual answer normalisation (lower case, no punctuation, no articles)."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def grounded(answer: str, messages: list) -> bool:
    """Is the answer string in some tool output of the transcript?  (yes/no and very short
    answers cannot be checked this way and count as grounded.)"""
    a = normalize(answer)
    if len(a) < 3 or a in ("yes", "no"):
        return True
    text = []
    for m in messages:
        if m["role"] == "tool":
            for line in json.loads(m["content"])["stdout"].split("\n"):
                try:
                    text.append(json.loads(line)["contents"])
                except (ValueError, TypeError, KeyError):
                    text.append(line)
    return f" {a} " in f" {normalize(' '.join(text))} "


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
        last = stages(c)[-1].split()[0]
        open_end = last not in ("head", "tail", "wc")       # e.g. `rg -F "the" corpus.jsonl`: gigabytes
        res = tools.run_tool(c + f" | head -c {MAX_BYTES}" if open_end else c, corpus_dir=corpus_dir, timeout=timeout,
                             max_tokens=10 ** 9, tokenizer=tokenizer, engine=engine)
        if open_end and last in ("rg", "grep") and res.exit_code == 0 and not res.stdout:
            res.exit_code = 1                   # what rg/grep itself returns without a match
        return res

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
    if "form" in v:
        notes.append(FORM)
    if "budget" in v:
        left = S.turns - S.turn                 # assistant turns after this tool response
        used = S.used + len(tokenizer(json.dumps(fields, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        S.commit = left <= 1 or CFG["context"] - used < LOW_ROOM
        if left <= 1:
            notes.append("This was your last tool call. Give your final answer now in an <answer> block: "
                         "the best answer supported by what you have read.")
        elif S.commit:
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


def head_k(stage: str) -> int | None:
    m = re.fullmatch(r"head(?: -n ?(\d+)| -(\d+))?", stage)
    return int(m.group(1) or m.group(2) or 10) if m else None


def search(run, cmd: str, v: set[str], notes: list[str]) -> tools.ToolResult:
    go = (lambda c: title_first(run, c)) if "title" in v else run
    st = stages(cmd)
    if "fold" in v:                             # fixed-string stages: non-ASCII as the corpus has it
        st = [s if not (f := flags(s)) or not f[1] & {"F", "--fixed-strings"} else
              "".join(c if ord(c) < 128 else json.dumps(c)[1:-1] for c in s) for s in st]
    res = go(" | ".join(st)) if "fold" in v else go(cmd)
    first = flags(st[0])
    if first is None or res.timed_out or res.exit_code not in (0, 1):
        return res
    if "soft" in v:
        res = soften(run, st, res, v, notes)
    if empty(res) and "fold" in v and not first[1] & {"i", "--ignore-case"}:
        st[0] = st[0].replace(first[0], first[0] + " -i", 1)
        more: list[str] = []
        alt = go(" | ".join(st))
        if "soft" in v:
            alt = soften(run, st, alt, v, more)
        if not empty(alt):
            notes += ["No match with this exact case; these are case-insensitive matches."] + more
            return alt
    if empty(res) and "relax" in v:
        filters = [i for i in range(1, len(st)) if flags(st[i])]
        for drop in dict.fromkeys((tuple(filters[-1:]), tuple(filters))):
            if drop:
                alt = go(" | ".join(s for i, s in enumerate(st) if i not in drop))
                if not empty(alt):
                    notes.append("No line matches all the patterns; these lines match without: "
                                 + " | ".join(st[i] for i in drop))
                    return alt
    return res


# ----------------------------------------------------------------------------- soft filters
CANDIDATES = 2000      # lines of the first pattern (file order) that are scored


def soften(run, st: list[str], res: tools.ToolResult, v: set[str], notes: list[str]) -> tools.ToolResult:
    """Ranked instead of Boolean filters, for `rg -F "X" corpus.jsonl | rg -F "f1" | ... | head -n K`.
    The lines matching every filter come first, as before.  When there are fewer than K of
    them, the rest of the K lines are the passages matching X that satisfy the most filters
    (with `title`: the article X first among equals; then file order)."""
    k = head_k(st[-1])
    tests = [line_test(s) for s in st[1:-1]]
    if k is None or not tests or None in tests or flags(st[0])[2][-1:] != [tools.CORPUS_FILENAME_LOGICAL]:
        return res
    full = [ln for ln in res.stdout.split("\n") if ln]
    if len(full) >= k:
        return res
    cand = run(f"{st[0]} | head -n {CANDIDATES}")
    if cand.timed_out or cand.exit_code not in (0, 1):
        return res
    plan = title_plan(" | ".join(st)) if "title" in v else None
    tiers = plan[2] if plan else []
    seen = set(full)
    rest = [ln for ln in dict.fromkeys(cand.stdout.split("\n")) if ln and ln not in seen]
    rest.sort(key=lambda ln: [-sum(t(ln) for t in tests)] + [not p.match(ln) for p in tiers])
    lines = (full + rest)[:k]
    if len(lines) > len(full):
        which = {0: "No line matches", 1: "Only the first line matches"}.get(len(full), f"Only the first {len(full)} lines match")
        notes.append(f"{which} all the patterns; the {'others' if full else 'lines shown'} match fewer of them, best first.")
        res.stdout = "\n".join(lines) + "\n"
    return res


def line_test(stage: str):
    """The filter `rg|grep -F [-i] [-w] "pattern"` as a predicate on a line; None for any other stage."""
    f = flags(stage)
    if f is None or f[1] - {"F", "i", "w"} or "F" not in f[1] or len(f[2]) != 1:
        return None
    rx = re.compile((r"\b%s\b" if "w" in f[1] else "%s") % re.escape(f[2][0]), re.IGNORECASE if "i" in f[1] else 0)
    return lambda line: bool(rx.search(line))


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
        k = head_k(s)
        if k is None or i != len(st) - 1:
            return None
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
