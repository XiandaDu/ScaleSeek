"""GrepSeek+ vs GrepSeek on the Wiki test subsets: outcome categories, whether the gold answer
was in a tool output, what the variants did, and cost."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, final_answer, mcnemar_exact, normalize_answer
from transformers import AutoTokenizer

R = Path("/scratch/a32du/scaleseek_v2/runs")
DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
tok = AutoTokenizer.from_pretrained("/scratch/a32du/scaleseek_v2/hf/hub/models--alireza7--GrepSeek-Qwen3.5-9B-GRPO/snapshots/a79563970cfdd2ced3cc5fde481737d0ebea6fa4")


def load(ds, run):
    return {r["id"]: r for r in map(json.loads, open(R / ds / run / "trajectories.jsonl"))}


def payloads(r):
    return [json.loads(t["content"]) for t in r["trajectory"]["turns"] or [] if t["role"] == "tool"]


def seen(r):
    text = []
    for p in payloads(r):
        for ln in p["stdout"].split("\n"):
            try:
                text.append(json.loads(ln)["contents"])
            except Exception:
                text.append(ln)
    text = " " + normalize_answer(" ".join(text)) + " "
    return any(f" {normalize_answer(g)} " in text for g in r["golden_answers"] if len(normalize_answer(g)) >= 3)


def category(r):
    if exact_match(final_answer(r), r["golden_answers"]):
        return "correct"
    if r["finish_reason"] != "answer":
        return "no answer"
    return "wrong, gold seen" if seen(r) else "wrong, gold not seen"


CATS = ["correct", "no answer", "wrong, gold seen", "wrong, gold not seen"]
tot = {run: Counter() for run in ("grepseek", "grepseek_plus")}
agg = {run: Counter() for run in ("grepseek", "grepseek_plus")}
print(f"{'dataset':16s} {'run':14s} " + " ".join(f"{c:>20s}" for c in CATS) + "   gold seen  EM|seen  EM|answered  calls/q  empty  tool tok/q  ctx tok/q  tool s/call")
for ds in DS:
    for run in ("grepseek", "grepseek_plus"):
        recs = load(ds, run)
        n = len(recs)
        c = Counter(category(r) for r in recs.values())
        s = {i: seen(r) for i, r in recs.items()}
        ok = {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in recs.items()}
        ans = [i for i, r in recs.items() if r["finish_reason"] == "answer"]
        ps = [p for r in recs.values() for p in payloads(r)]
        empty = sum(not p["stdout"].strip() and not p["timed_out"] and p["exit_code"] in (0, 1) for p in ps)
        tool_tok = sum(len(tok(t["content"], add_special_tokens=False)["input_ids"]) for r in recs.values() for t in r["trajectory"]["turns"] or [] if t["role"] == "tool")
        ctx_tok = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) for r in recs.values() for m in r["trajectory"]["messages"] or [])
        tool_s = [t["elapsed_s"] for r in recs.values() for t in r["trajectory"]["turns"] or [] if t["role"] == "tool" and "elapsed_s" in t]
        print(f"{ds:16s} {run:14s} " + " ".join(f"{100 * c[k] / n:19.1f}%" for k in CATS)
              + f"   {100 * sum(s.values()) / n:8.1f}%  {100 * sum(ok[i] for i in recs if s[i]) / max(1, sum(s.values())):6.1f}  {100 * sum(ok[i] for i in ans) / max(1, len(ans)):10.1f}"
              f"  {len(ps) / n:7.2f}  {100 * empty / max(1, len(ps)):4.1f}%  {tool_tok / n:10.0f}  {ctx_tok / n:9.0f}  {sum(tool_s) / max(1, len(tool_s)):9.1f}")
        tot[run].update(c)
        agg[run].update(n=n, seen=sum(s.values()), calls=len(ps), empty=empty, tool_tok=tool_tok, ctx_tok=ctx_tok, tool_s=sum(tool_s),
                        llm_s=sum(r["trajectory"]["llm_time_s"] or 0 for r in recs.values()), answered=len(ans),
                        thinking_long=sum(any(t["role"] == "assistant" and len(t["content"]) > 8000 for t in r["trajectory"]["turns"] or []) for r in recs.values()))
print()
for run in ("grepseek", "grepseek_plus"):
    a, n = agg[run], agg[run]["n"]
    print(f"{'ALL (3125)':16s} {run:14s} " + " ".join(f"{100 * tot[run][k] / n:19.1f}%" for k in CATS)
          + f"   {100 * a['seen'] / n:8.1f}%  calls/q {a['calls'] / n:.2f}  empty {100 * a['empty'] / a['calls']:.1f}%  tool tok/q {a['tool_tok'] / n:.0f}"
          f"  ctx tok/q {a['ctx_tok'] / n:.0f}  tool s/call {a['tool_s'] / a['calls']:.1f}  llm s/q {a['llm_s'] / n:.1f}  questions with a >8K-char turn {a['thinking_long']}")

# what the variants did (GrepSeek+ only)
notices, asked, heeded, heeded_ok = Counter(), 0, 0, 0
for ds in DS:
    for r in load(ds, "grepseek_plus").values():
        turns = r["trajectory"]["turns"] or []
        for t in turns:
            if t["role"] == "tool":
                n = json.loads(t["content"]).get("notice", "")
                for key in ("last tool call", "almost full", "case-insensitive", "match without"):
                    notices[key] += key in n
        told = [j for j, t in enumerate(turns) if t["role"] == "tool" and "final answer now" in json.loads(t["content"]).get("notice", "")]
        if told:
            asked += 1
            nxt = turns[told[0] + 1] if told[0] + 1 < len(turns) else None
            if nxt and nxt["role"] == "assistant" and nxt["parse"]["answer"] is not None:
                heeded += 1
                heeded_ok += exact_match(final_answer(r), r["golden_answers"])
print(f"\nGrepSeek+ notices: {dict(notices)}; told to answer in {asked} questions, answered on the next turn in {heeded} ({heeded_ok} of them correct)")

# the questions GrepSeek left unanswered
un = fixed = ans = 0
for ds in DS:
    a, b = load(ds, "grepseek_plus"), load(ds, "grepseek")
    for i, r in b.items():
        if r["finish_reason"] != "answer":
            un += 1
            ans += a[i]["finish_reason"] == "answer"
            fixed += exact_match(final_answer(a[i]), a[i]["golden_answers"])
print(f"GrepSeek left {un} questions unanswered; GrepSeek+ answers {ans} of them, {fixed} correctly")

# vs the strongest baseline
pa, pb = {}, {}
for ds in DS:
    a, b = load(ds, "grepseek_plus"), load(ds, "search_r1_14b__qwen3_emb_4b")
    pa.update({f"{ds}/{i}": exact_match(final_answer(r), r["golden_answers"]) for i, r in a.items()})
    pb.update({f"{ds}/{i}": exact_match(final_answer(r), r["golden_answers"]) for i, r in b.items()})
mc = mcnemar_exact(pa, pb)
print(f"GrepSeek+ vs Search-R1-14B + Qwen3-Emb-4B, pooled: {100 * sum(pa.values()) / len(pa):.1f} vs {100 * sum(pb.values()) / len(pb):.1f}  won {mc['only_a']} lost {mc['only_b']} p={mc['p_value']:.3f}")
