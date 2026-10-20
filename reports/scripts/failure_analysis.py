"""Failure analysis of the formal runs (compute node only).

Writes $ROOT/analysis/failures/:
  summary.json                 aggregate tables (GrepSeek taxonomy, cross-method, other methods)
  grepseek_features.jsonl      one row per GrepSeek question (all 8 datasets)
  others_features.jsonl        one row per question for the strong other runs
  cases/grepseek/<ds>/<category>/<id>.txt   readable trajectories, sampled per category
  cases/others/<run>/<ds>/<category>/<id>.txt
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.config import load_base
from scaleseek.corpus import DocStore, corpus_dir
from scaleseek.hub import snapshot_dir
from scaleseek.metrics import exact_match, f1_score, final_answer, finish_reason, normalize_answer
from scaleseek.retrieval.service import display_function

ROOT = Path("/scratch/a32du/scaleseek_v2")
RUNS = ROOT / "runs"
OUT = ROOT / "analysis" / "failures"
DATASETS = ["bamboogle", "popqa", "triviaqa", "nq", "hotpotqa", "2wikimultihopqa", "musique", "browsecomp_plus"]
ALL_RUNS = ["direct", "rag__bm25", "rag__e5", "rag__qwen3_emb_4b", "search_r1_7b__bm25", "search_r1_7b__e5",
            "search_r1_7b__qwen3_emb_4b", "search_r1_14b__bm25", "search_r1_14b__e5", "search_r1_14b__qwen3_emb_4b",
            "search_o1__bm25", "search_o1__e5", "search_o1__qwen3_emb_4b", "grepseek", "agentir__agentir_4b"]
STRONG = ["search_r1_14b__e5", "search_r1_14b__qwen3_emb_4b", "search_o1__qwen3_emb_4b", "rag__qwen3_emb_4b",
          "agentir__agentir_4b"]
random.seed(20261019)
base = load_base()


# ----------------------------------------------------------------------------- helpers
def norm_text(s: str) -> str:
    s = s.replace("\\n", " ").replace('\\"', '"').replace("\\u2013", "-")
    return " " + normalize_answer(s) + " "


def usable_golds(golds: list[str]) -> list[str]:
    out = []
    for g in golds:
        ng = normalize_answer(g)
        if len(ng) >= 3 and ng not in ("yes", "no"):
            out.append(" " + ng + " ")
    return out


def gold_in(ntext: str, ngolds: list[str]) -> bool:
    return any(g in ntext for g in ngolds)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def correctness(ds: str, run: str) -> dict[str, bool]:
    recs = load(RUNS / ds / run / "trajectories.jsonl")
    if ds == "browsecomp_plus":
        judged = {r["id"]: bool(r["judge_result"].get("correct")) for r in load(RUNS / ds / run / "judge__bcp_judge.jsonl")}
        return {r["id"]: judged.get(r["id"], False) for r in recs}
    return {r["id"]: exact_match(final_answer(r), r["golden_answers"]) for r in recs}


def clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n * 2 // 3] + f" …[{len(s) - n} chars cut]… " + s[-n // 3:]


def mark(s: str, golds: list[str]) -> str:
    for g in sorted(set(golds), key=len, reverse=True):
        if len(g) >= 3 and g.lower() not in ("yes", "no"):
            s = re.sub(re.escape(g), lambda m: f">>>{m.group(0)}<<<", s, flags=re.IGNORECASE)
    return s


# ----------------------------------------------------------------------------- tokenizer
from transformers import AutoTokenizer

import yaml
gs_yaml = yaml.safe_load(open("/home/a32du/ScaleSeek/configs/methods/grepseek.yaml"))
TOK = AutoTokenizer.from_pretrained(str(snapshot_dir(base, gs_yaml["generator"]["model"], gs_yaml["generator"]["revision"])))


def ntok(s: str) -> int:
    return len(TOK(s or "", add_special_tokens=False)["input_ids"])


# ----------------------------------------------------------------------------- cross-method
correct = {ds: {run: correctness(ds, run) for run in ALL_RUNS} for ds in DATASETS}
cross = {}
for ds in DATASETS:
    g = correct[ds]["grepseek"]
    ids = list(g)
    row = {"n": len(ids), "grepseek_correct": sum(g.values())}
    for run in ALL_RUNS:
        if run == "grepseek":
            continue
        o = correct[ds][run]
        row[run] = {"gs_wrong_other_right": sum(1 for q in ids if not g[q] and o.get(q)),
                    "gs_right_other_wrong": sum(1 for q in ids if g[q] and not o.get(q))}
    row["gs_wrong_any_strong_right"] = sum(1 for q in ids if not g[q] and any(correct[ds][r].get(q) for r in STRONG))
    row["gs_wrong_ge3_strong_right"] = sum(1 for q in ids if not g[q] and sum(bool(correct[ds][r].get(q)) for r in STRONG) >= 3)
    row["gs_right_all_strong_wrong"] = sum(1 for q in ids if g[q] and not any(correct[ds][r].get(q) for r in STRONG))
    row["oracle_any_run"] = sum(1 for q in ids if any(correct[ds][r].get(q) for r in ALL_RUNS))
    cross[ds] = row

# ----------------------------------------------------------------------------- GrepSeek features
CMD_FLAGS = [("-F", r"(^|\s)-[a-zA-Z]*F"), ("-i", r"(^|\s)-[a-zA-Z]*i"), ("-w", r"(^|\s)-[a-zA-Z]*w"),
             ("-o", r"(^|\s)-[a-zA-Z]*o"), ("wc", r"\|\s*wc\b"), ("head", r"\|\s*head\b")]


def cmd_features(cmd: str) -> dict:
    stages = [s.strip() for s in cmd.split("|")]
    first = stages[0].split()[0] if stages and stages[0] else ""
    f = {k: bool(re.search(p, cmd)) for k, p in CMD_FLAGS}
    m = re.search(r"\|\s*head\s+-n\s*(\d+)", cmd)
    f.update(prog=first, stages=len(stages), head_n=int(m.group(1)) if m else None)
    pat = re.findall(r'"([^"]*)"', stages[0]) if stages else []
    f["first_pattern"] = pat[0] if pat else None
    f["first_pattern_words"] = len(pat[0].split()) if pat else 0
    return f


gs_rows = []
for ds in DATASETS:
    golds_by_id = {}
    for rec in load(RUNS / ds / "grepseek" / "trajectories.jsonl"):
        golds = rec["golden_answers"]
        ngolds = usable_golds(golds)
        turns = rec["trajectory"]["turns"]
        calls, thinks, seen_turn, pred_seen = [], [], None, False
        npred = norm_text(rec["prediction"]) if rec["prediction"] else None
        asst_i = 0
        payload_tok = stdout_tok = 0
        synthetic = []
        for t in turns:
            if t["role"] == "assistant":
                asst_i += 1
                txt = t["content"] or ""
                think = txt.split("</think>")[0] if "</think>" in txt else ""
                thinks.append(ntok(think))
            elif t["role"] == "tool":
                p = json.loads(t["content"])
                if t.get("synthetic"):
                    synthetic.append(p["stderr"])
                    continue
                st = p["stdout"] or ""
                nt = norm_text(st)
                hit = gold_in(nt, ngolds) if ngolds else None
                if hit and seen_turn is None:
                    seen_turn = asst_i
                if npred and npred.strip() and npred in nt:
                    pred_seen = True
                pt, stt = ntok(t["content"]), ntok(st)
                payload_tok += pt
                stdout_tok += stt
                calls.append({"cmd": p["command"], "exit": p["exit_code"], "timed_out": p["timed_out"],
                              "truncated": p["truncated"], "empty": not st.strip(), "stdout_tok": stt,
                              "payload_tok": pt, "lines": len(p["information_lines"]), "gold_hit": hit,
                              "stderr": (p["stderr"] or "")[:200], **cmd_features(p["command"])})
        fr = finish_reason(rec)
        ans = final_answer(rec)
        em = exact_match(ans, golds)
        f1 = f1_score(ans, golds)
        ok = correct[ds]["grepseek"][rec["id"]]
        if ok:
            cat = "correct"
        elif fr == "answer":
            if ngolds and seen_turn is not None:
                cat = "wrong_answer_gold_seen"
            elif ngolds:
                cat = "wrong_answer_gold_never_seen"
            else:
                cat = "wrong_answer_gold_unmatchable"
        else:
            cat = fr                      # max_turns / context_exhausted / parse_error / api_error
        cmds = [c["cmd"] for c in calls]
        row = {"ds": ds, "id": rec["id"], "question": rec["question"], "golds": golds, "prediction": rec["prediction"],
               "finish": fr, "correct": ok, "em": em, "f1": f1, "category": cat,
               "n_assistant": asst_i, "n_calls": len(calls), "n_synthetic": len(synthetic), "synthetic": synthetic,
               "gold_matchable": bool(ngolds), "gold_seen_turn": seen_turn, "pred_in_outputs": pred_seen,
               "think_tok": thinks, "payload_tok": payload_tok, "stdout_tok": stdout_tok,
               "context_tok_est": ntok(rec["trajectory"]["turns"][0]["content"]) + ntok(rec["question"]) + sum(thinks)
                                  + payload_tok + 40 * asst_i,
               "n_empty": sum(c["empty"] for c in calls), "n_trunc": sum(c["truncated"] for c in calls),
               "n_timeout": sum(c["timed_out"] for c in calls),
               "n_err": sum(1 for c in calls if c["exit"] not in (0, 1) and not c["timed_out"]),
               "n_repeat": len(cmds) - len(set(cmds)), "calls": calls,
               "others_correct": [r for r in ALL_RUNS if r != "grepseek" and correct[ds][r].get(rec["id"])],
               "strong_correct": [r for r in STRONG if correct[ds][r].get(rec["id"])]}
        gs_rows.append(row)


def gs_summary(rows):
    s = {"n": len(rows), "correct": sum(r["correct"] for r in rows), "categories": Counter(r["category"] for r in rows)}
    calls = [c for r in rows for c in r["calls"]]
    s["calls"] = {"n": len(calls), "per_q": round(len(calls) / max(len(rows), 1), 2),
                  "empty": sum(c["empty"] for c in calls), "truncated": sum(c["truncated"] for c in calls),
                  "timeout": sum(c["timed_out"] for c in calls),
                  "error_exit": sum(1 for c in calls if c["exit"] not in (0, 1) and not c["timed_out"]),
                  "flags": {k: sum(c[k] for c in calls) for k, _ in CMD_FLAGS},
                  "prog": Counter(c["prog"] for c in calls).most_common(8),
                  "head_n": Counter(c["head_n"] for c in calls).most_common(6),
                  "stages": Counter(c["stages"] for c in calls).most_common(5),
                  "pattern_words": Counter(min(c["first_pattern_words"], 6) for c in calls).most_common(7),
                  "stdout_tok_mean": round(sum(c["stdout_tok"] for c in calls) / max(len(calls), 1)),
                  "payload_tok_mean": round(sum(c["payload_tok"] for c in calls) / max(len(calls), 1))}
    wa = [r for r in rows if r["category"].startswith("wrong_answer")]
    s["wrong_answers"] = {"n": len(wa), "f1_gt0": sum(r["f1"] > 0 for r in wa),
                          "pred_in_outputs": sum(r["pred_in_outputs"] for r in wa)}
    bud = [r for r in rows if r["finish"] in ("max_turns", "context_exhausted")]
    s["budget_failures"] = {"n": len(bud), "gold_seen": sum(r["gold_seen_turn"] is not None for r in bud),
                            "gold_matchable": sum(r["gold_matchable"] for r in bud)}
    ce = [r for r in rows if r["finish"] == "context_exhausted"]
    if ce:
        s["context_exhausted"] = {"n": len(ce), "mean_calls": round(sum(r["n_calls"] for r in ce) / len(ce), 2),
                                  "mean_think_tok": round(sum(sum(r["think_tok"]) for r in ce) / len(ce)),
                                  "mean_payload_tok": round(sum(r["payload_tok"] for r in ce) / len(ce)),
                                  "mean_stdout_tok": round(sum(r["stdout_tok"] for r in ce) / len(ce)),
                                  "gold_seen": sum(r["gold_seen_turn"] is not None for r in ce)}
    s["think_tok_per_turn_mean"] = round(sum(sum(r["think_tok"]) for r in rows) / max(sum(len(r["think_tok"]) for r in rows), 1))
    s["synthetic_errors"] = Counter(e[:90] for r in rows for e in r["synthetic"]).most_common(5)
    s["tool_error_stderr"] = Counter(c["stderr"][:90] for c in calls if c["exit"] not in (0, 1) and not c["timed_out"]).most_common(6)
    s["gs_wrong_strong_right_by_cat"] = Counter(r["category"] for r in rows if not r["correct"] and r["strong_correct"])
    return s


summary = {"cross": cross, "grepseek": {ds: gs_summary([r for r in gs_rows if r["ds"] == ds]) for ds in DATASETS}}
summary["grepseek"]["ALL_WIKI"] = gs_summary([r for r in gs_rows if r["ds"] != "browsecomp_plus"])

# ----------------------------------------------------------------------------- other strong runs
stores = {"wiki18": DocStore(corpus_dir(base, "wiki18")), "bcp": DocStore(corpus_dir(base, "bcp"))}
show_bcp = display_function(base, "bcp")
_doc_cache: dict = {}


def doc_text(corpus: str, did: str) -> str:
    key = (corpus, did)
    if key not in _doc_cache:
        t = stores[corpus].get(did)["contents"]
        _doc_cache[key] = show_bcp(t) if corpus == "bcp" else t
    return _doc_cache[key]


OTHER = ["search_r1_14b__qwen3_emb_4b", "search_r1_14b__e5", "search_o1__qwen3_emb_4b", "rag__qwen3_emb_4b",
         "agentir__agentir_4b"]
ot_rows = []
for run in OTHER:
    for ds in DATASETS:
        corpus = "bcp" if ds == "browsecomp_plus" else "wiki18"
        for rec in load(RUNS / ds / run / "trajectories.jsonl"):
            golds = rec["golden_answers"]
            ngolds = usable_golds(golds)
            queries, seen_at, n_search, extra = [], None, 0, {}
            if run.startswith("agentir"):
                res = rec["trajectory"]["official_run"].get("result") or []
                for it in res:
                    if it.get("type") == "tool_call":
                        n_search += 1
                        try:
                            queries.append(json.loads(it.get("arguments") or "{}").get("query"))
                        except Exception:
                            queries.append(str(it.get("arguments"))[:200])
                        if ngolds and seen_at is None and gold_in(norm_text(str(it.get("output") or "")), ngolds):
                            seen_at = n_search
                texts = [it.get("output") for it in res if it.get("type") == "output_text" and it.get("output")]
                extra["final_text_words"] = len((texts[-1] if texts else "").split())
                extra["pred_words"] = len((rec["prediction"] or "").split())
                extra["dropped_parallel"] = sum((c.get("dropped_parallel_calls") or 0) for c in rec["trajectory"]["calls"])
            else:
                for t in rec["trajectory"]:
                    if t.get("type") == "search":
                        n_search += 1
                        queries.append(t.get("query"))
                        if ngolds and seen_at is None and t.get("hits"):
                            if gold_in(norm_text(" ".join(doc_text(corpus, h["id"]) for h in t["hits"])), ngolds):
                                seen_at = n_search
                if run.startswith("search_o1"):
                    rid = [t for t in rec["trajectory"] if t.get("type") == "llm" and t.get("role") == "reason_in_documents"]
                    extra["rid_calls"] = len(rid)
                    extra["rid_no_info"] = sum("No helpful information found" in (t.get("text") or "") for t in rid)
                llm = [t for t in rec["trajectory"] if t.get("type") == "llm"]
                extra["gen_tok"] = sum(t.get("completion_tokens") or 0 for t in llm)
            fr = finish_reason(rec)
            ans = final_answer(rec)
            ok = correct[ds][run][rec["id"]]
            if ok:
                cat = "correct"
            elif fr == "answer":
                cat = ("wrong_answer_gold_retrieved" if seen_at else "wrong_answer_gold_not_retrieved") if ngolds else "wrong_answer_gold_unmatchable"
            else:
                cat = fr
            ot_rows.append({"run": run, "ds": ds, "id": rec["id"], "question": rec["question"], "golds": golds,
                            "prediction": rec["prediction"], "finish": fr, "correct": ok, "category": cat,
                            "em": exact_match(ans, golds), "f1": f1_score(ans, golds), "n_search": n_search,
                            "queries": queries, "gold_seen_search": seen_at, "gold_matchable": bool(ngolds),
                            "grepseek_correct": correct[ds]["grepseek"].get(rec["id"]), **extra})

summary["others"] = {}
for run in OTHER:
    summary["others"][run] = {}
    for ds in DATASETS:
        rows = [r for r in ot_rows if r["run"] == run and r["ds"] == ds]
        wa = [r for r in rows if r["category"].startswith("wrong_answer")]
        d = {"n": len(rows), "correct": sum(r["correct"] for r in rows), "categories": Counter(r["category"] for r in rows),
             "searches_per_q": round(sum(r["n_search"] for r in rows) / max(len(rows), 1), 2),
             "wrong_answer_f1_gt0": sum(r["f1"] > 0 for r in wa),
             "budget_fail_gold_seen": sum(1 for r in rows if r["finish"] in ("max_turns", "context_exhausted") and r["gold_seen_search"])}
        if run.startswith("agentir"):
            ans = [r for r in rows if r["finish"] == "answer"]
            d["pred_words_mean"] = round(sum(r["pred_words"] for r in ans) / max(len(ans), 1), 1)
            d["dropped_parallel_q"] = sum(1 for r in rows if r["dropped_parallel"])
            d["dropped_parallel_calls"] = sum(r["dropped_parallel"] for r in rows)
        if run.startswith("search_o1"):
            d["rid_no_info_share"] = round(sum(r["rid_no_info"] for r in rows) / max(sum(r["rid_calls"] for r in rows), 1), 3)
        summary["others"][run][ds] = d

# RAG vs Direct on the multi-hop sets
summary["rag_vs_direct"] = {}
for ds in DATASETS:
    if ds == "browsecomp_plus":
        continue
    d, rq = correct[ds]["direct"], correct[ds]["rag__qwen3_emb_4b"]
    rag_rows = {r["id"]: r for r in ot_rows if r["run"] == "rag__qwen3_emb_4b" and r["ds"] == ds}
    broke = [q for q in d if d[q] and not rq[q]]
    summary["rag_vs_direct"][ds] = {"direct_right_rag_wrong": len(broke), "rag_right_direct_wrong": sum(1 for q in d if rq[q] and not d[q]),
                                    "broke_with_gold_retrieved": sum(1 for q in broke if rag_rows[q]["gold_seen_search"])}

# ----------------------------------------------------------------------------- case files
def render_grepseek(r: dict, rec: dict) -> str:
    golds = r["golds"]
    lines = [f"=== {r['ds']} / grepseek / id {r['id']} / {r['category']}",
             f"Q: {r['question']}", f"GOLD: {golds}", f"PRED: {r['prediction']!r}  finish={r['finish']}  EM={r['em']}  F1={r['f1']:.2f}",
             f"strong runs correct: {r['strong_correct']}", f"all other runs correct: {len(r['others_correct'])}/14",
             f"calls={r['n_calls']} empty={r['n_empty']} trunc={r['n_trunc']} timeout={r['n_timeout']} err={r['n_err']} "
             f"repeat={r['n_repeat']} gold_seen_turn={r['gold_seen_turn']} pred_in_outputs={r['pred_in_outputs']} "
             f"think_tok={r['think_tok']} payload_tok={r['payload_tok']} ctx_est={r['context_tok_est']}", ""]
    k = 0
    for t in rec["trajectory"]["turns"][2:]:
        if t["role"] == "assistant":
            k += 1
            txt = t["content"] or ""
            think, _, act = txt.partition("</think>")
            lines.append(f"--- A{k} think({len(think)}c): {clip(think.replace('<think>', '').strip(), 900)}")
            lines.append(f"    action: {clip(act.strip(), 400)}")
        else:
            p = json.loads(t["content"])
            if t.get("synthetic"):
                lines.append(f"    [synthetic tool turn] {p['stderr']}")
                continue
            lines.append(f"    TOOL exit={p['exit_code']} timed_out={p['timed_out']} truncated={p['truncated']} lines={len(p['information_lines'])}"
                         + (f" stderr={p['stderr'][:150]!r}" if p['stderr'] else ""))
            for ln in (p["stdout"] or "").split("\n")[:10]:
                if ln.strip():
                    lines.append("      | " + mark(clip(ln.replace('\\"', '"').replace("\\n", " / "), 420), golds))
    return "\n".join(lines) + "\n"


def render_other(r: dict, rec: dict, corpus: str) -> str:
    golds = r["golds"]
    lines = [f"=== {r['ds']} / {r['run']} / id {r['id']} / {r['category']}", f"Q: {r['question']}", f"GOLD: {golds}",
             f"PRED: {r['prediction']!r}  finish={r['finish']}  F1={r['f1']:.2f}  grepseek_correct={r['grepseek_correct']}",
             f"searches={r['n_search']} gold_seen_search={r['gold_seen_search']}", ""]
    if r["run"].startswith("agentir"):
        for it in rec["trajectory"]["official_run"].get("result") or []:
            if it.get("type") == "reasoning":
                lines.append(f"--- reasoning: {clip(str(it.get('output')), 700)}")
            elif it.get("type") == "tool_call":
                lines.append(f"    SEARCH {it.get('arguments')}")
                try:
                    hits = json.loads(it.get("output") or "[]")
                    for h in hits[:3]:
                        lines.append("      | " + mark(clip(str(h.get("snippet")).replace("\n", " / "), 300), golds))
                except Exception:
                    lines.append("      | " + clip(str(it.get("output")), 300))
            elif it.get("type") == "output_text":
                lines.append(f"--- FINAL: {clip(str(it.get('output')), 900)}")
    else:
        for t in rec["trajectory"]:
            if t.get("type") == "llm":
                lines.append(f"--- llm{' (RiD)' if t.get('role') else ''} [{t.get('finish_reason')}, {t.get('completion_tokens')} tok]: "
                             f"{clip(t.get('text') or '', 900)}")
            elif t.get("type") == "search":
                lines.append(f"    SEARCH {t.get('query')!r}")
                for h in (t.get("hits") or [])[:3]:
                    lines.append("      | " + mark(clip(doc_text(corpus, h["id"]).replace("\n", " / "), 300), golds))
    return "\n".join(lines) + "\n"


def sample(rows, k):
    rows = list(rows)
    random.shuffle(rows)
    return rows[:k]


recs_cache = {}


def rec_of(ds, run, qid):
    key = (ds, run)
    if key not in recs_cache:
        recs_cache.clear()
        recs_cache[key] = {r["id"]: r for r in load(RUNS / ds / run / "trajectories.jsonl")}
    return recs_cache[key][qid]


CATS_GS = {"wrong_answer_gold_seen": 5, "wrong_answer_gold_never_seen": 5, "max_turns": 4, "context_exhausted": 4,
           "parse_error": 2, "strong_right": 5}
for ds in DATASETS:
    rows = [r for r in gs_rows if r["ds"] == ds]
    picks = []
    for cat, k in CATS_GS.items():
        pool = [r for r in rows if not r["correct"] and len(r["strong_correct"]) >= 3] if cat == "strong_right" else \
               [r for r in rows if r["category"] == cat]
        picks += [(cat, r) for r in sample(pool, k)]
    picks += [("grepseek_unique_win", r) for r in sample([r for r in rows if r["correct"] and not r["strong_correct"]], 3)]
    for cat, r in picks:
        d = OUT / "cases" / "grepseek" / ds / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{r['id']}.txt").write_text(render_grepseek(r, rec_of(ds, "grepseek", r["id"])))

for run in OTHER:
    for ds in DATASETS:
        corpus = "bcp" if ds == "browsecomp_plus" else "wiki18"
        rows = [r for r in ot_rows if r["run"] == run and r["ds"] == ds and not r["correct"]]
        by = defaultdict(list)
        for r in rows:
            by[r["category"]].append(r)
        picks = [(c, r) for c, rs in by.items() for r in sample(rs, 2)]
        picks += [("grepseek_wrong_this_right", r) for r in sample([r for r in ot_rows if r["run"] == run and r["ds"] == ds
                                                                   and r["correct"] and not r["grepseek_correct"]], 2)]
        for cat, r in picks:
            d = OUT / "cases" / "others" / run / ds / cat
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{r['id']}.txt").write_text(render_other(r, rec_of(ds, run, r["id"]), corpus))

# ----------------------------------------------------------------------------- write
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "grepseek_features.jsonl", "w") as f:
    for r in gs_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "others_features.jsonl", "w") as f:
    for r in ot_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=lambda o: dict(o)))
print("done", len(gs_rows), len(ot_rows))
