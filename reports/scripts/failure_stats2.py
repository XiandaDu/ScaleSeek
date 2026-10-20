"""Second-pass GrepSeek statistics (compute node only).  Reads analysis/failures/*.jsonl and
the GrepSeek trajectories; writes analysis/failures/stats2.json."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
import yaml
from transformers import AutoTokenizer

from scaleseek.config import load_base
from scaleseek.hub import snapshot_dir
from scaleseek.metrics import normalize_answer

ROOT = Path("/scratch/a32du/scaleseek_v2")
A = ROOT / "analysis" / "failures"
base = load_base()
gy = yaml.safe_load(open("/home/a32du/ScaleSeek/configs/methods/grepseek.yaml"))
TOK = AutoTokenizer.from_pretrained(str(snapshot_dir(base, gy["generator"]["model"], gy["generator"]["revision"])))
WINDOW = {"browsecomp_plus": 131072}


def ntok(s):
    return len(TOK(s or "", add_special_tokens=False)["input_ids"])


def form(pred, golds):
    """Answer-form relation of a wrong prediction to the gold answers."""
    p = " " + normalize_answer(pred or "") + " "
    gs = [" " + normalize_answer(g) + " " for g in golds if normalize_answer(g)]
    if any(g in p for g in gs):
        return "pred_contains_gold"          # too long / hedged / list
    if p.strip() and any(p in g for g in gs):
        return "gold_contains_pred"          # too short / missing part
    return "other"


gs = [json.loads(l) for l in open(A / "grepseek_features.jsonl")]
ot = [json.loads(l) for l in open(A / "others_features.jsonl")]
out = {}

# 1. answer form of wrong answers: GrepSeek vs the strong others (Wiki only; BCP is judged)
forms = defaultdict(Counter)
for r in gs:
    if r["ds"] != "browsecomp_plus" and r["finish"] == "answer" and not r["correct"]:
        forms["grepseek"][form(r["prediction"], r["golds"])] += 1
for r in ot:
    if r["ds"] != "browsecomp_plus" and r["finish"] == "answer" and not r["correct"]:
        forms[r["run"]][form(r["prediction"], r["golds"])] += 1
out["wrong_answer_form_wiki"] = forms
out["answer_words_wiki"] = {
    "grepseek": round(sum(len((r["prediction"] or "").split()) for r in gs if r["ds"] != "browsecomp_plus" and r["finish"] == "answer")
                      / max(sum(1 for r in gs if r["ds"] != "browsecomp_plus" and r["finish"] == "answer"), 1), 2),
    **{run: round(sum(len((r["prediction"] or "").split()) for r in ot if r["run"] == run and r["ds"] != "browsecomp_plus" and r["finish"] == "answer")
                  / max(sum(1 for r in ot if r["run"] == run and r["ds"] != "browsecomp_plus" and r["finish"] == "answer"), 1), 2)
       for run in sorted({r["run"] for r in ot})}}

# 2. command habits
calls = [(r["ds"], c) for r in gs for c in r["calls"]]
out["commands"] = {}
for ds in sorted({d for d, _ in calls}):
    cs = [c for d, c in calls if d == ds]
    sat = [c for c in cs if c["head_n"] and c["lines"] >= c["head_n"] and not c["truncated"]]
    out["commands"][ds] = {
        "n": len(cs),
        "title_anchored": sum(1 for c in cs if "contents" in c["cmd"]),
        "regex_no_F_first_stage": sum(1 for c in cs if not c["-F"]),
        "head_saturated": len(sat),
        "empty_after_filter": sum(1 for c in cs if c["empty"] and c["stages"] >= 3),
        "empty_single_stage": sum(1 for c in cs if c["empty"] and c["stages"] <= 2),
        "timeouts": sum(c["timed_out"] for c in cs),
        "timeout_stages": Counter(c["stages"] for c in cs if c["timed_out"]),
    }

# 3. grounded vs ungrounded answers
g = Counter()
for r in gs:
    if r["finish"] == "answer":
        key = ("grounded" if r["pred_in_outputs"] else "ungrounded")
        g[key + "_n"] += 1
        g[key + "_correct"] += int(r["correct"])
out["grounding"] = g

# 4. thinking-length blow-ups and the turns that produced no action
traj = {}
for ds in sorted({r["ds"] for r in gs}):
    for rec in map(json.loads, open(ROOT / "runs" / ds / "grepseek" / "trajectories.jsonl")):
        traj[ds, rec["id"]] = rec
blow = Counter()
noaction = Counter()
for r in gs:
    t = r["think_tok"]
    big = max(t) if t else 0
    blow["n"] += 1
    blow["max_think_ge_2000"] += big >= 2000
    blow["max_think_ge_2000_correct"] += big >= 2000 and r["correct"]
    blow["max_think_lt_2000_correct"] += big < 2000 and r["correct"]
    blow["max_think_lt_2000"] += big < 2000
    turns = traj[r["ds"], r["id"]]["trajectory"]["turns"]
    for i, tt in enumerate(turns):
        if tt.get("synthetic"):
            prev = turns[i - 1]
            th = (prev["content"] or "").split("</think>")[0]
            noaction["n"] += 1
            noaction["prev_finish_" + str(prev.get("finish_reason"))] += 1
            noaction["prev_think_ge_2000_tok"] += ntok(th) >= 2000
            noaction["prev_has_close_think"] += "</think>" in (prev["content"] or "")
out["thinking_blowup"] = blow
out["no_action_turns"] = noaction

# 5. the duplicated tool payload and the context-window failures
dup = Counter()
for r in gs:
    if r["finish"] != "context_exhausted":
        continue
    rec = traj[r["ds"], r["id"]]
    saved = 0
    for t in rec["trajectory"]["turns"]:
        if t["role"] == "tool" and not t.get("synthetic"):
            p = json.loads(t["content"])
            lean = dict(p)
            lean.pop("information_lines")
            saved += ntok(t["content"]) - ntok(json.dumps(lean, ensure_ascii=False))
    window = WINDOW.get(r["ds"], 16384)
    key = "bcp" if r["ds"] == "browsecomp_plus" else "wiki"
    dup[key + "_n"] += 1
    dup[key + "_saved_tok_sum"] += saved
    dup[key + "_fits_without_dup"] += (r["context_tok_est"] - saved) < window - 1024
out["dup_payload_context_exhausted"] = dup

# 6. gold seen but the run ended without an answer: at which call, and what came after
late = Counter()
for r in gs:
    if r["finish"] in ("max_turns", "context_exhausted") and r["gold_seen_turn"] is not None:
        late["n"] += 1
        late["calls_after_gold"] += r["n_calls"] - r["gold_seen_turn"]
out["budget_fail_after_gold"] = late

(A / "stats2.json").write_text(json.dumps(out, indent=1, default=dict))
print(json.dumps(out, indent=1, default=dict))
