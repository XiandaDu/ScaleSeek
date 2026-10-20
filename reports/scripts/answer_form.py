"""How much of each strong run's Wiki EM loss is answer form?

For every wrong scored answer, bucket the (normalised) prediction against the golds:
  list_with_gold      "X and Y" / "X, Y" containing a gold (several candidates given)
  year_of_date        a bare year where the gold is a full date containing it
  pred_contains_gold  over-specified ("Athina Onassis" vs "Athina")
  gold_contains_pred  under-specified ("three different relationships" vs "... he had in the past")
  other
and report EM next to cover-EM (a gold appears in the prediction), the lenient "acc" some papers use.
Writes analysis/failures/answer_form.json."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import normalize_answer

A = Path("/scratch/a32du/scaleseek_v2/analysis/failures")
YEAR = re.compile(r"^\d{4}$")


def pad(s):
    return f" {normalize_answer(s)} "


def bucket(pred, golds):
    p, gs = pad(pred), [pad(g) for g in golds if normalize_answer(g)]
    inside = [g for g in gs if g in p]
    if inside and re.search(r"\band\b|,|&|/|;", pred):
        return "list_with_gold"
    if YEAR.match(p.strip()) and any(p in g and len(g.split()) > 1 for g in gs):
        return "year_of_date"
    if inside:
        return "pred_contains_gold"
    if any(p in g for g in gs):
        return "gold_contains_pred"
    return "other"


rows = [json.loads(l) for l in open(A / "grepseek_features.jsonl")]
rows = [{**r, "run": "grepseek"} for r in rows] + [json.loads(l) for l in open(A / "others_features.jsonl")]
form, em, cover, n = defaultdict(Counter), Counter(), Counter(), Counter()
examples = defaultdict(list)
for r in rows:
    if r["ds"] == "browsecomp_plus":
        continue
    key = r["run"]
    n[key] += 1
    if r["finish"] != "answer":
        continue
    pred = r["prediction"] or ""
    em[key] += r["correct"]
    cover[key] += r["correct"] or any(pad(g) in pad(pred) for g in r["golds"] if normalize_answer(g))
    if not r["correct"]:
        b = bucket(pred, r["golds"])
        form[key][b] += 1
        if b != "other" and len(examples[(key, b)]) < 6:
            examples[(key, b)].append(f"{r['ds']}/{r['id']}: {pred!r} vs {r['golds'][:3]}")

out = {k: {"n": n[k], "em": em[k], "cover_em": cover[k], "em_pct": round(100 * em[k] / n[k], 1),
           "cover_em_pct": round(100 * cover[k] / n[k], 1), "wrong_answer_form": dict(form[k])} for k in n}
out["examples"] = {f"{k}|{b}": v for (k, b), v in examples.items()}
(A / "answer_form.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k in n:
    print(k, json.dumps(out[k], ensure_ascii=False))
for k, v in out["examples"].items():
    if k.startswith("grepseek"):
        print(k, *v, sep="\n  ")
