"""One ledger per run and dataset: every question lands in exactly one bucket.

  correct
  unanswered: max_turns / context_exhausted (incl. the BCP 10K cap) / no_answer (incl. parse errors)
  answered but wrong, first matching class wins:
    abstain      the answer is a refusal ("Not mentioned", "Information not available", ...)
    form         Wiki only: semantically close, EM says no (list with a gold, bare year of a gold
                 date, prediction contains a gold, a gold contains the prediction)
    gold_seen    a gold string appeared in some tool output / retrieved document
    gold_unseen  it never did
    unmatchable  gold too short or yes/no, so seen/unseen cannot be checked
Every question stays in the denominator: an unanswered question scores 0.
Writes failure_accounting.json next to this script."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import normalize_answer

A = Path("/scratch/a32du/scaleseek_v2/analysis/failures")
YEAR = re.compile(r"^\d{4}$")
REFUSAL = re.compile(r"not (mentioned|available|provided|specified|found|stated|given|determinable)|cannot be determined"
                     r"|insufficient|no information|unable to determine|^unknown$|^none$|^n/?a$")
UNANSWERED = {"max_turns": "max_turns", "context_exhausted": "context_exhausted", "max_tokens_cap": "context_exhausted",
              "no_answer": "no_answer", "parse_error": "no_answer"}


def pad(s):
    return f" {normalize_answer(s)} "


def is_form(pred, golds):
    p, gs = pad(pred), [pad(g) for g in golds if normalize_answer(g)]
    return any(g in p or p in g for g in gs) or (YEAR.match(p.strip()) and any(p in g for g in gs))


def bucket(r):
    if r["correct"]:
        return "correct"
    if r["category"] in UNANSWERED:
        return UNANSWERED[r["category"]]
    pred = r["prediction"] or ""
    if REFUSAL.search(pred.strip().lower()):
        return "abstain"
    if r["ds"] != "browsecomp_plus" and normalize_answer(pred) and is_form(pred, r["golds"]):
        return "form"
    cat = r["category"]
    if cat.endswith(("gold_seen", "gold_retrieved")):
        return "gold_seen"
    if cat.endswith(("never_seen", "not_retrieved")):
        return "gold_unseen"
    return "unmatchable"


rows = [{**json.loads(l), "run": "grepseek"} for l in open(A / "grepseek_features.jsonl")]
rows += [json.loads(l) for l in open(A / "others_features.jsonl")]
ledger = defaultdict(lambda: defaultdict(Counter))
for r in rows:
    ledger[r["run"]][r["ds"]][bucket(r)] += 1
out = {run: {ds: dict(c) for ds, c in by_ds.items()} for run, by_ds in ledger.items()}
Path(__file__).with_name("failure_accounting.json").write_text(json.dumps(out, indent=1))
for run, by_ds in out.items():
    wiki = Counter()
    for ds, c in by_ds.items():
        if ds != "browsecomp_plus":
            wiki.update(c)
    print(run, "WIKI", dict(wiki.most_common()), "BCP", by_ds["browsecomp_plus"])
