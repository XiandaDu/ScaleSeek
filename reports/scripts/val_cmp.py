"""Validation runs (500 NQ dev + 500 HotpotQA dev, v3 protocol) against the untrained v3
(val_v3_base): strict EM by source and overall, F1, paired exact McNemar, turns, and how
often the last (forced) turn was reached.
Usage: val_cmp.py <run name>...   (dirs under train_data/teacher/)"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

T = Path("/scratch/a32du/scaleseek_v2/train_data/teacher")


def load(name):
    return {r["id"]: r for r in map(json.loads, open(T / name / "trajectories.jsonl"))}


ref = load("val_v3_base")
em = lambda run, ids: {i: exact_match(final_answer(run[i]), run[i]["golden_answers"]) for i in ids}
for name in ["val_v3_base"] + sys.argv[1:]:
    run = load(name)
    ids = sorted(set(run) & set(ref))
    a, b = em(run, ids), em(ref, ids)
    mc = mcnemar_exact(a, b)
    rs = [run[i] for i in ids]
    turns = json.loads((T / name / "config.json").read_text())["profile"]["max_assistant_turns"]
    by = {s: 100 * sum(a[i] for i in ids if i.startswith(s)) / sum(i.startswith(s) for i in ids) for s in ("nq", "hotpotqa")}
    calls = Counter(t["type"] for r in rs for t in r["trajectory"] if t["type"] in ("find", "search"))
    print(f"{name:24s} n={len(ids)}  EM {100 * sum(a.values()) / len(ids):5.1f} (NQ {by['nq']:.1f}, HotpotQA {by['hotpotqa']:.1f})"
          f"  F1 {100 * sum(f1_score(final_answer(r), r['golden_answers']) for r in rs) / len(ids):5.1f}"
          f"  vs base: won {mc['only_a']} lost {mc['only_b']} p={mc['p_value']:.3f}"
          f"  turns {sum(r['n_turns'] for r in rs) / len(ids):.2f}  last turn {100 * sum(r['n_turns'] == turns for r in rs) / len(ids):.1f}%"
          f"  find/search per traj {calls['find'] / len(ids):.2f}/{calls['search'] / len(ids):.2f}"
          f"  errors {sum(any(r['errors'].values()) for r in rs)}")
