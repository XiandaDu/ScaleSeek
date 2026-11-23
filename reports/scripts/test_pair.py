"""Runs on the 7 Wiki test subsets, each compared with the first one: strict EM per dataset,
the three averages (subset micro, macro over 7, weighted by full dataset sizes), F1, paired exact McNemar.
Usage: test_pair.py <reference run> <run>...   (dirs under runs/<dataset>/)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

R = Path("/scratch/a32du/scaleseek_v2/runs")
FULL = {"nq": 3610, "triviaqa": 11313, "popqa": 14267, "hotpotqa": 7405, "2wikimultihopqa": 12576, "musique": 2417, "bamboogle": 125}


def load(run):
    out = {}
    for ds in FULL:
        for r in map(json.loads, open(R / ds / run / "trajectories.jsonl")):
            out[f"{ds}/{r['id']}"] = r
    return out


def row(run, recs, fn):
    per = {ds: [fn(r) for i, r in recs.items() if i.startswith(ds + "/")] for ds in FULL}
    mean = {ds: 100 * sum(v) / len(v) for ds, v in per.items()}
    micro = 100 * sum(map(sum, per.values())) / sum(map(len, per.values()))
    weighted = sum(mean[ds] * n for ds, n in FULL.items()) / sum(FULL.values())
    return f"{run:28s}" + "".join(f"{mean[ds]:7.1f}" for ds in FULL) + f" |{micro:6.1f}{sum(mean.values()) / 7:7.2f}{weighted:7.2f}"


em = lambda r: exact_match(final_answer(r), r["golden_answers"])
ref = load(sys.argv[1])
runs = {name: load(name) for name in sys.argv[1:]}
for label, fn in (("EM", em), ("F1", lambda r: f1_score(final_answer(r), r["golden_answers"]))):
    print(f"{label:28s}" + "".join(f"{ds[:6]:>7s}" for ds in FULL) + " | micro macro7 weighted")
    for name, recs in runs.items():
        print(row(name, recs, fn))
for name, recs in list(runs.items())[1:]:
    assert set(recs) == set(ref), name
    mc = mcnemar_exact({i: em(r) for i, r in recs.items()}, {i: em(r) for i, r in ref.items()})
    turns = sum(r["n_turns"] for r in recs.values()) / len(recs)
    print(f"{name} vs {sys.argv[1]}: won {mc['only_a']} lost {mc['only_b']} p={mc['p_value']:.3g}   turns {turns:.2f} (ref {sum(r['n_turns'] for r in ref.values()) / len(ref):.2f})"
          f"   answered {100 * sum(r['finish_reason'] == 'answer' for r in recs.values()) / len(recs):.1f}%")
