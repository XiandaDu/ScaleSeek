"""A run vs GrepSeek (and the other baselines) on the 7 Wiki test subsets: strict EM and F1,
paired exact McNemar vs GrepSeek per dataset and pooled, finish reasons.
Usage: test_cmp.py <run name, e.g. grepseek_plus>"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact

R = Path("/scratch/a32du/scaleseek_v2/runs")
DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
RUNS = ["direct", "rag__e5", "search_o1__qwen3_emb_4b", "search_r1_7b__e5", "search_r1_14b__e5",
        "search_r1_14b__qwen3_emb_4b", "grepseek", sys.argv[1]]


def load(ds, run):
    p = R / ds / run / "trajectories.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else None


for metric, fn in (("strict EM", lambda r: float(exact_match(final_answer(r), r["golden_answers"]))),
                   ("token F1", lambda r: f1_score(final_answer(r), r["golden_answers"]))):
    print(f"{metric:28s}" + "".join(f"{d[:8]:>9s}" for d in DS) + f"{'micro':>9s}")
    for run in RUNS:
        cells, tot, n = [], 0.0, 0
        for ds in DS:
            recs = load(ds, run)
            if recs is None:
                cells.append(f"{'-':>9s}")
                continue
            v = [fn(r) for r in recs.values()]
            cells.append(f"{100 * sum(v) / len(v):9.1f}")
            tot, n = tot + sum(v), n + len(v)
        print(f"{run:28s}" + "".join(cells) + (f"{100 * tot / n:9.1f}" if n == 3125 else f"   (n={n})"))
    print()

print(f"{sys.argv[1]} vs grepseek, per dataset")
pa, pb = {}, {}
for ds in DS:
    a, b = load(ds, sys.argv[1]), load(ds, "grepseek")
    if a is None:
        continue
    ea = {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in a.items()}
    eb = {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in b.items()}
    mc = mcnemar_exact(ea, eb)
    fa, fb = Counter(r["finish_reason"] for r in a.values()), Counter(r["finish_reason"] for r in b.values())
    n = len(a)
    print(f"  {ds:16s} n={n}  EM {100 * sum(ea.values()) / n:5.1f} vs {100 * sum(eb.values()) / n:5.1f} ({100 * (sum(ea.values()) - sum(eb.values())) / n:+5.1f}; "
          f"won {mc['only_a']:3d} lost {mc['only_b']:3d} p={mc['p_value']:.4f})  answered {100 * fa['answer'] / n:5.1f}% vs {100 * fb['answer'] / n:5.1f}%"
          f"  calls/q {sum(r['n_retrievals'] for r in a.values()) / n:.2f} vs {sum(r['n_retrievals'] for r in b.values()) / n:.2f}"
          f"  tool errors {sum(r['errors']['tool'] for r in a.values())} vs {sum(r['errors']['tool'] for r in b.values())}"
          f"  finish {dict(fa.most_common())}")
    pa.update({f"{ds}/{i}": v for i, v in ea.items()})
    pb.update({f"{ds}/{i}": v for i, v in eb.items()})
if pa:
    mc = mcnemar_exact(pa, pb)
    print(f"  pooled n={len(pa)}  EM {100 * sum(pa.values()) / len(pa):.1f} vs {100 * sum(pb.values()) / len(pb):.1f}  won {mc['only_a']} lost {mc['only_b']} p={mc['p_value']:.2e}")
