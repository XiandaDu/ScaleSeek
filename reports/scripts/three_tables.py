"""Our results table under different ways of averaging (EM and F1 per dataset, BCP judge accuracy):
  ours      micro over the 500-question subsets of the 7 QA datasets (what tmp.txt reports)
  paper     macro over 8 datasets, BCP included (the GrepSeek paper's Table 1 / Table 6 "Average")
  README    7 QA datasets weighted by their full sizes (the GrepSeek README's "micro-average")"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, f1_score, final_answer

R = Path("/scratch/a32du/scaleseek_v2/runs")
DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
FULL = dict(zip(DS, [3610, 11313, 14267, 7405, 12576, 2417, 125]))
RUNS = [("Direct", "direct", None), ("RAG BM25", "rag__bm25", None), ("RAG E5", "rag__e5", None), ("RAG Qwen3-4B", "rag__qwen3_emb_4b", None),
        ("Search-O1 BM25", "search_o1__bm25", None), ("Search-O1 E5", "search_o1__e5", None), ("Search-O1 Qwen3-4B", "search_o1__qwen3_emb_4b", None),
        ("Search-R1 7B BM25", "search_r1_7b__bm25", None), ("Search-R1 7B E5", "search_r1_7b__e5", None), ("Search-R1 7B Qwen3-4B", "search_r1_7b__qwen3_emb_4b", None),
        ("Search-R1 14B BM25", "search_r1_14b__bm25", None), ("Search-R1 14B E5", "search_r1_14b__e5", None), ("Search-R1 14B Qwen3-4B", "search_r1_14b__qwen3_emb_4b", None),
        ("AgentIR", "agentir__agentir_4b", None), ("GrepSeek", "grepseek", "grepseek_passages"), ("GrepSeek+", "grepseek_plus", "grepseek_plus_passages"),
        ("ScaleSeek v3 run 1", "scaleseek_v3_prompt__e5", None), ("ScaleSeek v3 run 2", "scaleseek_v3_prompt_r2__e5", None)]


def bcp(run):
    f = R / "browsecomp_plus" / run / "judge__bcp_judge.jsonl"
    if not f.exists():
        return None
    v = [bool(r["judge_result"] and r["judge_result"].get("correct")) for r in map(json.loads, open(f))]
    return 100 * sum(v) / len(v)


for metric, fn in (("EM", lambda r: float(exact_match(final_answer(r), r["golden_answers"]))), ("F1", lambda r: f1_score(final_answer(r), r["golden_answers"]))):
    print(f"\n=== {metric} ===  per dataset | BCP (whole pages; in brackets: 700-word passages) | averages: ours (subset micro, 7 QA) / macro-7 / paper (macro-8) / README (weighted, 7 QA)")
    for label, run, passages in RUNS:
        vals, n = {}, {}
        for ds in DS:
            recs = [json.loads(l) for l in open(R / ds / run / "trajectories.jsonl")]
            vals[ds], n[ds] = 100 * sum(map(fn, recs)) / len(recs), len(recs)
        b, bp = bcp(run), bcp(passages) if passages else None
        ours = sum(vals[d] * n[d] for d in DS) / sum(n.values())
        macro7 = sum(vals.values()) / 7
        b_used = bp if bp is not None else b
        macro8 = (sum(vals.values()) + b_used) / 8 if b_used is not None else float("nan")
        readme = sum(vals[d] * FULL[d] for d in DS) / sum(FULL.values())
        print(f"{label:24s} " + " ".join(f"{vals[d]:5.1f}" for d in DS) + f" | {b if b is not None else float('nan'):5.1f}" + (f" [{bp:4.1f}]" if bp is not None else "       ")
              + f" | {ours:5.1f} {macro7:5.1f} {macro8:5.1f} {readme:5.1f}")
