"""Two checks.
1. Denominator: EM/F1 over all questions (what we report) vs over answered questions only, for
   GrepSeek against the paper's F1, and answered-only EM for every run.
2. RAG refusals: how many RAG answers are refusals, how many of those questions Direct got right,
   and the EM RAG would have if each refusal were replaced by Direct's answer."""
import json
import sys
import re
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import f1_score, final_answer

R = Path("/scratch/a32du/scaleseek_v2/runs")
WIKI = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
PAPER_F1 = {"nq": 52.2, "triviaqa": 76.7, "popqa": 48.6, "hotpotqa": 62.3, "2wikimultihopqa": 51.8,
            "musique": 30.1, "bamboogle": 62.1}
REFUSAL = re.compile(r"not (mentioned|available|provided|specified|found|stated|given|determinable)|cannot be determined"
                     r"|insufficient|no information|unable to determine|^unknown$|^none$|^n/?a$")


def load(ds, run):
    m = json.loads((R / ds / run / "metrics.json").read_text())
    recs = {r["id"]: r for r in map(json.loads, open(R / ds / run / "trajectories.jsonl"))}
    return m, recs


print("1. GrepSeek F1: all questions (ours) / answered only / paper")
for ds in WIKI:
    m, recs = load(ds, "grepseek")
    f1 = {i: f1_score(final_answer(r), r["golden_answers"]) for i, r in recs.items()}
    ans = [i for i, r in recs.items() if r["finish_reason"] == "answer"]
    print(f"  {ds:16s} {100 * m['f1']['mean']:5.1f} / {100 * sum(f1[i] for i in ans) / len(ans):5.1f} / {PAPER_F1[ds]}")

print("\n   EM over answered questions only (Wiki, pooled)")
for run in ["grepseek", "search_r1_14b__qwen3_emb_4b", "search_o1__qwen3_emb_4b", "rag__qwen3_emb_4b",
            "agentir__agentir_4b", "direct"]:
    n = ok = answered = 0
    for ds in WIKI:
        m, recs = load(ds, run)
        em = m["per_question_em"]
        n += len(recs)
        ok += sum(em.values())
        answered += sum(r["finish_reason"] == "answer" for r in recs.values())
    print(f"  {run:30s} all {100 * ok / n:5.1f}   answered-only {100 * ok / answered:5.1f}   ({answered}/{n} answered)")

print("\n2. RAG refusals vs Direct on the same questions")
print("  run / dataset          EM   refusals  Direct right on them  EM with Direct fallback  Direct EM")
for run in ["rag__bm25", "rag__e5", "rag__qwen3_emb_4b"]:
    for ds in WIKI:
        m, recs = load(ds, run)
        md, _ = load(ds, "direct")
        em, dem = m["per_question_em"], md["per_question_em"]
        refused = [i for i, r in recs.items() if REFUSAL.search((r["prediction"] or "").strip().lower())]
        rescued = sum(dem[i] for i in refused)
        n = len(recs)
        print(f"  {run:18s} {ds:16s} {100 * m['em']['mean']:5.1f}  {len(refused):4d}     {rescued:4d}"
              f"                  {100 * (sum(em.values()) + rescued) / n:5.1f}                  {100 * md['em']['mean']:5.1f}")
