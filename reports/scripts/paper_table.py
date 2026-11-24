"""Numbers for the paper's QA tables: EM / F1 x100 per run and dataset, the micro average over the
seven Wiki datasets (all questions pooled), and the BCP judge accuracy.  Writes paper_table.json."""
import json
from pathlib import Path

R = Path("/scratch/a32du/scaleseek_v2/runs")
WIKI = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
RUNS = ["direct"] + [f"{m}__{r}" for m in ("rag", "search_o1", "search_r1_7b", "search_r1_14b")
                     for r in ("bm25", "e5", "qwen3_emb_4b")] + ["grepseek", "agentir__agentir_4b"]

out = {}
for run in RUNS:
    row, tot = {}, {"n": 0, "em": 0.0, "f1": 0.0}
    for ds in WIKI:
        m = json.loads((R / ds / run / "metrics.json").read_text())
        row[ds] = {"n": m["n"], "em": round(100 * m["em"]["mean"], 1), "f1": round(100 * m["f1"]["mean"], 1)}
        tot["n"] += m["n"]
        tot["em"] += m["em"]["mean"] * m["n"]
        tot["f1"] += m["f1"]["mean"] * m["n"]
    row["micro"] = {"n": tot["n"], "em": round(100 * tot["em"] / tot["n"], 1), "f1": round(100 * tot["f1"] / tot["n"], 1)}
    bcp = json.loads((R / "browsecomp_plus" / run / "metrics.json").read_text())
    row["bcp"] = {"n": bcp["n"], "acc": round(100 * bcp["judge"]["accuracy"]["mean"], 1)}
    out[run] = row
    print(f"{run:28s}", " ".join(f"{row[d]['em']:5.1f}/{row[d]['f1']:5.1f}" for d in WIKI + ["micro"]), f"BCP {row['bcp']['acc']}")
Path(__file__).with_name("paper_table.json").write_text(json.dumps(out, indent=1))
