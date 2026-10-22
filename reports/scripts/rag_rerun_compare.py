"""RAG with the own-knowledge fallback (runs/) vs without it (archive/rag_no_fallback/) vs the v1
paper table, with Direct for reference: EM (BCP: judge accuracy) and the number of refusals."""
import json
import re
from pathlib import Path

ROOT = Path("/scratch/a32du/scaleseek_v2")
DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "browsecomp_plus"]
V1 = {"bm25": {"nq": 32.8, "triviaqa": 66.6, "popqa": 33.5, "2wikimultihopqa": 36.2, "bamboogle": 43.2, "browsecomp_plus": 3.0},
      "e5": {"nq": 42.3, "triviaqa": 68.4, "popqa": 44.7, "2wikimultihopqa": 36.4, "bamboogle": 56.0, "browsecomp_plus": 0.1},
      "qwen3_emb_4b": {"nq": 40.8, "triviaqa": 68.0, "popqa": 46.5, "2wikimultihopqa": 38.8, "bamboogle": 49.6, "browsecomp_plus": 14.7}}
REFUSAL = re.compile(r"not (mentioned|available|provided|specified|found|stated|given|determinable)|cannot be determined"
                     r"|insufficient|no information|unable to determine|^unknown$|^none$|^n/?a$")


def score(run_dir: Path) -> float:
    m = json.loads((run_dir / "metrics.json").read_text())
    return 100 * (m["judge"]["accuracy"]["mean"] if "judge" in m else m["em"]["mean"])


def refusals(run_dir: Path) -> int:
    return sum(bool(REFUSAL.search((json.loads(l)["prediction"] or "").strip().lower()))
               for l in open(run_dir / "trajectories.jsonl"))


print("retriever  dataset           new (refusals) | without fallback (refusals) | v1 table | Direct")
for r in ("bm25", "e5", "qwen3_emb_4b"):
    for ds in DS:
        new, old = ROOT / "runs" / ds / f"rag__{r}", ROOT / "archive" / "rag_no_fallback" / ds / f"rag__{r}"
        v1 = V1[r].get(ds)
        print(f"{r:12s} {ds:16s} {score(new):5.1f} ({refusals(new):3d}) | {score(old):5.1f} ({refusals(old):3d})"
              f"            | {'' if v1 is None else v1:>5} | {score(ROOT / 'runs' / ds / 'direct'):5.1f}")
