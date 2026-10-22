"""presence_penalty 2.0 (formal) vs 0.0 (pilot) on the two tag-protocol failures:
Search-R1-14B e5 stopping mid-sentence (2Wiki, BCP) and Search-o1 bm25 misspelling its search tags (NQ).
BCP pilots are not judged: only how runs end is compared there."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
sys.path.insert(0, str(Path(__file__).parent))
from scaleseek.metrics import mcnemar_exact, run_metrics
from sr1_no_answer import profile as eos_profile
from tag_errors import profile as tag_profile

R = Path("/scratch/a32du/scaleseek_v2")
PAIRS = [("search_r1_14b__e5", "2wikimultihopqa", "sr1_14b_e5_2wikimultihopqa_pp0", eos_profile),
         ("search_r1_14b__e5", "browsecomp_plus", "sr1_14b_e5_browsecomp_plus_pp0", eos_profile),
         ("search_o1__bm25", "nq", "so1_bm25_nq_pp0", tag_profile)]

for run, ds, pilot, profile in PAIRS:
    dirs = {"pp2.0": R / "runs" / ds / run, "pp0.0": R / "dev" / "pilots" / pilot}
    if not (dirs["pp0.0"] / "trajectories.jsonl").exists():
        print(f"{run} {ds}: pilot not run")
        continue
    em = {}
    for label, d in dirs.items():
        recs = [json.loads(l) for l in open(d / "trajectories.jsonl")]
        m = run_metrics(recs, 1.96, 0.0)
        em[label] = m["per_question_em"]
        p = profile(d / "trajectories.jsonl")
        keep = {k: p[k] for k in ("no_answer", "dead_eos_per_1k_turns", "dead_len", "questions_with_bad_tag",
                                  "bad_turns", "main_turns", "finish_of_bad_questions") if k in p}
        print(f"{run} {ds} {label}: n={len(recs)} EM={100 * m['em']['mean']:.1f} "
              f"finish={dict(Counter(r['finish_reason'] for r in recs))} {json.dumps(keep, default=dict)}")
    if ds != "browsecomp_plus":
        print("   McNemar (EM):", mcnemar_exact(em["pp2.0"], em["pp0.0"]))
