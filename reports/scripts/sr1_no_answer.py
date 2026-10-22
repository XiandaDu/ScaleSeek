"""Search-R1 `no_answer` records: how does the last generation end, and is it presence_penalty?

A no_answer record is one whose last generation stopped on EOS without an <answer>.  For every
Search-R1 run (formal: presence_penalty 2.0) and the Search-R1-7B bm25 pilots without the penalty
(pp0), count generations that end on EOS with neither </search> nor <answer> ("dead" turns),
how long they are, and how often an answer turn carries text after </answer>.
Writes analysis/failures/sr1_no_answer.json.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/scratch/a32du/scaleseek_v2")
DS = ["bamboogle", "popqa", "triviaqa", "nq", "hotpotqa", "2wikimultihopqa", "musique", "browsecomp_plus"]
RUNS = [f"search_r1_{m}__{r}" for m in ("7b", "14b") for r in ("bm25", "e5", "qwen3_emb_4b")]


def bucket(ct):
    return "1" if ct <= 1 else "2" if ct == 2 else "3-20" if ct <= 20 else "21-200" if ct <= 200 else ">200"


def profile(path):
    n = no_answer = turns = dead = answer_turns = junk_after_answer = 0
    dead_len, samples = Counter(), []
    for line in open(path):
        r = json.loads(line)
        n += 1
        no_answer += r["finish_reason"] == "no_answer"
        for t in r["trajectory"]:
            if t["type"] != "llm":
                continue
            turns += 1
            text = t["text"]
            if "</answer>" in text:
                answer_turns += 1
                junk_after_answer += bool(text.split("</answer>", 1)[1].strip())
            elif t["finish_reason"] == "stop" and t["stop_reason"] is None:
                dead += 1
                dead_len[bucket(t["completion_tokens"])] += 1
                if len(samples) < 6:
                    samples.append({"id": r["id"], "ct": t["completion_tokens"], "prompt": t["prompt_tokens"],
                                    "tail": text[-70:]})
    return {"n": n, "no_answer": no_answer, "llm_turns": turns, "dead_eos_turns": dead,
            "dead_eos_per_1k_turns": round(1000 * dead / turns, 1), "dead_len": dead_len,
            "answer_turns": answer_turns, "junk_after_answer": junk_after_answer, "samples": samples}


if __name__ == "__main__":
    out = {}
    for ds in DS:
        for run in RUNS:
            out[f"{ds}/{run}"] = profile(ROOT / "runs" / ds / run / "trajectories.jsonl")
    for ds in ("bamboogle", "popqa", "hotpotqa", "2wikimultihopqa"):
        out[f"{ds}/search_r1_7b__bm25 pp0 pilot"] = profile(ROOT / "dev" / "pilots" / f"sr1_7b_bm25_{ds}_pp0" / "trajectories.jsonl")
    (ROOT / "analysis" / "failures" / "sr1_no_answer.json").write_text(json.dumps(out, indent=1, default=dict))
    for k, v in out.items():
        print(f"{k:55s} n={v['n']:4d} no_answer={v['no_answer']:4d} dead/1k={v['dead_eos_per_1k_turns']:6.1f} "
              f"dead_len={dict(v['dead_len'])} junk_after_answer={v['junk_after_answer']}/{v['answer_turns']}")
