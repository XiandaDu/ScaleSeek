"""SFT rows from teacher trajectories (train/teacher_rollouts.py): rejection sampling.

Kept: trajectories that end in an `answer` call that is correct under the shared rule
(strict EM) with no tool or format error -- at most `--per-question` of them per question,
the ones with the fewest turns first.
Each is cut into the segments train/scaleseek_rl/sft_dataset.py trains on: the prompt and
the tool turns (no loss), each assistant turn plus its EOS (loss).

    python train/build_sft.py <teacher run dir> <out_dir> [--per-question 1]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleseek.config import load_base
from scaleseek.hub import snapshot_dir
from scaleseek.metrics import exact_match, final_answer

SEED = 20261105


GEN = "<|im_start|>assistant\n<think>\n"       # the chat template's generation prompt (Qwen3.5, thinking on)


def segments(rec: dict, eos: str) -> list[dict]:
    """The final sequence cut at the generation prompts: what the harness wrote (the prompt,
    the tool turns: no loss) and what the model wrote in each assistant turn, up to its EOS
    (loss).  A forced last turn (`commit`) is one assistant turn like any other, so SFT
    teaches the model to write that answer call itself."""
    final = rec["trajectory"][-1]["text"]                    # the final_sequence entry
    segs, pos = [], 0
    while (i := final.find(GEN, pos)) != -1:
        start = i + len(GEN)
        end = final.index(eos, start) + len(eos)
        segs += [{"text": final[pos:start], "train": False}, {"text": final[start:end], "train": True}]
        pos = end
    assert pos == len(final), rec["id"]
    return segs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("teacher_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--per-question", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=12000, help="longer trajectories are left out")
    a = p.parse_args()
    teacher = json.loads((a.teacher_dir / "config.json").read_text())["teacher"]
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(load_base(), teacher["model"], teacher["revision"])))
    recs = [json.loads(l) for l in open(a.teacher_dir / "trajectories.jsonl")]
    by_q = defaultdict(list)
    stats = Counter()
    for r in recs:
        ok = (r["finish_reason"] == "answer" and not any(r["errors"].values())
              and exact_match(final_answer(r), r["golden_answers"]))
        stats["kept" if ok else f"dropped {r['finish_reason']}"] += 1
        if ok:
            by_q[r["id"].split("#")[0]].append(r)
    rng = random.Random(SEED)
    rows = []
    for qid, rs in sorted(by_q.items()):
        rng.shuffle(rs)
        for r in sorted(rs, key=lambda r: r["n_turns"])[: a.per_question]:      # the shortest correct ones: efficient search
            segs = segments(r, tok.eos_token)
            if len(tok.encode("".join(x["text"] for x in segs), add_special_tokens=False)) > a.max_tokens:
                stats["dropped too long"] += 1
                continue
            rows.append({"id": r["id"], "source": qid.split("_")[0], "turns": r["n_turns"], "segments": segs})
    rng.shuffle(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out_dir / "train.parquet")
    summary = {"teacher_dir": str(a.teacher_dir), "per_question": a.per_question, "trajectories": len(recs),
               "questions_solved": len(by_q), "rows": len(rows), "by_source": Counter(r["source"] for r in rows),
               "turns": dict(sorted(Counter(r["turns"] for r in rows).items())),
               "filter": dict(stats)}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
