#!/usr/bin/env python3
"""Generate ScaleSeek SFT cold-start trajectories with a teacher model.

Reads canonical QA rows ({"id","question","golden_answers"}) and writes one
trajectory per line (messages + assistant-only loss mask + status/meta), using
``train.sft.coldstart`` and the same BM25 tool environment as evaluation.

Smoke run (single 4090, tiny local index):
    python scripts/make_smoke_corpus.py --out-dir .smoke --build-index
    python scripts/generate_sft_data.py \
        --questions .smoke/questions.jsonl \
        --index-dir .smoke/bm25_index \
        --teacher hf:Qwen/Qwen3-4B \
        --out .smoke/sft_trajectories.jsonl

Cluster run (vLLM teacher endpoint + full wiki-18 index):
    python scripts/generate_sft_data.py --questions $DATA/rl_data/train.jsonl \
        --index-dir $BM25_INDEX_DIR --teacher openai:http://127.0.0.1:8000/v1 \
        --out $DATA/sft/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.bm25_retriever import BM25Retriever
from train.sft.coldstart import ColdStartConfig, build_trajectory
from train.sft.teacher import build_teacher


def _load_questions(path: str, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("question") and r.get("golden_answers"):
                rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def _load_tokenizer(spec: str | None):
    if not spec:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec, trust_remote_code=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="JSONL of {id,question,golden_answers}")
    ap.add_argument("--out", required=True, help="output trajectories JSONL")
    ap.add_argument("--teacher", required=True, help="teacher spec: 'hf:<model>' or 'openai:<base_url>'")
    ap.add_argument("--teacher-model", default=None, help="model name for openai backend")
    ap.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"),
                    help="BM25 Lucene index dir ($BM25_INDEX_DIR)")
    ap.add_argument("--student-tokenizer", default=None,
                    help="tokenizer used to budget tool responses (aligns train/inference); "
                         "default: reuse the teacher tokenizer if local")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-refine", type=int, default=2)
    ap.add_argument("--tool-response-tokens", type=int, default=512)
    ap.add_argument("--strict", action="store_true", help="skip examples whose hops don't verify")
    ap.add_argument("--no-quality-judge", action="store_true")
    ap.add_argument("--param-policy", choices=["heuristic", "search", "teacher"], default="heuristic",
                    help="'heuristic': assign k1/b/top_k/mode from query features; "
                         "'search': grid-search BM25 params and teach the ones that best rank the "
                         "target passage (empirically grounded); 'teacher': keep what the teacher emitted")
    args = ap.parse_args()

    if not args.index_dir:
        sys.exit("Set --index-dir or $BM25_INDEX_DIR")

    os.environ["BM25_INDEX_DIR"] = args.index_dir
    retriever = BM25Retriever(index_dir=args.index_dir)
    teacher = build_teacher(args.teacher, model=args.teacher_model)
    tokenizer = _load_tokenizer(args.student_tokenizer)
    if tokenizer is None:
        tokenizer = getattr(teacher, "tokenizer", None)

    cfg = ColdStartConfig(
        max_refine=args.max_refine,
        tool_response_tokens=args.tool_response_tokens,
        strict=args.strict,
        run_quality_judge=not args.no_quality_judge,
        param_policy=args.param_policy,
    )

    questions = _load_questions(args.questions, args.limit)
    print(f"[generate_sft_data] {len(questions)} questions | teacher={args.teacher} | index={args.index_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "quality_fail": 0, "skipped": 0}
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i, ex in enumerate(questions):
            traj = build_trajectory(ex, teacher, retriever, tokenizer=tokenizer, cfg=cfg)
            counts[traj.status] = counts.get(traj.status, 0) + 1
            fout.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            n_written += 1
            print(f"  [{i+1}/{len(questions)}] {ex.get('id')}: {traj.status} "
                  f"(hops={traj.meta.get('n_hops')}, tools={traj.meta.get('n_tool_calls')}, "
                  f"verified={traj.meta.get('all_hops_verified')})")

    summary = {
        "questions": len(questions), "written": n_written, "counts": counts,
        "teacher": args.teacher, "index_dir": args.index_dir,
        "kept_for_sft": counts.get("ok", 0),
    }
    (out_path.parent / (out_path.stem + ".summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\n[generate_sft_data] {counts} -> {out_path}")
    print(f"  SFT-eligible (status=ok): {counts.get('ok', 0)}")


if __name__ == "__main__":
    main()
