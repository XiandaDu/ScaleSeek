#!/usr/bin/env python3
"""Independent Qwen3.5-9B BrowseComp-Plus Appendix-F judge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.browsecomp_plus_judge import BCP_F_JUDGE_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.model != "Qwen/Qwen3.5-9B":
        raise SystemExit("Formal BCP judge is frozen to Qwen/Qwen3.5-9B")
    expected_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    if args.model_revision != expected_revision:
        raise SystemExit(f"Judge revision must be {expected_revision}")
    prompt_hash = hashlib.sha256(BCP_F_JUDGE_PROMPT.encode()).hexdigest()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    prior = {}
    if args.output.exists() and not args.force:
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row.get("judge_model_revision") != expected_revision \
                    or row.get("judge_prompt_sha256") != prompt_hash:
                raise SystemExit("Existing judge output has mismatched provenance")
            prior[str(row["id"])] = row
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    output = []
    for row in rows:
        rid = str(row["id"])
        if rid in prior:
            output.append(prior[rid]); continue
        gold = row.get("gold_answer") or (row.get("gold_answers") or [""])[0]
        prompt = BCP_F_JUDGE_PROMPT.format(
            question=row.get("question", ""),
            response=row.get("prediction") or row.get("final_text") or "[empty]",
            correct_answer=gold)
        response = None
        for attempt in range(args.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, temperature=0.0, top_p=1.0,
                    max_tokens=4096)
                break
            except Exception:
                if attempt == args.retries:
                    raise
        raw = response.choices[0].message.content or "{}"
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            verdict = {"correct": "no", "reasoning": "judge_json_parse_error",
                       "raw": raw, "extracted_final_answer": "None", "confidence": 0}
        output.append({"id": rid, "judge_model": args.model,
                       "judge_model_revision": expected_revision,
                       "judge_prompt": "BCP Appendix F (verbatim)",
                       "judge_prompt_sha256": prompt_hash,
                       "judge_temperature": 0.0, "judge_top_p": 1.0,
                       "verdict": verdict, "raw_judge_output": raw})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                       for item in output))


if __name__ == "__main__":
    main()
