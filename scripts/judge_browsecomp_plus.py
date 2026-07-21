#!/usr/bin/env python3
"""Independent Qwen3.5-9B BrowseComp-Plus Appendix-F judge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Return ONLY a compact JSON object with keys: extracted_final_answer, reasoning, correct, confidence."""


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
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
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
        prompt = PROMPT.format(question=row.get("question", ""),
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
