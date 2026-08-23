#!/usr/bin/env python3
"""Extract concise answers from DCI free-prose reports for EM/F1 scoring.

Reads the normalized DCI rows (prediction = full final report), asks the
frozen generator (temperature 0, thinking disabled) to extract the answer
span each report commits to, and writes a sibling normalized file with
prediction replaced by that span -- every other field, and especially the
provenance block, is carried over unchanged, plus audit fields:

  raw_final_text        the untouched report the span came from
  extraction_prompt_sha256
  extraction_model / extraction_revision
  extraction_status     extracted | no_answer | empty_input | parse_error

Gold answers are never shown to the extractor. Rows whose report is empty
skip the LLM entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts.dci_answer_extraction import PROMPT  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    session = requests.Session()

    def extract(row: dict) -> dict:
        out = dict(row)
        report = (row.get("prediction") or "").strip()
        out["raw_final_text"] = row.get("prediction")
        out["extraction_prompt_sha256"] = prompt_sha
        out["extraction_model"] = args.model
        out["extraction_revision"] = args.revision
        if not report:
            out["prediction"] = ""
            out["extraction_status"] = "empty_input"
            return out
        user = f"QUESTION:\n{row.get('question','')}\n\nREPORT:\n{report}"
        for attempt in range(3):
            try:
                r = session.post(
                    f"{args.base_url}/chat/completions",
                    json={
                        "model": args.model,
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": args.max_tokens,
                        "messages": [{"role": "system", "content": PROMPT},
                                     {"role": "user", "content": user}],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=300,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    out["prediction"] = ""
                    out["extraction_status"] = f"parse_error: {type(exc).__name__}"
                    return out
        matches = ANSWER_RE.findall(content)
        if not matches:
            out["prediction"] = ""
            out["extraction_status"] = "parse_error"
            return out
        span = matches[-1].strip()
        if span.upper() == "NO_ANSWER" or not span:
            out["prediction"] = ""
            out["extraction_status"] = "no_answer"
        else:
            out["prediction"] = span
            out["extraction_status"] = "extracted"
        return out

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(extract, rows))

    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    from collections import Counter
    stats = Counter(r["extraction_status"].split(":")[0] for r in results)
    print(f"[extract] {len(results)} rows -> {args.output}: {dict(stats)}")
    if stats.get("parse_error", 0) > 0.02 * len(results):
        sys.exit("EXTRACT-FAIL: >2% parse errors")


if __name__ == "__main__":
    main()
