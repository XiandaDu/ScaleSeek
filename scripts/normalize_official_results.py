#!/usr/bin/env python3
"""Schema-only adapter for outputs produced by pinned official harnesses."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.config import resolved_method  # noqa: E402


def get(row, path: str, default=None):
    value = row
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def load_rows(path: Path) -> list[tuple[str, dict]]:
    if path.is_file() and path.suffix == ".jsonl":
        return [(f"{path}:{i + 1}", json.loads(line)) for i, line in enumerate(
            path.read_text().splitlines()) if line.strip()]
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    return [(str(file), json.loads(file.read_text())) for file in files]


def _canonical_id(raw: str, prefix: str) -> str:
    """Map a harness-side id onto our canonical id space.

    Harnesses fed the raw FlashRAG file emit its ids verbatim ("test_8"), while
    our canonical dataset namespaces them per dataset ("popqa_test_8"). Without
    this the exact-ID-set check below rejects a perfectly good run: GrepSeek job
    7291 evaluated all 14,267 questions and its recovery then failed on
    "ID set differs from the canonical dataset" for nothing but a missing prefix.

    Idempotent on purpose -- an id that already carries the prefix is left alone,
    so passing --id-prefix to a harness that was fed the normalized file is a
    no-op rather than producing "popqa_popqa_test_8".
    """
    if not prefix or raw.startswith(prefix):
        return raw
    return prefix + raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["grepseek", "dci", "dr_dci", "rise", "agentir"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True,
                        help="Canonical normalized dataset JSONL for exact ID-set validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--id-prefix", default="",
                        help="Prepended to each harness id unless already present, "
                             "to reach our canonical namespace (e.g. 'popqa_' for a "
                             "harness fed the raw FlashRAG file, whose ids are "
                             "'test_N' while ours are 'popqa_test_N'). Leave empty "
                             "for harnesses fed our own converted datasets.")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--gold-field", default="gold_answers")
    parser.add_argument("--gold-split", default="",
                        help="Separator to split a joined gold string back into a "
                             "list. The dci-lite dataset conversion joins multiple "
                             "golds with ' / ' into one answer string, and the "
                             "harness echoes it verbatim; scoring against the "
                             "joined string would fail every multi-gold question.")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--answer-regex", default="",
                        help="Multiline regex whose group(1), from the LAST match, "
                             "becomes the prediction (e.g. the official DR-DCI "
                             "answer format's '^Exact Answer: ...' line). No match "
                             "keeps the full text; answer_extracted records which "
                             "path was taken. Format conversion only -- the raw "
                             "final_text stays in the trajectory files.")
    parser.add_argument("--finish-field", default="finish_reason")
    parser.add_argument("--turns-field", default="turns")
    parser.add_argument("--tool-calls-field", default="n_tool_calls")
    args = parser.parse_args()
    answer_re = re.compile(args.answer_regex, re.MULTILINE) if args.answer_regex else None
    run_record = json.loads(args.run_record.read_text())
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    config = resolved_method(args.method)
    if run_record.get("method") != args.method:
        raise SystemExit("Run record method mismatch")
    output = []
    for source, row in load_rows(args.input):
        gold = get(row, args.gold_field, [])
        if isinstance(gold, str):
            gold = [gold]
        if args.gold_split:
            gold = [part.strip() for g in gold for part in g.split(args.gold_split)
                    if part.strip()]
        prediction = get(row, args.prediction_field)
        answer_extracted = False
        if answer_re is not None and isinstance(prediction, str):
            matches = answer_re.findall(prediction)
            if matches:
                prediction = matches[-1].strip()
                answer_extracted = True
        turns = get(row, args.turns_field, []) or []
        if isinstance(turns, int):
            n_turns, turns = turns, []
        else:
            n_turns = len(turns) if isinstance(turns, list) else 0
        output.append({
            "id": _canonical_id(str(get(row, args.id_field, "")), args.id_prefix),
            "question": get(row, args.question_field, ""),
            "gold_answers": gold,
            "prediction": prediction,
            "answer_extracted": answer_extracted,
            "finish_reason": get(row, args.finish_field, "unknown"),
            "n_turns": n_turns,
            "n_tool_calls": int(get(row, args.tool_calls_field, 0) or 0),
            "turns": turns,
            "trajectory_path": source,
            "dataset_manifest": dataset_manifest,
            "method_config_id": config["method_config_id"],
            "generator_revision": run_record["generator_revision"],
            "harness_commit": run_record["harness_commit"],
            "harness_source_sha256": run_record["harness_source_sha256"],
            "official_argv": run_record["argv"],
        })
    ids = [row["id"] for row in output]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("Normalized official output has empty or duplicate IDs")
    # The reference file is our canonical dataset for full-split methods, and the
    # capped subset for the call-bound ones (dci / dr_dci / rise). Those two file
    # shapes name the id differently -- "id" in the canonical jsonl, "query_id" in
    # the dci-lite conversion -- so accept either rather than KeyError on one.
    def _row_id(line: str) -> str:
        row = json.loads(line)
        return str(row.get("id") or row.get("query_id") or "")

    expected_ids = {_row_id(line) for line in
                    args.dataset_jsonl.read_text().splitlines() if line.strip()}
    if set(ids) != expected_ids:
        raise SystemExit("Normalized official output ID set differs from the canonical dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    print(f"wrote {len(output)} rows -> {args.output}")


if __name__ == "__main__":
    main()
