"""Datasets: normalisation, preflight and the fixed-seed subset (BASELINES.md §2).

Layout under paths.data/datasets/<name>/:
    normalized.jsonl        {"id", "question", "golden_answers"} in source order
    preflight.json          source pin, counts, duplicate/empty checks, SHA256s
    extra.json              BCP only: evidence/gold doc ids per query (for smoke corpora)
    subsets/seed<S>_n<N>/   subset.jsonl + manifest.json  (written once, never overwritten)

Questions and answers are copied verbatim (no stripping, no case change): any per-method
preprocessing (e.g. Search-R1 appending '?') happens inside that method, as officially.
"""

from __future__ import annotations

import base64
import glob
import hashlib
from pathlib import Path

from .hub import snapshot_dir
from .utils import (PipelineError, canonical_json, iter_jsonl, log, now_iso, read_json,
                    read_jsonl, sha256_file, sha256_text, write_json, write_json_new, write_jsonl)


def dataset_dir(base: dict, name: str) -> Path:
    return Path(base["paths"]["data"]) / "datasets" / name


# ============================================================================= sources
def _flashrag_rows(base: dict, cfg: dict) -> tuple[list[dict], dict]:
    path = snapshot_dir(base, cfg["hf_repo"], cfg["revision"], "dataset") / cfg["file"]
    raw_lines = 0
    rows = []
    for obj in iter_jsonl(path):
        raw_lines += 1
        golds = obj.get("golden_answers")
        if not isinstance(golds, list) or not all(isinstance(g, str) for g in golds):
            raise PipelineError(f"{path}: id {obj.get('id')!r} has non-list/non-str golden_answers")
        rows.append({"id": str(obj["id"]), "question": obj["question"], "golden_answers": golds})
    source = {"path": str(path), "raw_rows": raw_lines, "raw_sha256": sha256_file(path)}
    return rows, source


# --- BrowseComp-Plus decryption: the procedure published with the benchmark
# (identical in texttron/AgentIR evaluation/prepare_data.py): XOR with a SHA256-derived
# key of the canary string; `query_id` is stored in clear.
def _derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode("utf-8")).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def _decrypt_obj(obj, password: str, skip: set[str]):
    if isinstance(obj, str):
        return _decrypt(obj, password)
    if isinstance(obj, list):
        return [_decrypt_obj(v, password, skip) for v in obj]
    if isinstance(obj, dict):
        return {k: (v if k in skip else _decrypt_obj(v, password, skip)) for k, v in obj.items()}
    return obj


def _bcp_rows(base: dict, cfg: dict) -> tuple[list[dict], dict, dict]:
    import pyarrow.parquet as pq

    snap = snapshot_dir(base, cfg["hf_repo"], cfg["revision"], "dataset")
    files = sorted(glob.glob(str(snap / cfg["file"])))
    rows, extra, raw_rows = [], {}, 0
    for f in files:
        # Only the columns we need: the evidence documents' TEXT is large and unused here.
        table = pq.read_table(f)
        wanted = [c for c in ("query_id", "query", "answer", "evidence_docs", "gold_docs")
                  if c in table.column_names]
        for rec in table.select(wanted).to_pylist():
            raw_rows += 1
            # Exactly the official transform: decrypt every string except query_id.
            dec = _decrypt_obj(rec, cfg["canary"], {"query_id"})
            qid = str(dec["query_id"])
            rows.append({"id": qid, "question": dec["query"], "golden_answers": [dec["answer"]]})
            extra[qid] = {col: [str(d["docid"]) for d in (dec.get(col) or [])]
                          for col in ("evidence_docs", "gold_docs")}
    source = {"files": files, "raw_rows": raw_rows,
              "raw_sha256": {Path(f).name: sha256_file(f) for f in files}}
    return rows, source, extra


# ============================================================================= preflight
def prepare_dataset(base: dict, name: str) -> dict:
    cfg = base["datasets"][name]
    out = dataset_dir(base, name)
    extra = None
    if cfg["source"] == "flashrag":
        rows, source = _flashrag_rows(base, cfg)
    else:                                       # browsecomp_plus
        rows, source, extra = _bcp_rows(base, cfg)

    ids = [r["id"] for r in rows]
    seen, dups = set(), []
    for i in ids:
        if i in seen:
            dups.append(i)
        seen.add(i)
    empty_q = [r["id"] for r in rows if not isinstance(r["question"], str) or not r["question"].strip()]
    empty_a = [r["id"] for r in rows if not any(isinstance(g, str) and g.strip() for g in r["golden_answers"])]
    write_jsonl(out / "normalized.jsonl", rows)
    report = {
        "dataset": name, "created": now_iso(),
        "source": {"hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "file": cfg["file"],
                   "split": cfg["split"], **source},
        "expected_rows": cfg["expected_rows"], "raw_rows": source["raw_rows"],
        "normalized_rows": len(rows), "unique_ids": len(seen),
        "duplicate_ids": dups[:50], "n_duplicate_ids": len(dups),
        "empty_questions": empty_q[:50], "n_empty_questions": len(empty_q),
        "empty_answers": empty_a[:50], "n_empty_answers": len(empty_a),
        "normalized_sha256": sha256_file(out / "normalized.jsonl"),
    }
    problems = []
    if source["raw_rows"] != cfg["expected_rows"]:
        problems.append(f"raw rows {source['raw_rows']} != expected {cfg['expected_rows']}")
    if len(rows) != source["raw_rows"]:
        problems.append(f"normalized {len(rows)} != raw {source['raw_rows']}")
    if dups:
        problems.append(f"{len(dups)} duplicate ids (e.g. {dups[:5]})")
    if empty_q:
        problems.append(f"{len(empty_q)} empty questions")
    if empty_a:
        problems.append(f"{len(empty_a)} rows without a non-empty gold answer")
    report["ok"] = not problems
    report["problems"] = problems
    write_json(out / "preflight.json", report)
    if extra is not None:
        write_json(out / "extra.json", extra)
    if problems:
        raise PipelineError(f"preflight FAILED for {name} (see {out/'preflight.json'}): "
                            + "; ".join(problems) + " -- stop and investigate, do not truncate")
    log(f"preflight ok: {name} rows={len(rows)} sha256={report['normalized_sha256'][:12]}")
    return report


def load_preflight(base: dict, name: str) -> dict:
    rep = read_json(dataset_dir(base, name) / "preflight.json")
    if not rep["ok"]:
        raise PipelineError(f"{name}: preflight failed: {rep['problems']}")
    if sha256_file(dataset_dir(base, name) / "normalized.jsonl") != rep["normalized_sha256"]:
        raise PipelineError(f"{name}: normalized.jsonl changed since its preflight")
    return rep


# ============================================================================= subset
def subset_dir(base: dict, name: str) -> Path:
    s = base["sampling"]
    return dataset_dir(base, name) / "subsets" / f"seed{s['seed']}_n{s['max_questions']}"


def _rank_key(seed: int, qid: str) -> str:
    return sha256_text(f"{seed}:{qid}")


def compute_subset(base: dict, name: str) -> tuple[list[dict], dict]:
    pre = load_preflight(base, name)
    rows = read_jsonl(dataset_dir(base, name) / "normalized.jsonl")
    seed, cap = base["sampling"]["seed"], base["sampling"]["max_questions"]
    n = min(len(rows), cap)
    chosen = set(sorted((r["id"] for r in rows), key=lambda q: _rank_key(seed, q))[:n])
    subset = [r for r in rows if r["id"] in chosen]
    ids = [r["id"] for r in subset]
    manifest = {
        "dataset": name,
        "rule": "the min(N, max_questions) ids with the smallest sha256('<seed>:<id>'), "
                "in dataset order" if n < len(rows) else "full dataset (N <= max_questions)",
        "seed": seed, "max_questions": cap, "n": n, "full_rows": len(rows),
        "full_normalized_sha256": pre["normalized_sha256"],
        "full_raw_sha256": pre["source"]["raw_sha256"],
        "source": {k: pre["source"][k] for k in ("hf_repo", "revision", "file", "split")},
        "ids": ids, "ids_sha256": sha256_text(canonical_json(ids)),
    }
    return subset, manifest


def make_subset(base: dict, name: str) -> dict:
    subset, manifest = compute_subset(base, name)
    out = subset_dir(base, name)
    mpath = out / "manifest.json"
    if mpath.exists():
        old = read_json(mpath)
        keep = {k: v for k, v in old.items() if k not in ("created", "subset_sha256")}
        if keep != manifest:
            raise PipelineError(f"{mpath} exists and differs from a recomputation; a new seed "
                                "or cap must go to a NEW manifest, never overwrite")
        log(f"subset exists and verifies: {name} n={old['n']}")
        return old
    write_jsonl(out / "subset.jsonl", subset)
    manifest["subset_sha256"] = sha256_file(out / "subset.jsonl")
    manifest["created"] = now_iso()
    write_json_new(mpath, manifest)
    log(f"subset written: {name} n={manifest['n']} ids_sha256={manifest['ids_sha256'][:12]}")
    return manifest


def load_subset(base: dict, name: str) -> tuple[list[dict], dict]:
    out = subset_dir(base, name)
    manifest = read_json(out / "manifest.json")
    if sha256_file(out / "subset.jsonl") != manifest["subset_sha256"]:
        raise PipelineError(f"{name}: subset.jsonl does not match its manifest")
    return read_jsonl(out / "subset.jsonl"), manifest
