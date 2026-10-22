"""AgentIR-4B index, built the OFFICIAL way (evaluation/scripts/embed_bcp.sh).

    python -m tevatron.retriever.driver.encode --model_name_or_path Tevatron/AgentIR-4B
        --pooling eos --normalize --passage_max_len 4096 --passage_prefix "" --fp16
        --per_device_eval_batch_size 24 ...  ->  corpus.shard.<i>.pkl  (reps, lookup)

run in envs/agentir (the official lock, incl. the locked tevatron commit).  The official
script shards an HF dataset with --dataset_number_of_shards; we give each job its own
contiguous slice of tevatron.jsonl instead (written to $SLURM_TMPDIR) -- the same rows,
the same encoder, and the lookup ids are checked against the corpus afterwards.  The
official searcher loads these pickles into a FLAT inner-product index (it cannot load
HNSW), so that is what AgentIR searches (docs/deviations.md).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..corpus import DocStore, corpus_dir, load_corpus_manifest
from ..hub import local_snapshot
from ..utils import (PipelineError, ensure_agentir_env, git_state, log, now_iso, read_json,
                     repo_root, sha256_file, write_json)
from .common import index_dir
from .dense import shard_bounds


def encode_command(base: dict, slice_file: Path, out_pkl: Path) -> list[str]:
    cfg = base["retrievers"]["agentir_4b"]
    cmd = [ensure_agentir_env(base), "-m", "tevatron.retriever.driver.encode",
           "--model_name_or_path", str(local_snapshot(base, cfg["model"], cfg["revision"])),
           "--dataset_name", "json", "--dataset_path", str(slice_file),
           "--encode_output_path", str(out_pkl),
           "--passage_max_len", str(cfg["passage_max_len"]),
           "--pooling", cfg["pooling"], "--passage_prefix", cfg["passage_prefix"],
           "--per_device_eval_batch_size", str(cfg["per_device_eval_batch_size"]),
           "--dataset_number_of_shards", "1", "--dataset_shard_index", "0"]
    if cfg["normalize"]:
        cmd.append("--normalize")
    if cfg["fp16"]:
        cmd.append("--fp16")
    return cmd


def encode_shard(base: dict, corpus: str, shard: int, n_shards: int, smoke: bool) -> dict:
    cman = load_corpus_manifest(base, corpus, smoke)
    out = index_dir(base, "agentir_4b", corpus, smoke)
    pkl = out / f"corpus.shard.{shard:04d}.pkl"
    meta_path = out / f"corpus.shard.{shard:04d}.json"
    if meta_path.exists() and pkl.exists() and read_json(meta_path)["corpus_sha256"] == cman["sha256"]:
        log(f"agentir shard {shard}/{n_shards} already encoded")
        return read_json(meta_path)
    cdir = corpus_dir(base, corpus, smoke)
    store = DocStore(cdir)
    lo, hi = shard_bounds(store.n, shard, n_shards)
    scratch = Path(os.environ.get("SLURM_TMPDIR") or out) / f"agentir_slice_{shard:04d}"
    scratch.mkdir(parents=True, exist_ok=True)
    slice_file = scratch / "slice.jsonl"
    # The slice is written from corpus.jsonl rows (seeking by byte offset, not scanning)
    # exactly as tevatron.jsonl was written -- same docid, same text, same serialisation.
    with open(slice_file, "w", encoding="utf-8") as dst:
        for row in store.iter_rows(lo, hi):
            dst.write(json.dumps({"docid": row["id"], "text": row["contents"]}, ensure_ascii=False) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    tmp_pkl = scratch / pkl.name
    cmd = encode_command(base, slice_file, tmp_pkl)
    # The official encoder is single-GPU (embed_bcp.sh pins CUDA_VISIBLE_DEVICES=$shard).
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    env = dict(os.environ, HF_DATASETS_CACHE=str(scratch / "hf_datasets"), CUDA_VISIBLE_DEVICES=gpu)
    log("agentir encode: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    with open(tmp_pkl, "rb") as f:
        reps, lookup = pickle.load(f)
    want = [store.id_of(r) for r in range(lo, hi)]
    if list(map(str, lookup)) != want:
        raise PipelineError(f"agentir shard {shard}: lookup ids do not match corpus rows {lo}..{hi}")
    reps = np.asarray(reps)
    shutil.move(str(tmp_pkl), str(pkl))
    meta = {"retriever": "agentir_4b", "corpus_sha256": cman["sha256"], "shard": shard,
            "n_shards": n_shards, "rows": [lo, hi], "shape": list(reps.shape), "dtype": str(reps.dtype),
            "command": cmd, "seconds": round(time.time() - t0, 1), "sha256": sha256_file(pkl),
            "created": now_iso()}
    write_json(meta_path, meta)
    return meta


def finalize(base: dict, corpus: str, n_shards: int, smoke: bool) -> dict:
    cman = load_corpus_manifest(base, corpus, smoke)
    out = index_dir(base, "agentir_4b", corpus, smoke)
    metas = [read_json(out / f"corpus.shard.{s:04d}.json") for s in range(n_shards)]
    extra = sorted(set(glob.glob(str(out / "corpus*.pkl"))) -
                   {str(out / f"corpus.shard.{s:04d}.pkl") for s in range(n_shards)})
    if extra:
        raise PipelineError(f"stray pickles would be loaded by the official glob: {extra}")
    if metas[0]["rows"][0] != 0 or metas[-1]["rows"][1] != cman["docs"] or any(
            a["rows"][1] != b["rows"][0] for a, b in zip(metas, metas[1:])):
        raise PipelineError("agentir shards do not tile the corpus")
    if any(m["corpus_sha256"] != cman["sha256"] for m in metas):
        raise PipelineError("agentir shard from another corpus")
    cfg = base["retrievers"]["agentir_4b"]
    manifest = {"retriever": "agentir_4b", "corpus": cman["corpus"], "corpus_sha256": cman["sha256"],
                "harness_commit": git_state(repo_root())["commit"],
                "docs": cman["docs"], "index": "official tevatron pickles -> FaissFlatSearcher (flat IP)",
                "params": {k: cfg[k] for k in cfg if k != "kind"},
                "shards": [{k: m[k] for k in ("shard", "rows", "shape", "dtype", "sha256")} for m in metas],
                "command_example": metas[0]["command"], "created": now_iso()}
    write_json(out / "manifest.json", manifest)
    log(f"agentir index ok: {cman['docs']} docs in {n_shards} shards")
    return manifest
