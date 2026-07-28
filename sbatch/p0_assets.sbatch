#!/usr/bin/env bash
#SBATCH -J p0_assets
#SBATCH -p rali
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-1 server acceptance, wave 0 (CPU): pinned model downloads, full PopQA
# download + preflight, corpus manifest, index-manifest backfill + validation,
# DCI-Lite dataset conversion. Everything is idempotent/resumable.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

echo "== 1. pinned model snapshots =="
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
pins = [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3",
     "395b18f1fecee52f1b51fb22f898c220f0a08ec3"),
    ("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3",
     "d65f11c88d3c129a01466f0c154aaad7d9b09225"),
    ("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"),
    ("alireza7/GrepSeek-Qwen3.5-9B-GRPO", "a79563970cfdd2ced3cc5fde481737d0ebea6fa4"),
    ("intfloat/e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"),
    ("Tevatron/AgentIR-4B", "e31abb637caa227c4b7d04176a24ecff1bcb10f4"),
]
for repo, rev in pins:
    for attempt in range(3):
        try:
            path = snapshot_download(repo, revision=rev)
            print(f"OK {repo}@{rev[:12]} -> {path}", flush=True)
            break
        except Exception as exc:
            print(f"retry {attempt+1} {repo}: {exc}", flush=True)
    else:
        raise SystemExit(f"FAILED to download {repo}@{rev}")
PYEOF

echo "== 2. full PopQA download (raw pinned file, bypassing datasets.load_dataset) =="
if [ ! -f "$POPQA_RAW" ]; then
  "$PY" - <<'PYEOF'
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "RUC-NLPIR/FlashRAG_datasets", "popqa/test.jsonl", repo_type="dataset",
    revision="bcafb8dd07d453be3cbeeeb3f78be1841bddf92c")
dst = os.path.join(os.environ["DATASETS"], "popqa", "test.jsonl")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(path, dst)
print("raw popqa ->", dst)
PYEOF
fi
echo "raw sha256:"; sha256sum "$POPQA_RAW"
echo "f976372fce3d7fe01b070357040391f10bd2710eeaa7c40d21c6574ffcad6cb8  $POPQA_RAW" | sha256sum -c -

echo "== 3. PopQA preflight + canonical normalized file + manifest =="
"$PY" - <<'PYEOF'
import json, os
from eval.datasets import load_dataset, validate_complete_dataset
rows = load_dataset("popqa")
manifest = validate_complete_dataset("popqa", rows)
print(json.dumps(manifest, indent=2))
base = os.path.join(os.environ["DATASETS"], "popqa")
with open(os.path.join(base, "test.normalized.jsonl"), "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n")
with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
assert manifest["count"] == manifest["unique_ids"] == 14267
PYEOF

echo "== 4. corpus manifest =="
if [ ! -f "$CORPUS_MANIFEST" ]; then
  "$PY" scripts/build_corpus_manifest.py --corpus "$CORPUS_FILE" --out "$CORPUS_MANIFEST"
fi
cat "$CORPUS_MANIFEST"

echo "== 5. backfill + validate index manifests (bm25, e5) =="
"$PY" scripts/backfill_index_manifest.py --backend e5 --index-dir "$E5_INDEX_DIR" \
  --corpus-manifest "$CORPUS_MANIFEST" --out "$E5_INDEX_DIR/index_manifest.json"
"$PY" scripts/backfill_index_manifest.py --backend bm25 --index-dir "$BM25_INDEX_DIR/index" \
  --corpus-manifest "$CORPUS_MANIFEST" --out "$BM25_INDEX_DIR/index_manifest.json"
"$PY" scripts/validate_index_manifests.py --corpus-manifest "$CORPUS_MANIFEST" \
  --index-manifest "$E5_INDEX_DIR/index_manifest.json" \
  --index-manifest "$BM25_INDEX_DIR/index_manifest.json"

echo "== 6. DCI-Lite PopQA conversion (full + sanctioned 1/10 subset) =="
"$PY" scripts/make_dcilite_datasets.py --datasets popqa \
  --out-dir "$DATA/dcilite_datasets"
wc -l "$DATA/dcilite_datasets/popqa_full.jsonl"
# DCI / DR-DCI / RISE evaluate the deterministic 1/10 subset (TASK.md
# "1/10 规模例外"): at 300/300/100 model calls per question the full split is
# ~32 days per row. All three read this SAME file, so they stay comparable to
# each other; the manifest fingerprints the id set and compute_metrics refuses
# a run that does not match it.
"$PY" scripts/make_dcilite_datasets.py --datasets popqa --fraction 10 \
  --out-dir "$DATA/dcilite_datasets"
wc -l "$DATA/dcilite_datasets/popqa_decile1of10.jsonl"

echo "== 7. AgentIR query TSV =="
"$PY" - <<'PYEOF'
import json, os
base = os.path.join(os.environ["DATASETS"], "popqa")
out = os.path.join(os.environ["DATA"], "agentir_queries")
os.makedirs(out, exist_ok=True)
with open(os.path.join(base, "test.normalized.jsonl"), encoding="utf-8") as fh, \
     open(os.path.join(out, "popqa_full.tsv"), "w", encoding="utf-8") as tsv, \
     open(os.path.join(out, "popqa_full_qa.jsonl"), "w", encoding="utf-8") as qa:
    for line in fh:
        row = json.loads(line)
        q = " ".join(str(row["question"]).split())
        tsv.write(f"{row['id']}\t{q}\n")
        qa.write(json.dumps({"query_id": str(row["id"]), "query": row["question"],
                             "answer": " / ".join(map(str, row.get("golden_answers", [])))},
                            ensure_ascii=False) + "\n")
print("wrote", out)
PYEOF

echo "ALL_P0_ASSETS_DONE"
