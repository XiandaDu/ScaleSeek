#!/usr/bin/env bash
#SBATCH -J p1_qwen3emb_index
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-1 acceptance: build the frozen Qwen3-Embedding-4B full-corpus index on
# the otherwise-idle octal25 (1x3090). Resumable; checkpoints every 1M vectors.
# Finishes with the same-query stability probe + three-way manifest validation.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

"$PY" scripts/build_qwen3_embedding_index.py \
  --corpus "$CORPUS_FILE" \
  --out "$QWEN3_EMB_INDEX_DIR" \
  --device cuda \
  --batch-size 512 \
  --resume \
  --checkpoint-every 1000000 \
  --corpus-manifest "$CORPUS_MANIFEST"

echo "== manifest =="
if [ ! -f "$QWEN3_EMB_INDEX_DIR/index_manifest.json" ]; then
  "$PY" scripts/backfill_index_manifest.py --backend qwen3_emb_4b \
    --index-dir "$QWEN3_EMB_INDEX_DIR" --corpus-manifest "$CORPUS_MANIFEST" \
    --out "$QWEN3_EMB_INDEX_DIR/index_manifest.json"
fi
"$PY" scripts/validate_index_manifests.py --corpus-manifest "$CORPUS_MANIFEST" \
  --index-manifest "$E5_INDEX_DIR/index_manifest.json" \
  --index-manifest "$BM25_INDEX_DIR/index_manifest.json" \
  --index-manifest "$QWEN3_EMB_INDEX_DIR/index_manifest.json"

echo "== stability probe (all three backends) =="
mkdir -p results/smoke
"$PY" scripts/probe_retriever_stability.py --backends bm25,e5,qwen3_emb_4b \
  --device cuda --out results/smoke/stability_all3.json

echo "ALL_P1_QWEN3EMB_DONE"
# Rorqual: the qwen3_emb_4b cells are submitted by submit_phase3.sh with
# --dependency=afterok on this job; no lane seeding here.
