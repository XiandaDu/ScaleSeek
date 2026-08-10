#!/usr/bin/env bash
#SBATCH -J p0_e5_index
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Rorqual: build the frozen intfloat/e5-base-v2 full-corpus FAISS index
# (official query:/passage: prefixes, masked-mean pooling, L2 normalize —
# all inside scripts/build_e5_index.py). Resumable via vector checkpoints.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

[ -f "$CORPUS_MANIFEST" ] || { echo "FATAL: corpus manifest missing (run p0_corpus_unzip)"; exit 4; }

"$PY" scripts/build_e5_index.py \
  --corpus "$CORPUS_FILE" \
  --out "$E5_INDEX_DIR" \
  --backend e5 \
  --device cuda \
  --batch-size 1024 \
  --resume \
  --checkpoint-every 2000000 \
  --corpus-manifest "$CORPUS_MANIFEST"

[ -f "$E5_INDEX_DIR/index_manifest.json" ] || \
  "$PY" scripts/backfill_index_manifest.py --backend e5 --index-dir "$E5_INDEX_DIR" \
    --corpus-manifest "$CORPUS_MANIFEST" --out "$E5_INDEX_DIR/index_manifest.json"
"$PY" scripts/validate_index_manifests.py --corpus-manifest "$CORPUS_MANIFEST" \
  --index-manifest "$E5_INDEX_DIR/index_manifest.json"
echo "ALL_P0_E5_INDEX_DONE"
