#!/usr/bin/env bash
#SBATCH -J p2_agentir
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 24
#SBATCH --mem=230G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-2 AgentIR: official reasoning-aware OSS agent loop. vLLM Qwen3.5-9B on
# GPUs 0-1; the AgentIR-4B query encoder takes GPU 2. The flat FAISS index over
# 21M x 2560 float32 embeddings needs most of octal30's RAM, hence --mem=230G.
source /home/a32du/ScaleSeek/sbatch/common.sh
# SCOPE=cap500 switches this row to the sanctioned deterministic subset
# (2026-08-27 user decision); default remains the full split.
SCOPE=${SCOPE:-full}
if [ "$SCOPE" = "cap500" ]; then SCOPE=cap500; fi
trap 'stop_vllm; cleanup_tmpdir' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
SNAP=$HF_HUB_CACHE/models--Tevatron--AgentIR-4B/snapshots/e31abb637caa227c4b7d04176a24ecff1bcb10f4
OUT=$REPO/$RESULTS_DIR/agentir_raw
mkdir -p "$RESULTS_DIR" "$OUT"

ls "$DATA"/agentir_official_index/corpus.*.pkl >/dev/null 2>&1 || {
  echo "FATAL: official AgentIR pkl shards missing; run p2_agentir_encode first"; exit 4; }

export VLLM_GPUS=0
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 1 Qwen/Qwen3.5-9B --reasoning-parser qwen3
waitp "$PORT" 3600

cd "$REPO"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_official_baseline.py agentir \
  --repo-root "$OFFICIAL_ROOT" --full-eval --model-revision "$GEN_REV" \
  --record $RESULTS_DIR/${DS}_agentir.run.json -- \
  --model Qwen/Qwen3.5-9B --model-url "http://127.0.0.1:$PORT/v1" \
  --searcher-type faiss --index-path "$DATA/agentir_official_index/corpus.*.pkl" \
  --model-name "$SNAP" --reasoning-aware --normalize \
  --k 5 --max-iterations 100 \
  --query "$DATA/agentir_queries/${DS}_${SCOPE:-full}.tsv" \
  --QA-jsonl "$DATA/agentir_queries/${DS}_${SCOPE:-full}_qa.jsonl" \
  --num-threads "${CONC:-8}" \
  --output-dir "$OUT"
stop_vllm

echo "== normalize + metrics =="
"$PY" scripts/normalize_official_results.py --method agentir \
  --input "$OUT" \
  --run-record $RESULTS_DIR/${DS}_agentir.run.json \
  --dataset-manifest "$DS_MANIFEST" \
  --dataset-jsonl "$([ "$SCOPE" = full ] && echo "$DS_NORM" || echo "$DATA/dcilite_datasets/${DS}_${SCOPE}.jsonl")" \
  --output $RESULTS_DIR/${DS}_agentir.jsonl
sane $RESULTS_DIR/${DS}_agentir.jsonl
"$PY" scripts/compute_metrics.py --results $RESULTS_DIR/${DS}_agentir.jsonl \
  --dataset "$DS" --agent agentir \
  $([ "$SCOPE" = full ] || echo "--expect-ids $DATA/dcilite_datasets/${DS}_${SCOPE}.manifest.json") \
  --out $RESULTS_DIR/${DS}_agentir.metrics.json
echo "ALL_DONE ${DS}_agentir"
