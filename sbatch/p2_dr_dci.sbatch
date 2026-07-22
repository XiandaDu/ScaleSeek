#!/usr/bin/env bash
#SBATCH -J p2_dr_dci
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 DR-DCI (Wiki-18 official E5 pull route). The official
# searchr1_wiki18_dci_server maps FAISS position -> corpus line; our frozen E5
# index is provably in corpus order (ordered_doc_ids_sha256 matches the corpus
# manifest), and its query prefix/pooling/normalization match the server's
# encoder exactly, so the manifest-pinned project E5 index backs the official
# /retrieve + /document service (recorded in the run record).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
export PATH=/data/rech/mofengra/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
PULL_PORT=$((PORT + 1))
RAW=results/phase2/dr_dci_raw
mkdir -p results/phase2 "$RAW"

echo "== official wiki18 E5 retriever service (project E5 index) =="
SVC_LOG=$REPO/logs/dr_dci_retriever_$SLURM_JOB_ID.log
( cd "$OFFICIAL_ROOT/dr_dci" && uv run python tools/dense_retriever/searchr1_wiki18_dci_server.py \
    --index-path "$E5_INDEX_DIR/index.faiss" \
    --corpus-path "$CORPUS_FILE" \
    --model-name-or-path intfloat/e5-base-v2 \
    --encoder-device cpu \
    --host 127.0.0.1 --port "$PULL_PORT" \
    --max-top-k 1000 > "$SVC_LOG" 2>&1 ) &
SVC_WAIT=0
until curl -sf "http://127.0.0.1:$PULL_PORT/health" >/dev/null 2>&1; do
  SVC_WAIT=$((SVC_WAIT+10))
  [ "$SVC_WAIT" -ge 1800 ] && { echo "FATAL: retriever service not healthy; see $SVC_LOG"; exit 4; }
  sleep 10
done
echo "service healthy after ${SVC_WAIT}s; probe /retrieve:"
curl -sf -X POST "http://127.0.0.1:$PULL_PORT/retrieve" \
  -H 'Content-Type: application/json' -d '{"query": "capital of France", "top_k": 3}' \
  | head -c 500 || { echo "FATAL: /retrieve probe failed"; exit 4; }
echo

start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B --reasoning-parser qwen3
waitp "$PORT" 3600

export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export DCI_PULL_BASE_URL="http://127.0.0.1:$PULL_PORT/retrieve"
export DCI_PULL_DOCUMENT_BASE_URL="http://127.0.0.1:$PULL_PORT/document"
mkdir -p "$OFFICIAL_ROOT/dr_dci/corpus/wiki18_empty"

"$PY" scripts/run_official_baseline.py dr_dci --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --record results/phase2/popqa_dr_dci.run.json -- \
  --model Qwen/Qwen3.5-9B --provider openai \
  --max-turns 300 --runtime-context-level level3 \
  --tools read,bash,pull \
  --pull-view-mode hardlink --pull-terminal-tools --pull-backend local \
  --pull-base-url "$DCI_PULL_BASE_URL" \
  --pull-document-base-url "$DCI_PULL_DOCUMENT_BASE_URL" \
  --pull-layout root --pull-prompt-mode rank_aware \
  --pull-materialization-mode root_flat_disclosed \
  --pull-min-top-k 300 --pull-max-top-k 600 --pull-max-queries 10 \
  --pull-preview-mode ranked --pull-preview-limit 20 \
  --max-concurrency "${CONC:-4}" --skip-judge \
  --pi-thinking-level high --node-max-old-space-size-mb 8192 \
  --dataset "$DATA/dcilite_datasets/popqa_full.jsonl" \
  --corpus-dir "$OFFICIAL_ROOT/dr_dci/corpus/wiki18_empty" \
  --output-root "$RAW"
stop_vllm

echo "== normalize + metrics =="
INPUT=$(find "$RAW" -name "*.jsonl" | head -1)
"$PY" scripts/normalize_official_results.py --method dr_dci \
  --input "$INPUT" \
  --run-record results/phase2/popqa_dr_dci.run.json \
  --dataset-manifest "$POPQA_MANIFEST" --dataset-jsonl "$POPQA_NORM" \
  --output results/phase2/popqa_dr_dci.jsonl
sane results/phase2/popqa_dr_dci.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dr_dci.jsonl \
  --dataset popqa --agent dr_dci --out results/phase2/popqa_dr_dci.metrics.json
echo "ALL_DONE popqa_dr_dci"
