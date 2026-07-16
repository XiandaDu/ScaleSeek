#!/bin/bash
# #15-4 IRCoT + #15-3 dci-agent-lite（都用 GPU0 的 4B @8000）
# 顺序：IRCoT bm25(top3) → IRCoT e5(top3) → dci-lite popqa n=50（官方 harness grep）
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
SS=/data/rech/mofengra/ScaleSeek
DD=/data/rech/mofengra/dr_dci_official
cd $SS || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda

curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 || { echo "[id] 4B@8000 not up, ABORT"; exit 1; }

# ---- IRCoT BM25 top-3 ----
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
echo "[id] ircot_bm25 start @ $(date +%m%d-%H:%M)"
$PY -m eval.run_eval --dataset popqa_full --agent ircot -n 1500 --concurrency 16 \
  --bm25-top-k 3 --max-turns 6 --max-tokens 2048 --resume \
  --output results/popqa_full_ircot_bm25.jsonl \
  && $PY scripts/compute_metrics.py --results results/popqa_full_ircot_bm25.jsonl \
       --out results/popqa_full_ircot_bm25.metrics.json \
  && echo "[id] ircot_bm25 done @ $(date +%m%d-%H:%M)" || echo "[id] ircot_bm25 FAILED"

# ---- IRCoT E5 top-3 ----
echo "[id] ircot_e5 start @ $(date +%m%d-%H:%M)"
$PY -m eval.run_eval --dataset popqa_full --agent ircot -n 1500 --concurrency 16 \
  --retrieval-backend e5 --bm25-top-k 3 --max-turns 6 --max-tokens 2048 --resume \
  --output results/popqa_full_ircot_e5.jsonl \
  && $PY scripts/compute_metrics.py --results results/popqa_full_ircot_e5.jsonl \
       --out results/popqa_full_ircot_e5.metrics.json \
  && echo "[id] ircot_e5 done @ $(date +%m%d-%H:%M)" || echo "[id] ircot_e5 FAILED"

# ---- dci-agent-lite popqa n=50（官方 harness，read,bash grep 14GB 语料）----
cd $DD || exit 1
set -a; source .env 2>/dev/null; set +a
export DCI_VIEW_CACHE_ROOT=$DD/.view_cache_dcilite
export DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses
export DCI_JUDGE_MAX_OUTPUT_TOKENS=2048
echo "[id] dci-lite popqa n=50 start @ $(date +%m%d-%H:%M)"
$DD/.venv/bin/python scripts/bcplus_eval/run_bcplus_eval.py \
  --dataset "$DD/data/dci-bench/data/popqa/test.jsonl" \
  --output-root "$DD/outputs/qa/popqa_dcilite_vllm4b_n50" \
  --corpus-dir "$DD/corpus/wiki_corpus" \
  --package-dir "$DD/pi-mono/packages/coding-agent" \
  --agent-dir "$DD/pi-mono/.pi/agent" \
  --provider vllm --model agent --judge-model agent \
  --tools read,bash --max-turns 300 --max-concurrency 4 --limit 50 \
  --runtime-context-level level3 --pi-thinking-level high \
  --node-max-old-space-size-mb 8192 \
  && echo "[id] dci-lite done @ $(date +%m%d-%H:%M)" || echo "[id] dci-lite FAILED"
echo "IRCOT_DCI_ALL_DONE @ $(date +%m%d-%H:%M)"
