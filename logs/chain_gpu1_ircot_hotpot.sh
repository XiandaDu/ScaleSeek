#!/bin/bash
# GPU1：4B@8005 → IRCoT hotpotqa {BM25, E5} top3（扩 IRCoT 到多跳 + Gold R@W）
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db

CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8005 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 > logs/vllm8005.log 2>&1 &
ok=0
for t in $(seq 1 120); do curl -sf -m 5 localhost:8005/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || { echo "[ih] 4B@8005 not up, ABORT"; exit 1; }
echo "[ih] 4B@8005 up @ $(date +%m%d-%H:%M)"

export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
run_one() {  # $1=out $2=backend
  local out=$1 backend=$2
  echo "[ih] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset hotpotqa --agent ircot \
    -n 1500 --concurrency 16 --port 8005 --retrieval-backend $backend \
    --bm25-top-k 3 --max-turns 6 --max-tokens 2048 --resume --output results/$out.jsonl \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl \
         --title-index-db $TITLE_DB --out results/$out.metrics.json \
    && echo "[ih] $out done @ $(date +%m%d-%H:%M)" || echo "[ih] $out FAILED"
}
run_one hotpotqa_ircot_bm25 bm25
run_one hotpotqa_ircot_e5 e5
pkill -f "[v]llm.*8005"
echo "IRCOT_HOTPOT_ALL_DONE @ $(date +%m%d-%H:%M)"
