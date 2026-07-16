#!/bin/bash
# octal30 GPU3：3B search_r1 全铺 5 数据集（BM25 + E5）。等 search_r1_e5 hotpot 完先。
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
PORT=8011

until grep -q "SR1_HOTPOT_DONE" logs/chain_o30_sr1_hotpot.log 2>/dev/null; do sleep 300; done
sleep 15
CUDA_VISIBLE_DEVICES=3 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $SR1 --served-model-name search_r1 --port $PORT \
  --gpu-memory-utilization 0.80 --max-model-len 8192 > logs/vllm${PORT}_sr1roll.log 2>&1 &
ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[rsr1] 3B not up ABORT"; exit 1; }
echo "[rsr1] 3B up @ $(date +%m%d-%H:%M)"

run(){ # $1 ds $2 out $3 backend
  local ds=$1 out=$2 be=$3
  [ -f results/$out.metrics.json ] && { echo "[rsr1] $out 已存在,跳过"; return; }
  echo "[rsr1] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval --dataset $ds --agent search_r1 -n 1500 \
    --concurrency 16 --search-r1-port $PORT --retrieval-backend $be --bm25-top-k 3 --max-turns 4 \
    --resume --output results/$out.jsonl \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl --out results/$out.metrics.json \
    && echo "[rsr1] $out done @ $(date +%m%d-%H:%M)" || echo "[rsr1] $out FAILED"; }

for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
  run $ds ${ds}_search_r1_bm25 bm25
  run $ds ${ds}_search_r1_e5   e5
done
pkill -f "[v]llm.*$PORT"
echo "ROLLOUT_SR1_ALL_DONE @ $(date +%m%d-%H:%M)"
