#!/bin/bash
# search_o1 修复重跑（宽松标记识别）：等 GPU1 IRCoT-hotpot 完 → 4B@8005 → 重跑
# 写 _v2 文件保留旧的做前后对比。修复只影响 24% 畸形标记；76% 无标记是模型行为。
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda

until grep -q "IRCOT_HOTPOT_ALL_DONE" logs/chain_gpu1_ircot_hotpot.log 2>/dev/null; do sleep 300; done
sleep 15
# IRCoT-hotpot 链结尾 pkill 了 8005；重起一个
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8005 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 > logs/vllm8005_so1.log 2>&1 &
ok=0
for t in $(seq 1 120); do curl -sf -m 5 localhost:8005/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || { echo "[so1fix] 4B@8005 not up, ABORT"; exit 1; }
echo "[so1fix] 4B up @ $(date +%m%d-%H:%M)"

export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
run_one() {  # $1=out $2=backend
  local out=$1 backend=$2
  echo "[so1fix] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset popqa_full --agent search_o1 \
    -n 1500 --concurrency 16 --port 8005 --retrieval-backend $backend --bm25-top-k 5 \
    --resume --output results/$out.jsonl \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl --out results/$out.metrics.json \
    && echo "[so1fix] $out done @ $(date +%m%d-%H:%M)" || echo "[so1fix] $out FAILED"
}
run_one popqa_full_search_o1_v2 bm25
run_one popqa_full_search_o1_e5_v2 e5
pkill -f "[v]llm.*8005"
echo "SO1FIX_ALL_DONE @ $(date +%m%d-%H:%M)"
