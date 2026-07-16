#!/bin/bash
# GPU0 链：SearchR1-3B @8001 → search_r1_e5（论文口径 E5+top3+4轮）
#          → 撤 3B → R1-Distill-7B @8004 → search_o1_7b {BM25, E5} top-3
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

wait_port() {
  local ok=0
  for t in $(seq 1 300); do
    curl -sf -m 5 localhost:$1/v1/models >/dev/null 2>&1 && { ok=1; break; }
    sleep 10
  done
  [ $ok = 1 ] || { echo "[gpu0] :$1 not up after 50min, ABORT"; tail -8 $2; exit 1; }
}

# ---- SearchR1-3B → search_r1_e5 ----
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $SR1 --served-model-name search_r1 --port 8001 \
  --gpu-memory-utilization 0.45 --max-model-len 8192 > logs/vllm8001_sr1.log 2>&1 &
wait_port 8001 logs/vllm8001_sr1.log
echo "[gpu0] SearchR1-3B up @ $(date +%m%d-%H:%M)"
CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset popqa_full --agent search_r1 \
  -n 1500 --concurrency 16 --retrieval-backend e5 --bm25-top-k 3 --max-turns 4 \
  --resume --output results/popqa_full_search_r1_e5.jsonl \
  && $PY scripts/compute_metrics.py --results results/popqa_full_search_r1_e5.jsonl \
       --out results/popqa_full_search_r1_e5.metrics.json \
  && echo "[gpu0] search_r1_e5 done @ $(date +%m%d-%H:%M)" \
  || echo "[gpu0] search_r1_e5 FAILED"
pkill -f "[v]llm.*8001"
sleep 20

# ---- R1-Distill-7B → search_o1_7b × {BM25, E5} ----
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $R1 --served-model-name agent7b --port 8004 \
  --gpu-memory-utilization 0.80 --max-model-len 32768 > logs/vllm8004_r1_7b.log 2>&1 &
wait_port 8004 logs/vllm8004_r1_7b.log
echo "[gpu0] R1-Distill-7B up @ $(date +%m%d-%H:%M)"
run_so1() {
  local out=$1; shift
  echo "[gpu0] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset popqa_full --agent search_o1 \
    -n 1500 --concurrency 16 --port 8004 --model agent7b --temperature 0.6 \
    --bm25-top-k 3 --resume --output results/$out.jsonl "$@" \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl \
         --out results/$out.metrics.json \
    && echo "[gpu0] $out done @ $(date +%m%d-%H:%M)" || echo "[gpu0] $out FAILED"
}
run_so1 popqa_full_search_o1_7b_bm25
run_so1 popqa_full_search_o1_7b_e5 --retrieval-backend e5
pkill -f "[v]llm.*8004"
echo "GPU0_CHAIN_ALL_DONE @ $(date +%m%d-%H:%M)"
