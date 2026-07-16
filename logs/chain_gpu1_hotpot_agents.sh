#!/bin/bash
# GPU1：补齐 hotpotqa 多跳表 —— search_o1 修复版(BM25+E5) + search_r1_e5(训练域内)
# 顺序：4B@8005 → search_o1 hotpot {bm25,e5} → 撤4B → 3B@8001 → search_r1 hotpot {e5}
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo

wait_port(){ local ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:$1/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done; [ $ok = 1 ] || { echo "[ha] :$1 not up ABORT"; exit 1; }; }
run(){ local ag=$1 out=$2 be=$3 port=$4; shift 4
  echo "[ha] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset hotpotqa --agent $ag -n 1500 \
    --concurrency 16 --port $port --retrieval-backend $be --bm25-top-k ${TOPK:-5} \
    --max-tokens 2048 --resume --output results/$out.jsonl "$@" \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl --title-index-db $TITLE_DB \
         --out results/$out.metrics.json \
    && echo "[ha] $out done @ $(date +%m%d-%H:%M)" || echo "[ha] $out FAILED"; }

# ---- 4B search_o1（修复版）----
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8005 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 > logs/vllm8005_ha.log 2>&1 &
wait_port 8005; echo "[ha] 4B up @ $(date +%m%d-%H:%M)"
run search_o1 hotpotqa_search_o1_v2 bm25 8005
run search_o1 hotpotqa_search_o1_e5_v2 e5 8005
pkill -f "[v]llm.*8005"; sleep 20

# ---- 3B search_r1_e5（hotpotqa 训练域内，top-3 4 轮）----
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $SR1 --served-model-name search_r1 --port 8001 \
  --gpu-memory-utilization 0.45 --max-model-len 8192 > logs/vllm8001_ha.log 2>&1 &
wait_port 8001; echo "[ha] 3B up @ $(date +%m%d-%H:%M)"
TOPK=3 run search_r1 hotpotqa_search_r1_e5 e5 8001 --max-turns 4
pkill -f "[v]llm.*8001"
echo "HOTPOT_AGENTS_ALL_DONE @ $(date +%m%d-%H:%M)"
