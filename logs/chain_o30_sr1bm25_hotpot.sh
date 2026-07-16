#!/bin/bash
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $SR1 --served-model-name search_r1 --port 8009 \
  --gpu-memory-utilization 0.80 --max-model-len 8192 > logs/vllm8009_sr1bm25.log 2>&1 &
ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:8009/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[sr1bm] 3B@8009 not up ABORT"; exit 1; }
echo "[sr1bm] 3B up @ $(date +%m%d-%H:%M)"
CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset hotpotqa --agent search_r1 -n 1500 \
  --concurrency 16 --search-r1-port 8009 --bm25-top-k 3 --max-turns 4 \
  --resume --output results/hotpotqa_search_r1_bm25.jsonl \
  && $PY scripts/compute_metrics.py --results results/hotpotqa_search_r1_bm25.jsonl \
       --title-index-db $TITLE_DB --out results/hotpotqa_search_r1_bm25.metrics.json \
  && echo "[sr1bm] done @ $(date +%m%d-%H:%M)" || echo "[sr1bm] FAILED"
pkill -f "[v]llm.*8009"
echo "SR1BM25_HOTPOT_DONE @ $(date +%m%d-%H:%M)"
