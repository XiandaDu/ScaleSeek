#!/bin/bash
# octal30 GPU2：4B search_o1 hotpotqa（修复版，BM25+E5，resume）
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8006 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm8006_so1.log 2>&1 &
ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:8006/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[so1h] 4B@8006 not up ABORT"; exit 1; }
echo "[so1h] 4B up @ $(date +%m%d-%H:%M)"
run(){ local out=$1 be=$2; shift 2
  echo "[so1h] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset hotpotqa --agent search_o1 -n 1500 \
    --concurrency 16 --port 8006 --retrieval-backend $be --bm25-top-k 5 --max-tokens 2048 \
    --resume --output results/$out.jsonl \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl --title-index-db $TITLE_DB \
         --out results/$out.metrics.json \
    && echo "[so1h] $out done @ $(date +%m%d-%H:%M)" || echo "[so1h] $out FAILED"; }
run hotpotqa_search_o1_v2 bm25
run hotpotqa_search_o1_e5_v2 e5
pkill -f "[v]llm.*8006"
echo "SO1_HOTPOT_DONE @ $(date +%m%d-%H:%M)"
