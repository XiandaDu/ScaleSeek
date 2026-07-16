#!/bin/bash
# octal30 GPU2：search_o1 修复版 @ BCP（走 BCP 小索引，零 IO 竞争 grepseek）
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8008 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm8008_bcp.log 2>&1 &
ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:8008/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[so1bcp] 4B@8008 not up ABORT"; exit 1; }
echo "[so1bcp] 4B up @ $(date +%m%d-%H:%M)"
echo "[so1bcp] search_o1 BCP start @ $(date +%m%d-%H:%M)"
CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset browsecomp_plus --agent search_o1 -n 830 \
  --concurrency 16 --port 8008 --bm25-top-k 5 --max-tokens 4096 \
  --resume --output results/bcp_search_o1_v2.jsonl \
  && $PY scripts/compute_metrics.py --results results/bcp_search_o1_v2.jsonl $QRELS \
       --out results/bcp_search_o1_v2.metrics.json \
  && echo "[so1bcp] done @ $(date +%m%d-%H:%M)" || echo "[so1bcp] FAILED"
pkill -f "[v]llm.*8008"
echo "SO1_BCP_DONE @ $(date +%m%d-%H:%M)"
