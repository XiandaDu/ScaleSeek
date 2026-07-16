#!/bin/bash
# octal35 GPU1：3B search_r1 —— 先收尾 hotpot E5(resume 900)，再全铺 5 数据集 ×{BM25,E5}
case "$(hostname)" in octal30*|octal35*|octal4*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TDB=/data/rech/mofengra/data/corpus_title_index.db
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
P=8021
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $SR1 \
  --served-model-name search_r1 --port $P --gpu-memory-utilization 0.80 --max-model-len 8192 \
  > logs/o35_sr1.log 2>&1 &
ok=0; for t in $(seq 1 150); do curl -sf -m 5 localhost:$P/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[o35sr1] 3B not up ABORT"; exit 1; }
echo "[o35sr1] 3B up @ $(date +%m%d-%H:%M)"
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
run(){ local ds=$1 out=$2 be=$3
  [ -f results/$out.metrics.json ] && { echo "[o35sr1] $out 跳过"; return; }
  echo "[o35sr1] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset $ds --agent search_r1 -n 1500 \
    --concurrency 16 --search-r1-port $P --retrieval-backend $be --bm25-top-k 3 --max-turns 4 \
    --resume --output results/$out.jsonl \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
    && echo "[o35sr1] $out done @ $(date +%m%d-%H:%M)" || echo "[o35sr1] $out FAILED"; }
run hotpotqa hotpotqa_search_r1_e5 e5
for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
  run $ds ${ds}_search_r1_bm25 bm25
  run $ds ${ds}_search_r1_e5   e5
done
pkill -f "[v]llm.*$P"
echo "O35_SR1_ALL_DONE @ $(date +%m%d-%H:%M)"
