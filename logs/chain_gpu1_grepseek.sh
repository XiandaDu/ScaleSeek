#!/bin/bash
# GPU1 链：GrepSeek-9B @8003 → BCP grepseek 830（--resume，从 400 续）
#          → grepseek tool_max_tokens {1024,4096} @popqa_full 1500
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
WIKI=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
BCP=$DATASETS/browsecomp_plus/corpus.jsonl

CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $GS --served-model-name grepseek --port 8003 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm8003_gs9b.log 2>&1 &
ok=0
for t in $(seq 1 300); do
  curl -sf -m 5 localhost:8003/v1/models >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
[ $ok = 1 ] || { echo "[gpu1] 9B not up after 50min, ABORT"; tail -8 logs/vllm8003_gs9b.log; exit 1; }
echo "[gpu1] GrepSeek-9B up @ $(date +%m%d-%H:%M)"

export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
echo "[gpu1] BCP grepseek 830 (resume) start @ $(date +%m%d-%H:%M)"
$PY -m eval.run_eval --dataset browsecomp_plus --agent grepseek -n 830 --concurrency 16 \
  --corpus-path $BCP --grepseek-port 8003 --grepseek-tokenizer $GS \
  --resume --output results/bcp_grepseek_830.jsonl \
  && $PY scripts/compute_metrics.py --results results/bcp_grepseek_830.jsonl \
       --bcp-qrels $DATASETS/browsecomp_plus/qrels.json \
       --bcp-doclen $DATASETS/browsecomp_plus/doclen.json \
       --out results/bcp_grepseek_830.metrics.json \
  && echo "[gpu1] BCP grepseek done @ $(date +%m%d-%H:%M)" \
  || echo "[gpu1] BCP grepseek FAILED"

export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
for T in 1024 4096; do
  echo "[gpu1] grepseek tool_max_tokens=$T start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset popqa_full --agent grepseek -n 1500 --concurrency 16 \
    --corpus-path $WIKI --grepseek-port 8003 --grepseek-tokenizer $GS \
    --grepseek-tool-max-tokens $T --resume \
    --output results/popqa_full_grepseek_tooltok$T.jsonl \
    && $PY scripts/compute_metrics.py --results results/popqa_full_grepseek_tooltok$T.jsonl \
         --out results/popqa_full_grepseek_tooltok$T.metrics.json \
    && echo "[gpu1] tool_max_tokens=$T done @ $(date +%m%d-%H:%M)" \
    || echo "[gpu1] tool_max_tokens=$T FAILED"
done
pkill -f "[v]llm.*8003"
echo "GPU1_CHAIN_ALL_DONE @ $(date +%m%d-%H:%M)"
