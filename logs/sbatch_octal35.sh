#!/bin/bash
#SBATCH --job-name=ss_o35
#SBATCH --partition=rali
#SBATCH --nodelist=octal[35]
#SBATCH --gres=gpu:rtx_a6000:2
#SBATCH --mem=50G
#SBATCH --time=10-23:14:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal35_%j.log
# octal35 (2×A6000/48G, 小内存) 分担：grep 型 agent（grepseek + dci）全铺 5 新数据集。
# 与 octal30 rollout 零重叠（octal30 未跑 grep 型 agent）。全部 --resume。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
# A6000 48G：9B 单卡无 TP。GPU0=9B grepseek，GPU1=4B dci（都 grep 14GB，共享页缓存）
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $GS --served-model-name grepseek --port 8103 \
  --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o35_9b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8100 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o35_4b.log 2>&1 &
for p in 8103 8100; do for t in $(seq 1 180); do curl -sf localhost:$p/v1/models >/dev/null 2>&1 && break; sleep 10; done; done
tf(){ [ "$1" = 2wikimultihopqa ] && echo "--title-index-db $TDB"; }
( for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset $ds --agent grepseek -n 1500 \
      --concurrency 12 --grepseek-port 8103 --grepseek-tokenizer $GS --corpus-path $CORPUS \
      --resume --output results/${ds}_grepseek.jsonl
    $PY scripts/compute_metrics.py --results results/${ds}_grepseek.jsonl $(tf $ds) --out results/${ds}_grepseek.metrics.json
  done ) &
for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset $ds --agent dci -n 1500 \
    --concurrency 2 --port 8100 --corpus-path $CORPUS \
    --resume --output results/${ds}_dci.jsonl
  $PY scripts/compute_metrics.py --results results/${ds}_dci.jsonl $(tf $ds) --out results/${ds}_dci.metrics.json
done
wait
