#!/bin/bash
#SBATCH --job-name=ss_o40
#SBATCH --partition=rali
#SBATCH --nodelist=octal[40]
#SBATCH --gres=gpu:ls40:4
#SBATCH --mem=256G
#SBATCH --time=10-23:14:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal40_%j.log
# octal40 (4×L40S/48G, 515G RAM, idle) 分担：grep 型 agent（grepseek + dci）全铺 5 新数据集。
# 4 卡并行：grepseek 拆两卡、dci 拆两卡（数据集对半）。与 octal30 rollout 零重叠。全部 --resume。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
tf(){ [ "$1" = 2wikimultihopqa ] && echo "--title-index-db $TDB"; }
# L40S 48G：9B 单卡无 TP。4 服务：GPU0/2=9B grepseek，GPU1/3=4B dci
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
  --served-model-name grepseek --port 8103 --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o40_9b_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
  --served-model-name grepseek --port 8203 --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o40_9b_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B \
  --served-model-name agent --port 8100 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o40_4b_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B \
  --served-model-name agent --port 8300 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.88 --max-model-len 32768 > logs/o40_4b_b.log 2>&1 &
for p in 8103 8203 8100 8300; do for t in $(seq 1 180); do curl -sf localhost:$p/v1/models >/dev/null 2>&1 && break; sleep 10; done; done

gs(){ local gpu=$1 port=$2; shift 2; for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu $PY -m eval.run_eval --dataset $ds --agent grepseek -n 1500 \
    --concurrency 16 --grepseek-port $port --grepseek-tokenizer $GS --corpus-path $CORPUS \
    --resume --output results/${ds}_grepseek.jsonl
  $PY scripts/compute_metrics.py --results results/${ds}_grepseek.jsonl $(tf $ds) --out results/${ds}_grepseek.metrics.json
done; }
dc(){ local gpu=$1 port=$2; shift 2; for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu $PY -m eval.run_eval --dataset $ds --agent dci -n 1500 \
    --concurrency 2 --port $port --corpus-path $CORPUS \
    --resume --output results/${ds}_dci.jsonl
  $PY scripts/compute_metrics.py --results results/${ds}_dci.jsonl $(tf $ds) --out results/${ds}_dci.metrics.json
done; }

gs 0 8103 nq triviaqa 2wikimultihopqa &
gs 2 8203 musique bamboogle &
dc 1 8100 nq triviaqa 2wikimultihopqa &
dc 3 8300 musique bamboogle &
wait
echo "O40_ALL_DONE @ $(date +%m%d-%H:%M)"
