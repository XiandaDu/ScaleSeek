#!/bin/bash
#SBATCH --job-name=ss_o41
#SBATCH --partition=rali
#SBATCH --nodelist=octal[41]
#SBATCH --gres=gpu:ls40:4
#SBATCH --mem=256G
#SBATCH --time=10-23:14:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal41_%j.log
# octal41 (4×L40S/48G, 515G RAM) 补矩阵：node30 只跑 prompt agent 的 BM25 版，
# 这里跑 E5 版 + agentir。全 indexed 检索、非 grep，与 node30/node40 零重叠。全部 --resume。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
AIDX=/data/rech/mofengra/data/agentir_index_v2
TDB=/data/rech/mofengra/data/corpus_title_index.db
DS="nq triviaqa 2wikimultihopqa musique bamboogle"
tf(){ [ "$1" = 2wikimultihopqa ] && echo "--title-index-db $TDB"; }
serve4b(){ CUDA_VISIBLE_DEVICES=$1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $2 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.80 \
  --max-model-len 32768 > logs/o41_$2.log 2>&1 & }
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return; sleep 10; done; }

serve4b 0 8401; serve4b 1 8402; serve4b 2 8403
for p in 8401 8402 8403; do waitp $p; done

# GPU0: search_o1_e5   GPU1: ircot_e5   GPU2: scaleseek_e5  （全 --retrieval-backend e5）
( for ds in $DS; do
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset $ds --agent search_o1 -n 1500 \
      --concurrency 16 --port 8401 --retrieval-backend e5 --bm25-top-k 5 --max-tokens 2048 \
      --resume --output results/${ds}_search_o1_e5.jsonl
    $PY scripts/compute_metrics.py --results results/${ds}_search_o1_e5.jsonl $(tf $ds) --out results/${ds}_search_o1_e5.metrics.json
  done ) &
( for ds in $DS; do
    CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset $ds --agent ircot -n 1500 \
      --concurrency 16 --port 8402 --retrieval-backend e5 --bm25-top-k 3 --max-turns 6 --max-tokens 2048 \
      --resume --output results/${ds}_ircot_e5.jsonl
    $PY scripts/compute_metrics.py --results results/${ds}_ircot_e5.jsonl $(tf $ds) --out results/${ds}_ircot_e5.metrics.json
  done ) &
( for ds in $DS; do
    CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset $ds --agent scaleseek -n 1500 \
      --concurrency 16 --port 8403 --retrieval-backend e5 --max-tokens 2048 \
      --resume --output results/${ds}_scaleseek_e5.jsonl
    $PY scripts/compute_metrics.py --results results/${ds}_scaleseek_e5.jsonl $(tf $ds) --out results/${ds}_scaleseek_e5.metrics.json
  done ) &

# GPU3: agentir —— 先预计算(AgentIR-4B 编码器) 5 集，再起 4B reader 评测
( for ds in $DS; do
    [ -f results/${ds}_agentir_retrieval.jsonl ] || \
    CUDA_VISIBLE_DEVICES=3 $PY scripts/precompute_agentir_retrieval.py --dataset $ds -n 1500 \
      --index-root $AIDX --top-k 5 --device cuda --out results/${ds}_agentir_retrieval.jsonl
  done
  serve4b 3 8404; waitp 8404
  for ds in $DS; do
    CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval --dataset $ds --agent agentir_rag -n 1500 \
      --concurrency 16 --port 8404 --agentir-cache results/${ds}_agentir_retrieval.jsonl \
      --max-tokens 2048 --resume --output results/${ds}_agentir.jsonl
    $PY scripts/compute_metrics.py --results results/${ds}_agentir.jsonl $(tf $ds) --out results/${ds}_agentir.metrics.json
  done ) &
wait
echo "O41_ALL_DONE @ $(date +%m%d-%H:%M)"
