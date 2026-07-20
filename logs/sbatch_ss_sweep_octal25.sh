#!/bin/bash
#SBATCH --job-name=ss_sweep25
#SBATCH --partition=rali
#SBATCH --nodelist=octal25
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_ss_sweep25_%j.log
#
# #7 ScaleSeek 检索参数研究 —— 这是验收里唯一的实质空白（此前盘上的 k1/b 扫跑的是
# bm25_rag agent，不是 scaleseek，而且只在 popqa 单跳一个数据集上，没有难度轴）。
#
# 设计：两条独立主效应轴，不做全交叉（27 格太贵，先看主效应）
#   top-k 轴（k1/b 固定在默认 1.5/0.75）：  3 / 5 / 10
#   k1,b 轴（top-k 固定在默认 5）：         (0.9,0.4) / (1.5,0.75) / (25,1.0)
#   两轴共享 (top-k=5, 1.5/0.75) 这个基线格 → 每数据集 5 格
# 难度轴：本作业跑 popqa_full(单跳) + 2wikimultihopqa(多跳)；musique(最难)在 abaque02
#   上并行跑，两个作业的输出文件名不重叠，无写冲突。
# n=500（EM 的 SE≈2 个点，够看主效应；解读时别把 <2 点的差当结论）。
#
# ⚠ octal25 只有 16 核，比 octal30/40 少得多。BM25 走倒排索引（不是 faiss 穷举），
#   CPU 需求低，14 核够用；但别在这台机上排 E5/faiss 的活。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=6
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TDB=/data/rech/mofengra/data/corpus_title_index.db
PORT=8025
N=500
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[sw25] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[sw25] $out api_error $e/$t 过半 —— 无效"; return 1; }; }

CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > logs/sw25_4b.log 2>&1 &
ok=0; for t in $(seq 1 180); do curl -sf -m 5 localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[sw25] FATAL: 4B 没起来"; exit 1; }
echo "[sw25] 4B@$PORT up @ $(date +%m%d-%H:%M)"

# $1=out $2=dataset $3...=检索参数
run(){ local out=$1 ds=$2; shift 2
  [ -f results/$out.metrics.json ] && { echo "[sw25] $out 已完成，跳过"; return; }
  echo "[sw25] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset $ds --agent scaleseek -n $N \
    --concurrency 16 --port $PORT --retrieval-backend bm25 --max-tokens 2048 "$@" \
    --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
    && echo "[sw25] $out done @ $(date +%m%d-%H:%M)" || echo "[sw25] $out FAILED"; }

for ds in popqa_full 2wikimultihopqa; do
  # --- top-k 轴（k1/b = 默认 1.5/0.75）---
  run ss_sw_${ds}_topk3  $ds --bm25-top-k 3
  run ss_sw_${ds}_topk5  $ds --bm25-top-k 5      # 基线格，两轴共用
  run ss_sw_${ds}_topk10 $ds --bm25-top-k 10
  # --- k1,b 轴（top-k = 默认 5）---
  run ss_sw_${ds}_k1-0.9_b-0.4 $ds --bm25-top-k 5 --bm25-k1 0.9 --bm25-b 0.4
  run ss_sw_${ds}_k1-25_b-1.0  $ds --bm25-top-k 5 --bm25-k1 25  --bm25-b 1.0
done
pkill -f "[v]llm.entrypoints.*--port $PORT"
echo "SW25_ALL_DONE @ $(date +%m%d-%H:%M)"
