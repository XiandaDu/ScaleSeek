#!/bin/bash
#SBATCH --job-name=ss_o30f
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_o30f_%j.log
#
# 把 n=500 上"方向一致但不显著"的两个发现，放大到 n=1500 定案。
#
# ① top-k=3 vs 5：n=500 的扫描里 top-k=3 在 popqa/2wiki/musique **三个数据集全部最好**
#    （+2.0 / +1.4 / +3.4 EM），但每个单独看都在 ±2 的噪声里。3/3 同向值得验。
#    对照组不用跑：现有的 ${ds}_scaleseek 就是 n=1500、top-k 默认 5、k1/b 默认、max-tokens 2048，
#    与本作业**只差 top-k 一个变量**，直接可比。
# ② search_o1 / dci 的 4096：n=500 配对检验 净+14 (p=.076) 和 净+12 (p=.088)，
#    都是"生成长"的 agent，2048 很可能真的截断了它们。n=1500 能把不一致题数放大 3 倍。
#    其余 4 个 agent（direct/bm25_rag/ircot/agentir）净变化恰好为 0、p=1.000，
#    已确认 2048 够用，不再跑。
#
# ⚠ 全部放在 A5000（bf16）上跑。**不能**挪到 abaque02 的 2080 Ti：Turing 只有 fp16，
#   而所有既有 n=1500 结果都是 bf16 —— 混着比会把数值精度和被测变量混淆。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o30f] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o30f] $out api_error $e/$t 过半 —— 无效"; return 1; }; }
srv(){ CUDA_VISIBLE_DEVICES=$1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $2 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > logs/o30f_$2.log 2>&1 & }
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o30f] FATAL: 端口 $1 没起来，放弃该 lane"; return 1; }
cell(){ local out=$1 ds=$2 g=$3 p=$4; shift 4
  [ -f results/$out.metrics.json ] && { echo "[o30f] $out 已完成，跳过"; return; }
  echo "[o30f] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=$g $PY -m eval.run_eval --dataset $ds -n 1500 --port $p "$@" \
    --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
    && echo "[o30f] $out done @ $(date +%m%d-%H:%M)" || echo "[o30f] $out FAILED"; }

( srv 0 8091; waitp 8091 || exit 1; echo "[o30f] laneA up @ $(date +%m%d-%H:%M)"
  cell ss_topk3_popqa_full_n1500 popqa_full 0 8091 --agent scaleseek --concurrency 16 \
    --retrieval-backend bm25 --bm25-top-k 3 --max-tokens 2048
  pkill -f "[v]llm.entrypoints.*--port 8091" ) & LA=$!
( srv 1 8092; waitp 8092 || exit 1; echo "[o30f] laneB up @ $(date +%m%d-%H:%M)"
  cell ss_topk3_2wikimultihopqa_n1500 2wikimultihopqa 1 8092 --agent scaleseek --concurrency 16 \
    --retrieval-backend bm25 --bm25-top-k 3 --max-tokens 2048
  pkill -f "[v]llm.entrypoints.*--port 8092" ) & LB=$!
( srv 2 8093; waitp 8093 || exit 1; echo "[o30f] laneC up @ $(date +%m%d-%H:%M)"
  cell mtok4096_search_o1_n1500 popqa_full 2 8093 --agent search_o1 --concurrency 16 \
    --retrieval-backend bm25 --bm25-top-k 5 --max-tokens 4096
  pkill -f "[v]llm.entrypoints.*--port 8093" ) & LC=$!
( srv 3 8094; waitp 8094 || exit 1; echo "[o30f] laneD up @ $(date +%m%d-%H:%M)"
  cell mtok4096_dci_n1500 popqa_full 3 8094 --agent dci --concurrency 4 \
    --corpus-path $CORPUS --max-tokens 4096
  pkill -f "[v]llm.entrypoints.*--port 8094" ) & LD=$!
wait $LA $LB $LC $LD
echo "O30F_ALL_DONE @ $(date +%m%d-%H:%M)"
