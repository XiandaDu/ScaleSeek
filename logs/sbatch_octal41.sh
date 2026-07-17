#!/bin/bash
#SBATCH --job-name=ss_o41
#SBATCH --partition=rali
#SBATCH --nodelist=octal[41]
#SBATCH --gres=gpu:ls40:4
# ⚠ 不写 --cpus-per-task 只给 1 核(=2 线程)，节点 62 核全闲着 —— E5/faiss lane 会被饿死。
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=10-23:14:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal41_%j.log
# octal41 (4×L40S/48G, 515G RAM)：接手 octal30 宕机后遗留的 4B rollout（缺 28 格）。
# 按数据集分 4 卡并行；每卡一个 4B 服务，跑该数据集的 6 个 agent。
# 与 octal35（grepseek全量 + search_r1）零重叠。全部 --resume。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
# ⚠ L40S(sm_89) 上 vLLM 起不来的两连坑（octal30/35 是 sm_86，JIT 缓存已编好所以碰不到）：
#  1) octal40/41 没有 /usr/local/cuda，CUDA_HOME 不设 → flashinfer JIT 找不到 nvcc。
#  2) pip 里 nvcc 13.2 vs CUDA 头文件 13.0（torch 是 +cu130）→ flashinfer cccl 硬检查
#     在编 renorm.cu（采样算子）时报 headers incompatible。
# → 关掉 flashinfer 采样器，退回 torch 原生采样（注意力本来就走 FLASH_ATTN）。
# 失败是静默的：服务死了 run_eval 照跑，每行 api_error（见 job 7159 的 10 格假数据）。
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TDB=/data/rech/mofengra/data/corpus_title_index.db
tf(){ [ "$1" = 2wikimultihopqa ] && echo "--title-index-db $TDB"; }
serve(){ CUDA_VISIBLE_DEVICES=$1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $2 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.80 \
  --max-model-len 32768 > logs/o41_$2.log 2>&1 & }
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o41] FATAL: 端口 $1 服务未起来，放弃该 lane（不写 api_error 垃圾）"; return 1; }
# 服务死了 run_eval 也会“跑完”，只是每行 api_error。出 metrics 前先验收。
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o41] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o41] $out api_error $e/$t 过半 —— 判为无效，不出 metrics"; return 1; }; }
# $1 gpu $2 port $3 ds $4 agent $5 out $6 backend ; rest extra
run(){ local g=$1 p=$2 ds=$3 ag=$4 out=$5 be=$6; shift 6
  [ -f results/$out.metrics.json ] && { echo "[o41] $out 跳过"; return; }
  echo "[o41] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=$g $PY -m eval.run_eval --dataset $ds --agent $ag -n 1500 \
    --concurrency 16 --port $p --retrieval-backend $be --max-tokens 2048 \
    --resume --output results/$out.jsonl "$@" \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
    && echo "[o41] $out done @ $(date +%m%d-%H:%M)" || echo "[o41] $out FAILED"; }
# 一卡跑一(组)数据集的 6 个 agent
suite(){ local g=$1 p=$2; shift 2
  waitp $p || { echo "[o41] lane $p 无服务，整条 lane 放弃"; return 1; }
  for ds in "$@"; do
    run $g $p $ds direct    ${ds}_direct     bm25
    run $g $p $ds bm25_rag  ${ds}_bm25_rag   bm25 --bm25-k1 1.2 --bm25-b 0.75
    run $g $p $ds bm25_rag  ${ds}_rag_e5     e5   --bm25-top-k 5
    run $g $p $ds scaleseek ${ds}_scaleseek  bm25
    run $g $p $ds ircot     ${ds}_ircot_bm25 bm25 --bm25-top-k 3 --max-turns 6
    run $g $p $ds search_o1 ${ds}_search_o1  bm25 --bm25-top-k 5
  done; }
serve 0 8041; serve 1 8042; serve 2 8043; serve 3 8044
# ⚠ 只 wait 这四条 lane 的 PID。裸 wait 会把 serve 起的 vLLM 也等进去 —— 它们永不退出，
# job 跑完 30 格后照样挂着白占 4 张卡（2026-07-16 job 7173 空转 9.5 小时）。
suite 0 8041 nq &                    L0=$!
suite 1 8042 triviaqa &              L1=$!
suite 2 8043 2wikimultihopqa &       L2=$!
suite 3 8044 musique bamboogle &     L3=$!
wait $L0 $L1 $L2 $L3
for p in 8041 8042 8043 8044; do pkill -f "[v]llm.*--port $p"; done
echo "O41_ALL_DONE @ $(date +%m%d-%H:%M)"
