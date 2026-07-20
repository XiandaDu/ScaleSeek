#!/bin/bash
#SBATCH --job-name=ss_o30e
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_o30e_%j.log
#
# 接收两批被 abaque01 驱动故障挤掉的活（原计划跑在 2080 Ti 上）：
#   Lane A (GPU0): ScaleSeek 参数扫的 musique 档（最难）—— 补齐难度轴第三档。
#                  popqa/2wiki 两档在 octal25 的 job 7196 上跑，输出名不重叠、无写冲突。
#   Lane B (GPU1): 4096 探针 —— direct / bm25_rag / ircot
#   Lane C (GPU2): 4096 探针 —— search_o1 / agentir / dci
# GPU3 空着不排：dci lane 的 grep 很吃 CPU，48 核给三条 lane 已经紧。
#
# abaque01 为什么不能用（2026-07-19 查证）：`nvidia-smi` 报
#   "Failed to initialize NVML: Driver/library version mismatch"（NVML 库 595.84 与
#   已加载的内核模块不匹配，典型的驱动升级后没重启）。vLLM 平台探测走 NVML → 直接
#   "Failed to infer device type"。这是节点级故障，需要 root 重载 nvidia 模块，我们修不了。
#   ⚠ 注意这**不是** Turing 的问题：torch 2.11+cu130 的 arch list 含 sm_75，
#   我原来那套 fp16 + TP=2 + 关 P2P 的设计是对的，纯粹是节点坏了。
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
N=500
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o30e] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o30e] $out api_error $e/$t 过半 —— 无效"; return 1; }; }
srv(){ CUDA_VISIBLE_DEVICES=$1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $2 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > logs/o30e_$2.log 2>&1 & }
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o30e] FATAL: 端口 $1 没起来，放弃该 lane"; return 1; }
run(){ local out=$1 g=$2 p=$3; shift 3
  [ -f results/$out.metrics.json ] && { echo "[o30e] $out 已完成，跳过"; return; }
  echo "[o30e] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=$g $PY -m eval.run_eval --dataset popqa_full -n $N \
    --concurrency 12 --port $p "$@" --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl --out results/$out.metrics.json \
    && echo "[o30e] $out done @ $(date +%m%d-%H:%M)" || echo "[o30e] $out FAILED"; }

# ---- Lane A: musique 参数扫（难度轴第三档）----
( srv 0 8081; waitp 8081 || exit 1; echo "[o30e] lane A up @ $(date +%m%d-%H:%M)"
  ss(){ local out=$1; shift
    [ -f results/$out.metrics.json ] && { echo "[o30e] $out 已完成，跳过"; return; }
    echo "[o30e] $out start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset musique --agent scaleseek -n $N \
      --concurrency 16 --port 8081 --retrieval-backend bm25 --max-tokens 2048 "$@" \
      --resume --output results/$out.jsonl \
      && sane $out \
      && $PY scripts/compute_metrics.py --results results/$out.jsonl --out results/$out.metrics.json \
      && echo "[o30e] $out done @ $(date +%m%d-%H:%M)" || echo "[o30e] $out FAILED"; }
  ss ss_sw_musique_topk3        --bm25-top-k 3
  ss ss_sw_musique_topk5        --bm25-top-k 5
  ss ss_sw_musique_topk10       --bm25-top-k 10
  ss ss_sw_musique_k1-0.9_b-0.4 --bm25-top-k 5 --bm25-k1 0.9 --bm25-b 0.4
  ss ss_sw_musique_k1-25_b-1.0  --bm25-top-k 5 --bm25-k1 25  --bm25-b 1.0
  pkill -f "[v]llm.entrypoints.*--port 8081" ) &
LA=$!

# ---- Lane B / C: 4096 探针（对照组用现有 1500 题 2048 结果的前 500 行重算，不另跑）----
( srv 1 8082; waitp 8082 || exit 1; echo "[o30e] lane B up @ $(date +%m%d-%H:%M)"
  run mtok4096_direct   1 8082 --agent direct   --retrieval-backend bm25 --max-tokens 4096
  run mtok4096_bm25_rag 1 8082 --agent bm25_rag --retrieval-backend bm25 --bm25-k1 1.2 --bm25-b 0.75 --max-tokens 4096
  run mtok4096_ircot    1 8082 --agent ircot    --retrieval-backend bm25 --bm25-top-k 3 --max-turns 6 --max-tokens 4096
  pkill -f "[v]llm.entrypoints.*--port 8082" ) &
LB=$!
( srv 2 8083; waitp 8083 || exit 1; echo "[o30e] lane C up @ $(date +%m%d-%H:%M)"
  run mtok4096_search_o1 2 8083 --agent search_o1 --retrieval-backend bm25 --bm25-top-k 5 --max-tokens 4096
  run mtok4096_agentir   2 8083 --agent agentir_rag --agentir-cache results/popqa_full_agentir_retrieval.jsonl --max-tokens 4096
  run mtok4096_dci       2 8083 --agent dci --corpus-path $CORPUS --concurrency 4 --max-tokens 4096
  pkill -f "[v]llm.entrypoints.*--port 8083" ) &
LC=$!
wait $LA $LB $LC
echo "O30E_ALL_DONE @ $(date +%m%d-%H:%M)"
