#!/bin/bash
#SBATCH --job-name=ss_o40
#SBATCH --partition=rali
#SBATCH --nodelist=octal[40]
#SBATCH --gres=gpu:ls40:4
#SBATCH --mem=256G
#SBATCH --time=10-23:14:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal40_%j.log
# octal40 第二轮（首轮 grepseek+dci 已 O40_ALL_DONE）。
# ⚠ octal35 的 CUDA 坏了（cuInit=999，节点级故障），其工作全部搬来这里：
#   GPU0 = grepseek 全量 14267 铁证（原 octal35）
#   GPU1 = search_r1 全铺（原 octal35）
#   GPU2 = search_o1_e5 + ircot_e5      GPU3 = scaleseek_e5 + agentir
# octal41 跑 BM25 版 rollout_4b —— 零重叠。全部 --resume。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
# ⚠ octal40/41 没有 /usr/local/cuda，CUDA_HOME 不设的话 flashinfer JIT 找不到 nvcc，
# vLLM EngineCore 起不来（job 7159 就是这样跑出 10 格 100% api_error 的假结果）。
# nvcc 来自 pip 包 nvidia-cuda-nvcc。
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
AIDX=/data/rech/mofengra/data/agentir_index_v2
TDB=/data/rech/mofengra/data/corpus_title_index.db
DS="nq triviaqa 2wikimultihopqa musique bamboogle"
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o40b] FATAL: 端口 $1 服务未起来，放弃该 lane（不写 api_error 垃圾）"; return 1; }
# 服务死了也会“跑完”，只是每行都是 api_error。出 metrics 前先验收。
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o40b] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o40b] $out api_error $e/$t 过半 —— 判为无效，不出 metrics"; return 1; }; }
serve4b(){ CUDA_VISIBLE_DEVICES=$1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $2 --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.80 \
  --max-model-len 32768 > logs/o40b_$2.log 2>&1 & }

# ---- GPU0: 9B grepseek 全量铁证（L40S 48G 单卡无 TP）----
( CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
    --served-model-name grepseek --port 8061 --gpu-memory-utilization 0.85 --max-model-len 32768 \
    > logs/o40b_9b.log 2>&1 &
  waitp 8061 || exit 1; echo "[o40b] 9B up @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset popqa_full14267 --agent grepseek \
    --concurrency 24 --corpus-path $CORPUS --grepseek-port 8061 --grepseek-tokenizer $GS \
    --resume --output results/popqa_grepseek_FULL14267.jsonl \
    && sane popqa_grepseek_FULL14267 \
    && $PY scripts/compute_metrics.py --results results/popqa_grepseek_FULL14267.jsonl \
         --out results/popqa_grepseek_FULL14267.metrics.json \
    && echo "[o40b] GSFULL done @ $(date +%m%d-%H:%M)" || echo "[o40b] GSFULL FAILED"
  # 7159 那 10 格 grepseek 是假数据（服务没起来），重跑：9B 服务在本卡，顺手接上
  for ds in $DS; do
    o=${ds}_grepseek; [ -f results/$o.metrics.json ] && continue
    echo "[o40b] $o start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset $ds --agent grepseek -n 1500 \
      --concurrency 16 --corpus-path $CORPUS --grepseek-port 8061 --grepseek-tokenizer $GS \
      --resume --output results/$o.jsonl \
      && sane $o \
      && $PY scripts/compute_metrics.py --results results/$o.jsonl $(tf $ds) --out results/$o.metrics.json \
      && echo "[o40b] $o done @ $(date +%m%d-%H:%M)" || echo "[o40b] $o FAILED"
  done ) &

# ---- GPU1: 3B search_r1（hotpot 收尾 + 全铺 10 格）----
( CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $SR1 \
    --served-model-name search_r1 --port 8062 --gpu-memory-utilization 0.80 --max-model-len 8192 \
    > logs/o40b_3b.log 2>&1 &
  waitp 8062 || exit 1; echo "[o40b] 3B up @ $(date +%m%d-%H:%M)"
  sr1(){ local ds=$1 out=$2 be=$3
    [ -f results/$out.metrics.json ] && return
    echo "[o40b] $out start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset $ds --agent search_r1 -n 1500 \
      --concurrency 16 --search-r1-port 8062 --retrieval-backend $be --bm25-top-k 3 --max-turns 4 \
      --resume --output results/$out.jsonl \
      && sane $out \
      && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
      && echo "[o40b] $out done @ $(date +%m%d-%H:%M)" || echo "[o40b] $out FAILED"; }
  sr1 hotpotqa hotpotqa_search_r1_e5 e5
  for ds in $DS; do sr1 $ds ${ds}_search_r1_bm25 bm25; sr1 $ds ${ds}_search_r1_e5 e5; done ) &

# ---- GPU2: search_o1_e5 + ircot_e5 ----
( serve4b 2 8063; waitp 8063 || exit 1; echo "[o40b] 4B@8063 up @ $(date +%m%d-%H:%M)"
  e5run(){ local ds=$1 ag=$2 out=$3; shift 3
    [ -f results/$out.metrics.json ] && return
    echo "[o40b] $out start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset $ds --agent $ag -n 1500 \
      --concurrency 16 --port 8063 --retrieval-backend e5 --max-tokens 2048 \
      --resume --output results/$out.jsonl "$@" \
      && sane $out \
      && $PY scripts/compute_metrics.py --results results/$out.jsonl $(tf $ds) --out results/$out.metrics.json \
      && echo "[o40b] $out done @ $(date +%m%d-%H:%M)" || echo "[o40b] $out FAILED"; }
  for ds in $DS; do e5run $ds search_o1 ${ds}_search_o1_e5 --bm25-top-k 5; done
  for ds in $DS; do e5run $ds ircot ${ds}_ircot_e5 --bm25-top-k 3 --max-turns 6; done ) &

# ---- GPU3: scaleseek_e5 + agentir（先预计算再起 reader，避免显存打架）----
( serve4b 3 8064; waitp 8064 || exit 1; echo "[o40b] 4B@8064 up @ $(date +%m%d-%H:%M)"
  for ds in $DS; do
    o=${ds}_scaleseek_e5; [ -f results/$o.metrics.json ] && continue
    echo "[o40b] $o start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval --dataset $ds --agent scaleseek -n 1500 \
      --concurrency 16 --port 8064 --retrieval-backend e5 --max-tokens 2048 \
      --resume --output results/$o.jsonl \
      && sane $o \
      && $PY scripts/compute_metrics.py --results results/$o.jsonl $(tf $ds) --out results/$o.metrics.json \
      && echo "[o40b] $o done @ $(date +%m%d-%H:%M)" || echo "[o40b] $o FAILED"
  done
  for ds in $DS; do
    [ -f results/${ds}_agentir_retrieval.jsonl ] || \
    CUDA_VISIBLE_DEVICES=3 $PY scripts/precompute_agentir_retrieval.py --dataset $ds -n 1500 \
      --index-root $AIDX --top-k 5 --device cuda --out results/${ds}_agentir_retrieval.jsonl
    o=${ds}_agentir; [ -f results/$o.metrics.json ] && continue
    echo "[o40b] $o start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval --dataset $ds --agent agentir_rag -n 1500 \
      --concurrency 16 --port 8064 --agentir-cache results/${ds}_agentir_retrieval.jsonl \
      --max-tokens 2048 --resume --output results/$o.jsonl \
      && sane $o \
      && $PY scripts/compute_metrics.py --results results/$o.jsonl $(tf $ds) --out results/$o.metrics.json \
      && echo "[o40b] $o done @ $(date +%m%d-%H:%M)" || echo "[o40b] $o FAILED"
  done
  # 7159 那 10 格里的 dci 5 格同样是假数据，重跑（4B 服务在本卡）
  for ds in $DS; do
    o=${ds}_dci; [ -f results/$o.metrics.json ] && continue
    echo "[o40b] $o start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval --dataset $ds --agent dci -n 1500 \
      --concurrency 4 --port 8064 --corpus-path $CORPUS --max-tokens 2048 \
      --resume --output results/$o.jsonl \
      && sane $o \
      && $PY scripts/compute_metrics.py --results results/$o.jsonl $(tf $ds) --out results/$o.metrics.json \
      && echo "[o40b] $o done @ $(date +%m%d-%H:%M)" || echo "[o40b] $o FAILED"
  done ) &
wait
echo "O40B_ALL_DONE @ $(date +%m%d-%H:%M)"
