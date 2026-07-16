#!/bin/bash
# GPU0 补充：hotpotqa（多跳）第二数据集，跑吃 4B 服务的 agent。
# 目的：① 给 8-agent 对比加多跳数据集 ② 验证 E5>>BM25 是否推广到多跳
#       ③ hotpotqa 有 gold → 出 Gold R@W（popqa 出不了）
# 不碰 GPU1 的 9B grepseek 链。全部 --resume（抗重启）。
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export LLM_TOKENIZER=Qwen/Qwen3-4B
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db

# 4B agent @8000 (GPU0)
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 > logs/vllm8000.log 2>&1 &
ok=0
for t in $(seq 1 120); do
  curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
[ $ok = 1 ] || { echo "[hotpot] 4B not up, ABORT"; tail -6 logs/vllm8000.log; exit 1; }
echo "[hotpot] 4B up @ $(date +%m%d-%H:%M)"

run_one() {  # $1=agent $2=out $3=backend(bm25|e5) $4...=extra
  local agent=$1 out=$2 backend=$3; shift 3
  echo "[hotpot] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset hotpotqa --agent $agent \
    -n 1500 --concurrency 16 --retrieval-backend $backend --max-tokens 2048 \
    --resume --output results/$out.jsonl "$@" \
    || { echo "[hotpot] $out FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl \
    --title-index-db $TITLE_DB --out results/$out.metrics.json
  echo "[hotpot] $out done @ $(date +%m%d-%H:%M)"
}

run_one direct    hotpotqa_direct_4b   bm25
run_one bm25_rag  hotpotqa_bm25_rag    bm25 --bm25-k1 1.2 --bm25-b 0.75
run_one bm25_rag  hotpotqa_rag_e5      e5   --bm25-top-k 5
run_one scaleseek hotpotqa_scaleseek   bm25
run_one search_o1 hotpotqa_search_o1   bm25 --bm25-top-k 5
pkill -f "[v]llm.*8000"
echo "HOTPOT_CHAIN_ALL_DONE @ $(date +%m%d-%H:%M)"
