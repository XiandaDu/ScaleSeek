#!/bin/bash
# octal30 GPU2：4B agent 全铺 5 数据集（direct/bm25_rag/rag_e5/scaleseek/search_o1/ircot）
# indexed 检索，零 grep（不抢 grepseek 全量的 14GB IO）。全部 --resume。
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db
PORT=8010

CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm${PORT}_rollout.log 2>&1 &
ok=0; for t in $(seq 1 120); do curl -sf -m 5 localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[r4b] 4B not up ABORT"; exit 1; }
echo "[r4b] 4B up @ $(date +%m%d-%H:%M)"

run(){ # $1 ds $2 agent $3 out $4 backend; rest extra
  local ds=$1 ag=$2 out=$3 be=$4; shift 4
  [ -f results/$out.metrics.json ] && { echo "[r4b] $out 已存在,跳过"; return; }
  echo "[r4b] $out start @ $(date +%m%d-%H:%M)"
  local tflag=""; case "$ds" in 2wikimultihopqa) tflag="--title-index-db $TITLE_DB";; esac
  CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset $ds --agent $ag -n 1500 \
    --concurrency 16 --port $PORT --retrieval-backend $be --max-tokens 2048 \
    --resume --output results/$out.jsonl "$@" \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $tflag --out results/$out.metrics.json \
    && echo "[r4b] $out done @ $(date +%m%d-%H:%M)" || echo "[r4b] $out FAILED"; }

for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
  run $ds direct    ${ds}_direct     bm25
  run $ds bm25_rag  ${ds}_bm25_rag   bm25 --bm25-k1 1.2 --bm25-b 0.75
  run $ds bm25_rag  ${ds}_rag_e5     e5   --bm25-top-k 5
  run $ds scaleseek ${ds}_scaleseek  bm25
  run $ds ircot     ${ds}_ircot_bm25 bm25 --bm25-top-k 3 --max-turns 6
  run $ds search_o1 ${ds}_search_o1  bm25 --bm25-top-k 5
done
pkill -f "[v]llm.*$PORT"
echo "ROLLOUT_4B_ALL_DONE @ $(date +%m%d-%H:%M)"
