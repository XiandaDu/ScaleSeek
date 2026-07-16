#!/bin/bash
# E5 接力：等 build_e5_index 完工（21,015,324 全量）→ 在 popqa_full 上自动跑
#   1) rag_e5 top-5（与我们 RAG(BM25) 行同口径）
#   2) rag_e5 top-3（GrepSeek Table 1 RAG+E5 对照口径，论文参照 F1 .4468）
#   3) search_r1 论文口径（E5 + top-3 + 4 轮；对标原论文 PopQA EM .413）
#   4) search_o1 dense 后端（E5 + top-5）
# 前提：vLLM 4B @ :8000 常驻（GPU0）。查询编码器走 GPU1。
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index
FULL=21015324

# 等构建完成：log 出现 Done. 且进程退出
until grep -q "^Done\." logs/e5_build.log 2>/dev/null; do sleep 600; done
while pgrep -f "build_e5_index" >/dev/null; do sleep 60; done
n=$(wc -l < $E5_INDEX_DIR/doc_ids.txt 2>/dev/null || echo 0)
[ "$n" -eq $FULL ] || { echo "[e5-relay] doc_ids=$n != $FULL, ABORT"; exit 1; }
echo "[e5-relay] index complete, starting evals @ $(date +%m%d-%H:%M)"

run_one() {  # $1=out-name, rest=run_eval flags
  local out=$1; shift
  echo "[e5-relay] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset popqa_full \
    --retrieval-backend e5 --concurrency 16 \
    --output results/$out.jsonl "$@" \
    || { echo "[e5-relay] $out FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl \
    --out results/$out.metrics.json
  echo "[e5-relay] $out done @ $(date +%m%d-%H:%M)"
}

run_one popqa_full_rag_e5           --agent bm25_rag  --bm25-top-k 5
run_one popqa_full_rag_e5_top3      --agent bm25_rag  --bm25-top-k 3
run_one popqa_full_search_r1_e5     --agent search_r1 --bm25-top-k 3 --max-turns 4
run_one popqa_full_search_o1_e5     --agent search_o1 --bm25-top-k 5
echo "E5_RELAY_ALL_DONE @ $(date +%m%d-%H:%M)"
