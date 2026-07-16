#!/bin/bash
# 接力编排：shard4 完工 → 起 4B vLLM → BM25 四组补扫(popqa_full)
#          shard5 完工 → agentir 逐片检索预计算 → agentir_rag 评测
# 全程日志：launch 时重定向到 logs/pipeline.log；vLLM 单独 logs/vllm8000.log
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING to run on non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index DATASETS=/data/rech/mofengra/datasets
IDX=/data/rech/mofengra/data/agentir_index_v2
FULL=3502554

wait_shard() {  # $1=shard 号：doc_ids 满行数 且 对应 builder 进程退出
  local i=$1
  until [ "$(wc -l < $IDX/shard$i/doc_ids.txt 2>/dev/null || echo 0)" -ge $FULL ]; do sleep 300; done
  while pgrep -f "build_agentir_index.*shard$i" >/dev/null; do sleep 60; done
  echo "[pipeline] shard$i complete @ $(date +%m%d-%H:%M)"
}

# ---- 阶段 1：等 shard4（GPU0 释放）→ vLLM → BM25 四组补扫 ----
wait_shard 4

CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 > logs/vllm8000.log 2>&1 &
ok=0
for t in $(seq 1 120); do
  curl -sf localhost:8000/v1/models >/dev/null && { ok=1; break; }
  sleep 10
done
[ $ok = 1 ] || { echo "[pipeline] vLLM FAILED to come up after 20min, ABORT"; exit 1; }
echo "[pipeline] vLLM up @ $(date +%m%d-%H:%M)"

for kb in "0.9 0.4" "1.5 0.75" "16 1.0" "25 1.0"; do set -- $kb
  echo "[pipeline] bm25 sweep k1=$1 b=$2 start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset popqa_full --agent bm25_rag --concurrency 32 \
    --bm25-k1 $1 --bm25-b $2 --output results/popqa_full_bm25_k1-$1_b-$2.jsonl
  $PY scripts/compute_metrics.py --results results/popqa_full_bm25_k1-$1_b-$2.jsonl \
    --out results/popqa_full_bm25_k1-$1_b-$2.metrics.json
done
echo "[pipeline] BM25_SWEEP_DONE @ $(date +%m%d-%H:%M)"

# ---- 阶段 2：等 shard5（GPU1 释放）→ 6/6 校验 → 预计算 → agentir_rag ----
wait_shard 5
for i in 0 1 2 3 4 5; do
  n=$(wc -l < $IDX/shard$i/doc_ids.txt 2>/dev/null || echo 0)
  [ "$n" -eq $FULL ] || { echo "[pipeline] shard$i count=$n != $FULL, ABORT"; exit 1; }
done
echo "[pipeline] all 6 shards verified @ $(date +%m%d-%H:%M)"

CUDA_VISIBLE_DEVICES=1 $PY scripts/precompute_agentir_retrieval.py \
  --dataset popqa_full -n 1500 --index-root $IDX \
  --top-k 5 --device cuda --out results/popqa_full_agentir_retrieval.jsonl \
  || { echo "[pipeline] precompute FAILED, ABORT"; exit 1; }
echo "[pipeline] precompute done @ $(date +%m%d-%H:%M)"

$PY -m eval.run_eval --dataset popqa_full --agent agentir_rag --concurrency 32 \
  --agentir-cache results/popqa_full_agentir_retrieval.jsonl \
  --output results/popqa_full_agentir.jsonl
$PY scripts/compute_metrics.py --results results/popqa_full_agentir.jsonl \
  --out results/popqa_full_agentir.metrics.json
echo "PIPELINE_ALL_DONE @ $(date +%m%d-%H:%M)"
