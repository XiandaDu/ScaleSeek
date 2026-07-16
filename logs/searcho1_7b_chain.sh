#!/bin/bash
# 任务 #10：search-o1 官方口径 7B 推理模型基线
# 模型：DeepSeek-R1-Distill-Qwen-7B（官方 repo 无自有 ckpt，是 prompt 框架；
#       其 to-do 点名 DeepSeek-R1 系为 backbone 方向 → 取其 7B 蒸馏版）
# 检索：对齐原则（grepseek 优先）→ top-3；BM25 与 E5 两个变体，
#       对应 GrepSeek Table 1 的 Search-O1+BM25 .4003 / +E5 .4322 两格。
# 门控：等 GPU1 的 grepseek tool_max_tokens 消融完 → 撤 9B → 上 7B @8004。
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

until grep -q "ABLATE_GS_TOOLTOK_ALL_DONE" logs/ablate_gs_tooltok.log 2>/dev/null; do sleep 600; done
pkill -f "[v]llm.*8003"
sleep 20
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $R1 --served-model-name agent7b --port 8004 \
  --gpu-memory-utilization 0.80 --max-model-len 32768 > logs/vllm8004_r1_7b.log 2>&1 &
ok=0
for t in $(seq 1 90); do
  curl -sf -m 5 localhost:8004/v1/models >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
[ $ok = 1 ] || { echo "[so1-7b] 7B vLLM FAILED, ABORT"; tail -5 logs/vllm8004_r1_7b.log; exit 1; }
echo "[so1-7b] R1-Distill-7B up @ $(date +%m%d-%H:%M)"

run_one() {  # $1=outname, rest flags
  local out=$1; shift
  echo "[so1-7b] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset popqa_full --agent search_o1 \
    -n 1500 --concurrency 16 --port 8004 --model agent7b --temperature 0.6 \
    --bm25-top-k 3 --output results/$out.jsonl "$@" \
    || { echo "[so1-7b] $out FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl \
    --out results/$out.metrics.json
  echo "[so1-7b] $out done @ $(date +%m%d-%H:%M)"
}

run_one popqa_full_search_o1_7b_bm25
run_one popqa_full_search_o1_7b_e5 --retrieval-backend e5
echo "SEARCHO1_7B_ALL_DONE @ $(date +%m%d-%H:%M)"
