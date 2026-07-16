#!/bin/bash
# GPU1 尾链（取代 searcho1_7b_chain.sh）：
#   门控：grepseek tool_max_tokens 消融完（ABLATE_GS_TOOLTOK_ALL_DONE）
#   → 撤 9B → 3B SearchR1 ckpt @8001 → 补跑 search_r1_e5（论文口径，
#     上次因 :8001 未起而 100% api_error）
#   → 撤 3B → R1-Distill-7B @8004 → search_o1_7b × {BM25, E5} top-3
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
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

wait_port() {  # $1=port $2=logfile
  local ok=0
  for t in $(seq 1 270); do
    curl -sf -m 5 localhost:$1/v1/models >/dev/null 2>&1 && { ok=1; break; }
    sleep 10
  done
  [ $ok = 1 ] || { echo "[gpu1-tail] :$1 not up after 45min, ABORT"; tail -5 $2; exit 1; }
}

until grep -q "ABLATE_GS_TOOLTOK_ALL_DONE" logs/ablate_gs_tooltok.log 2>/dev/null; do sleep 600; done
pkill -f "[v]llm.*8003"
sleep 20

# ---- 3B SearchR1 @8001 → search_r1_e5 ----
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $SR1 --served-model-name search_r1 --port 8001 \
  --gpu-memory-utilization 0.45 --max-model-len 8192 > logs/vllm8001_sr1.log 2>&1 &
wait_port 8001 logs/vllm8001_sr1.log
echo "[gpu1-tail] SearchR1-3B up @ $(date +%m%d-%H:%M)"
CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset popqa_full --agent search_r1 \
  -n 1500 --concurrency 16 --retrieval-backend e5 --bm25-top-k 3 --max-turns 4 \
  --output results/popqa_full_search_r1_e5.jsonl \
  && $PY scripts/compute_metrics.py --results results/popqa_full_search_r1_e5.jsonl \
       --out results/popqa_full_search_r1_e5.metrics.json \
  && echo "[gpu1-tail] search_r1_e5 done @ $(date +%m%d-%H:%M)" \
  || echo "[gpu1-tail] search_r1_e5 FAILED"
pkill -f "[v]llm.*8001"
sleep 20

# ---- R1-Distill-7B @8004 → search_o1_7b × 2 ----
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $R1 --served-model-name agent7b --port 8004 \
  --gpu-memory-utilization 0.80 --max-model-len 32768 > logs/vllm8004_r1_7b.log 2>&1 &
wait_port 8004 logs/vllm8004_r1_7b.log
echo "[gpu1-tail] R1-Distill-7B up @ $(date +%m%d-%H:%M)"

run_so1() {  # $1=outname, rest flags
  local out=$1; shift
  echo "[gpu1-tail] $out start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset popqa_full --agent search_o1 \
    -n 1500 --concurrency 16 --port 8004 --model agent7b --temperature 0.6 \
    --bm25-top-k 3 --output results/$out.jsonl "$@" \
    || { echo "[gpu1-tail] $out FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl \
    --out results/$out.metrics.json
  echo "[gpu1-tail] $out done @ $(date +%m%d-%H:%M)"
}
run_so1 popqa_full_search_o1_7b_bm25
run_so1 popqa_full_search_o1_7b_e5 --retrieval-backend e5
echo "GPU1_TAIL_ALL_DONE @ $(date +%m%d-%H:%M)"
