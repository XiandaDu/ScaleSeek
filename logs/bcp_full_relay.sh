#!/bin/bash
# BCP 全量接力（830 题）：
#   阶段 A：等 DR-DCI 830 完（vLLM:8000 腾出吞吐）→ direct / bm25(25/1) /
#           bm25(1.2/.75) / scaleseek 全量（--max-tokens 4096：冒烟发现 2048
#           被 thinking 吃光，parse_err >50%）
#   阶段 B：等 E5 接力完（GPU1 腾空）→ 停 faiss_searcher → GPU1 起 GrepSeek-9B
#           @:8003 → grepseek 全量（冒烟时 8002 被检索服务占用而全错，此处补齐）
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
export LLM_TOKENIZER=Qwen/Qwen3-4B
BCP_CORPUS=$DATASETS/browsecomp_plus/corpus.jsonl
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"
GS_MODEL=alireza7/GrepSeek-Qwen3.5-9B-GRPO

run_one() {  # $1=agent $2=outfile $3...=extra flags
  local agent=$1 out=$2; shift 2
  echo "[bcp-full] $agent -> $out start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset browsecomp_plus --agent $agent -n 830 \
    --concurrency 16 --output results/$out.jsonl "$@" \
    || { echo "[bcp-full] $agent FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl $QRELS \
    --out results/$out.metrics.json
  echo "[bcp-full] $agent done @ $(date +%m%d-%H:%M)"
}

# ---- 阶段 A：等 DR-DCI 830 ----
until grep -q "Finished bcplus eval" logs/drdci_full830.log 2>/dev/null; do sleep 600; done
echo "[bcp-full] DR-DCI done, phase A start @ $(date +%m%d-%H:%M)"

run_one direct    bcp_direct_830    --max-tokens 4096
run_one bm25_rag  bcp_bm25_25_1_830 --bm25-k1 25 --bm25-b 1.0 --max-tokens 4096
run_one bm25_rag  bcp_bm25_12_075_830 --bm25-k1 1.2 --bm25-b 0.75 --max-tokens 4096
run_one scaleseek bcp_scaleseek_830 --max-tokens 4096
echo "[bcp-full] phase A done @ $(date +%m%d-%H:%M)"

# ---- 阶段 B：等 E5 接力完 → GPU1 腾空起 9B → grepseek ----
until grep -q "E5_RELAY_ALL_DONE" logs/e5_relay.log 2>/dev/null; do sleep 600; done
pkill -f "[f]aiss_searcher" 2>/dev/null
sleep 10
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $GS_MODEL --served-model-name grepseek --port 8003 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm8003_gs9b.log 2>&1 &
ok=0
for t in $(seq 1 120); do
  curl -sf localhost:8003/v1/models >/dev/null && { ok=1; break; }
  sleep 10
done
[ $ok = 1 ] || { echo "[bcp-full] 9B vLLM FAILED to come up, ABORT"; exit 1; }
echo "[bcp-full] GrepSeek-9B up @ $(date +%m%d-%H:%M)"

run_one grepseek bcp_grepseek_830 --corpus-path $BCP_CORPUS \
  --grepseek-port 8003 --grepseek-tokenizer $GS_MODEL
echo "BCP_FULL_ALL_DONE @ $(date +%m%d-%H:%M)"
