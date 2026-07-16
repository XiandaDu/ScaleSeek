#!/bin/bash
# 补跑：hotpotqa scaleseek 用当前配置、同 1500 样本（旧 7405 文件缺 trace 字段，已移开）
# 门控：等 hotpot 主链完（HOTPOT_CHAIN_ALL_DONE，8000 被 pkill）→ 自起 4B → 跑
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index LLM_TOKENIZER=Qwen/Qwen3-4B
TITLE_DB=/data/rech/mofengra/data/corpus_title_index.db

until grep -q "HOTPOT_CHAIN_ALL_DONE" logs/chain_gpu0_hotpot.log 2>/dev/null; do sleep 300; done
sleep 15
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 > logs/vllm8000.log 2>&1 &
ok=0
for t in $(seq 1 120); do curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || { echo "[ss-rerun] 4B not up, ABORT"; exit 1; }
echo "[ss-rerun] 4B up, scaleseek 1500 start @ $(date +%m%d-%H:%M)"
CUDA_VISIBLE_DEVICES=0 $PY -m eval.run_eval --dataset hotpotqa --agent scaleseek \
  -n 1500 --concurrency 16 --max-tokens 2048 --resume \
  --output results/hotpotqa_scaleseek.jsonl \
  && $PY scripts/compute_metrics.py --results results/hotpotqa_scaleseek.jsonl \
       --title-index-db $TITLE_DB --out results/hotpotqa_scaleseek.metrics.json \
  && echo "[ss-rerun] scaleseek done @ $(date +%m%d-%H:%M)" || echo "[ss-rerun] FAILED"
pkill -f "[v]llm.*8000"
echo "SS_RERUN_ALL_DONE @ $(date +%m%d-%H:%M)"
