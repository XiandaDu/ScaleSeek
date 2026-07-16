#!/bin/bash
# grepseek tool_max_tokens 消融（任务 #12）：{1024, 4096} @ popqa_full 1500
# 2048 基线 = 已有 popqa_full_grepseek（.3440/.3870）。
# 门控：等 BCP grepseek 830 完（9B@:8003 空出吞吐），复用同一服务器。
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl

until [ -f results/bcp_grepseek_830.metrics.json ]; do sleep 600; done
curl -sf -m 10 localhost:8003/v1/models >/dev/null || { echo "[ablate-gs] 9B@:8003 not up, ABORT"; exit 1; }

for T in 1024 4096; do
  echo "[ablate-gs] tool_max_tokens=$T start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset popqa_full --agent grepseek -n 1500 --concurrency 16 \
    --corpus-path $CORPUS --grepseek-port 8003 --grepseek-tokenizer $GS \
    --grepseek-tool-max-tokens $T \
    --output results/popqa_full_grepseek_tooltok$T.jsonl \
    || { echo "[ablate-gs] T=$T FAILED"; continue; }
  $PY scripts/compute_metrics.py --results results/popqa_full_grepseek_tooltok$T.jsonl \
    --out results/popqa_full_grepseek_tooltok$T.metrics.json
  echo "[ablate-gs] tool_max_tokens=$T done @ $(date +%m%d-%H:%M)"
done
echo "ABLATE_GS_TOOLTOK_ALL_DONE @ $(date +%m%d-%H:%M)"
