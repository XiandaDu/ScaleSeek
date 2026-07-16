#!/bin/bash
# GitHub issue 铁证：grepseek 全 14,267 popqa（消除"前 1500 顺序偏差"变量）
# GPU0，9B@8003，我们的 harness（已验证 ==官方 Δ0.003），--resume 抗重启
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl

# 节点感知：octal30=A5000/24G 需 TP=2 装 9B；octal35=A6000/48G 单卡即可
if [[ "$(hostname)" == octal30* ]]; then
  GS_GPUS=0,1; GS_EXTRA="--tensor-parallel-size 2 --disable-custom-all-reduce"; export NCCL_P2P_DISABLE=1
else
  GS_GPUS=0; GS_EXTRA=""
fi
CUDA_VISIBLE_DEVICES=$GS_GPUS setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model $GS --served-model-name grepseek --port 8003 $GS_EXTRA \
  --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/vllm8003_gs9b_gpu0.log 2>&1 &
ok=0
for t in $(seq 1 300); do curl -sf -m 5 localhost:8003/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
[ $ok = 1 ] || { echo "[gsfull] 9B not up, ABORT"; tail -8 logs/vllm8003_gs9b_gpu0.log; exit 1; }
echo "[gsfull] 9B up @ $(date +%m%d-%H:%M), running FULL 14267 (resume-safe)"

# 无 -n → 全 14,267；--resume 抗重启（会话反复重启，进度不丢）
$PY -m eval.run_eval --dataset popqa_full14267 --agent grepseek --concurrency 24 \
  --corpus-path $CORPUS --grepseek-port 8003 --grepseek-tokenizer $GS \
  --resume --output results/popqa_grepseek_FULL14267.jsonl \
  && $PY scripts/compute_metrics.py --results results/popqa_grepseek_FULL14267.jsonl \
       --out results/popqa_grepseek_FULL14267.metrics.json \
  && echo "GSFULL_DONE @ $(date +%m%d-%H:%M)" || echo "GSFULL_FAILED"
pkill -f "[v]llm.*8003"
