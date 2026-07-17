#!/bin/bash
# 手动在 octal40 上把闲置的 GPU2 用起来：跑 5 个数据集 grepseek（主表矩阵必需，
# 本来堵在 GPU0 的 grepseek-全量14267 后面要等 ~20h 才轮到）。
# GPU0 那条 lane 之后也会跑同样 5 格，但顺序相同(nq→...→bamboogle)且本脚本早 ~20h 起步，
# 每格 GPU2 都先写出 metrics.json，GPU0 的循环 `[ -f metrics.json ] && continue` 会全部跳过 —— 无写冲突。
set -u
# —— 硬规则：绝不在跳板机(arcade*)跑重活；必须在 octal40 上 ——
case "$(hostname)" in
  octal40*) : ;;
  *) echo "REFUSING: 当前在 $(hostname)，本脚本只能在 octal40 跑（GPU2 在那）"; exit 1;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
# L40S(sm_89) 两连坑修复（不设这两个 9B 服务起不来，会静默跑出 100% api_error）
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
PORT=8065
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[g2] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[g2] $out api_error $e/$t 过半 —— 无效，不出 metrics"; return 1; }; }

# 释放 GPU2：干掉那条已完成 lane 遗留的 4B 服务(8063)。只 kill 端口 8063，不碰 8061/8062/8064。
echo "[g2] 释放 GPU2：kill 8063 空闲 4B ..."
pkill -f "[v]llm.entrypoints.*--port 8063" 2>/dev/null; sleep 8

# 在 GPU2 起 9B grepseek 服务（独立端口 8065，不干扰 GPU0 的 8061）
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
  --served-model-name grepseek --port $PORT --gpu-memory-utilization 0.85 --max-model-len 32768 \
  > logs/gpu2_9b.log 2>&1 &
ok=0; for t in $(seq 1 180); do curl -sf -m 5 localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[g2] 9B 未起来（查 logs/gpu2_9b.log 有没有 nvcc/headers 报错）ABORT"; exit 1; }
echo "[g2] 9B up @ $(date +%m%d-%H:%M)"

for ds in nq triviaqa 2wikimultihopqa musique bamboogle; do
  o=${ds}_grepseek
  [ -f results/$o.metrics.json ] && { echo "[g2] $o 已完成，跳过"; continue; }
  echo "[g2] $o start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval --dataset $ds --agent grepseek -n 1500 \
    --concurrency 24 --corpus-path $CORPUS --grepseek-port $PORT --grepseek-tokenizer $GS \
    --resume --output results/$o.jsonl \
    && sane $o \
    && $PY scripts/compute_metrics.py --results results/$o.jsonl $(tf $ds) --out results/$o.metrics.json \
    && echo "[g2] $o done @ $(date +%m%d-%H:%M)" || echo "[g2] $o FAILED"
done
pkill -f "[v]llm.entrypoints.*--port $PORT"
echo "G2_ALL_DONE @ $(date +%m%d-%H:%M)"
