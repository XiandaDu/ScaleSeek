#!/bin/bash
# usage: driver_gpu.sh <gpu_id> "<shard:skip> <shard:skip> ..."
GPU=$1; shift
# HARD GUARD: heavy jobs run ONLY on GPU nodes, NEVER on the jump host (arcade*).
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "[driver$GPU] REFUSING to run on non-GPU host: $(hostname)" \
       >> /data/rech/mofengra/ScaleSeek/logs/driver$GPU.log; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd /data/rech/mofengra/ScaleSeek
for spec in $@; do
  i=${spec%%:*}; skip=${spec##*:}
  for attempt in 1 2 3; do
    echo "[driver$GPU] shard$i attempt$attempt @ $(date +%m%d-%H:%M)" >> logs/driver$GPU.log
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/build_agentir_index.py \
      --corpus /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \
      --out /data/rech/mofengra/data/agentir_index_v2/shard$i \
      --skip-passages $skip --max-passages 3502554 --resume \
      --device cuda --batch-size 512 --max-length 256 --index-type sq8_flat \
      >> logs/agentir_shard$i.log 2>&1 && { echo "[driver$GPU] shard$i DONE" >> logs/driver$GPU.log; break; }
    echo "[driver$GPU] shard$i attempt$attempt FAILED; retry in 120s" >> logs/driver$GPU.log
    sleep 120
  done
done
echo "DRIVER${GPU}_ALL_DONE @ $(date +%m%d-%H:%M)" >> logs/driver$GPU.log
