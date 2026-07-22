#!/usr/bin/env bash
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=56G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Parametrized full-PopQA Phase-2 run for the locally-run methods.
# Submit with:  sbatch -J p2_<name> --export=ALL,AGENT=...,RET=...,SR1=... p2_local.sbatch
#   AGENT: direct | rag | scaleseek | search_o1 | search_r1
#   RET:   none | bm25 | e5 | qwen3_emb_4b        (qwen3_emb_4b needs --gres=gpu:3 --mem=110G)
#   SR1:   "" | 7b | 14b                          (search_r1 only)
# The output file name comes from the job name: p2_<x> -> results/phase2/popqa_<x>.jsonl
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT

AGENT=${AGENT:?export AGENT=}
RET=${RET:-none}
SR1=${SR1:-}
PORT=$((10000 + SLURM_JOB_ID % 20000))
OUT=results/phase2/popqa_${SLURM_JOB_NAME#p2_}.jsonl
mkdir -p results/phase2

RETARGS=()
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | wc -l)
case "$RET" in
  none) ;;
  qwen3_emb_4b)
    # vLLM on all GPUs but the last; the 4B query encoder gets the last one.
    export VLLM_GPUS=$(seq -s, 0 $((NGPU - 2)))
    RETARGS=(--retriever qwen3_emb_4b --retriever-device "cuda:$((NGPU - 1))")
    ;;
  *) RETARGS=(--retriever "$RET" --retriever-device cpu);;
esac

RESUME=()
[ -f "$OUT" ] && RESUME=(--resume)

if [ "$AGENT" = search_r1 ]; then
  case "$SR1" in
    7b)  M=$SR1_7B;  R=$SR1_7B_REV;  CONC=${CONC:-8};;
    14b) M=$SR1_14B; R=$SR1_14B_REV; CONC=${CONC:-6}; export VLLM_MEM_UTIL=0.93;;
    *) echo "FATAL: search_r1 requires SR1=7b|14b"; exit 1;;
  esac
  start_vllm "$M" "$R" "$PORT" 2 "search-r1-$SR1"
  waitp "$PORT" 3600
  "$PY" -m eval.run_eval --dataset popqa --full-eval --agent search_r1 \
    "${RETARGS[@]}" \
    --search-r1-host 127.0.0.1 --search-r1-port "$PORT" \
    --search-r1-model "search-r1-$SR1" --search-r1-tokenizer "$M" \
    --generator-revision "$R" --concurrency "$CONC" --retries 3 \
    --output "$OUT" "${RESUME[@]}"
else
  start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B --reasoning-parser qwen3
  waitp "$PORT" 3600
  EXTRA=()
  case "$AGENT" in
    search_o1|scaleseek) EXTRA=(--tokenizer Qwen/Qwen3.5-9B);;
  esac
  "$PY" -m eval.run_eval --dataset popqa --full-eval --agent "$AGENT" \
    "${RETARGS[@]}" "${EXTRA[@]}" \
    --host 127.0.0.1 --port "$PORT" --model Qwen/Qwen3.5-9B \
    --generator-revision "$GEN_REV" --concurrency "${CONC:-16}" --retries 3 \
    --output "$OUT" "${RESUME[@]}"
fi
stop_vllm

sane "$OUT"
"$PY" scripts/compute_metrics.py --results "$OUT" --dataset popqa --agent "$AGENT" \
  --out "${OUT%.jsonl}.metrics.json"
echo "ALL_DONE $OUT"
