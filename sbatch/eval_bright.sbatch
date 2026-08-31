#!/usr/bin/env bash
# 用 SFT/RL checkpoint 在 BRIGHT 上评测（vLLM + 按任务隔离的 BM25 索引）。
#
#   sbatch sbatch/eval_bright.sbatch                    # 默认 60 题探测
#   EVAL_MODEL=<ckpt> EVAL_N=1384 sbatch ...            # 全量
#
# 覆盖项：EVAL_MODEL [最新 SFT ckpt] / EVAL_N [60] / EVAL_CONC [4] / EVAL_TAG [sft]
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_eval_bright
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
require_full_gpus 1
trap 'stop_vllm; cleanup_tmpdir' EXIT

export BRIGHT_INDEX_ROOT="${BRIGHT_INDEX_ROOT:-$DATA/bright_indexes}"
[ -d "$BRIGHT_INDEX_ROOT" ] || { echo "FATAL: 缺 BRIGHT 索引 — 先跑 sbatch/p0_bright.sbatch"; exit 1; }

# EVAL_RL_STEP=<n> 评 RL checkpoint（自动合并 FSDP 分片）；不设则用最新 SFT
MODEL="${EVAL_MODEL:-}"
if [ -z "$MODEL" ] && [ -n "${EVAL_RL_STEP:-}" ]; then
  MODEL=$(bash scripts/merge_rl_ckpt.sh "$EVAL_RL_STEP")
fi
if [ -z "$MODEL" ]; then
  MODEL=$(find "$SFT_CKPT_DIR" -type d -name huggingface -printf '%T@ %p\n' 2>/dev/null \
          | sort -n | tail -1 | cut -d' ' -f2-)
fi
[ -f "$MODEL/config.json" ] || { echo "FATAL: 找不到 checkpoint: $MODEL"; exit 1; }

TAG="${EVAL_TAG:-sft}"
OUT=$RESULTS/bright_scaleseek_${TAG}_${SLURM_JOB_ID}.jsonl
mkdir -p "$RESULTS"
echo "[eval_bright] model=$MODEL  indexes=$BRIGHT_INDEX_ROOT  n=${EVAL_N:-60}"

# 与 eval_bcp 同样不用 common.sh 的 start_vllm（它固定传 --revision，本地路径会崩）
VLLM_LOG=$REPO/logs/vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype bfloat16 \
    --host 127.0.0.1 --port "$LLM_PORT" --tensor-parallel-size 1 \
    --served-model-name "$LLM_MODEL" \
    --max-model-len "${VLLM_MAX_LEN:-32768}" \
    --gpu-memory-utilization "${VLLM_MEM_UTIL:-0.90}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
waitp "$LLM_PORT" 2400

"$PY" -m eval.run_eval --dataset bright --agent scaleseek --retriever bm25 \
    --model "$LLM_MODEL" -n "${EVAL_N:-60}" \
    --concurrency "${EVAL_CONC:-4}" --output "$OUT"

stop_vllm
echo "[eval_bright] 完成 -> $OUT"
