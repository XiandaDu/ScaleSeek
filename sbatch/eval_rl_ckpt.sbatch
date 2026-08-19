#!/usr/bin/env bash
# 合并一个 RL FSDP checkpoint 成 HF 格式，然后与 SFT 基线同条件对照评测。
#
#   RL_STEP=40 sbatch sbatch/eval_rl_ckpt.sbatch
#
# 覆盖项：
#   RL_STEP    要评的 global_step  [自动取最新]
#   RL_RUN     run 名              [grpo_9b]
#   EVAL_N     dev 题数            [16，与格式闸门同规模，可直接对照]
#
# 为什么要合并：verl 存的是 4 份 FSDP 分片（model_world_size_4_rank_*.pt），
# actor/huggingface/ 下只有 config 和 tokenizer，没有权重。
# 对照基线（SFT global_step_159，同样 16 题 dev）：format_pass 15/16 (94%)、EM 10/16 (62%)
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_eval_rl
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
require_full_gpus 1
trap cleanup_tmpdir EXIT

RUN="${RL_RUN:-grpo_9b}"
SRC=$RL_CKPT_DIR/$RUN
STEP="${RL_STEP:-}"
if [ -z "$STEP" ]; then
  STEP=$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null)
fi
[ -n "$STEP" ] || { echo "FATAL: 找不到 checkpoint step"; exit 1; }
ACTOR=$SRC/global_step_$STEP/actor
MERGED=$SRC/merged_step_$STEP
[ -d "$ACTOR" ] || { echo "FATAL: $ACTOR 不存在"; exit 1; }

echo "[eval_rl] run=$RUN step=$STEP"
if [ -f "$MERGED/model.safetensors" ] || [ -f "$MERGED/model.safetensors.index.json" ]; then
  echo "[eval_rl] 已合并 — 跳过"
else
  echo "[eval_rl] 合并 FSDP 分片 -> $MERGED"
  "$PY" -m verl.model_merger merge --backend fsdp \
      --local_dir "$ACTOR" --target_dir "$MERGED"
fi
[ -f "$MERGED/config.json" ] || { echo "FATAL: 合并失败"; exit 1; }

# 与格式闸门完全同条件，好和 SFT 基线直接比
"$PY" scripts/run_scaleseek_smoke_eval.py \
    --model "$MERGED" \
    --index-dir "$BM25_INDEX_DIR" \
    --questions "$RL_DATA_DIR/dev.jsonl" \
    --limit "${EVAL_N:-16}" \
    --out "$RESULTS/rl_step${STEP}_dev.jsonl"

echo
echo "[eval_rl] 对照 SFT 基线 (global_step_159, 同 16 题): format_pass 94%, EM 62%"
