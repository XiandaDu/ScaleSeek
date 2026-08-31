#!/usr/bin/env bash
# 把一个 RL FSDP checkpoint 合并成 HF 格式，回显合并后的路径。
#   MERGED=$(bash scripts/merge_rl_ckpt.sh [step])
# verl 存的是 model_world_size_N_rank_*.pt；actor/huggingface/ 只有 config 和
# tokenizer，没有权重，必须先合并才能给 vLLM 或 transformers 加载。
set -euo pipefail
RUN="${RL_RUN:-grpo_9b}"
SRC=$RL_CKPT_DIR/$RUN
STEP="${1:-${RL_STEP:-}}"
[ -n "$STEP" ] || STEP=$(cat "$SRC/latest_checkpointed_iteration.txt")
ACTOR=$SRC/global_step_$STEP/actor
MERGED=$SRC/merged_step_$STEP
if [ ! -f "$MERGED/config.json" ]; then
  echo "[merge] $ACTOR -> $MERGED" >&2
  "$PY" -m verl.model_merger merge --backend fsdp \
      --local_dir "$ACTOR" --target_dir "$MERGED" >&2
fi
[ -f "$MERGED/config.json" ] || { echo "FATAL: 合并失败" >&2; exit 1; }
echo "$MERGED"
