#!/usr/bin/env bash
# Phase-2 的 E/F 阶段（pilot_v2 + 汇总），可独立重跑。完全后台：
#   source setup_env.sh; setsid nohup bash scripts/phase2_pilot.sh > logs/phase2_pilot.log 2>&1 < /dev/null &
# 状态：logs/phase2_status.txt；进度：bash scripts/rl_progress.sh pilot_v2；汇总：logs/phase2_F_summary.txt
set -uo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"; mkdir -p logs
STATUS=logs/phase2_status.txt
mark() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
V2=$DATA/rl_data_v2; PILOT_STEPS="${PILOT_STEPS:-40}"
HF_CKPT="${SCALESEEK_MODEL_PATH:-$(find "$SFT_CKPT_DIR/v2" -type d -name huggingface 2>/dev/null | sort | tail -1 || true)}"
[ -n "$HF_CKPT" ] && [ -f "$HF_CKPT/config.json" ] || { mark "FATAL: no SFT v2 checkpoint"; exit 1; }
wait_gpus() {
  local need=$1 t=0
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l)
    [ "$free" -ge "$need" ] && return 0
    [ "$t" -eq 0 ] && mark "waiting for $need free GPUs (now $free free)"
    sleep 120; t=$((t+2)); [ "$t" -ge $(( ${GPU_WAIT_H:-24} * 60 )) ] && { mark "FATAL: GPUs busy for ${GPU_WAIT_H:-24}h"; exit 1; }
  done
}
wait_gpus 4
# 闸门阈值 0.6 而非 0.7：2026-09-06 19:54 新 SFT 在 16 题贪心上 11/16 作答（5 题跑满 8 轮），
# 差一题被拦。闸门防的是"全零奖励组"，69% 作答率离那很远；RL 里没作答 = 0 分，本身就是
# 推动作答的信号。
mark "[E] pilot_v2 start (steps=$PILOT_STEPS, model=$HF_CKPT)"
SCALESEEK_RL_CONFIG=grpo_trainer_v2 RL_RUN_NAME=pilot_v2 RL_TOTAL_STEPS="$PILOT_STEPS" RL_BATCH=48 \
RL_SAVE_FREQ=10 RL_TEST_FREQ=20 RL_VAL_BEFORE_TRAIN=true RL_MAX_ATTEMPTS=4 RL_MIN_FORMAT_PASS="${RL_MIN_FORMAT_PASS:-0.6}" \
SCALESEEK_MODEL_PATH="$HF_CKPT" SCALESEEK_TRAIN_FILES="$V2/train.jsonl" SCALESEEK_VAL_FILES="$V2/dev.jsonl" \
bash scripts/run_rl_local.sh > logs/phase2_E_pilot.log 2>&1; RC_E=$?
mark "[E] done rc=$RC_E ckpt=$(cat "$RL_CKPT_DIR/pilot_v2/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
{
  echo "== pilot_v2 progress"; bash scripts/rl_progress.sh pilot_v2 60
  echo; echo "== behaviour audit (last 3 train rollout dumps)"
  ls -t "$RL_CKPT_DIR"/pilot_v2/train_rollouts/*.jsonl 2>/dev/null | head -3 | xargs -r "$PY" scripts/analyze_rollouts.py --show 0
  for f in "$RL_CKPT_DIR"/pilot_v2/val_rollouts/*.jsonl; do [ -f "$f" ] || continue; echo; echo "== val $f"; "$PY" scripts/analyze_rollouts.py "$f" --show 0 | head -12; done
} > logs/phase2_F_summary.txt 2>&1
mark "pilot finished; summary in logs/phase2_F_summary.txt"
