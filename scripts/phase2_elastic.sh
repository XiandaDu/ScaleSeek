#!/usr/bin/env bash
# 弹性放量：有几张空卡用几张（2 或 4），2 卡跑着时若另外两张空出来（连续 3 次、每 5 min 一次
# 检查都 <2G），就优雅停掉、合并最新 checkpoint、换 4 卡续跑。
#
# 为什么要合并而不是 resume：verl 的 FSDP 分片按 world size 存
# （model_world_size_N_rank_*.pt），换卡数加载不了。每次切换 = 合并成 HF 权重 → 新 run 目录
# 热启动：优化器状态重置，KL 参考变成切换时的权重（周期性 ref reset），约 15 min 重启开销。
# 总步数按各段 checkpoint 累加，run 名带起始步：v2_<n>gpu_from<step>。
#
#   source setup_env.sh; setsid nohup bash scripts/phase2_elastic.sh > logs/phase2_elastic.log 2>&1 < /dev/null &
#   停：bash scripts/gpu_pause.sh（它会杀 phase2_* 链和训练进程）
set -uo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"; mkdir -p logs
STATUS=logs/phase2_status.txt
mark() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
V2=$DATA/rl_data_v2
TRAIN_FILE="${TRAIN_OVERRIDE:-$V2/train.jsonl}"      # v3 用 pool.jsonl（未筛的 50k 多跳池）
FULL_STEPS="${FULL_STEPS:-300}"
RL_CONFIG="${RL_CONFIG:-grpo_trainer_v2}"          # v2.1 起用 grpo_trainer_v2_1（KL 锚定 SFT v2，覆盖率系数 0.5）
export SCALESEEK_REF_PATH="${SCALESEEK_REF_PATH:-$(find "$SFT_CKPT_DIR/v2" -type d -name huggingface | sort | tail -1)}"
CUR_RUN="${CUR_RUN:-pilot_v2}"
STEP=$(cat "$RL_CKPT_DIR/$CUR_RUN/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
OVERALL="${OVERALL_START:-$STEP}"
mark "[elastic] start: run=$CUR_RUN local_step=$STEP overall=$OVERALL target=$FULL_STEPS config=$RL_CONFIG ref=$SCALESEEK_REF_PATH"

free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}' | paste -sd, -; }
n_of() { [ -z "$1" ] && echo 0 || echo "$1" | tr ',' '\n' | grep -c .; }
stop_run() {  # 优雅停：先杀 launcher（免得它 resume），再 TERM 训练进程，等 checkpoint 写完
  pkill -f "^bash scripts/run_rl_local.sh" 2>/dev/null; pkill -f "^bash scripts/run_rl.sh" 2>/dev/null
  pkill -TERM -f "verl.trainer.main_ppo" 2>/dev/null; sleep 60
  pkill -KILL -f "verl.trainer.main_ppo" 2>/dev/null; pkill -f "^ray::" 2>/dev/null; sleep 20
  pkill -KILL -f "^ray::|raylet|gcs_server" 2>/dev/null; sleep 10
}

while [ "$OVERALL" -lt "$FULL_STEPS" ]; do
  # ── pick GPUs ──
  while :; do
    F=$(free_gpus); N=$(n_of "$F")
    if [ "$N" -ge 4 ]; then MODE=4; GPUS=$(echo "$F" | cut -d, -f1-4); break
    elif [ "$N" -ge 2 ]; then MODE=2; GPUS=$(echo "$F" | cut -d, -f1-2); break
    else [ "${waited:-0}" -eq 0 ] && mark "[elastic] waiting for >=2 free GPUs (free: '$F')"; waited=1; sleep 120; fi
  done
  waited=0
  # ── merge the latest checkpoint of CUR_RUN -> HF weights for the warm start ──
  LOCAL=$(cat "$RL_CKPT_DIR/$CUR_RUN/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ "$CUR_RUN" = "__none__" ] && LOCAL=0
  if [ "$CUR_RUN" = "__none__" ]; then
    MODEL="${SCALESEEK_V3_MODEL:?SCALESEEK_V3_MODEL 未设}"   # 冷启动：直接用 SFT checkpoint
  elif [ "$LOCAL" -gt 0 ]; then
    MODEL=$(RL_RUN="$CUR_RUN" bash scripts/merge_rl_ckpt.sh "$LOCAL" 2>>logs/phase2_elastic_merge.log) || { mark "FATAL: merge of $CUR_RUN step $LOCAL failed"; exit 1; }
  else
    MODEL=$(find "$SFT_CKPT_DIR/v2" -type d -name huggingface | sort | tail -1)
  fi
  NEW_RUN="v2_${MODE}gpu_from${OVERALL}"
  # 2 卡时每卡要放下一半的 FSDP 分片，20480-token 的 micro-batch 在 update_actor 里 OOM
  # （2026-09-15 16:07，v2_2gpu_from40 第一步）：token 预算减半，vLLM 显存占比也降一点。
  EXTRA=""
  if [ "$MODE" -eq 2 ]; then
    EXTRA="actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
  fi
  REMAIN=$(( FULL_STEPS - OVERALL ))
  mark "[elastic] launching $NEW_RUN on GPUs $GPUS ($MODE) from $MODEL, $REMAIN steps to go"
  CUDA_VISIBLE_DEVICES="$GPUS" SCALESEEK_RL_CONFIG="$RL_CONFIG" RL_RUN_NAME="$NEW_RUN" RL_GPUS="$MODE" NPROC="$MODE" \
  ULYSSES_SP="${ULYSSES_SP:-1}" RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-1}" \
  RL_TOTAL_STEPS="$REMAIN" RL_BATCH=48 RL_SAVE_FREQ=5 RL_TEST_FREQ=40 RL_VAL_BEFORE_TRAIN=false RL_MAX_ATTEMPTS=6 \
  RL_SKIP_FORMAT_GATE=1 SCALESEEK_MODEL_PATH="$MODEL" SCALESEEK_TRAIN_FILES="$TRAIN_FILE" SCALESEEK_VAL_FILES="$V2/dev.jsonl" \
  bash scripts/run_rl_local.sh trainer.total_epochs=10 $EXTRA > "logs/phase2_elastic_${NEW_RUN}.log" 2>&1 &
  LPID=$!
  # ── watch: finished, or (2-GPU mode) the other pair became free for 3 checks ──
  idle_other=0
  while kill -0 "$LPID" 2>/dev/null; do
    sleep 300
    if [ "$MODE" -eq 2 ]; then
      OTHER=$(free_gpus); NO=$(n_of "$OTHER")
      if [ "$NO" -ge 2 ]; then idle_other=$((idle_other+1)); else idle_other=0; fi
      if [ "$idle_other" -ge 3 ]; then
        mark "[elastic] other GPUs ($OTHER) free for 15 min -> switching to 4 GPUs"
        stop_run; break
      fi
    fi
  done
  wait "$LPID" 2>/dev/null; RC=$?
  DONE=$(cat "$RL_CKPT_DIR/$NEW_RUN/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  OVERALL=$(( OVERALL + DONE )); CUR_RUN="$NEW_RUN"
  mark "[elastic] $NEW_RUN ended rc=$RC local_steps=$DONE overall=$OVERALL"
  [ "$DONE" -eq 0 ] && { mark "FATAL: $NEW_RUN made no checkpoint — stopping (see logs/phase2_elastic_${NEW_RUN}.log)"; exit 1; }
done
mark "[elastic] reached $OVERALL steps; final run $CUR_RUN"
{
  for r in pilot_v2 $(ls -d "$RL_CKPT_DIR"/v2_*gpu_from* 2>/dev/null | xargs -n1 basename); do
    for f in "$RL_CKPT_DIR"/$r/val_rollouts/*.jsonl; do [ -f "$f" ] || continue; echo "== $r $(basename $f)"; "$PY" scripts/analyze_rollouts.py "$f" --show 0 | grep -E "EM overall|EM by source|tool calls per"; done
  done
} > logs/phase2_full_summary.txt 2>&1
mark "[elastic] summary in logs/phase2_full_summary.txt"
