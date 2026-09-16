#!/usr/bin/env bash
# 本机（/data/xianda，4×A100-80G，无 Slurm）的 GRPO 启动器 —— sbatch/rl_train.sbatch 的
# 非 Slurm 版：同样的前置检查、看门狗和断路器，外加一个"崩了就 resume_mode=auto 续跑"的
# 外层循环，代替 chain_rl.sh 的 afterany 作业链。
#
#   source setup_env.sh
#   SCALESEEK_MODEL_PATH=<HF 格式权重> RL_RUN_NAME=grpo_9b_from190 \
#   RL_TOTAL_STEPS=310 bash scripts/run_rl_local.sh [额外 hydra 覆盖...]
#
# 覆盖项（与 rl_train.sbatch 同名）：
#   SCALESEEK_MODEL_PATH   热启动权重（HF 格式）    [必填；不填会从 base 冷启动并 FATAL]
#   RL_RUN_NAME            run 名 = 输出目录名      [grpo_9b]
#   RL_GPUS                GPU 数                   [4]
#   RL_BATCH               每步题数                 [64]（× rollout.n=5 = 320 条轨迹）
#   RL_TOTAL_STEPS         本 run 的总步数          [500]
#   RL_SAVE_FREQ / RL_TEST_FREQ                     [10 / 40]
#   RL_VAL_BEFORE_TRAIN    开跑前先验一次 dev       [false]
#   SCALESEEK_TRAIN_FILES / SCALESEEK_VAL_FILES     [$RL_DATA_DIR/train.jsonl / dev.jsonl]
#   ULYSSES_SP / RL_ROLLOUT_TP                      [2 / 2]，都必须整除 RL_GPUS
#   RL_SKIP_FORMAT_GATE    跳过格式闸门             [0]
#   RL_MAX_ATTEMPTS        崩溃后自动续跑的最大次数 [6]
#   RL_STALL_MIN / RL_HARD_MAX                      看门狗阈值（分钟）[100 / 210]
#
# 产物：$RL_CKPT_DIR/$RL_RUN_NAME/global_step_*/ ；日志 logs/rl_<run>_attempt<k>.log
set -euo pipefail

: "${REPO:?先 source setup_env.sh}"
: "${RL_CKPT_DIR:?先 source setup_env.sh}"
cd "$REPO"
[ "${SCALESEEK_SITE:-}" = "local" ] || echo "warn: SCALESEEK_SITE=${SCALESEEK_SITE:-?}，本脚本是为 local 站点写的"

export NPROC="${RL_GPUS:-4}"
have=$(nvidia-smi -L 2>/dev/null | grep -c .)
[ "$have" -ge "$NPROC" ] || { echo "FATAL: 需要 $NPROC 张 GPU，实际只有 $have"; exit 1; }
# 共享机器：别把别人的作业挤爆
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
[ "$busy" -eq 0 ] || echo "warn: 有 $busy 张 GPU 已被占用 >2G（nvidia-smi 看看是谁）"

export ULYSSES_SP="${ULYSSES_SP:-2}"
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-2}"
[ $(( NPROC % ULYSSES_SP )) -eq 0 ] || { echo "FATAL: ULYSSES_SP=$ULYSSES_SP 必须整除 NPROC=$NPROC"; exit 1; }
[ $(( NPROC % RL_ROLLOUT_TP )) -eq 0 ] || { echo "FATAL: RL_ROLLOUT_TP=$RL_ROLLOUT_TP 必须整除 NPROC=$NPROC"; exit 1; }

export RL_RUN_NAME="${RL_RUN_NAME:-grpo_9b}"
export SCALESEEK_OUTPUT_DIR="${SCALESEEK_OUTPUT_DIR:-$RL_CKPT_DIR/$RL_RUN_NAME}"
export SCALESEEK_TRAIN_FILES="${SCALESEEK_TRAIN_FILES:-$RL_DATA_DIR/train.jsonl}"
export SCALESEEK_VAL_FILES="${SCALESEEK_VAL_FILES:-$RL_DATA_DIR/dev.jsonl}"

# --- 前置检查 ---
[ -d "$GREPSEEK_VERL_DIR/verl/trainer" ] || { echo "FATAL: verl 不存在: $GREPSEEK_VERL_DIR"; exit 1; }
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: BM25 索引不存在: $BM25_INDEX_DIR/index"; exit 1; }
[ -f "$SCALESEEK_TRAIN_FILES" ] || { echo "FATAL: RL 训练集不存在: $SCALESEEK_TRAIN_FILES"; exit 1; }
[ -f "$SCALESEEK_VAL_FILES" ] || { echo "FATAL: RL 验证集不存在: $SCALESEEK_VAL_FILES"; exit 1; }
if [ -z "${SCALESEEK_MODEL_PATH:-}" ]; then
  echo "FATAL: SCALESEEK_MODEL_PATH 未设 —— 会从 base 模型冷启动。续跑请指向 step-190 权重："
  echo "       export SCALESEEK_MODEL_PATH=\$(\$PY -c 'from huggingface_hub import snapshot_download as s;print(s(\"$RL_STEP190_HF_MODEL\"))')"
  exit 1
fi
[ -f "$SCALESEEK_MODEL_PATH/config.json" ] || { echo "FATAL: $SCALESEEK_MODEL_PATH 下没有 config.json"; exit 1; }
"$PY" -c "import flash_attn" 2>/dev/null || { echo "FATAL: flash-attn 未装（bash scripts/setup_venv_local.sh）"; exit 1; }

mkdir -p "$SCALESEEK_OUTPUT_DIR" "$REPO/logs"
echo "[rl] model=$SCALESEEK_MODEL_PATH"
echo "[rl] train=$SCALESEEK_TRAIN_FILES ($(wc -l < "$SCALESEEK_TRAIN_FILES") 行)  val=$SCALESEEK_VAL_FILES"
echo "[rl] out=$SCALESEEK_OUTPUT_DIR  gpus=$NPROC ulysses_sp=$ULYSSES_SP rollout_tp=$RL_ROLLOUT_TP"
echo "[rl] batch=${RL_BATCH:-64} total_steps=${RL_TOTAL_STEPS:-500} save=${RL_SAVE_FREQ:-10} test=${RL_TEST_FREQ:-40}"

# --- 格式闸门（与 rl_train.sbatch 相同）---
if [ "${RL_SKIP_FORMAT_GATE:-0}" != "1" ]; then
  echo "[rl] 前置检查：格式遵守率（阈值 ${RL_MIN_FORMAT_PASS:-0.7}，${RL_GATE_N:-16} 题）"
  if ! CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_scaleseek_smoke_eval.py \
        --model "$SCALESEEK_MODEL_PATH" --index-dir "$BM25_INDEX_DIR" \
        --questions "$SCALESEEK_VAL_FILES" --limit "${RL_GATE_N:-16}" \
        --min-format-pass "${RL_MIN_FORMAT_PASS:-0.7}" \
        --out "$SCALESEEK_OUTPUT_DIR/format_gate.jsonl"; then
    echo "FATAL: 格式闸门未通过。确认要无视时用 RL_SKIP_FORMAT_GATE=1 重投。"; exit 1
  fi
fi

export HF_HUB_OFFLINE=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PYSPY="$SCALESEEK_VENV/bin/py-spy"
STALL_MIN="${RL_STALL_MIN:-100}"
HARD_MAX="${RL_HARD_MAX:-210}"
MAX_ATTEMPTS="${RL_MAX_ATTEMPTS:-6}"

latest_step() { cat "$SCALESEEK_OUTPUT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

attempt=0; last_seen=$(latest_step); no_progress=0
while :; do
  attempt=$((attempt+1))
  RL_LOG="$REPO/logs/rl_${RL_RUN_NAME}_attempt${attempt}.log"
  echo "[rl] ===== attempt $attempt/$MAX_ATTEMPTS  从 step $(latest_step) 起  log=$RL_LOG  $(date) ====="

  bash scripts/run_rl.sh \
      actor_rollout_ref.rollout.tensor_model_parallel_size="$RL_ROLLOUT_TP" \
      data.dataloader_num_workers="${RL_WORKERS:-8}" \
      data.train_batch_size="${RL_BATCH:-64}" \
      trainer.total_training_steps="${RL_TOTAL_STEPS:-500}" \
      trainer.save_freq="${RL_SAVE_FREQ:-10}" \
      trainer.test_freq="${RL_TEST_FREQ:-40}" \
      trainer.val_before_train="${RL_VAL_BEFORE_TRAIN:-false}" \
      trainer.resume_mode=auto \
      trainer.experiment_name="${RL_RUN_NAME}" \
      "$@" > "$RL_LOG" 2>&1 &
  MAIN_PID=$!

  # ── 看门狗 / 断路器（逻辑同 rl_train.sbatch；这个子 shell 里必须关 errexit）──
  (
    set +e
    t=0; idle=0; base_step=$(latest_step)
    while kill -0 "$MAIN_PID" 2>/dev/null; do
      sleep 900; t=$((t+15))
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
      steps=$(grep -ac "step:" "$RL_LOG" 2>/dev/null)
      echo "[watchdog] +${t}min gpu_util=[${util}] steps_logged=$steps ckpt=$(latest_step)"
      if [ -x "$PYSPY" ] && [ "$steps" -eq 0 ]; then
        for p in $(pgrep -f "ray::TaskRunner|verl.trainer.main_ppo" 2>/dev/null | head -1); do
          echo "[watchdog] py-spy dump pid=$p:"; "$PYSPY" dump --pid "$p" 2>&1 | head -25
        done
      fi
      mean_util=$(echo "$util" | tr ',' '\n' | awk 'NF{s+=$1;n++} END{print (n?int(s/n):0)}')
      if [ "${mean_util:-0}" -lt 10 ]; then idle=$((idle+1)); else idle=0; fi
      if [ "$steps" -eq 0 ]; then
        if [ "$t" -ge "$STALL_MIN" ] && [ "$idle" -ge 3 ]; then
          echo "[watchdog] FATAL: ${STALL_MIN} 分钟无训练步且 GPU 连续闲置 —— 判定挂死"
          kill -TERM "$MAIN_PID" 2>/dev/null; sleep 60; kill -KILL "$MAIN_PID" 2>/dev/null; break
        fi
        if [ "$t" -ge "$HARD_MAX" ] && [ "${mean_util:-0}" -lt 50 ]; then
          echo "[watchdog] FATAL: ${HARD_MAX} 分钟仍无训练步且 GPU 利用率仅 ${mean_util}% —— 硬兜底终止"
          kill -TERM "$MAIN_PID" 2>/dev/null; sleep 60; kill -KILL "$MAIN_PID" 2>/dev/null; break
        fi
      fi
    done
  ) > >(tee -a "$REPO/logs/rl_${RL_RUN_NAME}_watchdog.log") 2>&1 &
  WATCHDOG_PID=$!   # 进程替换而非管道：管道时 $! 是 tee 的 pid，看门狗子 shell 杀不掉

  RC=0; wait "$MAIN_PID" || RC=$?
  kill "$WATCHDOG_PID" 2>/dev/null || true
  # Ray 偶尔留下孤儿 worker，占着显存会让下一次 attempt 起不来
  "$PY" -c "import ray; ray.shutdown()" >/dev/null 2>&1 || true
  pkill -f "^ray::" 2>/dev/null || true; sleep 10

  now=$(latest_step)
  if [ "$RC" -eq 0 ]; then
    echo "[rl] 完成（exit 0），最终 checkpoint step=$now  $(date)"; break
  fi
  echo "[rl] attempt $attempt 退出码 $RC，checkpoint step $last_seen -> $now  $(date)"
  if [ "$now" -gt "$last_seen" ]; then no_progress=0; else no_progress=$((no_progress+1)); fi
  last_seen=$now
  if [ "$no_progress" -ge 2 ]; then
    echo "FATAL: 连续 $no_progress 次 attempt 没有推进 checkpoint —— 不再自动重试，看 $RL_LOG"; exit "$RC"
  fi
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "FATAL: 达到 RL_MAX_ATTEMPTS=$MAX_ATTEMPTS"; exit "$RC"
  fi
done

echo "[rl] checkpoints -> $SCALESEEK_OUTPUT_DIR"
echo "     合并: RL_RUN=$RL_RUN_NAME bash scripts/merge_rl_ckpt.sh $(latest_step)"
