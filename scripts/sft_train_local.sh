#!/usr/bin/env bash
# 本机版 sbatch/sft_train.sbatch：4 卡 verl SFT，产出 HF checkpoint。
#   source setup_env.sh; bash scripts/sft_train_local.sh
# 覆盖项：SCALESEEK_SFT_TRAIN [$SFT_TRAJ_DIR/v2/sft_train.parquet]  SCALESEEK_SFT_OUTPUT [$SFT_CKPT_DIR/v2]
#         SCALESEEK_SFT_EPOCHS [3]  SCALESEEK_SFT_BSZ [32]  SCALESEEK_SFT_MAXLEN [8192]
set -euo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"
export SCALESEEK_SFT_BASE="${SCALESEEK_SFT_BASE:-Qwen/Qwen3.5-9B}"
export SCALESEEK_SFT_TRAIN="${SCALESEEK_SFT_TRAIN:-$SFT_TRAJ_DIR/v2/sft_train.parquet}"
export SCALESEEK_SFT_VAL="${SCALESEEK_SFT_VAL:-null}"
export SCALESEEK_SFT_OUTPUT="${SCALESEEK_SFT_OUTPUT:-$SFT_CKPT_DIR/v2}"
export SCALESEEK_SFT_EPOCHS="${SCALESEEK_SFT_EPOCHS:-3}"
export SCALESEEK_SFT_MAXLEN="${SCALESEEK_SFT_MAXLEN:-8192}"
export SCALESEEK_SFT_BSZ="${SCALESEEK_SFT_BSZ:-32}"
export NPROC="${NPROC:-4}" PYBIN="$PY" HF_HUB_OFFLINE=1
[ -f "$SCALESEEK_SFT_TRAIN" ] || { echo "FATAL: 训练 parquet 不存在: $SCALESEEK_SFT_TRAIN"; exit 1; }
mkdir -p "$SCALESEEK_SFT_OUTPUT" logs
# find 在目录为空时退出码非零；set -e + pipefail 会让整段静默死掉（2026-09-05 06:10 链断在这里）
existing=$(find "$SCALESEEK_SFT_OUTPUT" -type d -name huggingface 2>/dev/null | sort | tail -1 || true)
if [ -n "$existing" ] && [ -f "$existing/config.json" ]; then echo "[sft_train] 已有 checkpoint: $existing — 跳过"; echo "$existing"; exit 0; fi
mkdir -p "$SCALESEEK_SFT_OUTPUT" logs
echo "[sft_train] base=$SCALESEEK_SFT_BASE train=$SCALESEEK_SFT_TRAIN out=$SCALESEEK_SFT_OUTPUT gpus=$NPROC  $(date -u)"
bash scripts/run_sft.sh \
    model.enable_gradient_checkpointing=true engine.use_torch_compile=false \
    trainer.experiment_name="local_v2" trainer.save_freq=200 "$@" > logs/sft_train_v2.log 2>&1 \
  || { echo "FATAL: run_sft.sh 失败，见 logs/sft_train_v2.log"; grep -aE "Error|error" logs/sft_train_v2.log | tail -5; exit 1; }
HF_CKPT=$(find "$SCALESEEK_SFT_OUTPUT" -type d -name huggingface | sort | tail -1)
[ -n "$HF_CKPT" ] && [ -f "$HF_CKPT/config.json" ] || { echo "FATAL: 没找到 HF checkpoint"; exit 1; }
echo "[sft_train] 完成: $HF_CKPT  $(date -u)"; echo "$HF_CKPT"
