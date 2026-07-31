#!/usr/bin/env bash
# 阶段 2/3：冷启动 SFT 微调（verl sft_trainer）。
#
#   sbatch sbatch/sft_train.sbatch
#
# 覆盖项：
#   SCALESEEK_SFT_BASE    学生模型      [Qwen/Qwen3.5-9B]
#   SCALESEEK_SFT_TRAIN   训练 parquet  [$SFT_TRAJ_DIR/sft_train.parquet]
#   SCALESEEK_SFT_EPOCHS  轮数          [3]
#   SCALESEEK_SFT_BSZ     全局 batch    [64]
#   SCALESEEK_SFT_MAXLEN  最大序列长度  [4096]
#   SFT_GPUS              GPU 数        [4]
#
# 产物：$SFT_CKPT_DIR/<experiment>/global_step_*/huggingface  <- RL 的热启动点
#
# 为什么 4 卡：9B 全参微调的显存账是 权重 18G(bf16) + 梯度 18G + AdamW 状态
# 72G(fp32 两份动量) ≈ 110G，再加激活。4 张 H100（320G）用 FSDP 分片够用且有余量。
# 想更快可以调到 8 卡整节点（--gres=gpu:h100:8 --cpus-per-task=112 --mem=0）。
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_sft_train
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
require_full_gpus 4
trap cleanup_tmpdir EXIT

export SCALESEEK_SFT_BASE="${SCALESEEK_SFT_BASE:-Qwen/Qwen3.5-9B}"
export SCALESEEK_SFT_TRAIN="${SCALESEEK_SFT_TRAIN:-$SFT_TRAJ_DIR/sft_train.parquet}"
export SCALESEEK_SFT_VAL="${SCALESEEK_SFT_VAL:-$SCALESEEK_SFT_TRAIN}"
export SCALESEEK_SFT_OUTPUT="${SCALESEEK_SFT_OUTPUT:-$SFT_CKPT_DIR/$SLURM_JOB_ID}"
export SCALESEEK_SFT_EPOCHS="${SCALESEEK_SFT_EPOCHS:-3}"
export SCALESEEK_SFT_MAXLEN="${SCALESEEK_SFT_MAXLEN:-4096}"
export SCALESEEK_SFT_BSZ="${SCALESEEK_SFT_BSZ:-64}"
export NPROC=4
export PYBIN="$PY"

[ -f "$SCALESEEK_SFT_TRAIN" ] || { echo "FATAL: 训练 parquet 不存在: $SCALESEEK_SFT_TRAIN
  先跑: sbatch sbatch/sft_data.sbatch"; exit 1; }

mkdir -p "$SCALESEEK_SFT_OUTPUT"
echo "[sft_train] base=$SCALESEEK_SFT_BASE  out=$SCALESEEK_SFT_OUTPUT"

# 多卡 + H100 上可以打开 flash-attn 路径（旧集群 A5000 因显存与 sm_86 关掉了），
# 但 remove_padding 与 Qwen3 的 per-turn <think> 模板有已知冲突，保持关闭。
bash scripts/run_sft.sh \
    model.enable_gradient_checkpointing=true \
    engine.use_torch_compile=false \
    trainer.experiment_name="nibi_${SLURM_JOB_ID}" \
    trainer.save_freq=200 \
    "$@"

HF_CKPT=$(find "$SCALESEEK_SFT_OUTPUT" -type d -name huggingface | sort | tail -1)
echo "[sft_train] 完成。HF checkpoint: ${HF_CKPT:-<未找到>}"
echo "下一步: SCALESEEK_MODEL_PATH=$HF_CKPT sbatch sbatch/rl_train.sbatch"
