#!/bin/bash
# =====================================================================================
# SFT warm start of the ScaleSeek agent (train/config/sft.yaml) on one 4xA100-40GB node.
#     bash submit.sh sft --name <run> --data <dir with train.parquet> [--model <hf dir>] [hydra overrides]
# Checkpoints (FSDP shards + an HF copy) go to $SCALESEEK_ROOT/train_runs/<run>; a
# resubmitted job resumes from the newest one.
# =====================================================================================
#SBATCH --job-name=ss_sft
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=24:00:00

set -euo pipefail
NAME="$1"; MODEL="$2"; DATA="$3"; shift 3
source setup_env.sh >/dev/null && deactivate   # SCALESEEK_ROOT and job caches; the job starts in the frozen checkout
bash setup_env.sh --job-train
R="$SCALESEEK_ROOT"; OUT="$R/train_runs/$NAME"; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname)  harness $(git rev-parse HEAD)  model $MODEL  data $DATA  $(date -Is)"
export VERL_DIR="$R/repos/grepseek/verl" SCALESEEK_REPO="$PWD"
export PYTHONPATH="$PWD/train:$PWD:$VERL_DIR"
export SCALESEEK_SFT_MODEL="$MODEL" SCALESEEK_SFT_DATA="$DATA" SCALESEEK_SFT_OUT="$OUT"
export TMPDIR="$SLURM_TMPDIR" TENSORBOARD_DIR="$OUT/tensorboard" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES               # Slurm sets it next to CUDA_VISIBLE_DEVICES; verl refuses both
"$SLURM_TMPDIR/scaleseek_env/train/bin/torchrun" --nnodes=1 --nproc_per_node=4 -m verl.trainer.sft_trainer \
    --config-path "$PWD/train/config" --config-name sft "$@"
