#!/bin/bash
# =====================================================================================
# GRPO of the ScaleSeek agent on one 4xA100-40GB node (train/config/grpo.yaml).
#     bash submit.sh rl --name <run> [--model <hf dir>] [hydra overrides...]
# The retrieval services run in the main env on this node (find = exact index on CPU,
# search = e5 on GPU 0 next to training); verl runs in the training env.  Checkpoints and
# rollouts go to $SCALESEEK_ROOT/train_runs/<run>; a resubmitted job resumes from the
# newest checkpoint there (verl resume_mode=auto).
# =====================================================================================
#SBATCH --job-name=ss_rl
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=24:00:00

set -euo pipefail
NAME="$1"; MODEL="$2"; shift 2
source setup_env.sh                       # main env (retrieval services); job starts in the frozen checkout
R="$SCALESEEK_ROOT"; OUT="$R/train_runs/$NAME"; mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname)  harness $(git rev-parse HEAD)  model $MODEL  $(date -Is)"

export JAVA_TOOL_OPTIONS=-Xmx32g
python run.py serve-retriever --retriever exact --corpus wiki18 --port 8101 --device cpu > "$OUT/find_service.$SLURM_JOB_ID.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 python run.py serve-retriever --retriever e5 --corpus wiki18 --port 8102 --device cuda:0 \
    > "$OUT/search_service.$SLURM_JOB_ID.log" 2>&1 &
for port in 8101 8102; do
  timeout 1800 bash -c "until curl -sf localhost:$port/health >/dev/null; do sleep 10; done"
done
echo "retrieval services up $(date -Is)"

deactivate                                # the services keep the main env; verl runs in the training env
bash setup_env.sh --job-train
export VERL_DIR="$R/repos/grepseek/verl" SCALESEEK_REPO="$PWD"
export PYTHONPATH="$PWD/train:$PWD:$VERL_DIR"
export SCALESEEK_FIND_URL=http://127.0.0.1:8101 SCALESEEK_SEARCH_URL=http://127.0.0.1:8102
export SCALESEEK_RL_MODEL="$MODEL" SCALESEEK_RL_DATA="$R/train_data/rl_v1" SCALESEEK_RL_OUT="$OUT"
export TMPDIR="$SLURM_TMPDIR" RAY_TMPDIR="$SLURM_TMPDIR/ray" TENSORBOARD_DIR="$OUT/tensorboard"
unset ROCR_VISIBLE_DEVICES               # Slurm sets it next to CUDA_VISIBLE_DEVICES; verl refuses both
"$SLURM_TMPDIR/scaleseek_env/train/bin/python" -m verl.trainer.main_ppo \
    --config-path "$PWD/train/config" --config-name grpo "$@"
