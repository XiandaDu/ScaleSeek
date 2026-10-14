#!/bin/bash
# =====================================================================================
# Job array over corpus shards: task K of N runs  run.py <cmd> ... --shard K/N
# N is the array size, or SCALESEEK_SHARDS when only some tasks are (re)run (submit.sh --tasks).
# =====================================================================================
#SBATCH --job-name=scaleseek-array
#SBATCH --account=def-CHANGEME
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
source setup_env.sh            # the job starts in the frozen checkout (sbatch --chdir)
N="${SCALESEEK_SHARDS:-${SLURM_ARRAY_TASK_COUNT:?not an array job}}"
K="${SLURM_ARRAY_TASK_ID}"
echo "array $SLURM_ARRAY_JOB_ID task $K/$N on $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-none}"
srun --ntasks=1 python run.py "$@" --shard "$K/$N"
