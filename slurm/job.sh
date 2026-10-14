#!/bin/bash
# =====================================================================================
# Generic job: activate the environment and run ONE run.py command.
#     bash submit.sh <stage> ...      <-- use this; it supplies account and resources
# The #SBATCH lines are defaults that submit.sh overrides on the command line.
# =====================================================================================
#SBATCH --job-name=scaleseek
#SBATCH --account=def-CHANGEME
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source setup_env.sh            # the job starts in the frozen checkout (sbatch --chdir)
echo "job $SLURM_JOB_ID on $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-none}  $(date -Is)"
echo "harness $(git rev-parse HEAD) in $PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
export JAVA_TOOL_OPTIONS="-Xmx${SCALESEEK_JAVA_HEAP:-32g}"
srun --ntasks=1 python run.py "$@"
