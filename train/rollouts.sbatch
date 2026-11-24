#!/bin/bash
# =====================================================================================
# Our agent on train questions (the SFT data) or validation questions (train/rollouts.py).
#     bash submit.sh rollouts --name <run> [--dev] [--model M --revision R --tp T] [--nq N --hotpotqa N --samples K]
# Output: $SCALESEEK_ROOT/train_data/rollouts/<run>; a resubmitted job resumes (records are
# appended one by one).
# =====================================================================================
#SBATCH --job-name=ss_rollouts
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=24:00:00

set -euo pipefail
NAME="$1"; shift
source setup_env.sh                       # the job starts in the frozen checkout
R="$SCALESEEK_ROOT"
echo "job $SLURM_JOB_ID on $(hostname)  harness $(git rev-parse HEAD)  $(date -Is)"
# what the find service reads at random (see train/rl.sbatch)
cat "$R/data/corpora/wiki18/corpus.jsonl" "$R/indexes/wiki18/exact/lucene/"* > /dev/null
echo "page cache warmed $(date -Is)"
export JAVA_TOOL_OPTIONS=-Xmx32g
python train/rollouts.py "$R/train_data/rollouts/$NAME" --exclude "$R/train_data/rl_v1/dev.parquet" "$@"
