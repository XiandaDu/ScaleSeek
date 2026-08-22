#!/usr/bin/env bash
#SBATCH -J p2_agentir_reshard
#SBATCH -p rali
#SBATCH -w ilar03
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# CPU-only re-shard of the AgentIR pkls (see scripts/reshard_agentir_pkls.py:
# the 4x51G layout's load peak of ~255G cannot fit a 257G node, and both 500G
# nodes are unavailable -- octal40 silently kills every job, octal41 is
# admin-drained). Peak here is one 51G shard + one ~13G piece; ilar03's 128G
# CPU box carries it without occupying a GPU node. Chains p2_agentir on
# success, exactly like p2_agentir_encode did.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; cleanup_tmpdir; exit $code' EXIT

"$PY" scripts/reshard_agentir_pkls.py \
  --in-dir "$DATA/agentir_official_index" \
  --out-dir "$DATA/agentir_official_index/resharded" \
  --pieces-per-shard 4
N=$(ls "$DATA/agentir_official_index/resharded"/corpus.reshard.*.pkl | wc -l)
[ "$N" -eq 16 ] || { echo "FATAL: expected 16 resharded pieces, found $N"; exit 1; }
echo "ALL_P2_AGENTIR_RESHARD_DONE"
submit_retry "$REPO/sbatch/p2_agentir.sbatch"
