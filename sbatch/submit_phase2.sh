#!/usr/bin/env bash
# Bring up the Phase-1 acceptance + Phase-2 chain under the 4-job QOS cap.
#
# The cluster QOS allows at most 4 queued/running jobs per user, so the matrix
# runs as self-advancing lanes: every job pops its successor from
# sbatch/queue_<lane>.txt when it ends (advance_lane in common.sh), and gate
# jobs submit the lane heads when they pass:
#   p0_assets -> p1_accept -> [laneA head, laneB head, p1_qwen3emb_index]
#   p1_qwen3emb_index -> laneD head
#   laneA tail (agentir encode) -> p2_agentir
# dr_dci and rise are wired interactively before joining a lane.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/phase2

cat > sbatch/queue_laneA.txt <<'EOF'
-J p2_rag_bm25 --export=ALL,AGENT=rag,RET=bm25,LANE=laneA sbatch/p2_local.sbatch
-J p2_rag_e5 --export=ALL,AGENT=rag,RET=e5,LANE=laneA sbatch/p2_local.sbatch
-J p2_scaleseek --export=ALL,AGENT=scaleseek,RET=bm25,LANE=laneA sbatch/p2_local.sbatch
-J p2_search_o1_bm25 --export=ALL,AGENT=search_o1,RET=bm25,LANE=laneA sbatch/p2_local.sbatch
-J p2_search_o1_e5 --export=ALL,AGENT=search_o1,RET=e5,LANE=laneA sbatch/p2_local.sbatch
-J p2_grepseek --export=ALL,LANE=laneA sbatch/p2_grepseek.sbatch
-J p2_dci --export=ALL,LANE=laneA sbatch/p2_dci.sbatch
-J p2_agentir_encode --export=ALL sbatch/p2_agentir_encode.sbatch
EOF

cat > sbatch/queue_laneB.txt <<'EOF'
-J p2_search_r1_7b_e5 --export=ALL,AGENT=search_r1,SR1=7b,RET=e5,LANE=laneB sbatch/p2_local.sbatch
-J p2_search_r1_14b_bm25 --export=ALL,AGENT=search_r1,SR1=14b,RET=bm25,LANE=laneB sbatch/p2_local.sbatch
-J p2_search_r1_14b_e5 --export=ALL,AGENT=search_r1,SR1=14b,RET=e5,LANE=laneB sbatch/p2_local.sbatch
EOF

cat > sbatch/queue_laneD.txt <<'EOF'
-J p2_search_o1_qwen3emb --gres=gpu:3 --mem=110G --export=ALL,AGENT=search_o1,RET=qwen3_emb_4b,LANE=laneD sbatch/p2_local.sbatch
-J p2_search_r1_7b_qwen3emb --gres=gpu:3 --mem=110G --export=ALL,AGENT=search_r1,SR1=7b,RET=qwen3_emb_4b,LANE=laneD sbatch/p2_local.sbatch
-J p2_search_r1_14b_qwen3emb --gres=gpu:3 --mem=110G --export=ALL,AGENT=search_r1,SR1=14b,RET=qwen3_emb_4b,LANE=laneD sbatch/p2_local.sbatch
EOF
echo "lane queue files written"

# (Re)submit the acceptance gate against the current p1_accept.sbatch.
PEND=$(squeue -h -u "$USER" -n p1_accept -o %i || true)
if [ -n "$PEND" ]; then
  echo "cancelling stale p1_accept $PEND (spooled copy lacks the lane epilogue)"
  scancel "$PEND"
fi
ASSETS=$(squeue -h -u "$USER" -n p0_assets -o %i | head -1 || true)
if [ -n "$ASSETS" ]; then
  sbatch --dependency=afterok:"$ASSETS" sbatch/p1_accept.sbatch
else
  sbatch sbatch/p1_accept.sbatch
fi
squeue -u "$USER" -o "%.9i %.22j %.9T %.12r %.20E"
