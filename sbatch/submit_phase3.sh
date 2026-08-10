#!/usr/bin/env bash
# Phase-3 orchestrator for Rorqual (cluster-packing-and-decile-scope-Rorqual).
# Default DS=triviaqa — benchmark #2 in the TASK.md phase-3 dataset list.
#
# Rorqual has no 2-running/4-queued QOS cap, so the RALI lane machinery is
# replaced by one SLURM dependency DAG submitted up front. Every job is
# idempotent (skips work whose outputs already exist), so re-running this
# script after a failure only re-queues what is still missing.
#
# PREREQUISITES (login node, once):
#   1. venv built (see requirements.rorqual.txt / branch notes)
#   2. models + FlashRAG data prefetched into $HF_HOME (pinned revisions)
#   3. scripts/rorqual_login_setup.sh completed (official repos, uv envs, node)
#
# DAG:
#   p0_corpus_unzip ─┬─ p0_bm25_index ──┬─ p0_assets ── p1_accept ─┬─ 14 per-cell jobs
#                    ├─ p0_e5_index ────┘                          ├─ p2_grepseek
#                    ├─ p1_qwen3emb_index ──────────── (packs D,E) ├─ p2_dr_dci
#                    ├─ p2_agentir_encode ───────────────────────── p2_agentir
#                    ├─ p0_dci_corpus ───────────────────────────── p2_dci
#                    └─ p0_rise_articles ── p1_rise_toc ─────────── p2_rise
set -euo pipefail
cd "$(dirname "$0")/.."
export DS=${DS:-triviaqa}
export PHASE=${PHASE:-phase3}
mkdir -p logs "results/$PHASE"
# advance_lane must find empty queues (lane machinery retired on Rorqual).
: > sbatch/queue_laneA.txt; : > sbatch/queue_laneB.txt; : > sbatch/queue_laneD.txt

OFFICIAL_ROOT=${OFFICIAL_ROOT:-/scratch/a32du/official-baselines}
for r in dci_agent_lite dr_dci rise agentir grepseek; do
  [ -d "$OFFICIAL_ROOT/$r" ] || {
    echo "FATAL: $OFFICIAL_ROOT/$r missing — run scripts/rorqual_login_setup.sh first"; exit 1; }
done

EXP=ALL,DS=$DS,PHASE=$PHASE
jid() { sbatch "$@" | grep -oP 'Submitted batch job \K\d+'; }

echo "== wave 0: corpus =="
UNZIP=$(jid --export="$EXP" sbatch/p0_corpus_unzip.sbatch)

echo "== wave 1: indexes + corpora (all after corpus) =="
BM25=$(jid   --export="$EXP" --dependency=afterok:$UNZIP sbatch/p0_bm25_index.sbatch)
E5=$(jid     --export="$EXP" --dependency=afterok:$UNZIP sbatch/p0_e5_index.sbatch)
Q3E=$(jid    --export="$EXP" --dependency=afterok:$UNZIP sbatch/p1_qwen3emb_index.sbatch)
AIRENC=$(jid --export="$EXP" --dependency=afterok:$UNZIP sbatch/p2_agentir_encode.sbatch)
DCICORP=$(jid --export="$EXP" --dependency=afterok:$UNZIP sbatch/p0_dci_corpus.sbatch)
RISEART=$(jid --export="$EXP" --dependency=afterok:$UNZIP sbatch/p0_rise_articles.sbatch)

echo "== wave 2: dataset assets + acceptance gate =="
ASSETS=$(jid --export="$EXP" --dependency=afterok:$BM25:$E5 sbatch/p0_assets.sbatch)
ACCEPT=$(jid --export="$EXP" --dependency=afterok:$ASSETS sbatch/p1_accept.sbatch)
RISETOC=$(jid --export="$EXP" --dependency=afterok:$RISEART:$ASSETS sbatch/p1_rise_toc.sbatch)

echo "== wave 3: the $DS matrix (one small backfill-friendly job per cell) =="
# Rorqual queue reality: a whole-node 4-GPU 4-day job waits ~2 weeks in
# gpubase_bynode_b5, while 1-2 GPU 48h jobs backfill in hours-days. There is no
# job-count cap here, so each cell gets its own job; p2_pack still provides the
# GPU planning, skip-if-finished and requeue logic for a CELLS list of one.
cell() {  # cell <name> <agent> <ret> [sr1] [ngpu]
  local name=$1 agent=$2 ret=$3 sr1=${4:-} ngpu=${5:-1}
  local spec="$name:$agent:$ret"; [ -n "$sr1" ] && spec="$spec:$sr1"
  local dep=afterok:$ACCEPT
  [ "$ret" = qwen3_emb_4b ] && dep=$dep:$Q3E
  jid -J "p3_$name" --gres=gpu:h100:$ngpu -c $((12 * ngpu)) --mem=$((100 * ngpu))G \
    -t 48:00:00 --export="$EXP,CELLS=$spec" --dependency=$dep sbatch/p2_pack.sbatch
}
cell direct            direct    none
cell rag_bm25          rag       bm25
cell rag_e5            rag       e5
cell scaleseek         scaleseek bm25
cell search_o1_bm25    search_o1 bm25
cell search_o1_e5      search_o1 e5
cell search_r1_7b_bm25 search_r1 bm25 7b
cell search_r1_7b_e5   search_r1 e5   7b
cell search_r1_14b_bm25 search_r1 bm25 14b
cell search_r1_14b_e5   search_r1 e5   14b
cell rag_qwen3emb           rag       qwen3_emb_4b "" 2
cell search_o1_qwen3emb     search_o1 qwen3_emb_4b "" 2
cell search_r1_7b_qwen3emb  search_r1 qwen3_emb_4b 7b 2
cell search_r1_14b_qwen3emb search_r1 qwen3_emb_4b 14b 2
jid -J p3_grepseek --export="$EXP" --dependency=afterok:$ACCEPT sbatch/p2_grepseek.sbatch
jid -J p3_dr_dci  --export="$EXP" --dependency=afterok:$ACCEPT sbatch/p2_dr_dci.sbatch
jid -J p3_agentir --export="$EXP" --dependency=afterok:$ACCEPT:$AIRENC sbatch/p2_agentir.sbatch
jid -J p3_dci     --export="$EXP" --dependency=afterok:$ACCEPT:$DCICORP sbatch/p2_dci.sbatch
jid -J p3_rise    --export="$EXP" --dependency=afterok:$ACCEPT:$RISETOC sbatch/p2_rise.sbatch

squeue -u "$USER" -o "%.9i %.22j %.9T %.12r %.20E"
