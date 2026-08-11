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
DATA_ROOT=/scratch/a32du/data
mkdir -p logs "results/$PHASE"
# advance_lane must find empty queues (lane machinery retired on Rorqual).
: > sbatch/queue_laneA.txt; : > sbatch/queue_laneB.txt; : > sbatch/queue_laneD.txt

OFFICIAL_ROOT=${OFFICIAL_ROOT:-/scratch/a32du/official-baselines}
for r in dci_agent_lite dr_dci rise agentir grepseek; do
  [ -d "$OFFICIAL_ROOT/$r" ] || {
    echo "FATAL: $OFFICIAL_ROOT/$r missing — run scripts/rorqual_login_setup.sh first"; exit 1; }
done

EXP=ALL,DS=$DS,PHASE=$PHASE
# Idempotent by JOB NAME: if a job of this name is already pending/running,
# reuse its id for downstream dependencies instead of double-submitting (a
# second copy of a resumable index build would fight the first one's
# checkpoints). Failed/cancelled jobs are not in PD/R, so re-running this
# script resubmits exactly the broken part of the DAG.
sub() {  # sub <job-name> <sbatch args...>
  local name=$1; shift
  local existing
  existing=$(squeue -h -u "$USER" -n "$name" -t PD,R -o %i 2>/dev/null | head -1)
  if [ -n "$existing" ]; then
    echo "reuse   $name -> $existing" >&2
    echo "$existing"; return 0
  fi
  local id
  id=$(sbatch -J "$name" "$@" | grep -oP 'Submitted batch job \K\d+')
  echo "submit  $name -> $id" >&2
  echo "$id"
}

# Stages whose outputs already exist need no job at all; represent them with
# dependency-free "done" and prune afterok terms accordingly.
dep() {  # dep <id-or-done>... -> "--dependency=afterok:..." or nothing
  local ids=""
  for x in "$@"; do [ "$x" != done ] && ids="$ids:$x"; done
  [ -n "$ids" ] && echo "--dependency=afterok:${ids#:}"
}

echo "== wave 0: corpus =="
if [ -f "$DATA_ROOT/wiki_18_corpus/corpus_manifest.json" ]; then UNZIP=done
else UNZIP=$(sub p0_corpus_unzip --export="$EXP" sbatch/p0_corpus_unzip.sbatch); fi

echo "== wave 1: indexes + corpora (all after corpus) =="
if [ -f "$DATA_ROOT/bm25_index/index_manifest.json" ]; then BM25=done
else BM25=$(sub p0_bm25_index --export="$EXP" $(dep $UNZIP) sbatch/p0_bm25_index.sbatch); fi
if [ -f "$DATA_ROOT/e5_index/index_manifest.json" ]; then E5=done
else E5=$(sub p0_e5_index --export="$EXP" $(dep $UNZIP) sbatch/p0_e5_index.sbatch); fi
if [ -f "$DATA_ROOT/qwen3_embedding_4b_index/index_manifest.json" ]; then Q3E=done
else Q3E=$(sub p1_qwen3emb_index --export="$EXP" $(dep $UNZIP) sbatch/p1_qwen3emb_index.sbatch); fi
if ls "$DATA_ROOT"/agentir_official_index/corpus.*.pkl >/dev/null 2>&1; then AIRENC=done
else AIRENC=$(sub p2_agentir_encode --export="$EXP" $(dep $UNZIP) sbatch/p2_agentir_encode.sbatch); fi
if [ -f "$DATA_ROOT/dci_wiki_corpus.tar.zst" ]; then DCICORP=done
else DCICORP=$(sub p0_dci_corpus --export="$EXP" $(dep $UNZIP) sbatch/p0_dci_corpus.sbatch); fi
if [ -f "$DATA_ROOT/rise_wiki_articles.tar.zst" ]; then RISEART=done
else RISEART=$(sub p0_rise_articles --export="$EXP" $(dep $UNZIP) sbatch/p0_rise_articles.sbatch); fi

echo "== wave 2: dataset assets + acceptance gate =="
if [ -f "results/$PHASE/.accept_passed_${DS}" ]; then
  echo "acceptance already passed for $DS; skipping assets+accept resubmission" >&2
  ASSETS=done; ACCEPT=done
else
  ASSETS=$(sub p0_assets --export="$EXP" $(dep $BM25 $E5) sbatch/p0_assets.sbatch)
  ACCEPT=$(sub p1_accept --export="$EXP" $(dep $ASSETS) sbatch/p1_accept.sbatch)
fi
# RISE TOC is sharded 4x (~0.07 docs/s per 2-GPU server; 98,582 candidates
# would be ~16 days serially). Shards share the persistent TOC_DIR and skip
# already-structured articles, so re-running any of them is safe.
# 8 shards x 60h instead of 4 x 7d: the 7-day requests sat unscheduled for a
# day in the small b5 bucket; 60h lands in b4 (twice the nodes, backfills).
TOCDEPS=""
for i in 0 1 2 3 4 5 6 7; do
  T=$(sub p1_rise_toc$i --export="$EXP,CAND_SHARD=$i,CAND_NSHARDS=8,TOC_WORKERS=24" \
      -t 60:00:00 $(dep $RISEART $ASSETS) sbatch/p1_rise_toc.sbatch)
  TOCDEPS="$TOCDEPS $T"
done
RISETOC=$TOCDEPS

echo "== wave 3: the $DS matrix (one small backfill-friendly job per cell) =="
# Rorqual queue reality: a whole-node 4-GPU 4-day job waits ~2 weeks in
# gpubase_bynode_b5, while 1-2 GPU 48h jobs backfill in hours-days. There is no
# job-count cap here, so each cell gets its own job; p2_pack still provides the
# GPU planning, skip-if-finished and requeue logic for a CELLS list of one.
cell() {  # cell <name> <agent> <ret> [sr1] [ngpu]
  local name=$1 agent=$2 ret=$3 sr1=${4:-} ngpu=${5:-1}
  # Finished rows need no job at all (the pack would only start, see the
  # metrics file and exit — wasting a queue slot each babysitter cycle).
  [ -f "results/$PHASE/${DS}_${name}.metrics.json" ] && {
    echo "done    p3_$name (metrics exist)" >&2; return 0; }
  local spec="$name:$agent:$ret"; [ -n "$sr1" ] && spec="$spec:$sr1"
  local deps="$ACCEPT"
  [ "$ret" = qwen3_emb_4b ] && deps="$deps $Q3E"
  # shellcheck disable=SC2046
  sub "p3_$name" --gres=gpu:h100:$ngpu -c $((12 * ngpu)) --mem=$((100 * ngpu))G \
    -t 48:00:00 --export="$EXP,CELLS=$spec" $(dep $deps) sbatch/p2_pack.sbatch >/dev/null
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
[ -f "results/$PHASE/${DS}_grepseek.metrics.json" ] && echo "done    p3_grepseek" >&2 || sub p3_grepseek --export="$EXP" $(dep $ACCEPT) sbatch/p2_grepseek.sbatch >/dev/null
[ -f "results/$PHASE/${DS}_dr_dci.metrics.json" ] && echo "done    p3_dr_dci" >&2 || sub p3_dr_dci   --export="$EXP" $(dep $ACCEPT) sbatch/p2_dr_dci.sbatch >/dev/null
[ -f "results/$PHASE/${DS}_agentir.metrics.json" ] && echo "done    p3_agentir" >&2 || sub p3_agentir  --export="$EXP" $(dep $ACCEPT $AIRENC) sbatch/p2_agentir.sbatch >/dev/null
[ -f "results/$PHASE/${DS}_dci.metrics.json" ] && echo "done    p3_dci" >&2 || sub p3_dci      --export="$EXP" $(dep $ACCEPT $DCICORP) sbatch/p2_dci.sbatch >/dev/null
[ -f "results/$PHASE/${DS}_rise.metrics.json" ] && echo "done    p3_rise" >&2 || sub p3_rise     --export="$EXP" $(dep $ACCEPT $RISETOC) sbatch/p2_rise.sbatch >/dev/null

squeue -u "$USER" -o "%.9i %.22j %.9T %.12r %.20E"
