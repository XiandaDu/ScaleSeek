#!/usr/bin/env bash
# =====================================================================================
# Submit any pipeline stage without editing a tracked file.  Pure bash (login nodes run
# no computation): resources are chosen here, every experimental parameter stays in yaml.
#
#   bash submit.sh prepare
#   bash submit.sh index-bm25   --corpus wiki18
#   bash submit.sh index-exact  --corpus wiki18          # the new agent's find tool
#   bash submit.sh encode       --retriever e5 --corpus wiki18 --shards 8
#   bash submit.sh build-hnsw   --retriever e5 --corpus wiki18 --shards 8
#   bash submit.sh encode-agentir   --corpus wiki18 --shards 32
#   bash submit.sh finalize-agentir --corpus wiki18 --shards 32
#   bash submit.sh check-retrievers --corpus wiki18
#   bash submit.sh eval  --method rag --dataset popqa --retriever bm25
#   bash submit.sh eval-all --dataset popqa          # the 15 runs of one dataset
#   bash submit.sh grepseek-pilot --name <run> [--variants budget,dedup] [--cap 2048] [--replay a,b --turns 1]   # GrepSeek+ on validation questions
#   bash submit.sh judge --dataset browsecomp_plus
#   bash submit.sh score --dataset popqa
#   bash submit.sh rl --name <run> [--model <hf dir>] [hydra overrides]   # GRPO of our agent
#   bash submit.sh teacher --name <run> [--nq N --hotpotqa N --samples K --take N --model M --revision R --tp 1]  # 27B trajectories for SFT
#   bash submit.sh sft --name <run> --data <dir> [--model <hf dir>] [hydra overrides]   # SFT warm start
#   ... --time 48:00:00 --dry-run --after <jobid> --after-any <jobid>   (extra sbatch controls, any stage)
# =====================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
[ -f cluster.env ] && . ./cluster.env
ACCOUNT="${SCALESEEK_ACCOUNT:?set SCALESEEK_ACCOUNT in cluster.env}"
GPU_TYPE="${SCALESEEK_GPU_TYPE:-a100}"
STAGE="${1:?usage: bash submit.sh <stage> [args]}"; shift

TIME=""; DRY=0; AFTER=""; AFTER_ANY=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --time) TIME="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --after) AFTER="$2"; shift 2 ;;
    --after-any) AFTER_ANY="$2"; shift 2 ;;     # start when it ends in any state (RL resume after a time limit)
    *) ARGS+=("$1"); shift ;;
  esac
done
arg() {  # value of --$1 among ARGS
  local i; for ((i=0; i<${#ARGS[@]}; i++)); do
    [ "${ARGS[$i]}" = "--$1" ] && { echo "${ARGS[$((i+1))]}"; return; }; done; echo ""; }
matrix() {  # "<method> <retriever|->" for every run, read from configs/methods/*.yaml
  local f m r
  for f in configs/methods/*.yaml; do
    m="$(basename "$f" .yaml)"; [ "$m" = bcp_judge ] && continue
    r="$(awk '/^  retrievers:/{gsub(/[][,]/," "); for(i=2;i<=NF;i++) print $i}' "$f")"
    if [ -z "$r" ]; then echo "$m -"; else for x in $r; do echo "$m $x"; done; fi
  done
}
yaml_tp() { awk '/tensor_parallel_size:/{print $2; exit}' "configs/methods/$1.yaml"; }

# Every job runs in a frozen checkout (git worktree) of the current commit, so jobs that
# start hours later never pick up later edits.  Uncommitted changes are refused.
if [ $DRY -eq 0 ] && [ -n "$(git status --porcelain)" ]; then
  echo "commit first; uncommitted changes:"; git status --short; exit 2
fi
REPO="$PWD"
COMMIT="$(git rev-parse HEAD)"
SNAP="${SCALESEEK_ROOT:-/scratch/$USER/scaleseek_v2}/harness/${COMMIT:0:12}"
[ -d "$SNAP" ] || git worktree add -q --detach "$SNAP" "$COMMIT"
cp -f cluster.env "$SNAP/"
mkdir -p logs
submit() {  # submit <name> <script> <sbatch resource flags...> -- <run.py args...>
  local name="$1" script="$2"; shift 2
  local res=(); while [ "$1" != "--" ]; do res+=("$1"); shift; done; shift
  local cmd=(sbatch --chdir="$SNAP" --account="$ACCOUNT" --job-name="ss_$name" --output="$REPO/logs/%x-%j.out")
  [ -n "$TIME" ] && res+=(--time="$TIME")
  [ -n "$AFTER" ] && cmd+=(--dependency="afterok:${AFTER//,/:}")   # --after 1,2,3 = all of them
  [ -n "$AFTER_ANY" ] && cmd+=(--dependency="afterany:${AFTER_ANY//,/:}")
  cmd+=("${res[@]}" "$script" "$@")
  echo "${cmd[*]}"
  [ $DRY -eq 1 ] || "${cmd[@]}"
}

case "$STAGE" in
  prepare)
    submit prepare slurm/job.sh --cpus-per-task=8 --mem=96G --time=12:00:00 -- prepare "${ARGS[@]}" ;;
  index-bm25)
    submit "bm25_$(arg corpus)" slurm/job.sh --cpus-per-task=32 --mem=128G --time=24:00:00 \
      --export=ALL,SCALESEEK_JAVA_HEAP=96g -- index-bm25 "${ARGS[@]}" ;;
  index-exact)
    submit "exact_$(arg corpus)" slurm/job.sh --cpus-per-task=32 --mem=128G --time=24:00:00 \
      --export=ALL,SCALESEEK_JAVA_HEAP=96g -- index-exact "${ARGS[@]}" ;;
  encode|encode-agentir)
    N="$(arg shards)"; [ -n "$N" ] || { echo "--shards N required"; exit 2; }
    TASKS="$(arg tasks)"; TASKS="${TASKS:-0-$((N-1))}"     # --tasks 13,15,22: re-run only those
    PASS=(); for ((i=0; i<${#ARGS[@]}; i++)); do
      case "${ARGS[$i]}" in --shards|--tasks) i=$((i+1)); continue ;; esac; PASS+=("${ARGS[$i]}"); done
    submit "${STAGE}_$(arg retriever)_$(arg corpus)" slurm/array.sh --array="$TASKS" --export=ALL,SCALESEEK_SHARDS="$N" \
      --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=8 --mem=96G --time=24:00:00 -- "$STAGE" "${PASS[@]}" ;;
  build-hnsw)
    # qwen3_emb_4b: 21M x 2560 float32 = 215 GB + graph -> a large-memory node.
    # (Alliance routes large --mem requests to large-memory nodes by itself.)
    if [ "$(arg corpus)" = bcp ]; then MEM=64G; elif [ "$(arg retriever)" = "qwen3_emb_4b" ]; then MEM=400G; else MEM=150G; fi
    submit "hnsw_$(arg retriever)_$(arg corpus)" slurm/job.sh --cpus-per-task=32 --mem="$MEM" \
      --time=72:00:00 -- build-hnsw "${ARGS[@]}" --threads 32 ;;
  finalize-agentir)
    submit "agentir_final_$(arg corpus)" slurm/job.sh --cpus-per-task=4 --mem=32G --time=02:00:00 -- finalize-agentir "${ARGS[@]}" ;;
  check-retrievers)
    if [ "$(arg corpus)" = bcp ]; then MEM=64G; else MEM=498G; fi   # wiki18: the qwen3 HNSW is ~220 GB
    submit "rcheck_$(arg corpus)" slurm/job.sh --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=12 --mem="$MEM" \
      --time=06:00:00 -- check-retrievers "${ARGS[@]}" ;;
  eval)
    M="$(arg method)"; TP="$(yaml_tp "$M")"; R="$(arg retriever)"
    # The flat AgentIR index and the qwen3 HNSW index each need ~220 GB of RAM.
    if [ "$M" = "agentir" ] || [ "$R" = "qwen3_emb_4b" ]; then MEM=498G; CPU=48; else MEM=$((64*TP + 64))G; CPU=$((12*TP)); fi
    submit "$(arg dataset)_${M}${R:+_$R}" slurm/job.sh --gpus-per-node="$GPU_TYPE:$TP" --cpus-per-task="$CPU" \
      --mem="$MEM" --time=72:00:00 -- eval "${ARGS[@]}" ;;
  eval-all)
    D="$(arg dataset)"
    while read -r M R; do
      if [ "$R" = "-" ]; then bash "$0" eval --method "$M" --dataset "$D" ${TIME:+--time $TIME} $([ $DRY -eq 1 ] && echo --dry-run)
      else bash "$0" eval --method "$M" --dataset "$D" --retriever "$R" ${TIME:+--time $TIME} $([ $DRY -eq 1 ] && echo --dry-run); fi
    done < <(matrix) ;;
  grepseek-pilot)
    # GrepSeek+ variants on validation questions (scaleseek/grepseek_pilot.py); resources = a GrepSeek eval.
    submit "gpilot_$(arg name)" slurm/job.sh --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=12 --mem=128G \
      --time=08:00:00 -- grepseek-pilot "${ARGS[@]}" ;;
  judge)
    TP="$(yaml_tp bcp_judge)"
    submit "judge_$(arg dataset)" slurm/job.sh --gpus-per-node="$GPU_TYPE:$TP" --cpus-per-task=$((12*TP)) \
      --mem=$((64*TP + 64))G --time=12:00:00 -- judge "${ARGS[@]}" ;;
  score)
    submit "score_$(arg dataset)" slurm/job.sh --cpus-per-task=2 --mem=16G --time=00:30:00 -- score "${ARGS[@]}" ;;
  rl)
    # GRPO of our agent (train/rl.sbatch); --model defaults to the pinned base Qwen3.5-9B.
    N="$(arg name)"; [ -n "$N" ] || { echo "--name required"; exit 2; }
    M="$(arg model)"; M="${M:-${SCALESEEK_ROOT:-/scratch/$USER/scaleseek_v2}/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
    PASS=(); for ((i=0; i<${#ARGS[@]}; i++)); do
      case "${ARGS[$i]}" in --name|--model) i=$((i+1)); continue ;; esac; PASS+=("${ARGS[$i]}"); done
    mkdir -p "${SCALESEEK_ROOT:-/scratch/$USER/scaleseek_v2}/train_runs"
    submit "rl_$N" train/rl.sbatch --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=48 --mem=498G \
      -- "$N" "$M" "${PASS[@]}" ;;
  sft)
    # SFT warm start (train/sft.sbatch); --model defaults to the pinned base Qwen3.5-9B.
    N="$(arg name)"; D="$(arg data)"; [ -n "$N" ] && [ -n "$D" ] || { echo "--name and --data required"; exit 2; }
    M="$(arg model)"; M="${M:-${SCALESEEK_ROOT:-/scratch/$USER/scaleseek_v2}/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
    PASS=(); for ((i=0; i<${#ARGS[@]}; i++)); do
      case "${ARGS[$i]}" in --name|--model|--data) i=$((i+1)); continue ;; esac; PASS+=("${ARGS[$i]}"); done
    submit "sft_$N" train/sft.sbatch --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=48 --mem=498G \
      -- "$N" "$M" "$D" "${PASS[@]}" ;;
  teacher)
    # Qwen3.5-27B trajectories of our agent on train questions (train/teacher.sbatch), for SFT.
    N="$(arg name)"; [ -n "$N" ] || { echo "--name required"; exit 2; }
    PASS=(); for ((i=0; i<${#ARGS[@]}; i++)); do
      case "${ARGS[$i]}" in --name) i=$((i+1)); continue ;; esac; PASS+=("${ARGS[$i]}"); done
    TP="$(arg tp)"; TP="${TP:-4}"            # --tp 1 for a 9B generator (--model/--revision)
    submit "teacher_$N" train/teacher.sbatch --gpus-per-node="$GPU_TYPE:$TP" --cpus-per-task=$((12*TP)) \
      --mem=$((64*TP + 64))G -- "$N" "${PASS[@]}" ;;
  *) echo "unknown stage $STAGE"; sed -n '2,21p' "$0"; exit 2 ;;
esac
