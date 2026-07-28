#!/usr/bin/env bash
#SBATCH -p rali
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
#
# Packed Phase-2 launcher: hold a WHOLE node and run one eval cell per GPU.
#
# Why this exists
# ---------------
# QOS `normal` is MaxJobsPU=2 / MaxSubmitPU=4 with **no MaxTRES at all** -- it
# caps the number of jobs, not GPUs. p2_local.sbatch asked for `--gres=gpu:2` on
# nodes that have 4, so two concurrent jobs used 4 of the partition's ~19 GPUs
# and the matrix ran 2-wide. The cells are independent (different output files,
# no shared state), so this serialisation bought nothing. Packing to a full node
# gives 4x per slot; two slots then run 8 cells at once.
#
# Each cell still runs `--full-eval` over the complete 14,267-row PopQA split.
# No `-n`, no `--offset`, no subsetting: this changes only how many cells run at
# the same time, never what any single cell evaluates. TASK.md is untouched.
#
# Usage
#   sbatch -J p2_packA --export=ALL,LANE=laneA,CELLS="rag_bm25:rag:bm25 rag_e5:rag:e5" \
#          sbatch/p2_pack.sbatch
#
#   CELLS = `+`- or space-separated `outname:agent:ret[:sr1]`
#     Use `+` when the value travels through `--export=ALL,CELLS=...` or a lane
#     queue file: `--export` splits on commas, and advance_lane word-splits the
#     queue line, so a space- or comma-separated list silently loses cells.
#     outname -> results/phase2/popqa_<outname>.jsonl   (must be unique per cell)
#     agent   -> direct | rag | scaleseek | search_o1 | search_r1
#     ret     -> none | bm25 | e5 | qwen3_emb_4b
#     sr1     -> 7b | 14b        (search_r1 only)
#
# A cell whose .metrics.json already exists is skipped (same convention as the
# manual GPU2 helper in ab35cde), so a requeued job resumes the remaining work.
# One cell failing does not abort its siblings; the job exit code is non-zero if
# any cell failed, and every cell's verdict is printed in the summary.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_pack; cleanup_tmpdir; advance_lane; exit $code' EXIT

CELLS=${CELLS:?export CELLS="name:agent:ret[:sr1]+..."}
CELLS=${CELLS//+/ }   # accept the queue-safe `+` separator
mkdir -p results/phase2

NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)
PER=$(gpus_per_cell)
echo "[pack] node=$(hostname) gpus=$NGPU per_cell=$PER cells='$CELLS'"

# ---------------------------------------------------------------------------
# Plan: assign GPUs to cells, skipping already-finished ones.
# ---------------------------------------------------------------------------
declare -a PLAN_NAME PLAN_AGENT PLAN_RET PLAN_SR1 PLAN_GPUS PLAN_PORT PLAN_ENC
next_gpu=0
for spec in $CELLS; do
  IFS=: read -r name agent ret sr1 <<<"$spec"
  ret=${ret:-none}
  out=results/phase2/popqa_${name}.jsonl
  if [ -f "${out%.jsonl}.metrics.json" ]; then
    echo "[pack] skip $name (metrics.json exists)"
    continue
  fi
  # qwen3_emb_4b keeps its query encoder on a separate GPU, so that cell needs
  # one more than a plain cell.
  need=$PER; enc=""
  [ "$ret" = qwen3_emb_4b ] && need=$((PER + 1))
  if [ $((next_gpu + need)) -gt "$NGPU" ]; then
    # Do NOT drop it: a silently skipped cell is a missing matrix row that
    # nobody notices until the results table is assembled.
    DEFERRED="${DEFERRED:-}+$spec"
    echo "[pack] DEFER $name: needs $need GPU(s), only $((NGPU - next_gpu)) left"
    continue
  fi
  gpus=$(seq -s, "$next_gpu" $((next_gpu + PER - 1)))
  [ "$ret" = qwen3_emb_4b ] && enc=$((next_gpu + PER))
  PLAN_NAME+=("$name"); PLAN_AGENT+=("$agent"); PLAN_RET+=("$ret")
  PLAN_SR1+=("${sr1:-}"); PLAN_GPUS+=("$gpus"); PLAN_ENC+=("$enc")
  PLAN_PORT+=($((10000 + (SLURM_JOB_ID + next_gpu * 137) % 20000)))
  next_gpu=$((next_gpu + need))
done


# Cells that did not fit this allocation are re-submitted as their own pack job
# (a 24GB-GPU node fits 2 cells at TP=2, a 48GB one fits 4 at TP=1, so the split
# depends on where SLURM placed us and cannot be decided at submit time).
# The follow-up job inherits LANE and will advance it when IT finishes, so this
# job must then stop owning the lane -- otherwise the queue advances twice and a
# matrix row is skipped.
requeue_deferred() {
  [ -n "${DEFERRED:-}" ] || return 0
  local tag=${SLURM_JOB_NAME:-p2_pack}_rest
  echo "[pack] re-submitting deferred cells as $tag:$DEFERRED"
  if submit_retry -J "$tag" --export=ALL,LANE="${LANE:-}",CELLS="${DEFERRED#+}" \
                  "$REPO/sbatch/p2_pack.sbatch"; then
    LANE=""   # ownership handed to the follow-up job
  else
    echo "FATAL: could not re-submit deferred cells:$DEFERRED (kick manually)"
  fi
}
requeue_deferred

N=${#PLAN_NAME[@]}
if [ "$N" -eq 0 ]; then
  echo "[pack] nothing to run here"
  exit 0
fi
echo "[pack] running $N cell(s) in parallel"

# ---------------------------------------------------------------------------
# One cell = one vLLM server + one full-eval, on its own GPU(s) and port.
# ---------------------------------------------------------------------------
run_cell() {
  local i=$1
  local name=${PLAN_NAME[$i]} agent=${PLAN_AGENT[$i]} ret=${PLAN_RET[$i]}
  local sr1=${PLAN_SR1[$i]} gpus=${PLAN_GPUS[$i]} port=${PLAN_PORT[$i]} enc=${PLAN_ENC[$i]}
  local out=results/phase2/popqa_${name}.jsonl
  local clog=$REPO/logs/cell_${SLURM_JOB_NAME:-pack}_${SLURM_JOB_ID:-0}_${name}.log

  {
    echo "=== cell $name (agent=$agent ret=$ret sr1=${sr1:-none} gpus=$gpus port=$port) ==="
    local RETARGS=() RESUME=() EXTRA=()
    case "$ret" in
      none) ;;
      qwen3_emb_4b) RETARGS=(--retriever qwen3_emb_4b --retriever-device "cuda:$enc");;
      *) RETARGS=(--retriever "$ret" --retriever-device cpu);;
    esac
    [ -f "$out" ] && RESUME=(--resume)

    local pid
    if [ "$agent" = search_r1 ]; then
      local M R CONC
      case "$sr1" in
        7b)  M=$SR1_7B;  R=$SR1_7B_REV;  CONC=${CONC:-8};;
        14b) M=$SR1_14B; R=$SR1_14B_REV; CONC=${CONC:-6};;
        *) echo "FATAL: search_r1 needs sr1=7b|14b"; return 1;;
      esac
      pid=$(start_vllm_on "$gpus" "$M" "$R" "$port" "$PER" "search-r1-$sr1")
      waitp_pid "$port" "$pid" 3600 || return 1
      "$PY" -m eval.run_eval --dataset popqa --full-eval --agent search_r1 \
        "${RETARGS[@]}" \
        --search-r1-host 127.0.0.1 --search-r1-port "$port" \
        --search-r1-model "search-r1-$sr1" --search-r1-tokenizer "$M" \
        --generator-revision "$R" --concurrency "$CONC" --retries 3 \
        --seed "${DECODE_SEED:-12345}" \
        --output "$out" "${RESUME[@]}" || return 1
    else
      export VLLM_USE_V2_MODEL_RUNNER=0
      pid=$(start_vllm_on "$gpus" Qwen/Qwen3.5-9B "$GEN_REV" "$port" "$PER" \
                          Qwen/Qwen3.5-9B --reasoning-parser qwen3)
      waitp_pid "$port" "$pid" 3600 || return 1
      case "$agent" in
        direct|rag|scaleseek)
          if ! "$PY" scripts/probe_thinking_closure.py \
                --base-url "http://127.0.0.1:$port/v1" --model Qwen/Qwen3.5-9B \
                --popqa "$POPQA_NORM" --n 16 --max-tokens 28000 \
                --gate-mode budget_think --agent "$agent"; then
            echo "FATAL: thinking-closure gate failed for $agent"; return 1
          fi;;
      esac
      case "$agent" in
        search_o1|scaleseek) EXTRA=(--tokenizer Qwen/Qwen3.5-9B);;
      esac
      "$PY" -m eval.run_eval --dataset popqa --full-eval --agent "$agent" \
        "${RETARGS[@]}" "${EXTRA[@]}" \
        --host 127.0.0.1 --port "$port" --model Qwen/Qwen3.5-9B \
        --generator-revision "$GEN_REV" --concurrency "${CONC:-16}" --retries 3 \
        --seed "${DECODE_SEED:-12345}" \
        --output "$out" "${RESUME[@]}" || return 1
    fi

    sane "$out" || return 1
    "$PY" scripts/compute_metrics.py --results "$out" --dataset popqa \
      --agent "$agent" --out "${out%.jsonl}.metrics.json" || return 1
    echo "ALL_DONE $out"
  } > "$clog" 2>&1
}

# `set -e` (from common.sh) would kill the job on the first failing cell; the
# whole point of packing is that siblings are independent.
set +e
declare -a CELL_PID
for i in $(seq 0 $((N - 1))); do
  run_cell "$i" &
  CELL_PID+=("$!")
  echo "[pack] launched ${PLAN_NAME[$i]} -> logs/cell_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${PLAN_NAME[$i]}.log"
  sleep 20   # stagger vLLM starts so 4 model loads do not thrash page cache
done

rc=0
for i in $(seq 0 $((N - 1))); do
  wait "${CELL_PID[$i]}"; s=$?
  if [ "$s" -eq 0 ]; then echo "[pack] OK   ${PLAN_NAME[$i]}"
  else echo "[pack] FAIL ${PLAN_NAME[$i]} (exit $s)"; rc=1; fi
done

echo "[pack] summary:"
for i in $(seq 0 $((N - 1))); do
  m=results/phase2/popqa_${PLAN_NAME[$i]}.metrics.json
  if [ -f "$m" ]; then
    printf '  %-28s %s\n' "${PLAN_NAME[$i]}" \
      "$("$PY" -c "import json,sys; d=json.load(open('$m')); a=d['answer_quality']; print(f\"n={d['n']} em={a['em']:.4f} f1={a['f1']:.4f}\")" 2>/dev/null || echo "metrics unreadable")"
  else
    printf '  %-28s %s\n' "${PLAN_NAME[$i]}" "NO METRICS"
  fi
done
exit $rc
