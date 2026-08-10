#!/usr/bin/env bash
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 12
#SBATCH --mem=56G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Parametrized full-dataset run ($DS, default popqa) for the locally-run methods.
# Submit with:  sbatch -J p2_<name> --export=ALL,AGENT=...,RET=...,SR1=... p2_local.sbatch
#   AGENT: direct | rag | scaleseek | search_o1 | search_r1
#   RET:   none | bm25 | e5 | qwen3_emb_4b        (qwen3_emb_4b needs --gres=gpu:h100:3 --mem=110G)
#   SR1:   "" | 7b | 14b                          (search_r1 only)
# The output file name comes from the job name: p2_<x> -> $RESULTS_DIR/${DS}_<x>.jsonl
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT

AGENT=${AGENT:?export AGENT=}
RET=${RET:-none}
SR1=${SR1:-}
PORT=$((10000 + SLURM_JOB_ID % 20000))
OUT=$RESULTS_DIR/${DS}_${SLURM_JOB_NAME#p2_}.jsonl
mkdir -p "$RESULTS_DIR"

RETARGS=()
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | wc -l)
case "$RET" in
  none) ;;
  qwen3_emb_4b)
    # vLLM on all GPUs but the last; the 4B query encoder gets the last one.
    export VLLM_GPUS=$(seq -s, 0 $((NGPU - 2)))
    RETARGS=(--retriever qwen3_emb_4b --retriever-device "cuda:$((NGPU - 1))")
    ;;
  *) RETARGS=(--retriever "$RET" --retriever-device cpu);;
esac

RESUME=()
[ -f "$OUT" ] && RESUME=(--resume)

if [ "$AGENT" = search_r1 ]; then
  case "$SR1" in
    7b)  M=$SR1_7B;  R=$SR1_7B_REV;  CONC=${CONC:-8};;
    14b) M=$SR1_14B; R=$SR1_14B_REV; CONC=${CONC:-6}; export VLLM_MEM_UTIL=0.93;;
    *) echo "FATAL: search_r1 requires SR1=7b|14b"; exit 1;;
  esac
  start_vllm "$M" "$R" "$PORT" 2 "search-r1-$SR1"
  waitp "$PORT" 3600
  "$PY" -m eval.run_eval --dataset "$DS" --full-eval --agent search_r1 \
    "${RETARGS[@]}" \
    --search-r1-host 127.0.0.1 --search-r1-port "$PORT" \
    --search-r1-model "search-r1-$SR1" --search-r1-tokenizer "$M" \
    --generator-revision "$R" --concurrency "$CONC" --retries 3 \
    --output "$OUT" "${RESUME[@]}"
else
  # thinking_token_budget (direct/rag forced thinking closure) is only wired
  # into vLLM 0.23's V1 model runner.
  export VLLM_USE_V2_MODEL_RUNNER=0
  start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B --reasoning-parser qwen3
  waitp "$PORT" 3600
  # Gate before burning a full run: the probe must produce parseable answers
  # under the exact production decoding mode or every row degenerates to
  # parse_error (2026-07-22: 12h of 100% empty predictions). On failure it
  # prints a mode comparison table for a human decision and the lane moves on.
  # Production mode for all three prompt agents = budget_think (no max_tokens cap
  # + vLLM-native forced thinking closure). The gate probes the agent's OWN prompt
  # (--agent), so scaleseek is scored on real <tool_call> turns.
  #
  # 2026-07-28: scaleseek was previously gated on greedy_nothink because the
  # 07-23/07-24 tables showed budget_think at 0/16. Those tables predate the
  # parser fix in 19198ee and are void — re-reading the same logs shows
  # budget_think emitted 12-15/16 valid closed answers with finish=length never
  # once triggered. enable_thinking=false is unfrozen pending a clean gate run.
  case "$AGENT" in
    direct|rag|scaleseek) GATE_MODE=budget_think;;
    *)                    GATE_MODE=;;
  esac
  if [ -n "$GATE_MODE" ]; then
    if ! "$PY" scripts/probe_thinking_closure.py \
          --base-url "http://127.0.0.1:$PORT/v1" --model Qwen/Qwen3.5-9B \
          --popqa "$DS_NORM" --n 16 --max-tokens 28000 \
          --gate-mode "$GATE_MODE" --agent "$AGENT"; then
      echo "FATAL: thinking-closure gate ($GATE_MODE) failed for $AGENT; config needs a user decision (see mode tables above)"
      exit 3
    fi
  fi
  EXTRA=()
  case "$AGENT" in
    search_o1|scaleseek) EXTRA=(--tokenizer Qwen/Qwen3.5-9B);;
  esac
  "$PY" -m eval.run_eval --dataset "$DS" --full-eval --agent "$AGENT" \
    "${RETARGS[@]}" "${EXTRA[@]}" \
    --host 127.0.0.1 --port "$PORT" --model Qwen/Qwen3.5-9B \
    --generator-revision "$GEN_REV" --concurrency "${CONC:-48}" --retries 3 \
    --output "$OUT" "${RESUME[@]}"
fi
stop_vllm

sane "$OUT"
"$PY" scripts/compute_metrics.py --results "$OUT" --dataset "$DS" --agent "$AGENT" \
  --out "${OUT%.jsonl}.metrics.json"
echo "ALL_DONE $OUT"
