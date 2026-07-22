#!/usr/bin/env bash
# Shared guards and helpers for every ScaleSeek sbatch job. `source sbatch/common.sh`.
# Encodes the hard operational lessons:
#   1. never run heavy work on a jump host (arcade*),
#   2. octal40/41 have no /usr/local/cuda -> unset CUDA_HOME kills vLLM silently,
#   3. /tmp on octal30 is tmpfs -> route TMPDIR to /var/tmp and clean it up,
#   4. a dead vLLM must abort the lane instead of producing 100% api_error rows.
set -euo pipefail

case "$(hostname)" in
  octal*|ilar*|abaque*|blg*) ;;
  *) echo "FATAL: refusing to run on $(hostname) (jump host?)"; exit 1;;
esac

source /u/mofengra/miniconda3/etc/profile.d/conda.sh
source /data/rech/mofengra/ScaleSeek/setup_env.sh
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python

export E5_INDEX_DIR=$DATA/e5_index
export QWEN3_EMB_INDEX_DIR=$DATA/qwen3_embedding_4b_index
export LLM_TOKENIZER=Qwen/Qwen3.5-9B
export OFFICIAL_ROOT=/data/rech/mofengra/official-baselines
export GEN_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export SR1_7B=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3
export SR1_7B_REV=395b18f1fecee52f1b51fb22f898c220f0a08ec3
export SR1_14B=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3
export SR1_14B_REV=d65f11c88d3c129a01466f0c154aaad7d9b09225
export GREPSEEK_MODEL=alireza7/GrepSeek-Qwen3.5-9B-GRPO
export GREPSEEK_REV=a79563970cfdd2ced3cc5fde481737d0ebea6fa4
export CORPUS_MANIFEST=$CORPUS_DIR/corpus_manifest.json
export POPQA_RAW=$DATASETS/popqa/test.jsonl
export POPQA_NORM=$DATASETS/popqa/test.normalized.jsonl
export POPQA_MANIFEST=$DATASETS/popqa/manifest.json

if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda /opt/cuda; do
    [ -x "$c/bin/nvcc" ] && export CUDA_HOME=$c && break
  done
fi
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")"); export CUDA_HOME
fi
[ -n "${CUDA_HOME:-}" ] && export PATH=$CUDA_HOME/bin:$PATH

export TMPDIR=/var/tmp/mofengra/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR"
cleanup_tmpdir() { rm -rf "$TMPDIR" 2>/dev/null || true; }

# A5000/3090 have no P2P; the flag is a small perf cost elsewhere but never wrong.
export NCCL_P2P_DISABLE=1

VLLM_PID=""
start_vllm() {  # start_vllm <model> <revision> <port> <tp> <served-name> [extra vllm args...]
  local model=$1 rev=$2 port=$3 tp=$4 name=$5; shift 5
  local log=$REPO/logs/vllm_${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-0}_$port.log
  echo "[vllm] serving $model@$rev on :$port tp=$tp as '$name' (log: $log)"
  CUDA_VISIBLE_DEVICES="${VLLM_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}" \
  "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" --revision "$rev" --dtype bfloat16 \
      --host 127.0.0.1 --port "$port" --tensor-parallel-size "$tp" \
      --served-model-name "$name" \
      --max-model-len "${VLLM_MAX_LEN:-32768}" \
      --gpu-memory-utilization "${VLLM_MEM_UTIL:-0.90}" \
      --disable-custom-all-reduce \
      "$@" > "$log" 2>&1 &
  VLLM_PID=$!
}

waitp() {  # waitp <port> [timeout_s] -- abandon the lane rather than write garbage
  local port=$1 timeout=${2:-2400} t=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "FATAL: vLLM exited before :$port opened"; return 1
    fi
    t=$((t+5))
    if [ "$t" -ge "$timeout" ]; then
      echo "FATAL: vLLM :$port not up after ${timeout}s"; return 1
    fi
    sleep 5
  done
  echo "[vllm] :$port up after ${t}s"
}

stop_vllm() {
  if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 5
  fi
}

sane() {  # sane <results.jsonl> -- refuse mostly-api_error or parroted outputs
  "$PY" - "$1" <<'PYEOF'
import collections, json, sys
path = sys.argv[1]
total = err = 0
preds = collections.Counter()
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        total += 1
        row = json.loads(line)
        if "api_error" in json.dumps(row).lower():
            err += 1
        preds[str(row.get("prediction", "")).strip().lower()] += 1
if total == 0:
    sys.exit(f"SANE-FAIL {path}: empty results")
frac = err / total
print(f"[sane] {path}: {total} rows, api_error rows {err} ({frac:.1%})")
if frac > 0.2:
    sys.exit(f"SANE-FAIL {path}: api_error fraction {frac:.1%} > 20%")
# Parrot guard: a template placeholder echoed as the "answer" makes most
# predictions identical (2026-07-22: direct/rag returned 'your answer here').
if total >= 8:
    top, top_n = preds.most_common(1)[0]
    if top_n / total > 0.8:
        sys.exit(f"SANE-FAIL {path}: {top_n}/{total} identical predictions ({top[:60]!r})")
PYEOF
}

submit_retry() {  # submit_retry <sbatch args...> -- survive QOSMaxSubmitJobPerUserLimit
  local tries=0
  while true; do
    if out=$(sbatch "$@" 2>&1); then echo "[submit] $out <- $*"; return 0; fi
    tries=$((tries+1))
    echo "[submit] blocked ($out); retry $tries in 300s: $*"
    # Give up after ~30 min: a job stuck here is holding its GPUs hostage.
    if [ "$tries" -ge 6 ]; then return 1; fi
    sleep 300
  done
}

advance_lane() {  # advance_lane -- pop the next job off this lane's queue file
  local lane=${LANE:-}
  [ -n "$lane" ] || return 0
  local qf=$REPO/sbatch/queue_${lane}.txt
  [ -s "$qf" ] || { echo "[lane $lane] queue empty"; return 0; }
  local line
  line=$(head -1 "$qf")
  tail -n +2 "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
  echo "[lane $lane] next: $line"
  # shellcheck disable=SC2086
  if ! submit_retry $line; then
    # Put the line back so the lane can be kicked manually; free our resources.
    printf '%s\n' "$line" | cat - "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
    echo "FATAL: lane $lane stalled at QOS submit cap; requeued line for manual kick"
  fi
}

echo "[common] host=$(hostname) job=${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-0}" \
     "gpus=${CUDA_VISIBLE_DEVICES:-none} CUDA_HOME=${CUDA_HOME:-unset} TMPDIR=$TMPDIR"
