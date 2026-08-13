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
# octal40/41 have no /usr/local/cuda and no nvcc on PATH, so both probes above
# miss and vLLM dies silently -> a whole run of 100% api_error rows. The pip
# CUDA package inside the env is the working CUDA_HOME there (commit 00e687f).
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13 \
           /u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu12; do
    [ -x "$c/bin/nvcc" ] && export CUDA_HOME=$c && break
  done
fi
[ -n "${CUDA_HOME:-}" ] && export PATH=$CUDA_HOME/bin:$PATH
if [ -z "${CUDA_HOME:-}" ]; then
  echo "FATAL: CUDA_HOME unresolved on $(hostname); vLLM would fail silently and"
  echo "       produce a full run of api_error rows. Refusing to start."
  exit 1
fi

# L40S (sm_89, octal40/41): flashinfer's JIT is built against mismatched CUDA
# headers and breaks sampling; torch-native sampling works (commit 1be8033).
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qiE "l40|ls40"; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo "[common] L40S detected -> VLLM_USE_FLASHINFER_SAMPLER=0"
fi

export TMPDIR=/var/tmp/mofengra/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR"
cleanup_tmpdir() { rm -rf "$TMPDIR" 2>/dev/null || true; }

# A5000/3090 have no P2P; the flag is a small perf cost elsewhere but never wrong.
export NCCL_P2P_DISABLE=1

# Static ripgrep for the DCI-family harnesses. The official dci-agent-lite
# setup.sh installs ripgrep via apt/brew; cluster nodes have neither rg nor
# sudo, so a static musl build lives here. Without it the agent's `rg` fails
# with exit 127 and the model falls back to `grep -r` over the 14GB corpus --
# job 7361 hung 4h49m on ONE smoke question that way. Pi's bash tool has no
# default timeout in the Lite checkout (`DCI_BASH_DEFAULT_TIMEOUT_SECONDS`
# only exists in the dr_dci fork's bash.ts), so nothing else bounds it.
export PATH=/data/rech/mofengra/tools/bin:$PATH

assert_gpus_usable() {  # assert_gpus_usable [expected_count]
  # Fail fast, and blame the node. octal35 repeatedly reported `idle` with tens
  # of GB free while every vLLM launch on it died several minutes later with
  # "CUDA unknown error ... Setting the available devices to be zero" -- it took
  # down dci/dr_dci twice (7336/7337, then 7347/7348). SLURM does not detect this,
  # so without a check here the scheduler keeps handing out a node that cannot
  # run anything, and the failure looks like a vLLM or config problem.
  local want=${1:-} got
  got=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ") || got=0
  if [ "${got:-0}" -eq 0 ]; then
    echo "FATAL: no usable GPU on $(hostname) -- nvidia-smi lists none."
    echo "       This is a node fault, not a config problem. Resubmit with"
    echo "       --exclude=$(hostname) and report the node."
    return 1
  fi
  if [ -n "$want" ] && [ "$got" -lt "$want" ]; then
    echo "FATAL: $(hostname) exposes $got GPU(s), need $want"; return 1
  fi
  # Counting is not enough: octal35 lists its GPUs in nvidia-smi while every
  # CUDA context creation dies with "CUDA unknown error" (7336/7337, 7347/7348,
  # and 7420 again after the count check passed). Prove init actually works.
  if ! "$PY" -c "import torch; torch.zeros(1, device='cuda')" >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi lists GPUs but CUDA init fails on $(hostname) --"
    echo "       the known octal35-class node fault. Resubmit with"
    echo "       --exclude=$(hostname) and report the node."
    return 1
  fi
  echo "[gpu] $(hostname): $got GPU(s) visible, CUDA init OK"
}

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

# --- Packed mode: several independent cells inside one job -------------------
# The QOS caps JOBS (2 running / 4 queued), not GPUs -- there is no MaxTRES at
# all. Asking for gpu:2 on a 4-GPU node therefore burned half of every
# allocation and serialised a matrix that has no data dependencies. These
# helpers let one job hold a whole node and run one cell per GPU.

gpu_mem_mib() {  # smallest GPU memory in the allocation, MiB
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | head -1
}

gpus_per_cell() {  # how many GPUs one Qwen3.5-9B server needs on this node
  # 9B bf16 needs ~18GB weights + KV; it fits on a 48GB L40S/A6000 at TP=1 but
  # not on a 24GB A5000, where TP=2 is mandatory (32 layers / 4 KV heads /
  # head_dim 256 = 128 KB per token, so a single 32k sequence alone is 4.0 GB).
  local mem; mem=$(gpu_mem_mib)
  if [ -n "$mem" ] && [ "$mem" -ge 40000 ]; then echo 1; else echo 2; fi
}

PACK_PIDS=()
PACK_LAST_PID=""
start_vllm_on() {  # start_vllm_on <gpus> <model> <rev> <port> <tp> <name> [extra...]
  # Returns the pid in the global PACK_LAST_PID, NOT on stdout. `pid=$(start_vllm_on
  # ...)` would capture this function's log lines too, and the caller would then
  # `kill -0` a string -- which fails, and reports as "vLLM exited before the port
  # opened" while vLLM is in fact still loading normally. That cost three cells on
  # 2026-07-31 (jobs 7284/7290 died in 52s with vLLM healthy).
  local gpus=$1 model=$2 rev=$3 port=$4 tp=$5 name=$6; shift 6
  local log=$REPO/logs/vllm_${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-0}_$port.log
  echo "[vllm] gpus=$gpus serving $model@$rev on :$port tp=$tp as '$name'"
  CUDA_VISIBLE_DEVICES="$gpus" \
  "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" --revision "$rev" --dtype bfloat16 \
      --host 127.0.0.1 --port "$port" --tensor-parallel-size "$tp" \
      --served-model-name "$name" \
      --max-model-len "${VLLM_MAX_LEN:-32768}" \
      --gpu-memory-utilization "${VLLM_MEM_UTIL:-0.90}" \
      --disable-custom-all-reduce \
      "$@" > "$log" 2>&1 &
  PACK_LAST_PID=$!
  PACK_PIDS+=("$PACK_LAST_PID")
}

stop_pack() {
  for p in "${PACK_PIDS[@]:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
  PACK_PIDS=()
  sleep 5
}

waitp_pid() {  # waitp_pid <port> <pid> [timeout] -- like waitp but for a given pid
  local port=$1 pid=$2 timeout=${3:-3600} t=0
  # Fail loudly on a non-numeric pid instead of letting `kill -0` fail and
  # reporting it as a dead server (see start_vllm_on).
  case "$pid" in
    ''|*[!0-9]*) echo "FATAL: waitp_pid got a non-numeric pid: ${pid:0:80}"; return 1;;
  esac
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FATAL: vLLM (pid $pid) exited before :$port opened"; return 1
    fi
    t=$((t+5))
    [ "$t" -ge "$timeout" ] && { echo "FATAL: vLLM :$port not up after ${timeout}s"; return 1; }
    sleep 5
  done
  echo "[vllm] :$port up after ${t}s"
}

pi_smoke_gate() {  # pi_smoke_gate <dataset.jsonl> <smoke_out_dir> <harness cmd...>
  # One question, checked before committing to 1500. Every Pi failure so far --
  # wrong provider location, reasoning_effort ignored, poisoned resume state --
  # produced blank answers that only surfaced 20-30 minutes into a full run, and
  # each time the fix looked obvious in hindsight. One question costs a minute.
  local ds=$1 out=$2; shift 2
  local one=$TMPDIR/smoke_one.jsonl
  head -1 "$ds" > "$one"
  rm -rf "$out"; mkdir -p "$out"
  echo "[smoke] one question through the real harness before the full run"
  "$@" --dataset "$one" --output-root "$out" >"$REPO/logs/pi_smoke_${SLURM_JOB_ID:-0}.log" 2>&1 || {
    echo "FATAL: smoke question failed; see logs/pi_smoke_${SLURM_JOB_ID:-0}.log"
    tail -20 "$REPO/logs/pi_smoke_${SLURM_JOB_ID:-0}.log" 2>/dev/null
    return 1; }
  local res=$out/results.jsonl
  [ -f "$res" ] || { echo "FATAL: smoke produced no results.jsonl"; return 1; }
  if grep -q '"final_text": ""' "$res"; then
    echo "FATAL: the smoke question came back with an EMPTY final_text."
    echo "       The agent runs but answers nothing -- exactly the state that"
    echo "       produced 1500 blank rows on 7340/7341/7351/7354/7355."
    echo "       First suspect: --tool-call-parser not matching the model's chat"
    echo "       template (Qwen3.5 emits XML function calls -> qwen3_coder; the"
    echo "       hermes parser silently swallows them: 7359/7360). Check the"
    echo "       conversation_full.json for turn_count 1 + content '\\n\\n'."
    echo "       Inspect: $out/*/result.json and $out/*/stderr.txt"
    grep -oE '"(turn_count|request_count|run_error)": *("[^"]{0,120}"|[0-9]+|null)' "$res" | head -3
    return 1
  fi
  echo "[smoke] OK -- non-empty answer:"
  grep -oE '"final_text": *"[^"]{0,120}"' "$res" | head -1
  rm -rf "$out"
}

assert_raw_not_poisoned() {  # assert_raw_not_poisoned <raw_dir>
  # The Pi launcher resumes: a per-question directory whose result.json says
  # run_status "completed" is skipped, returncode 2, "Cannot resume completed
  # run". That is correct behaviour -- but a run that finished with EMPTY answers
  # also writes "completed", so a broken run leaves state that makes every later
  # attempt silently skip all 1500 questions in minutes and produce nothing.
  # 7351 did exactly that on top of 7340's leftovers before anyone noticed.
  local raw=$1 res=$1/results.jsonl
  [ -f "$res" ] || return 0
  local total empty
  total=$(wc -l < "$res")
  empty=$(grep -c '"final_text": ""' "$res" || true)
  [ "${total:-0}" -gt 0 ] || return 0
  if [ "${empty:-0}" -eq "$total" ]; then
    echo "FATAL: $raw holds $total rows whose final_text is ALL empty, and the Pi"
    echo "       launcher will skip every one of them as already 'completed'."
    echo "       Archive it before re-running, e.g."
    echo "         mv $raw ${raw}_poisoned_\$(date +%m%d)"
    return 1
  fi
  echo "[raw] $raw: $total prior rows, $empty empty -- resume is meaningful"
}

pi_vllm_provider() {  # pi_vllm_provider <port> [model] -- point the Pi agent at our vLLM
  # The Pi coding agent (dci / dr_dci) does NOT read OPENAI_BASE_URL. Passing
  # `--provider openai` therefore sends it to api.openai.com, which answers 401,
  # and the launcher still exits 0 with run_status "completed" and an empty
  # final_text -- 1.2s per question, 1500/1500 blank. dci (7340) and dr_dci
  # (7341) both burned a slot producing exactly nothing that way.
  #
  # The official path needs no patching (assets/docs/setup.md "configure a local
  # vLLM provider"): declare a custom OpenAI-compatible provider in
  # $PI_CODING_AGENT_DIR/models.json and pass `--provider vllm`. The port is
  # per-job, so the file is generated per job rather than kept in ~/.pi.
  # The agent dir must be the checkout's own pi-mono/.pi/agent -- that is where
  # Pi looks, and where the July configuration lived alongside auth.json. Writing
  # models.json anywhere else (e.g. under $TMPDIR) leaves the provider undefined
  # and every question dies with `Unknown provider "vllm"`, turn_count 0,
  # returncode 1 -- 167/167 blank rows on job 7354 before this was spotted.
  # .pi/ is untracked in the checkout, so this does not disturb the commit gate.
  local port=$1 model=${2:-Qwen/Qwen3.5-9B} repo=${3:-$OFFICIAL_ROOT/dr_dci}
  export PI_CODING_AGENT_DIR=$repo/pi-mono/.pi/agent
  export VLLM_API_KEY=dummy
  mkdir -p "$PI_CODING_AGENT_DIR"
  # Shape copied from the configuration that actually ran DCI and DR-DCI in July
  # (/data/rech/mofengra/dr_dci_official/pi-mono/.pi/agent/models.json), not from
  # the setup docs. The docs' example omits reasoning/contextWindow/maxTokens and
  # writes apiKey as an env var NAME; the working file set a literal key and
  # declared the model's reasoning + window explicitly, which matters for a
  # thinking model.
  #
  # One deliberate difference: the July file used id "agent" (vLLM was served
  # under --served-model-name agent). Phase-2 pins the frozen generator, and
  # run_official_baseline._validate requires --model Qwen/Qwen3.5-9B under
  # --full-eval, so the id here matches the served-model-name we actually launch.
  cat > "$PI_CODING_AGENT_DIR/models.json" <<PIEOF
{
  "providers": {
    "vllm": {
      "baseUrl": "http://127.0.0.1:$port/v1",
      "apiKey": "EMPTY",
      "api": "openai-completions",
      "compat": {
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "$model",
          "name": "Qwen3.5-9B (local vLLM)",
          "reasoning": true,
          "contextWindow": ${VLLM_MAX_LEN:-32768},
          "maxTokens": 8192
        }
      ]
    }
  }
}
PIEOF
  export OPENAI_API_KEY=EMPTY
  echo "[pi] provider vllm -> http://127.0.0.1:$port/v1 (model $model)"
  echo "[pi] PI_CODING_AGENT_DIR=$PI_CODING_AGENT_DIR"

  # Both documented failure modes, checked in seconds instead of hours. dci ran
  # 2h03 and dr_dci 17min before anyone learned the agent could not talk to the
  # model at all; the harness reports run_status "completed" with an empty
  # final_text, so nothing downstream complains until `sane` sees 1500 identical
  # blank predictions.
  local probe
  probe=$(curl -sf -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer ${VLLM_API_KEY}" \
    -d "{\"model\":\"$model\",\"max_tokens\":16,\"tool_choice\":\"auto\",
         \"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],
         \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"noop\",
           \"description\":\"no op\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}" \
    2>&1) || {
      echo "FATAL: the model endpoint rejected a tool_choice=auto request."
      echo "       Pi sends exactly this and fails with '400 (no body)' when vLLM"
      echo "       lacks --enable-auto-tool-choice --tool-call-parser <parser>."
      echo "       response: ${probe:0:400}"
      return 1; }
  echo "[pi] tool-calling probe OK"

  # Whether Pi actually loaded this provider is proven by pi_smoke_gate, which
  # runs a real question and requires a real answer. An earlier version asserted
  # it here via `dci-agent-lite --list-models` -- a flag that does not exist; the
  # name came from a Pi error message referring to a different CLI, and the check
  # only ever printed usage and blocked the run. The smoke gate is strictly
  # stronger, so there is nothing left to add here.
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
  # Retrying here can only ever half-work, and it is worth being explicit about
  # why rather than tuning the number:
  #
  #   * This runs in the EXIT trap. stop_vllm has killed the server, but SLURM
  #     still holds the node+GPUs until the batch script exits, so every minute
  #     spent retrying is a minute of idle allocated GPUs.
  #   * The job also still counts toward MaxSubmitPU while it sits here, so at
  #     the cap it is partly blocking its own submission. Waiting succeeds only
  #     if some OTHER job happens to finish meanwhile.
  #
  # So the budget stays short (1h). The real fix is to stop submitting from the
  # trap at all -- pre-arm the successor at job start with
  # `--dependency=afterany:$SLURM_JOB_ID` -- which needs a lane-architecture
  # change and should be done deliberately, not in an EXIT handler.
  #
  # What IS fixed here is the silence. Giving up used to print one line into a
  # log nobody reads, and the lane simply stopped: laneA parked ~2 days on
  # 2026-07-28 and again on 2026-08-02. A stall now leaves a marker file that
  # sbatch/status.sh reports at the top, so it is visible on the next check.
  local tries=0
  while true; do
    if out=$(sbatch "$@" 2>&1); then
      echo "[submit] $out <- $*"
      rm -f "$REPO/logs/.lane_stalled_${LANE:-unknown}" 2>/dev/null || true
      return 0
    fi
    tries=$((tries+1))
    echo "[submit] blocked ($out); retry $tries/12 in 300s: $*"
    if [ "$tries" -ge 12 ]; then
      # Report the ACTUAL sbatch error, never a guessed cause. The first version
      # of this marker hardcoded "at QOS submit cap"; job 7303 in fact failed
      # with "Unable to open file sbatch/p2_grepseek_recover.sbatch" (a relative
      # path resolved from the wrong cwd), and the marker sent the reader looking
      # at the wrong thing for two days.
      printf 'LANE STALLED after 1h of retries\nlane=%s job=%s at %s\nsbatch said: %s\nkick with: sbatch %s\n' \
        "${LANE:-unknown}" "${SLURM_JOB_NAME:-?}/${SLURM_JOB_ID:-?}" \
        "$(date '+%Y-%m-%d %H:%M')" "$out" "$*" \
        > "$REPO/logs/.lane_stalled_${LANE:-unknown}" 2>/dev/null || true
      return 1
    fi
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
  # Queue lines name the script relatively ("sbatch/p2_x.sbatch"); sbatch resolves
  # that against the CURRENT directory, which is whatever the finishing job left
  # it as. Job 7303 died with "Unable to open file sbatch/p2_grepseek_recover.sbatch"
  # for exactly this reason and parked laneA for two days. Submit from $REPO.
  cd "$REPO" || true
  # shellcheck disable=SC2086
  if ! submit_retry $line; then
    # Put the line back so the lane can be kicked manually; free our resources.
    printf '%s\n' "$line" | cat - "$qf" > "$qf.tmp" && mv "$qf.tmp" "$qf"
    echo "FATAL: lane $lane stalled at QOS submit cap; requeued line for manual kick"
  fi
}

echo "[common] host=$(hostname) job=${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-0}" \
     "gpus=${CUDA_VISIBLE_DEVICES:-none} CUDA_HOME=${CUDA_HOME:-unset} TMPDIR=$TMPDIR"
