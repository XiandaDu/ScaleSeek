#!/usr/bin/env bash
# 每个 ScaleSeek sbatch 作业共用的守卫与工具函数。用法：`source sbatch/common.sh`
#
# Nibi (Alliance Canada / SHARCNET) 版本。沿用旧集群踩出来的教训：
#   1. 绝不在登录节点跑重活（Nibi 登录节点是 l1.nibi）；
#   2. CUDA_HOME 未设会让 vLLM 静默失败 —— 从 module 推导；
#   3. TMPDIR 必须指向节点本地 NVMe（$SLURM_TMPDIR），不是 /tmp 也不是 home；
#   4. vLLM 起不来要让整条 lane 中止，而不是写出 100% api_error 的结果。
# Nibi 特有：
#   5. 不设 NCCL_P2P_DISABLE —— 那是 A5000/3090 的绕行方案，H100 有 NVLink，
#      禁掉会让多卡训练慢数倍；
#   6. MIG 切片（g30-g37）只有 1 个逻辑设备且无卡间通信，多卡作业要拒绝启动。
set -euo pipefail

case "$(hostname -s)" in
  l[0-9]*) echo "FATAL: 拒绝在登录节点 $(hostname -s) 上运行"; exit 1;;
  c[0-9]*|g[0-9]*|o[0-9]*|u[0-9]*|m[0-9]*) ;;
  *) echo "FATAL: 未知主机 $(hostname -s)"; exit 1;;
esac

# shellcheck disable=SC1091
source /home/a32du/ScaleSeek/setup_env.sh

# --- CUDA ---
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${EBROOTCUDA:-}" ]; then
    export CUDA_HOME="$EBROOTCUDA"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")"); export CUDA_HOME
  fi
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

# --- TMPDIR：节点本地 NVMe，作业结束由 Slurm 自动清理 ---
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-manual}}"
mkdir -p "$TMPDIR"
cleanup_tmpdir() { [ -n "${SLURM_TMPDIR:-}" ] || rm -rf "$TMPDIR" 2>/dev/null || true; }

# --- MIG 上的 CUDA_VISIBLE_DEVICES 修正 ---
# Slurm 在 MIG 切片上把 CUDA_VISIBLE_DEVICES 设成 MIG UUID
# （CUDA_VISIBLE_DEVICES=MIG-df5d2c22-...）。torch 认这个格式，但 vLLM 0.23 会
# 拿它去 int() 解析，启动时直接崩：
#     ValueError: invalid literal for int() with base 10: 'MIG-df5d2c22-...'
# Slurm 的 device cgroup 已经把可见设备限制成那一个 MIG 实例，所以改写成序号
# 是安全的（torch 两种写法都报同一个 MIG 3g.40gb 设备）。
if [ "${SCALESEEK_ON_MIG:-0}" = "1" ] && [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
  echo "[mig] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> 0 (vLLM 无法解析 MIG UUID)"
  export CUDA_VISIBLE_DEVICES=0
fi

# --- ROCm 环境变量清理 ---
# Nibi 混布 NVIDIA(H100) 与 AMD(MI300A) 节点，Slurm 的 gres 插件在所有 GPU 作业
# 上同时导出 CUDA_VISIBLE_DEVICES 和 ROCR_VISIBLE_DEVICES。torch 2.11 检测到两者
# 并存会直接抛
#   ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.
# （2026-07-31 作业 18799125：4 卡 sft_train 四个 rank 66 秒内全灭）。
# 我们只跑 CUDA 节点，ROCm 变量一律清掉。
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

# --- GPU 数与 MIG 守卫 ---
detect_gpus() {  # detect_gpus -> 回显可用 GPU 数
  nvidia-smi -L 2>/dev/null | grep -c . || echo 0
}
require_full_gpus() {  # require_full_gpus <n> -- 多卡作业跑在 MIG 上必然失败，早点死
  local want=$1 have; have=$(detect_gpus)
  if [ "${SCALESEEK_ON_MIG:-0}" = "1" ] && [ "$want" -gt 1 ]; then
    echo "FATAL: 当前是 MIG 切片，无法做 $want 卡训练。请申请 --gres=gpu:h100:$want"
    exit 1
  fi
  if [ "$have" -lt "$want" ]; then
    echo "FATAL: 需要 $want 张 GPU，实际只有 $have"
    exit 1
  fi
  echo "[gpu] $have 张可用；本作业使用 $want"
}

VLLM_PID=""
start_vllm() {  # start_vllm <model> <revision> <port> <tp> <served-name> [额外 vllm 参数...]
  local model=$1 rev=$2 port=$3 tp=$4 name=$5; shift 5
  local log=$REPO/logs/vllm_${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-0}_$port.log
  echo "[vllm] serving $model@$rev on :$port tp=$tp as '$name' (log: $log)"
  CUDA_VISIBLE_DEVICES="${VLLM_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}" \
  "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" --revision "$rev" --dtype bfloat16 \
      --host 127.0.0.1 --port "$port" --tensor-parallel-size "$tp" \
      --served-model-name "$name" \
      --max-model-len "${VLLM_MAX_LEN:-32768}" \
      --gpu-memory-utilization "${VLLM_MEM_UTIL:-0.90}" \
      "$@" > "$log" 2>&1 &
  VLLM_PID=$!
}

waitp() {  # waitp <port> [timeout_s] -- 宁可中止 lane，也不要写垃圾结果
  local port=$1 timeout=${2:-2400} t=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "FATAL: vLLM 在 :$port 打开前就退出了"; return 1
    fi
    t=$((t+5))
    if [ "$t" -ge "$timeout" ]; then
      echo "FATAL: vLLM :$port ${timeout}s 后仍未就绪"; return 1
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
  local port=$1 model=${2:-Qwen/Qwen3.5-9B}
  export PI_CODING_AGENT_DIR=$TMPDIR/pi_agent
  export VLLM_API_KEY=dummy
  mkdir -p "$PI_CODING_AGENT_DIR"
  cat > "$PI_CODING_AGENT_DIR/models.json" <<PIEOF
{
  "providers": {
    "vllm": {
      "baseUrl": "http://127.0.0.1:$port/v1",
      "api": "openai-completions",
      "apiKey": "VLLM_API_KEY",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        { "id": "$model" }
      ]
    }
  }
}
PIEOF
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
      echo "       lacks --enable-auto-tool-choice --tool-call-parser hermes."
      echo "       response: ${probe:0:400}"
      return 1; }
  echo "[pi] tool-calling probe OK"
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
# 复读守卫：模板占位符被当成答案输出会让预测高度重复（2026-07-22:
# direct/rag 曾整批返回 'your answer here'）。
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

echo "[common] host=$(hostname -s) job=${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-0}" \
     "gpus=${CUDA_VISIBLE_DEVICES:-none} mig=${SCALESEEK_ON_MIG:-?}" \
     "CUDA_HOME=${CUDA_HOME:-unset} TMPDIR=$TMPDIR"
