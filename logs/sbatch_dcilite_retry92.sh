#!/bin/bash
#SBATCH --job-name=ss_dci92
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_dci92_%j.log
#
# dci-lite popqa 最后 92 题。前两轮失败的**真正根因**已查明（2026-07-19）：
#
#   pi coding-agent 的 bash 工具把 stdout 写到 `join(os.tmpdir(), "pi-bash-<id>.log")`
#   （pi-mono/packages/coding-agent/src/core/bash-executor.ts:92）。
#   octal30 的 /tmp 是 **tmpfs —— 占的是内存**。agent 每题 grep 14GB wiki 语料，
#   单个日志能长到 7.6GB；739 个日志累积占了 **101GB 内存**（跨作业不清，tmpfs 只有
#   重启才释放）。节点 251GB 内存被吃掉 40%，于是：
#     · 内核分配不出 socket 缓冲 → node 报 `write ENOBUFS`（第一轮 116 个）
#     · tmpfs 写失败 → node 报 `write` errno -122 → 工具输出 JSON 被腰斩 →
#       harness 解析报 `Unterminated string`（第二轮 80 个，截断位置 25~10931 随机，
#       正因为是"写到哪断到哪"而非固定缓冲）
#   一个原因解释了两轮完全不同的报错。第一轮 n=50 没事，是因为日志还没堆起来。
#
# 修复（已在节点上逐条验证）：
#   1) TMPDIR 指到本地 NVMe（/var/tmp 在 /dev/nvme1n1p3，384G 空闲，不是 tmpfs）。
#      已验证 `TMPDIR=... node -e 'os.tmpdir()'` 确实返回新路径 —— node 认这个变量。
#   2) 跑完就地清理，别再留给下一个作业。
#   3) 并发恢复到 4（之前降到 3 是在治标，内存压力才是真因）。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
SS=/data/rech/mofengra/ScaleSeek
DD=/data/rech/mofengra/dr_dci_official
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8

# ---- 核心修复：把 agent 的临时日志赶出 tmpfs ----
export TMPDIR=/var/tmp/mofengra_dcilite_$SLURM_JOB_ID
mkdir -p "$TMPDIR" || { echo "[d92] FATAL: 建不了 $TMPDIR"; exit 1; }
trap 'rm -rf "$TMPDIR"' EXIT
echo "[d92] TMPDIR=$TMPDIR ($(df -h $TMPDIR | tail -1 | awk "{print \$4}") 可用，本地盘非 tmpfs)"
# 顺手清掉别的作业留下的残骸（只删自己的、且没有进程持有的）
find /tmp -maxdepth 1 -name 'pi-bash-*.log' -user "$USER" -mmin +60 -delete 2>/dev/null
echo "[d92] /tmp 现况: $(df -h /tmp | tail -1)"

CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 40960 > $SS/logs/d92_4b.log 2>&1 &
ok=0; for t in $(seq 1 180); do curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[d92] FATAL: 8000 没起来"; exit 1; }
echo "[d92] 4B@8000 up @ $(date +%m%d-%H:%M)"

cd $DD || exit 1
set -a; source .env 2>/dev/null; set +a
export TMPDIR=/var/tmp/mofengra_dcilite_$SLURM_JOB_ID   # source .env 可能覆盖，重设一次
export DCI_VIEW_CACHE_ROOT=$TMPDIR/view_cache
export DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses
export DCI_JUDGE_MAX_OUTPUT_TOKENS=2048
echo "[d92] dci-lite retry92 start @ $(date +%m%d-%H:%M)"
$DD/.venv/bin/python scripts/bcplus_eval/run_bcplus_eval.py \
  --dataset "$DD/data/dci-bench/data/popqa/retry92.jsonl" \
  --output-root "$DD/outputs/qa/popqa_dcilite_vllm4b_retry92" \
  --corpus-dir "$DD/corpus/wiki_corpus" \
  --package-dir "$DD/pi-mono/packages/coding-agent" \
  --agent-dir "$DD/pi-mono/.pi/agent" \
  --provider vllm --model agent --judge-model agent \
  --tools read,bash --max-turns 300 --max-concurrency 4 --limit 92 \
  --runtime-context-level level3 --pi-thinking-level high \
  --node-max-old-space-size-mb 8192 \
  && echo "[d92] retry92 done @ $(date +%m%d-%H:%M)" || echo "[d92] retry92 FAILED"

echo "[d92] 收尾 /tmp: $(df -h /tmp | tail -1)"
pkill -f "[v]llm.entrypoints.*--port 8000"
echo "D92_ALL_DONE @ $(date +%m%d-%H:%M)"
