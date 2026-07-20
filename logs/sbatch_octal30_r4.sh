#!/bin/bash
#SBATCH --job-name=ss_o30d
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=230G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal30r4_%j.log
#
# 补跑 dci-lite popqa n=500 里失败的 142 题。原 run（0718，job 7193 lane C）：
#   correct 173 / total 500 → acc .346，但 failed_runs=116，另有 26 题报 400。
#   把 142 个失败剔掉后是 173/358 = .483 —— 差 14 个点，所以 .346 这个数不能直接用。
#
# 两类失败，都已定位：
#  ① 116 × node `write ENOBUFS`：内核 socket 缓冲被打爆。当时同一个节点上还并排跑着
#    9B(TP=2) + 4B×2 三条 lane，dci-lite 的 node 子进程抢不到 socket 缓冲。
#    → 本作业**独占整机**，且并发从 4 降到 3。
#  ② 26 × HTTP 400 "requested 8192 output tokens > max context 32768"：agent 自己要
#    8192 输出，撞上 32768 的窗口上限。
#    → --max-model-len 提到 40960（4B 在 24G 卡上放得下）。
#
# 只跑这 142 题（单独的 retry142.jsonl），跑完再和原 358 条合并，不重跑已成功的。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
SS=/data/rech/mofengra/ScaleSeek
DD=/data/rech/mofengra/dr_dci_official
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12

# ⚠ .env 把 OPENAI_BASE_URL 钉死在 127.0.0.1:8000。上一个作业(7194)想用别的端口、
#   在 source 之后 export 覆盖 —— **没生效**，node 子进程仍打 8000，142 题全 "Connection error"。
#   所以这里老老实实就用 8000，不再跟它斗。
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 40960 > $SS/logs/o30d_4b.log 2>&1 &
ok=0; for t in $(seq 1 180); do curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[o30d] FATAL: 8000 没起来，放弃"; exit 1; }
echo "[o30d] 4B@8000 up (max-model-len 40960) @ $(date +%m%d-%H:%M)"

cd $DD || exit 1
set -a; source .env 2>/dev/null; set +a
export DCI_VIEW_CACHE_ROOT=$DD/.view_cache_dcilite_retry
export DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses
export DCI_JUDGE_MAX_OUTPUT_TOKENS=2048
echo "[o30d] dci-lite retry142 start @ $(date +%m%d-%H:%M)"
$DD/.venv/bin/python scripts/bcplus_eval/run_bcplus_eval.py \
  --dataset "$DD/data/dci-bench/data/popqa/retry142.jsonl" \
  --output-root "$DD/outputs/qa/popqa_dcilite_vllm4b_retry142" \
  --corpus-dir "$DD/corpus/wiki_corpus" \
  --package-dir "$DD/pi-mono/packages/coding-agent" \
  --agent-dir "$DD/pi-mono/.pi/agent" \
  --provider vllm --model agent --judge-model agent \
  --tools read,bash --max-turns 300 --max-concurrency 3 --limit 142 \
  --runtime-context-level level3 --pi-thinking-level high \
  --node-max-old-space-size-mb 8192 \
  && echo "[o30d] retry142 done @ $(date +%m%d-%H:%M)" || echo "[o30d] retry142 FAILED"

pkill -f "[v]llm.entrypoints.*--port 8000"
echo "O30D_ALL_DONE @ $(date +%m%d-%H:%M)"
