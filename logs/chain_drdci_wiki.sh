#!/bin/bash
# #13 DR-DCI wiki 六集官方协议（各 50 题，pull→wiki-18 E5 检索，vllm/4B agent + judge）
# 门控：等 GPU0 搜索链完（GPU0_CHAIN_ALL_DONE）→ GPU0 腾空、E5 RAM 释放。
# 复用我们自建的 E5 索引（21M，与 wiki_corpus.jsonl 行序对齐，已校验）作为
# searchr1_wiki18_dci_server 的后端——零下载、零适配器。
case "$(hostname)" in octal30*|octal35*) : ;; *) echo "REFUSING $(hostname)"; exit 1;; esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
SS=/data/rech/mofengra/ScaleSeek
DD=/data/rech/mofengra/dr_dci_official
cd $DD || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1

until grep -q "GPU0_CHAIN_ALL_DONE" $SS/logs/chain_gpu0.log 2>/dev/null; do sleep 600; done
echo "[wiki] GPU0 freed, starting @ $(date +%m%d-%H:%M)"

# --- wiki-18 E5 检索端点 @18011 (GPU0) ---
CUDA_VISIBLE_DEVICES=0 setsid nohup $DD/.venv/bin/python \
  tools/dense_retriever/searchr1_wiki18_dci_server.py \
  --index-path /data/rech/mofengra/data/e5_index/index.faiss \
  --corpus-path /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \
  --model-name-or-path intfloat/e5-base-v2 \
  --port 18011 --encoder-device cuda:0 --encoder-fp16 --max-top-k 1000 \
  > $SS/logs/wiki_retriever.log 2>&1 &
# --- 4B agent @8000 (GPU0) ---
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.55 --max-model-len 32768 > $SS/logs/vllm8000.log 2>&1 &

for t in $(seq 1 300); do
  r=0; a=0
  curl -sf -m 5 localhost:18011/health >/dev/null 2>&1 && r=1
  curl -sf -m 5 localhost:8000/v1/models >/dev/null 2>&1 && a=1
  [ $r = 1 ] && [ $a = 1 ] && break
  sleep 10
done
[ $r = 1 ] && [ $a = 1 ] || { echo "[wiki] servers not up (retr=$r agent=$a), ABORT"; exit 1; }
echo "[wiki] retriever+agent up @ $(date +%m%d-%H:%M)"

set -a; source .env 2>/dev/null; set +a
export DCI_VIEW_CACHE_ROOT=$DD/.view_cache_wiki
export DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses
export DCI_JUDGE_MAX_OUTPUT_TOKENS=2048
export DCI_PULL_BASE_URL=http://127.0.0.1:18011/retrieve
export DCI_PULL_DOCUMENT_BASE_URL=http://127.0.0.1:18011/document
mkdir -p corpus/wiki18_empty

run_ds() {  # $1=dataset dir name
  local ds=$1
  echo "[wiki] $ds start @ $(date +%m%d-%H:%M)"
  $DD/.venv/bin/python scripts/bcplus_eval/run_bcplus_eval.py \
    --dataset "$DD/data/dci-bench/data/$ds/test.jsonl" \
    --output-root "$DD/outputs/qa/${ds}_wiki18_e5_vllm4b" \
    --corpus-dir "$DD/corpus/wiki18_empty" \
    --package-dir "$DD/pi-mono/packages/coding-agent" \
    --agent-dir "$DD/pi-mono/.pi/agent" \
    --provider vllm --model agent --judge-model agent \
    --tools read,bash,pull --pull-terminal-tools --pull-backend local \
    --pull-base-url "$DCI_PULL_BASE_URL" \
    --pull-layout root --pull-prompt-mode rank_aware \
    --pull-materialization-mode root_flat_disclosed \
    --pull-min-top-k 300 --pull-max-top-k 600 --pull-max-queries 1 \
    --pull-preview-mode ranked --pull-preview-limit 20 \
    --max-turns 300 --max-concurrency 8 --limit 50 \
    --runtime-context-level level3 --pi-thinking-level high \
    --full-corpus-doc-count 21015324 --node-max-old-space-size-mb 8192 \
    && echo "[wiki] $ds done @ $(date +%m%d-%H:%M)" || echo "[wiki] $ds FAILED"
}

for ds in nq triviaqa hotpotqa 2wikimultihopqa musique bamboogle; do run_ds $ds; done
pkill -f "[s]earchr1_wiki18_dci_server"
pkill -f "[v]llm.*8000"
echo "DRDCI_WIKI_ALL_DONE @ $(date +%m%d-%H:%M)"
