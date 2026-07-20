#!/bin/bash
#SBATCH --job-name=ss_o30b
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal30r2_%j.log
#
# octal30 第二轮。⚠ L40S 节点(octal40/41)已被别人整机占满，A5000 是现在唯一的算力，
# 所以 9B 必须在 24G 卡上跑通 —— 单卡不可行，见下。
#
# 【为什么必须 TP=2】GrepSeek-9B: 32 层 / 4 个 KV 头 / head_dim 256
#   KV = 2*32*4*256*2B = 128 KB/token → 32k 上下文单条序列就要 4.00 GB。
#   单卡 24G：权重 18G，即使 util 拉到 0.92 也只剩 4G KV = 只能跑 1 条并发 → 不可行。
#   TP=2：权重每卡 9G，0.85*24=20.4G → 每卡 11.4G KV，两卡合计 22.8G ≈ 5-6 条 32k 并发。
#
# 【上一轮 TP=2 为什么挂死】job 7184 的 lane A 停在 "vLLM is using nccl==2.28.9" 后再无输出，
#   GPU0/1 显存仅 421MiB 却 100% 利用率 —— NCCL 自旋死锁。A5000 没有 NVLink，P2P 走 PCIe，
#   RTX/A 系消费级卡上 vLLM 的 custom all-reduce + NCCL P2P 是已知会挂的组合。
#   → NCCL_P2P_DISABLE=1 + --disable-custom-all-reduce，退回走 shared memory 归约。
#   仍失败则自动重试一次并加 NCCL_SHM_DISABLE=1（退到最保守的路径）。
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
SS=/data/rech/mofengra/ScaleSeek
DD=/data/rech/mofengra/dr_dci_official
cd $SS
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
BCP_CORPUS=$DATASETS/browsecomp_plus/corpus.jsonl
BCP_E5=/data/rech/mofengra/data/bcp_e5_index
TDB=/data/rech/mofengra/data/corpus_title_index.db
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"
waitp(){ for t in $(seq 1 $2); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o30b] 端口 $1 服务未起来（等了 $(( $2 * 10 / 60 )) 分钟）"; return 1; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o30b] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o30b] $out api_error $e/$t 过半 —— 判为无效，不出 metrics"; return 1; }; }
cell(){ local out=$1; shift
  [ -f results/$out.metrics.json ] && { echo "[o30b] $out 已完成，跳过"; return; }
  echo "[o30b] $out start @ $(date +%m%d-%H:%M)"
  "$@" --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $METRIC_EXTRA --out results/$out.metrics.json \
    && echo "[o30b] $out done @ $(date +%m%d-%H:%M)" || echo "[o30b] $out FAILED"; }

# ================= Lane A：GPU0+1 TP=2 跑 9B —— 主表最后一个缺格 =================
( export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
  boot9b(){ CUDA_VISIBLE_DEVICES=0,1 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
      --served-model-name grepseek --port 8031 --tensor-parallel-size 2 --disable-custom-all-reduce \
      --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/o30b_9b$1.log 2>&1 & }
  boot9b ""
  if ! waitp 8031 150; then
    echo "[o30b] TP=2 第一次未起来，加 NCCL_SHM_DISABLE=1 重试 @ $(date +%m%d-%H:%M)"
    pkill -f "[v]llm.entrypoints.*--port 8031"; sleep 30
    export NCCL_SHM_DISABLE=1; boot9b "_retry"
    waitp 8031 150 || { echo "[o30b] FATAL: TP=2 两次都起不来，放弃 lane A（不写 api_error 垃圾）"; exit 1; }
  fi
  echo "[o30b] 9B(TP=2) up @ $(date +%m%d-%H:%M)"
  METRIC_EXTRA="--title-index-db $TDB"
  cell hotpotqa_grepseek env CUDA_VISIBLE_DEVICES=0,1 $PY -m eval.run_eval \
    --dataset hotpotqa --agent grepseek -n 1500 --concurrency 12 --corpus-path $CORPUS \
    --grepseek-port 8031 --grepseek-tokenizer $GS
  pkill -f "[v]llm.entrypoints.*--port 8031" ) &
LA=$!

# ============ Lane B：GPU2 —— 补最后一格 E5 轴，然后把 E5 铺到 BCP 第二个域 ============
# ⚠ 顺序是刻意的：先跑完 4B 的格子 → 杀掉 4B → **裸卡**建 BCP 的 E5 索引 → 再起 4B。
#   建索引要 GPU 编码 10 万条文档，跟 4B 抢显存必 OOM（job 7176 的 agentir 就是这么全挂的）。
( srv4b(){ CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name agent --port 8033 --enable-auto-tool-choice \
    --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.72 \
    --max-model-len 32768 > logs/o30b_4b_$1.log 2>&1 & }
  # --- B1: popqa 标准集最后一格 E5（wiki 索引）---
  export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
  export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
  srv4b wiki; waitp 8033 150 || exit 1; echo "[o30b] 4B@8033(wiki) up @ $(date +%m%d-%H:%M)"
  METRIC_EXTRA=""
  cell popqa_full_scaleseek_e5 env CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval \
    --dataset popqa_full --agent scaleseek -n 1500 --concurrency 16 --port 8033 \
    --retrieval-backend e5 --max-tokens 2048
  # --- B2: 腾空卡，建 BCP 的 E5 索引（10 万条，wiki 是 2100 万，这个很快）---
  pkill -f "[v]llm.entrypoints.*--port 8033"; sleep 20
  if [ ! -f $BCP_E5/index.faiss ]; then
    echo "[o30b] 建 BCP E5 索引 start @ $(date +%m%d-%H:%M)"
    CUDA_VISIBLE_DEVICES=2 $PY scripts/build_e5_index.py --corpus $BCP_CORPUS --out $BCP_E5 \
      --device cuda --batch-size 512 --max-length 256 --index-type sq8_flat \
      && echo "[o30b] BCP E5 索引 done @ $(date +%m%d-%H:%M)" || echo "[o30b] BCP E5 索引 FAILED"
  fi
  [ -f $BCP_E5/index.faiss ] || { echo "[o30b] 无 BCP E5 索引，跳过 B3"; exit 1; }
  # --- B3: BCP 上的 E5 变体 —— 让"检索器轴"在第二个域也有对照 ---
  export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
  export E5_INDEX_DIR=$BCP_E5
  srv4b bcp; waitp 8033 150 || exit 1; echo "[o30b] 4B@8033(bcp) up @ $(date +%m%d-%H:%M)"
  METRIC_EXTRA="$QRELS"
  cell bcp_rag_e5_830 env CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval \
    --dataset browsecomp_plus --agent bm25_rag -n 830 --concurrency 16 --port 8033 \
    --retrieval-backend e5 --bm25-top-k 5 --max-tokens 4096
  cell bcp_scaleseek_e5_830 env CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval \
    --dataset browsecomp_plus --agent scaleseek -n 830 --concurrency 16 --port 8033 \
    --retrieval-backend e5 --max-tokens 4096
  pkill -f "[v]llm.entrypoints.*--port 8033" ) &
LB=$!

# ====== Lane C：GPU3 —— #15-3 官方 dci-agent-lite 从 n=50 扩到 n=500 ======
# n=50 的 ±14 F1 抽样误差根本不足以判定"我们自实现的 dci 是否代表官方方法"，这是全项目
# 对标可信度的关键一环。n=50 用了 1h38m（约 2 分钟/题 @并发4）→ n=500 约 16 小时。
# 官方 harness 的 agent 和 judge 都读 .env 里的 OPENAI_BASE_URL=127.0.0.1:8000，端口写死。
( CUDA_VISIBLE_DEVICES=3 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name agent --port 8000 --enable-auto-tool-choice \
    --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
    --max-model-len 32768 > logs/o30b_4b_8000.log 2>&1 &
  waitp 8000 150 || { echo "[o30b] FATAL: 8000 没起来，放弃 lane C"; exit 1; }
  echo "[o30b] 4B@8000 up @ $(date +%m%d-%H:%M)"
  cd $DD || exit 1
  set -a; source .env 2>/dev/null; set +a
  export DCI_VIEW_CACHE_ROOT=$DD/.view_cache_dcilite_n500
  export DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses
  export DCI_JUDGE_MAX_OUTPUT_TOKENS=2048
  echo "[o30b] dci-lite popqa n=500 start @ $(date +%m%d-%H:%M)"
  $DD/.venv/bin/python scripts/bcplus_eval/run_bcplus_eval.py \
    --dataset "$DD/data/dci-bench/data/popqa/test.jsonl" \
    --output-root "$DD/outputs/qa/popqa_dcilite_vllm4b_n500" \
    --corpus-dir "$DD/corpus/wiki_corpus" \
    --package-dir "$DD/pi-mono/packages/coding-agent" \
    --agent-dir "$DD/pi-mono/.pi/agent" \
    --provider vllm --model agent --judge-model agent \
    --tools read,bash --max-turns 300 --max-concurrency 4 --limit 500 \
    --runtime-context-level level3 --pi-thinking-level high \
    --node-max-old-space-size-mb 8192 \
    && echo "[o30b] dci-lite n=500 done @ $(date +%m%d-%H:%M)" || echo "[o30b] dci-lite n=500 FAILED"
  pkill -f "[v]llm.entrypoints.*--port 8000" ) &
LC=$!

# ⚠ 只 wait 三条 lane，不能裸 wait（会把永不退出的 vLLM 也等进去，job 7173 空占卡 9.5h）
wait $LA $LB $LC
echo "O30B_ALL_DONE @ $(date +%m%d-%H:%M)"
