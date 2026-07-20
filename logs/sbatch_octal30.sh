#!/bin/bash
#SBATCH --job-name=ss_o30
#SBATCH --partition=rali
#SBATCH --nodelist=octal[30]
#SBATCH --gres=gpu:rtx_a5000:4
# ⚠ 不写 --cpus-per-task 只给 1 核(=2 线程)：faiss 穷举 21M 向量 + grep 14G 语料会被饿死
#   （2026-07-17 job 7172 实测 E5 lane 掉到 80 题/小时、GPU 0%）。
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/rech/mofengra/ScaleSeek/logs/sbatch_octal30_%j.log
#
# octal30 复活后接手剩下的活。⚠ 本节点是 4×RTX_A5000 **24G**，不是 octal40/41 的 L40S 48G，
# 显存预算完全不同，参数不能照抄：
#   - 9B GrepSeek 单卡放不下（权重 18G，剩 4.8G KV 撑不住 32k 上下文）→ 必须 TP=2 占两张卡。
#   - 4B 服务若照抄 0.80×24=19.2G，E5 编码器（约 1.3G + 激活）会挤不进同一张卡
#     —— 这正是 job 7176 agentir precompute 全 OOM 的同一个坑 → E5 lane 降到 0.72。
#
# 分工（与 octal40 job 7176 零重叠，那边在跑 grepseek全量/dci/agentir，都不碰这里的格子）：
#   GPU0+1 (TP=2, 9B)  hotpotqa_grepseek        ← 主表最后一个缺格
#   GPU2   (4B, E5)    hotpotqa_scaleseek_e5 / hotpotqa_search_o1_e5
#   GPU3   (4B→3B)     bcp_ircot / bcp_dci / bcp_search_r1  ← BCP 第二域从 5 agent 补到 8
set -u
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
# octal30 是 sm_86、flashinfer JIT 缓存早就编好，本来碰不到 octal40/41 那两个坑；
# 但设上无害且能防同样的静默失败（服务死了 run_eval 照跑，每行 api_error）。
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
GS=alireza7/GrepSeek-Qwen3.5-9B-GRPO
SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
BCP_CORPUS=$DATASETS/browsecomp_plus/corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"
waitp(){ for t in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done
  echo "[o30] FATAL: 端口 $1 服务未起来，放弃该 lane（不写 api_error 垃圾）"; return 1; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[o30] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[o30] $out api_error $e/$t 过半 —— 判为无效，不出 metrics"; return 1; }; }
# $1=out $2...=完整 run_eval 命令；METRIC_EXTRA 由各 lane 设好（title-index-db / bcp qrels）
cell(){ local out=$1; shift
  [ -f results/$out.metrics.json ] && { echo "[o30] $out 已完成，跳过"; return; }
  echo "[o30] $out start @ $(date +%m%d-%H:%M)"
  "$@" --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $METRIC_EXTRA --out results/$out.metrics.json \
    && echo "[o30] $out done @ $(date +%m%d-%H:%M)" || echo "[o30] $out FAILED"; }

# ---- Lane A：GPU0+GPU1 张量并行跑 9B（单张 24G 放不下 18G 权重 + 32k KV）----
( export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
  CUDA_VISIBLE_DEVICES=0,1 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $GS \
    --served-model-name grepseek --port 8031 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 32768 > logs/o30_9b.log 2>&1 &
  waitp 8031 || exit 1; echo "[o30] 9B(TP=2) up @ $(date +%m%d-%H:%M)"
  METRIC_EXTRA="--title-index-db $TDB"
  cell hotpotqa_grepseek env CUDA_VISIBLE_DEVICES=0,1 $PY -m eval.run_eval \
    --dataset hotpotqa --agent grepseek -n 1500 --concurrency 16 --corpus-path $CORPUS \
    --grepseek-port 8031 --grepseek-tokenizer $GS
  pkill -f "[v]llm.entrypoints.*--port 8031" ) &
LA=$!

# ---- Lane B：GPU2 跑 hotpotqa 的两格 E5 变体 ----
# ⚠ 0.72 而非 0.80：E5 编码器要和 4B 挤同一张 24G 卡，得给它留 ~6.7G。
( export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
  export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
  CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name agent --port 8033 --enable-auto-tool-choice \
    --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.72 \
    --max-model-len 32768 > logs/o30_4b_8033.log 2>&1 &
  waitp 8033 || exit 1; echo "[o30] 4B@8033 up @ $(date +%m%d-%H:%M)"
  METRIC_EXTRA="--title-index-db $TDB"
  cell hotpotqa_scaleseek_e5 env CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval \
    --dataset hotpotqa --agent scaleseek -n 1500 --concurrency 16 --port 8033 \
    --retrieval-backend e5 --max-tokens 2048
  cell hotpotqa_search_o1_e5 env CUDA_VISIBLE_DEVICES=2 $PY -m eval.run_eval \
    --dataset hotpotqa --agent search_o1 -n 1500 --concurrency 16 --port 8033 \
    --retrieval-backend e5 --bm25-top-k 5 --max-tokens 2048
  pkill -f "[v]llm.entrypoints.*--port 8033" ) &
LB=$!

# ---- Lane C：GPU3 把 BCP 第二域从 5 个 agent 补到 8 个 ----
# BCP 用自己的 BM25 索引和语料；--max-tokens 4096（冒烟发现 2048 被 thinking 吃光，parse_err>50%）。
# rag_e5 / agentir 不做：BCP 语料没有 E5 索引也没有 agentir 索引。
( export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
  METRIC_EXTRA="$QRELS"
  CUDA_VISIBLE_DEVICES=3 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name agent --port 8034 --enable-auto-tool-choice \
    --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
    --max-model-len 32768 > logs/o30_4b_8034.log 2>&1 &
  waitp 8034 || exit 1; echo "[o30] 4B@8034 up @ $(date +%m%d-%H:%M)"
  cell bcp_ircot_830 env CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval \
    --dataset browsecomp_plus --agent ircot -n 830 --concurrency 16 --port 8034 \
    --retrieval-backend bm25 --bm25-top-k 3 --max-turns 6 --max-tokens 4096
  cell bcp_dci_830 env CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval \
    --dataset browsecomp_plus --agent dci -n 830 --concurrency 4 --port 8034 \
    --corpus-path $BCP_CORPUS --max-tokens 4096
  # 换 3B search_r1：两个模型塞不进一张 24G 卡，只能串行换服务
  pkill -f "[v]llm.entrypoints.*--port 8034"; sleep 20
  CUDA_VISIBLE_DEVICES=3 setsid nohup $PY -m vllm.entrypoints.openai.api_server --model $SR1 \
    --served-model-name search_r1 --port 8035 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 > logs/o30_3b.log 2>&1 &
  waitp 8035 || exit 1; echo "[o30] 3B@8035 up @ $(date +%m%d-%H:%M)"
  cell bcp_search_r1_830 env CUDA_VISIBLE_DEVICES=3 $PY -m eval.run_eval \
    --dataset browsecomp_plus --agent search_r1 -n 830 --concurrency 16 \
    --search-r1-port 8035 --retrieval-backend bm25 --bm25-top-k 3 --max-turns 4
  pkill -f "[v]llm.entrypoints.*--port 8035" ) &
LC=$!

# ⚠ 只 wait 这三条 lane。裸 wait 会把 serve 起的 vLLM 也等进去，它们永不退出
#   （2026-07-16 job 7173 就这样跑完后空占 4 张卡 9.5 小时）。
wait $LA $LB $LC
echo "O30_ALL_DONE @ $(date +%m%d-%H:%M)"
