#!/usr/bin/env bash
# ScaleSeek 复现环境变量 —— 在计算节点上 `source setup_env.sh` 加载。
# （source 而非 bash，这样 export 的变量留在当前 shell）

# --- 核心路径 ---
export REPO=/data/rech/mofengra/ScaleSeek
export DATA=/data/rech/mofengra/data
export CKPT=/data/rech/mofengra/checkpoints
export DATASETS=/data/rech/mofengra/datasets

# --- HF 缓存（千万别放 ~ 下，与 GrepSeek 共享）---
export HF_HOME=$DATA/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

# --- 语料（与 GrepSeek 共享 wiki-18）---
export CORPUS_DIR=$DATA/wiki_18_corpus
export CORPUS_FILE=$CORPUS_DIR/wiki_corpus.jsonl

# --- BM25 索引（eval 前需先运行 scripts/build_bm25_index.py）---
export BM25_INDEX_DIR=$DATA/bm25_index

# --- 推理服务 (vLLM) ---
export LLM_HOST=127.0.0.1
export LLM_PORT=8000
export LLM_MODEL=agent
export OPENAI_API_KEY=dummy  # vLLM doesn't need a real key; pyserini imports openai at module load

# --- 编译 / 训练相关（A5000 = sm_86）---
export FLASH_ATTN_CUDA_ARCHS=86

# --- 创建目录 ---
mkdir -p "$DATA" "$CKPT" "$HF_HOME" "$HF_HUB_CACHE" "$BM25_INDEX_DIR" "$DATASETS"

echo "ScaleSeek env loaded:"
echo "  REPO=$REPO"
echo "  DATA=$DATA"
echo "  DATASETS=$DATASETS"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  BM25_INDEX_DIR=$BM25_INDEX_DIR"
echo "  LLM=$LLM_HOST:$LLM_PORT  model=$LLM_MODEL"
cd "$REPO" || echo "warn: REPO 目录不存在: $REPO"

conda activate scaleseek
# 并发 sbatch 作业同时 source 此文件；仅在显式要求时刷新 requirements，避免并发写坏文件。
if [ "${SCALESEEK_FREEZE:-0}" = "1" ]; then
  pip freeze > requirements.txt
fi
