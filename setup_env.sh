#!/usr/bin/env bash
# ScaleSeek 复现环境变量 —— Rorqual (Alliance/Calcul Québec) 版。
# 在计算节点或登录节点 `source setup_env.sh` 加载（source 而非 bash）。
#
# 与 RALI 版的结构性差异（cluster-packing-and-decile-scope-Rorqual 分支）：
#   * conda 在 Alliance 集群不受支持 -> module python/3.11 + ~/scaleseek_env venv。
#   * 计算节点无外网 -> 所有下载都在登录节点完成；SLURM 作业里强制 HF 离线模式。
#   * 数据在 /scratch（20TB / 1M 文件配额）；注意 1M 文件配额意味着 DCI 的
#     2100 万文件语料和 RISE 的全量文章语料必须走「$SLURM_TMPDIR 展开 + tar 归档」，
#     不能直接落在 /scratch（见 sbatch/p0_dci_corpus.sbatch、p0_rise_articles.sbatch）。

# --- 模块与 venv（幂等）---
module load python/3.11 java/21 2>/dev/null || true
if [ -f "$HOME/scaleseek_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/scaleseek_env/bin/activate"
else
  echo "warn: ~/scaleseek_env 不存在；先在登录节点运行 build_venv（见分支说明）"
fi

# --- 核心路径 ---
export REPO=$HOME/ScaleSeek
export DATA=/scratch/a32du/data
export CKPT=/scratch/a32du/checkpoints
export DATASETS=/scratch/a32du/datasets

# --- HF 缓存（scratch，不能放 ~：home 只有 50GB）---
export HF_HOME=$DATA/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

# --- 计算节点无外网：作业内强制离线，避免 HF 在 21M 次调用里反复撞超时 ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  # 官方 harness 的 `uv run` 只能复用登录节点 rorqual_login_setup.sh 建好的
  # .venv，绝不能在计算节点尝试联网 resolve/下载。
  export UV_OFFLINE=1
  export UV_PYTHON_PREFERENCE=only-system
fi

# --- 语料（FlashRAG wiki18_100w，21,015,324 段落）---
export CORPUS_DIR=$DATA/wiki_18_corpus
export CORPUS_FILE=$CORPUS_DIR/wiki_corpus.jsonl

# --- BM25 索引（eval 前需先运行 scripts/build_bm25_index.py）---
export BM25_INDEX_DIR=$DATA/bm25_index

# --- 推理服务 (vLLM) ---
export LLM_HOST=127.0.0.1
export LLM_PORT=8000
export LLM_MODEL=agent
export OPENAI_API_KEY=dummy  # vLLM doesn't need a real key; pyserini imports openai at module load

# --- 编译 / 训练相关（H100 = sm_90）---
export FLASH_ATTN_CUDA_ARCHS=90

# --- 创建目录 ---
mkdir -p "$DATA" "$CKPT" "$HF_HOME" "$HF_HUB_CACHE" "$BM25_INDEX_DIR" "$DATASETS"

echo "ScaleSeek env loaded (rorqual):"
echo "  REPO=$REPO"
echo "  DATA=$DATA"
echo "  DATASETS=$DATASETS"
echo "  CORPUS_FILE=$CORPUS_FILE"
echo "  BM25_INDEX_DIR=$BM25_INDEX_DIR"
echo "  LLM=$LLM_HOST:$LLM_PORT  model=$LLM_MODEL"
cd "$REPO" || echo "warn: REPO 目录不存在: $REPO"

# 并发 sbatch 作业同时 source 此文件；仅在显式要求时刷新 requirements，避免并发写坏文件。
# Rorqual 上冻结到 requirements.rorqual.txt，不覆盖 RALI 的 requirements.txt。
if [ "${SCALESEEK_FREEZE:-0}" = "1" ]; then
  pip freeze > requirements.rorqual.txt
fi
