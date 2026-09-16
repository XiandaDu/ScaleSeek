#!/usr/bin/env bash
# ScaleSeek 环境变量。`source setup_env.sh`（source 而非 bash，export 才留在当前 shell）。
#
# 两个站点，按主机自动选择（也可 SCALESEEK_SITE=local|nibi 强制）：
#   local  /data/xianda 的 4×A100-80G 单机（无 Slurm）。仿照 VIEScore2/setup_env.sh：
#          所有缓存、venv、Claude 配置都放 /data/xianda 下，不碰 ~。
#   nibi   Alliance Canada / SHARCNET 的 Slurm 集群（home 50G / scratch 1T / project 931G）。

if [ -z "${SCALESEEK_SITE:-}" ]; then
  if [ -d /scratch/a32du ]; then SCALESEEK_SITE=nibi
  elif [ -d /data/xianda ]; then SCALESEEK_SITE=local
  else echo "warn: 未识别的主机，按 local 处理"; SCALESEEK_SITE=local; fi
fi
export SCALESEEK_SITE

case "$SCALESEEK_SITE" in
# ═══════════════════════════════════════════════════════════════════════════
local)
  DATA_ROOT=/data/xianda
  export REPO=$DATA_ROOT/ScaleSeek
  export SCALESEEK_DATA=$DATA_ROOT/scaleseek_data     # 语料、索引、数据集、结果、官方仓库
  export DATA=$SCALESEEK_DATA/data
  export CKPT=$DATA_ROOT/checkpoints/scaleseek        # 与 VIEScore2 的 CHECKPOINT_BASE 同级
  export DATASETS=$DATA/datasets
  export RESULTS=$SCALESEEK_DATA/results
  export OFFICIAL_ROOT=$SCALESEEK_DATA/official-baselines
  export SCALESEEK_VENV=$DATA_ROOT/venvs/ScaleSeek    # scripts/setup_venv_local.sh 建

  # --- Claude Code 配置放仓库本地（复制自 VIEScore2/setup_env.sh）---
  export CLAUDE_CONFIG_DIR=$REPO/.claude

  # --- 缓存：全部放 /data/xianda ---
  export HF_HOME=$DATA_ROOT/hf_cache                  # 与 VIEScore2 共享（Allenda/* 已在此）
  export HF_HUB_CACHE=$HF_HOME/hub
  export PIP_CACHE_DIR=$DATA_ROOT/pip-cache
  export UV_CACHE_DIR=$DATA_ROOT/uv-cache
  export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/uv-pythons
  export TMPDIR=$DATA_ROOT/tmp
  export TRITON_CACHE_DIR=$SCALESEEK_DATA/.triton
  export TORCHINDUCTOR_CACHE_DIR=$SCALESEEK_DATA/.inductor
  export RAY_TMPDIR=$DATA_ROOT/tmp/ray

  # --- Java（pyserini/Lucene）：Temurin 21 解压在数据盘，系统没有 java ---
  export JAVA_HOME=$SCALESEEK_DATA/jdk/jdk-21.0.12.1+1
  export PATH=$JAVA_HOME/bin:$PATH

  # --- CUDA ---
  # 驱动 570 = CUDA 12.8。仓库 pin 的 torch 2.11.0+cu130 在这台机器上加载不了
  # （libcuda 太旧），所以 venv 装的是 torch/vLLM 的 cu129 构建（12.x 小版本兼容）。
  # 系统 nvcc 是 12.3（主版本一致，torch 的 cpp_extension 只会 warning），用它编 flash-attn。
  export CUDA_HOME=/usr/local/cuda-12.3
  export PATH=$CUDA_HOME/bin:$PATH
  export FLASH_ATTN_CUDA_ARCHS=80       # A100 = sm_80
  export TORCH_CUDA_ARCH_LIST=8.0
  # 这四张是 PCIe A100，卡间没有 P2P（nvidia-smi topo -p2p r 全是 NS）。不关掉 NCCL 会在
  # FSDP 第一次 broadcast 时死于 "Cuda failure 217 'peer access is not supported between
  # these two devices'"（2026-09-04 冒烟）。H100/NVLink 节点上千万别设，见 nibi 分支。
  export NCCL_P2P_DISABLE=1
  # 关了 P2P 仍不够：Ray 给每个 worker 只暴露一张卡（CUDA_VISIBLE_DEVICES=单卡）时，NCCL 2.28
  # 的 SHM 传输走 cuMem 句柄，同样死于 transport/shm.cc:590 的 cuda 217。四卡复现实验里
  # 只有 NCCL_CUMEM_ENABLE=0（或更慢的 NCCL_SHM_DISABLE=1）能过。
  export NCCL_CUMEM_ENABLE=0
  export SCALESEEK_ON_MIG=0
  export NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | grep -c .)}"
  ;;
# ═══════════════════════════════════════════════════════════════════════════
nibi)
  export REPO=/home/a32du/ScaleSeek
  export SCRATCH_ROOT=/scratch/a32du/scaleseek
  export PROJECT_ROOT=/project/def-amirhk/a32du/scaleseek
  export DATA=$SCRATCH_ROOT/data
  export CKPT=$SCRATCH_ROOT/checkpoints
  export DATASETS=$DATA/datasets
  export RESULTS=$PROJECT_ROOT/results
  export OFFICIAL_ROOT=$SCRATCH_ROOT/official-baselines
  export SCALESEEK_VENV=/scratch/a32du/venvs/scaleseek

  # HF 缓存绝不能放 home（50G / 500K 文件配额）
  export HF_HOME=$DATA/hf_cache
  export HF_HUB_CACHE=$HF_HOME/hub

  # 只 load nvcc 与 java；不 load python（venv 用 uv 的独立 CPython，见 scripts/setup_venv.sh）。
  # cuda/13.2 匹配 torch 2.11.0+cu130。
  if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 cuda/13.2 java/21 >/dev/null 2>&1 || \
      echo "warn: module load 失败，检查 module avail"
  fi
  export PATH=/scratch/a32du/bin:$PATH   # uv
  export UV_PYTHON_INSTALL_DIR=/scratch/a32du/uv/python
  export UV_CACHE_DIR=/scratch/a32du/uv/cache

  # 临时目录务必用节点本地 NVMe
  export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER}"
  export TRITON_CACHE_DIR=$TMPDIR/triton
  export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor

  # H100 = sm_90。不要设 NCCL_P2P_DISABLE（H100 有 NVLink）。
  export FLASH_ATTN_CUDA_ARCHS=90
  export TORCH_CUDA_ARCH_LIST=9.0

  # MIG 切片：只有 1 个逻辑设备且无卡间通信；vLLM 0.23 无法解析 MIG UUID。
  if nvidia-smi -L 2>/dev/null | grep -q MIG; then
    export SCALESEEK_ON_MIG=1
    export NPROC=1
    if [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
      echo "note: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> 0 (vLLM 无法解析 MIG UUID)"
      export CUDA_VISIBLE_DEVICES=0
    fi
    echo "note: 检测到 MIG 切片 —— 强制单卡 (NPROC=1)"
  else
    export SCALESEEK_ON_MIG=0
  fi
  if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; fi
  ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# 两站点共用
# ═══════════════════════════════════════════════════════════════════════════
mkdir -p "$TMPDIR" "${PIP_CACHE_DIR:-$TMPDIR}" 2>/dev/null

# --- 密钥（.env 不入库；至少包含 HF_TOKEN）---
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
export HF_XET_HIGH_PERFORMANCE=1

# --- 语料与索引 ---
export CORPUS_DIR=$DATA/wiki_18_corpus
export CORPUS_FILE=$CORPUS_DIR/wiki_corpus.jsonl
export CORPUS_MANIFEST=$CORPUS_DIR/corpus_manifest.json
export BM25_INDEX_DIR=$DATA/bm25_index
export E5_INDEX_DIR=$DATA/e5_index
export QWEN3_EMB_INDEX_DIR=$DATA/qwen3_embedding_4b_index

# --- 推理服务 (vLLM) ---
export LLM_HOST=127.0.0.1
export LLM_PORT=8000
export LLM_MODEL=agent
export LLM_TOKENIZER=Qwen/Qwen3.5-9B
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # vLLM 不需要真 key；pyserini import 时会读它

# --- venv ---
if [ -x "$SCALESEEK_VENV/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$SCALESEEK_VENV/bin/activate"
  export PY="$SCALESEEK_VENV/bin/python"
  hash -r
else
  export PY=python
  echo "warn: venv 不存在于 $SCALESEEK_VENV —— 先跑 bash scripts/setup_venv.sh (nibi) 或 scripts/setup_venv_local.sh (local)"
fi

# --- JVM 堆（pyserini / Lucene）---
# 检索走 mmap 不占堆，默认封顶 8G；建索引的作业显式设 SCALESEEK_JVM_HEAP=32g。
_heap="${SCALESEEK_JVM_HEAP:-}"
if [ -z "$_heap" ]; then
  if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    _heap=$(( SLURM_MEM_PER_NODE / 1024 / 4 ))
    [ "$_heap" -lt 2 ] && _heap=2
    [ "$_heap" -gt 8 ] && _heap=8
    _heap="${_heap}g"
  else
    _heap=8g
  fi
fi
export JAVA_TOOL_OPTIONS="-Xmx${_heap}"
unset _heap
export TOKENIZERS_PARALLELISM=true

# --- verl（来自 grepseek 仓库的 vendored 版本）---
export GREPSEEK_ROOT=$OFFICIAL_ROOT/grepseek
export GREPSEEK_VERL_DIR=$GREPSEEK_ROOT/verl
export SCALESEEK_ROOT=$REPO
# Alliance 的 StdEnv 会往 PYTHONPATH 塞 gentoo 编译的 site-packages，与 venv ABI 不兼容，剔除。
_clean_pp=""
IFS=':' read -ra _pp_parts <<< "${PYTHONPATH:-}"
for _p in "${_pp_parts[@]}"; do
  case "$_p" in
    ""|/cvmfs/soft.computecanada.ca/custom/python/site-packages*|"$REPO"|"$GREPSEEK_VERL_DIR") ;;
    *) _clean_pp="${_clean_pp:+$_clean_pp:}$_p";;
  esac
done
export PYTHONPATH="$REPO:$GREPSEEK_VERL_DIR${_clean_pp:+:$_clean_pp}"
unset _clean_pp _pp_parts _p

# --- 官方 baseline 模型 pin（沿用 Phase-1 冻结版本）---
export GEN_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export SR1_7B=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3
export SR1_7B_REV=395b18f1fecee52f1b51fb22f898c220f0a08ec3
export SR1_14B=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3
export SR1_14B_REV=d65f11c88d3c129a01466f0c154aaad7d9b09225
export GREPSEEK_MODEL=alireza7/GrepSeek-Qwen3.5-9B-GRPO
export GREPSEEK_REV=a79563970cfdd2ced3cc5fde481737d0ebea6fa4
export POPQA_RAW=$DATASETS/popqa/test.jsonl
export POPQA_NORM=$DATASETS/popqa/test.normalized.jsonl
export POPQA_MANIFEST=$DATASETS/popqa/manifest.json

# --- SFT / RL 阶段路径 ---
export SFT_TRAJ_DIR=$DATA/sft
export SFT_CKPT_DIR=$CKPT/sft
export RL_DATA_DIR=$DATA/rl_data
export RL_CKPT_DIR=$CKPT/rl
# HF 上的中间产物（合并后的 HF 权重；FSDP 分片与优化器状态未上传）
export SFT_HF_MODEL=Allenda/ScaleSeek-Qwen3.5-9B-SFT
export RL_STEP190_HF_MODEL=Allenda/ScaleSeek-Qwen3.5-9B-GRPO-step190

mkdir -p "$DATA" "$CKPT" "$HF_HUB_CACHE" "$DATASETS" "$BM25_INDEX_DIR" \
         "$RESULTS" "$OFFICIAL_ROOT" "$SFT_TRAJ_DIR" "$SFT_CKPT_DIR" \
         "$RL_DATA_DIR" "$RL_CKPT_DIR" "$REPO/logs" "$TRITON_CACHE_DIR" 2>/dev/null

cat <<EOT
ScaleSeek env loaded (site=$SCALESEEK_SITE):
  host=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-all}
  REPO=$REPO
  DATA=$DATA
  CKPT=$CKPT
  RESULTS=$RESULTS
  HF_HOME=$HF_HOME
  TMPDIR=$TMPDIR
  PY=$PY  ($("$PY" --version 2>&1 | cut -d' ' -f2))
EOT

cd "$REPO" 2>/dev/null || echo "warn: REPO 目录不存在: $REPO"
