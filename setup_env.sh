#!/usr/bin/env bash
# ScaleSeek 环境变量 —— Nibi (Alliance Canada / SHARCNET)。
# 在登录节点或计算节点上 `source setup_env.sh` 加载（source 而非 bash）。
#
# 存储分工（Nibi 配额：home 50G / scratch 1T / project 931G）：
#   $REPO     /home     仅代码，有备份，文件数配额紧
#   $DATA     /scratch  语料、索引、HF 缓存、RL 数据 —— 可重建，会被定期清理
#   $CKPT     /scratch  训练 checkpoint
#   $RESULTS  /project  评测结果与报告 —— 有备份，别放大文件
#   $TMPDIR   $SLURM_TMPDIR  节点本地 NVMe (11T)，作业结束即消失

# --- 核心路径 ---
export REPO=/home/a32du/ScaleSeek
export SCRATCH_ROOT=/scratch/a32du/scaleseek
export PROJECT_ROOT=/project/def-amirhk/a32du/scaleseek

export DATA=$SCRATCH_ROOT/data
export CKPT=$SCRATCH_ROOT/checkpoints
export DATASETS=$DATA/datasets
export RESULTS=$PROJECT_ROOT/results
export OFFICIAL_ROOT=$SCRATCH_ROOT/official-baselines
export SCALESEEK_VENV=/scratch/a32du/venvs/scaleseek

# --- HF 缓存（绝不能放 home，会撑爆 50G / 500K 文件配额）---
export HF_HOME=$DATA/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

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
export OPENAI_API_KEY=dummy   # vLLM 不需要真 key；pyserini 在 import 时会读它

# --- 模块 ---
# 只 load nvcc 与 java；不 load python/3.12 —— venv 用的是 uv 的独立 CPython，
# Alliance 的 gentoo python 不认 manylinux wheel（详见 scripts/setup_venv.sh），
# 把它放进 PATH 只会造成混淆。
#   cuda -> nvcc，flash-attn 等源码编译需要
#   java -> pyserini 的 Lucene 索引运行时需要
#
# cuda/13.2 而非 12.9：torch 是 2.11.0+cu130，flash-attn 也是对着 13.2 编的。
# 载 12.9 时 torch 的 cpp_extension 会警告
#   "No CUDA runtime is found, using CUDA_HOME=.../cudacore/12.9.1"
# 运行时无害（torch 自带 CUDA 运行时），但任何 JIT 编译的扩展都会对着错误版本的
# 头文件编。保持与 wheel 的 CUDA 大版本一致。
if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 cuda/13.2 java/21 >/dev/null 2>&1 || \
    echo "warn: module load 失败，检查 module avail"
fi
export PATH=/scratch/a32du/bin:$PATH   # uv
export UV_PYTHON_INSTALL_DIR=/scratch/a32du/uv/python
export UV_CACHE_DIR=/scratch/a32du/uv/cache

# --- venv ---
if [ -x "$SCALESEEK_VENV/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$SCALESEEK_VENV/bin/activate"
  export PY="$SCALESEEK_VENV/bin/python"
else
  export PY=python
  echo "warn: venv 不存在于 $SCALESEEK_VENV —— 先跑 bash scripts/setup_venv.sh"
fi

# --- 临时目录（务必用节点本地 NVMe，不要用 /tmp 或 home）---
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
# hf_transfer 已废弃（huggingface_hub 会 FutureWarning），改用 Xet 高速传输
export HF_XET_HIGH_PERFORMANCE=1

# --- GPU 架构 / 拓扑 ---
# Nibi 是 H100 = sm_90（旧集群 A5000 是 sm_86，flash-attn 编译目标必须改）。
export FLASH_ATTN_CUDA_ARCHS=90
export TORCH_CUDA_ARCH_LIST=9.0
# 注意：不要设 NCCL_P2P_DISABLE=1。那是 A5000/3090 无 P2P 时的绕行方案，
# H100 节点有 NVLink，禁掉 P2P 会让多卡训练慢数倍。

# MIG 切片上多卡通信不可用，且只会看到 1 个逻辑设备。
if nvidia-smi -L 2>/dev/null | grep -q MIG; then
  export SCALESEEK_ON_MIG=1
  export NPROC=1
  # Slurm 在 MIG 上把 CUDA_VISIBLE_DEVICES 设成 MIG UUID。torch 认这个格式，但
  # vLLM 0.23 会拿它去 int() 解析，启动即崩：
  #   ValueError: invalid literal for int() with base 10: 'MIG-ea872388-...'
  # device cgroup 已把可见设备限制成那一个 MIG 实例，改写成序号是安全的。
  # 放在 setup_env.sh 而非 sbatch/common.sh，交互会话才同样受益。
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    echo "note: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> 0 (vLLM 无法解析 MIG UUID)"
    export CUDA_VISIBLE_DEVICES=0
  fi
  echo "note: 检测到 MIG 切片 —— 强制单卡 (NPROC=1)，多卡训练请申请整卡 gpu:h100:N"
else
  export SCALESEEK_ON_MIG=0
fi

# --- JVM 堆（pyserini / Lucene）---
# Alliance 的 java module 会设 JAVA_TOOL_OPTIONS=-Xmx2g，建 21M passage 的索引会 OOM。
#
# 但**检索**不需要大堆：Lucene 用 MMapDirectory / MemorySegmentIndexInput，索引
# 走 mmap 活在 OS page cache 里，不在 JVM 堆上。大堆只对建索引有用。
# 早先按 mem/4 给（110G 作业 -> 27G 堆）是错的：在训练作业里白占预算，挤压
# vLLM、FSDP actor/ref 和 dataloader worker。
# 默认封顶 8G；建索引的作业显式设 SCALESEEK_JVM_HEAP=32g。
_heap="${SCALESEEK_JVM_HEAP:-}"
if [ -z "$_heap" ]; then
  if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    _heap=$(( SLURM_MEM_PER_NODE / 1024 / 4 ))
    [ "$_heap" -lt 2 ] && _heap=2
    [ "$_heap" -gt 8 ] && _heap=8
    _heap="${_heap}g"
  else
    _heap=4g
  fi
fi
export JAVA_TOOL_OPTIONS="-Xmx${_heap}"
unset _heap

# 线程数跟随 Slurm 分配，避免 OpenMP 抢占整机核心
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export TOKENIZERS_PARALLELISM=true

# --- verl（来自 grepseek 仓库的 vendored 版本）---
export GREPSEEK_ROOT=$OFFICIAL_ROOT/grepseek
export GREPSEEK_VERL_DIR=$GREPSEEK_ROOT/verl
export SCALESEEK_ROOT=$REPO
# Alliance 的 StdEnv 会往 PYTHONPATH 里塞 /cvmfs/.../custom/python/site-packages，
# 那里面是 gentoo python 编译的包，和 venv 里 uv 的独立 CPython ABI 不兼容，
# 混进来会造成难查的 import 崩溃。这里显式剔除。
_clean_pp=""
IFS=':' read -ra _pp_parts <<< "${PYTHONPATH:-}"
for _p in "${_pp_parts[@]}"; do
  case "$_p" in
    ""|/cvmfs/soft.computecanada.ca/custom/python/site-packages*) ;;
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

# --- 创建目录 ---
mkdir -p "$DATA" "$CKPT" "$HF_HUB_CACHE" "$DATASETS" "$BM25_INDEX_DIR" \
         "$RESULTS" "$OFFICIAL_ROOT" "$SFT_TRAJ_DIR" "$SFT_CKPT_DIR" \
         "$RL_DATA_DIR" "$RL_CKPT_DIR" "$REPO/logs" 2>/dev/null

cat <<EOF
ScaleSeek env loaded (Nibi):
  host=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-none}
  REPO=$REPO
  DATA=$DATA
  CKPT=$CKPT
  RESULTS=$RESULTS
  TMPDIR=$TMPDIR
  PY=$PY
EOF

cd "$REPO" 2>/dev/null || echo "warn: REPO 目录不存在: $REPO"
