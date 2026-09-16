#!/usr/bin/env bash
# 在 /data/xianda 的 4×A100 单机上构建 ScaleSeek 的 Python 环境（venv 名为 ScaleSeek）。
#
#   bash scripts/setup_venv_local.sh            # 建 venv 并装全部依赖
#   bash scripts/setup_venv_local.sh --recreate # 删掉重建
#
# 与 Nibi 的 scripts/setup_venv.sh 的差异（都是这台机器逼出来的，不是偏好）：
#   * 驱动 570.133 = CUDA 12.8。requirements.txt 冻结的 torch 2.11.0+cu130 / vllm 0.23.0(cu130)
#     在这里 import 即死（libcudart.so.13 找不到 / 驱动太旧）。改装 PyTorch 官方 cu129 轮子和
#     vLLM 官方 wheels.vllm.ai 的 0.23.0+cu129 构建 —— CUDA 12.x 小版本兼容，实测 A100 可用。
#     版本号（torch 2.11.0 / vllm 0.23.0 / transformers 5.12.1 / datasets 5.0.0 / pyserini 1.6.0）
#     与 constraints-core.txt 完全一致，只是 CUDA 后缀不同。
#   * 系统没有 java：Temurin 21 解压到 $SCALESEEK_DATA/jdk（setup_env.sh 负责 PATH）。
#   * flash-attn 没有 torch 2.11 的预编译轮子，用系统 nvcc 12.3 编（主版本与 cu129 一致，
#     torch cpp_extension 只警告不报错）。252 核 / 1.9T 内存，MAX_JOBS=32 约 25 分钟。
#     它是硬依赖：vendored verl 的 FSDP 路径 use_remove_padding=true 走 varlen 内核。
#   * flash-linear-attention：Qwen3.5 的 gated-delta-net 层用它的 triton 内核，
#     没有会回落到 transformers 的纯 torch 实现（能跑，慢）。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/data/xianda
VENV=$DATA_ROOT/venvs/ScaleSeek
export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/uv-pythons
export UV_CACHE_DIR=$DATA_ROOT/uv-cache
export TMPDIR=$DATA_ROOT/tmp
LOG=$REPO/logs/setup_venv_local.$(date +%Y%m%d-%H%M%S).log
mkdir -p "$REPO/logs" "$(dirname "$VENV")" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$TMPDIR"
command -v uv >/dev/null || { echo "uv 不在 PATH（~/.local/bin/uv）"; exit 1; }

[ "${1:-}" = "--recreate" ] && { echo "[venv] 删除 $VENV"; rm -rf "$VENV"; }

uv python install 3.12 2>&1 | tail -1 || true
if [ ! -x "$VENV/bin/python" ]; then
  echo "[venv] 创建 $VENV (uv CPython 3.12, prompt=ScaleSeek)"
  uv venv --python 3.12 --prompt ScaleSeek "$VENV"
fi
PY=$VENV/bin/python
UVI="uv pip install --python $PY"

echo "[venv] 1/5 torch 2.11.0 cu129"
$UVI torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
     --index-url https://download.pytorch.org/whl/cu129 2>&1 | tee -a "$LOG" | tail -2

echo "[venv] 2/5 vLLM 0.23.0+cu129（constraints 保住 torch）"
$UVI --constraint "$REPO/constraints-core.txt" "vllm==0.23.0+cu129" \
     --find-links https://wheels.vllm.ai/0.23.0/cu129/vllm/ 2>&1 | tee -a "$LOG" | tail -2

echo "[venv] 3/5 eval 栈核心 pin + verl 训练栈"
$UVI --constraint "$REPO/constraints-core.txt" \
     transformers==5.12.1 datasets==5.0.0 pyserini==1.6.0 faiss-cpu==1.14.3 \
     huggingface_hub==1.20.1 tokenizers==0.22.2 safetensors==0.8.0 hf-xet \
     py-spy==0.4.2 pytest PyYAML python-dotenv openai anthropic==0.111.0 \
     ninja packaging flash-linear-attention==0.4.1 \
     -r "$REPO/requirements-verl.txt" 2>&1 | tee -a "$LOG" | tail -2

echo "[venv] 4/5 flash-attn 2.8.3（源码编译，sm_80）"
if ! $PY -c "import flash_attn" 2>/dev/null; then
  export CUDA_HOME=/usr/local/cuda-12.3 PATH=/usr/local/cuda-12.3/bin:$PATH
  export MAX_JOBS=32 NVCC_THREADS=4 FLASH_ATTN_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST=8.0
  $UVI --no-build-isolation --no-cache --constraint "$REPO/constraints-core.txt" \
       "flash-attn==2.8.3" 2>&1 | tee -a "$LOG" | tail -2
fi

echo "[venv] 5/5 校验"
export PYTHONPATH="$REPO:$DATA_ROOT/scaleseek_data/official-baselines/grepseek/verl"
export JAVA_HOME=$DATA_ROOT/scaleseek_data/jdk/jdk-21.0.12.1+1 PATH=$DATA_ROOT/scaleseek_data/jdk/jdk-21.0.12.1+1/bin:$PATH
export OPENAI_API_KEY=dummy
$PY - <<'PYEOF'
import importlib, sys
ok = True
for mod in ("torch", "transformers", "datasets", "vllm", "pyserini", "ray",
            "faiss", "peft", "tensordict", "verl", "flash_attn", "fla"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {mod:14s} {type(e).__name__}: {e}")
import torch
print(f"  torch cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
sys.exit(0 if ok else 1)
PYEOF
echo "[venv] 完成 -> $VENV；source setup_env.sh 会自动激活"
