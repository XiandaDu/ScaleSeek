#!/usr/bin/env bash
# 在 Nibi 上构建 ScaleSeek 的 Python 环境。
#
#   bash scripts/setup_venv.sh            # 建 venv 并装全部 pinned 依赖
#   bash scripts/setup_venv.sh --recreate # 删掉重建
#
# ── 为什么用 uv 而不是 module load python + pip ──────────────────────────────
# Alliance 的 gentoo 版 Python 不接受 manylinux wheel（`pip debug --verbose`
# 只列出 linux_x86_64 标签，42 个，零个 manylinux）。PyPI 上几乎所有二进制包
# 都是 manylinux，于是 pip 会报 "No matching distribution"：
#     faiss-cpu==1.14.3  -> (from versions: 1.12.0)
#     cuda-bindings==13.3.1 -> (from versions: none)
# 这正是 Alliance 自建 wheelhouse 的原因，但那个 wheelhouse 里 torch 只有
# 1.9.1，没有 vllm / faiss / flash-attn，撑不起这个栈。
#
# uv 自带的独立 CPython（python-build-standalone）是标准 glibc 构建，认
# manylinux，因此能直接装 requirements.txt 里的精确版本。
#
# uv cache 与 venv 必须在同一文件系统（都放 /scratch），这样 uv 用硬链接而不是
# 复制，装十几 GB 的 torch/vLLM 才不会被 /scratch 的小文件写入拖死。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/scratch/a32du/venvs/scaleseek
UV=/scratch/a32du/bin/uv
export UV_PYTHON_INSTALL_DIR=/scratch/a32du/uv/python
export UV_CACHE_DIR=/scratch/a32du/uv/cache
LOG=$REPO/logs/setup_venv.$(date +%Y%m%d-%H%M%S).log
mkdir -p "$REPO/logs" "$(dirname "$VENV")" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

[ "${1:-}" = "--recreate" ] && { echo "[venv] 删除 $VENV"; rm -rf "$VENV"; }

# nvcc（flash-attn 等源码编译）与 java（pyserini 的 Lucene 索引）仍需 module。
# 不 load python —— venv 自带解释器，load 反而会往 PATH 里塞一个不认 manylinux 的 python。
module load StdEnv/2023 cuda/12.9 java/21

if [ ! -x "$UV" ]; then
  echo "[venv] 安装 uv -> $UV"
  mkdir -p "$(dirname "$UV")"
  tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/uv.XXXXXX")
  curl -sLo "$tmp/uv.tar.gz" \
    https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar xzf "$tmp/uv.tar.gz" -C "$tmp"
  install -m755 "$tmp"/uv-*/uv "$tmp"/uv-*/uvx "$(dirname "$UV")"/
  rm -rf "$tmp"
fi
echo "[venv] $($UV --version)"

"$UV" python install 3.12 2>&1 | tail -1 || true

if [ ! -x "$VENV/bin/python" ]; then
  echo "[venv] 创建 $VENV (uv 独立 CPython 3.12)"
  "$UV" venv --python 3.12 "$VENV"
fi

# 确认这个解释器真的认 manylinux —— 不然后面几百个包会一个个失败
"$VENV/bin/python" - <<'PY'
import sys, sysconfig
plat = sysconfig.get_platform()
print(f"  interpreter {sys.version.split()[0]}  platform {plat}")
PY

echo "[venv] 日志 -> $LOG"

# 第 1 步：eval 栈（旧集群的精确 freeze —— torch/vLLM/transformers/pyserini/faiss）
echo "[venv] 1/2 安装 requirements.txt ($(wc -l < "$REPO/requirements.txt") 个包)"
"$UV" pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt" 2>&1 \
  | tee -a "$LOG" | tail -5

# 第 2 步：verl 训练栈（ray/peft/tensordict/…）。requirements.txt 里没有这些，
# 因为那份 freeze 来自只跑评测的机器。用 constraint 挡住解析器顺手改动 torch。
echo "[venv] 2/2 安装 requirements-verl.txt（训练栈，constraint 保护核心 pin）"
"$UV" pip install --python "$VENV/bin/python" \
  --constraint "$REPO/constraints-core.txt" \
  -r "$REPO/requirements-verl.txt" 2>&1 | tee -a "$LOG" | tail -5

echo
echo "[venv] 校验："
# verl 是 vendored 的，不在 site-packages 里，校验时要把它加进路径
export PYTHONPATH="$REPO:/scratch/a32du/scaleseek/official-baselines/grepseek/verl"
"$VENV/bin/python" - <<'PY'
import importlib, sys
ok = True
for mod in ("torch", "transformers", "datasets", "vllm", "pyserini", "ray",
            "faiss", "peft", "tensordict", "verl"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {mod:14s} {type(e).__name__}: {e}")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  cuda ok  {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
    else:
        print("  cuda 不可用（登录节点上正常；计算节点上则是问题）")
except Exception as e:
    print(f"  cuda 检查失败: {e}")
sys.exit(0 if ok else 1)
PY

echo "[venv] 完成 -> $VENV"
echo "       后续 source setup_env.sh 会自动激活它。"
