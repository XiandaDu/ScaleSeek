set -euo pipefail
cd /home/a32du/ScaleSeek
module load StdEnv/2023 cuda/13.2 >/dev/null 2>&1   # 匹配 torch 的 cu130
export CUDA_HOME=${EBROOTCUDA:?}
export PATH=$CUDA_HOME/bin:$PATH
export UV_CACHE_DIR=/scratch/a32du/uv/cache
# grepseek TRAINING_ENV.md 的教训：$(nproc) 会让并行 nvcc OOM，必须限流
export MAX_JOBS=4 NVCC_THREADS=4
export FLASH_ATTN_CUDA_ARCHS=90          # H100 sm_90，只编这一个架构省时间
echo "nvcc: $(nvcc --version | tail -2 | head -1)"
echo "g++ : $(g++ --version | head -1)"
/scratch/a32du/bin/uv pip install --python /scratch/a32du/venvs/scaleseek/bin/python \
    --no-build-isolation --constraint constraints-core.txt flash-attn
