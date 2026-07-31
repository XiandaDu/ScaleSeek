#!/usr/bin/env bash
# ScaleSeek GRPO RL training launch script.
#
# Prerequisites:
#   1. source setup_env.sh           (sets REPO, DATA, BM25_INDEX_DIR, etc.)
#   2. python scripts/prepare_rl_data.py --out_dir $DATA/rl_data
#   3. vLLM + verl installed in the scaleseek conda env
#
# Required env vars (set before running, or via setup_env.sh):
#   REPO            ScaleSeek repo root
#   BM25_INDEX_DIR  directory containing /index (built by build_bm25_index.py)
#
# Optional:
#   SCALESEEK_MODEL_PATH  SFT checkpoint path [default: base Qwen3-8B]
#   SCALESEEK_TRAIN_FILES [default: $DATA/rl_data/train.jsonl]
#   SCALESEEK_VAL_FILES   [default: $DATA/rl_data/dev.jsonl]
#   SCALESEEK_OUTPUT_DIR  [default: $REPO/checkpoints/rl/<timestamp>]
#   NPROC                 number of GPUs [default: all visible]
#   WANDB_API_KEY + WANDB_ENTITY  (both needed to enable W&B logging)
#
# Example:
#   source setup_env.sh
#   export SCALESEEK_MODEL_PATH=$CKPT/sft/global_step_200/huggingface
#   bash scripts/run_rl.sh
#
# Extra CLI args are forwarded to main_ppo, e.g.:
#   bash scripts/run_rl.sh actor_rollout_ref.actor.optim.lr=2e-6

set -euo pipefail

# --- repo root check ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALESEEK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export SCALESEEK_ROOT

if [[ ! -f "${SCALESEEK_ROOT}/setup_env.sh" ]]; then
    echo "error: run from the ScaleSeek repo root or set SCALESEEK_ROOT" >&2
    exit 1
fi

# --- verl path (use GrepSeek's vendored verl) ---
GREPSEEK_ROOT="${GREPSEEK_ROOT:-$(dirname "${SCALESEEK_ROOT}")/grepseek}"
export GREPSEEK_VERL_DIR="${GREPSEEK_ROOT}/verl"
if [[ ! -d "${GREPSEEK_VERL_DIR}/verl/trainer" ]]; then
    echo "error: verl not found at ${GREPSEEK_VERL_DIR}" >&2
    echo "       Set GREPSEEK_ROOT to the grepseek repo directory." >&2
    exit 1
fi
export PYTHONPATH="${SCALESEEK_ROOT}:${GREPSEEK_VERL_DIR}:${PYTHONPATH:-}"

# --- BM25 index ---
: "${BM25_INDEX_DIR:?Please source setup_env.sh or set BM25_INDEX_DIR}"
if [[ ! -d "${BM25_INDEX_DIR}/index" ]]; then
    echo "error: Lucene index not found at ${BM25_INDEX_DIR}/index" >&2
    echo "       Run: python scripts/build_bm25_index.py" >&2
    exit 1
fi
export BM25_INDEX_DIR

# --- RL data files ---
DATA_DIR="${DATA:-$(dirname "${SCALESEEK_ROOT}")/data}"
SCALESEEK_TRAIN_FILES="${SCALESEEK_TRAIN_FILES:-${DATA_DIR}/rl_data/train.jsonl}"
SCALESEEK_VAL_FILES="${SCALESEEK_VAL_FILES:-${DATA_DIR}/rl_data/dev.jsonl}"
if [[ ! -f "${SCALESEEK_TRAIN_FILES}" ]]; then
    echo "error: train file not found: ${SCALESEEK_TRAIN_FILES}" >&2
    echo "       Run: python scripts/prepare_rl_data.py --out_dir ${DATA_DIR}/rl_data" >&2
    exit 1
fi
export SCALESEEK_TRAIN_FILES SCALESEEK_VAL_FILES

# --- model / output / GPUs ---
export SCALESEEK_MODEL_PATH="${SCALESEEK_MODEL_PATH:-Qwen/Qwen3.5-9B}"
[[ "${SCALESEEK_MODEL_PATH}" == "Qwen/Qwen3.5-9B" ]] && \
    echo "note: SCALESEEK_MODEL_PATH unset — initializing RL from BASE model (no SFT warm-start)"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NPROC="${NPROC:-4}"
ULYSSES_SP="${ULYSSES_SP:-2}"   # must divide NPROC

export SCALESEEK_OUTPUT_DIR="${SCALESEEK_OUTPUT_DIR:-${SCALESEEK_ROOT}/checkpoints/rl/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${SCALESEEK_OUTPUT_DIR}"

# --- caches ---
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$$}"
mkdir -p "${HF_HUB_CACHE}" "${TRITON_CACHE_DIR}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — incompatible with
# vLLM's CuMemAllocator hybrid engine.
export TOKENIZERS_PARALLELISM=true
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # pyserini needs a non-empty key at import

# --- W&B ---
if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
    LOGGER='[console,wandb]'; WANDB_ENABLE=true
    export WANDB_PROJECT="${WANDB_PROJECT:-scaleseek_grpo}"
else
    LOGGER='[console]'; WANDB_ENABLE=false
fi

# --- sanity checks ---
python -c "
import sys
sys.path.insert(0, '${SCALESEEK_ROOT}')
from train.agent_loop import ScaleSeekAgentLoop
print('ScaleSeekAgentLoop registered.')
from train.reward import scaleseek_reward
print('scaleseek_reward importable.')
from eval.bm25_retriever import BM25Retriever
print('BM25Retriever importable.')
"

echo "============================================================"
echo "ScaleSeek GRPO training"
echo "model:       ${SCALESEEK_MODEL_PATH}"
echo "train:       ${SCALESEEK_TRAIN_FILES}"
echo "val:         ${SCALESEEK_VAL_FILES}"
echo "BM25 index:  ${BM25_INDEX_DIR}/index"
echo "output:      ${SCALESEEK_OUTPUT_DIR}"
echo "GPUs:        ${NPROC}  (ulysses_sp=${ULYSSES_SP})"
echo "logger:      ${LOGGER}"
echo "============================================================"

python -m verl.trainer.main_ppo \
    --config-path "${SCALESEEK_ROOT}/train/config" \
    --config-name grpo_trainer \
    "hydra.searchpath=[file://${GREPSEEK_VERL_DIR}/verl/trainer/config]" \
    trainer.n_gpus_per_node="${NPROC}" \
    ray_kwargs.ray_init.num_gpus="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    trainer.logger="${LOGGER}" \
    wandb.enable="${WANDB_ENABLE}" \
    "$@"

echo "done. checkpoints → ${SCALESEEK_OUTPUT_DIR}"
