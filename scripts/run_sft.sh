#!/usr/bin/env bash
# ScaleSeek cold-start SFT launcher (verl SFT trainer, verl>=0.8).
#
# Fine-tunes a small Qwen3 student on the cold-start trajectories produced by
# scripts/generate_sft_data.py, emitting an HF-format checkpoint that
# scripts/run_rl.sh consumes via SCALESEEK_MODEL_PATH.
#
# Pipeline:
#   1. python scripts/make_smoke_corpus.py --out-dir .smoke --build-index      # smoke corpus+index
#   2. python scripts/generate_sft_data.py --questions ... --out TRAJ.jsonl     # teacher trajectories
#   3. python -m train.sft_dataset --in TRAJ.jsonl --out TRAIN.parquet          # -> verl parquet
#   4. bash scripts/run_sft.sh                                                  # this script
#
# Env vars (all have smoke-friendly defaults):
#   SCALESEEK_SFT_BASE      base/student model            [Qwen/Qwen3-1.7B]
#   SCALESEEK_SFT_TRAIN     train parquet                 [.smoke/sft_train.parquet]
#   SCALESEEK_SFT_VAL       val parquet                   [= train]
#   SCALESEEK_SFT_OUTPUT    checkpoint output dir         [.smoke/sft_ckpt]
#   SCALESEEK_SFT_EPOCHS    epochs                        [3]
#   SCALESEEK_SFT_MAXLEN    max sequence length           [4096]
#   SCALESEEK_SFT_BSZ       global train batch size       [8]
#   NPROC                   GPUs                           [1]
#   PYBIN                   python interpreter            [.venv/bin/python or python]
#
# NOTE: pad_mode must be `no_padding`. The vendored verl's FSDP engine hard-asserts
# it (verl/workers/engine/fsdp/transformer_impl.py:883 —
# `assert pad_mode == DatasetPadMode.NO_PADDING`), so the older
# "single-GPU friendly" combination of pad_mode=right + use_remove_padding=false
# fails at the first training step with `AssertionError: pad_mode right not supported`.
# no_padding packs variable-length sequences, which needs flash-attn's varlen
# kernels — flash-attn is therefore a hard requirement here, not an optimization.
# On the cluster, source setup_env.sh and point SCALESEEK_SFT_* at $DATA/$CKPT;
# raise NPROC and batch size.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYBIN="${PYBIN:-$( [ -x "${ROOT}/.venv/bin/python" ] && echo "${ROOT}/.venv/bin/python" || echo python )}"
BASE="${SCALESEEK_SFT_BASE:-Qwen/Qwen3-1.7B}"
TRAIN="${SCALESEEK_SFT_TRAIN:-.smoke/sft_train.parquet}"
VAL="${SCALESEEK_SFT_VAL:-${TRAIN}}"
OUTPUT="${SCALESEEK_SFT_OUTPUT:-.smoke/sft_ckpt}"
EPOCHS="${SCALESEEK_SFT_EPOCHS:-3}"
MAXLEN="${SCALESEEK_SFT_MAXLEN:-4096}"
BSZ="${SCALESEEK_SFT_BSZ:-8}"
NPROC="${NPROC:-1}"

if [[ ! -f "${TRAIN}" ]]; then
  echo "error: train parquet not found: ${TRAIN}" >&2
  echo "       build it: ${PYBIN} -m train.sft_dataset --in <trajectories.jsonl> --out ${TRAIN}" >&2
  exit 1
fi

echo "[run_sft] base=${BASE}  train=${TRAIN}"
echo "[run_sft] output=${OUTPUT}  epochs=${EPOCHS}  maxlen=${MAXLEN}  bsz=${BSZ}  nproc=${NPROC}"

# verl multi-turn SFT over the `messages` column (MultiTurnSFTDataset is the default
# when data.custom_cls.path is null). ignore_input_ids_mismatch handles the Qwen3
# per-turn <think> template mismatch documented in verl's sft_trainer_engine.yaml.
"${PYBIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN}" \
  data.val_files="${VAL}" \
  data.messages_key=messages \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=false \
  data.train_batch_size="${BSZ}" \
  data.micro_batch_size_per_gpu=2 \
  data.max_length="${MAXLEN}" \
  data.truncation=right \
  data.ignore_input_ids_mismatch=true \
  model.path="${BASE}" \
  model.trust_remote_code=true \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  engine.use_torch_compile=false \
  optim.lr=1e-5 \
  checkpoint.save_contents='[model,hf_model,extra]' \
  trainer.default_local_dir="${OUTPUT}" \
  trainer.project_name=scaleseek_sft \
  trainer.experiment_name=smoke \
  trainer.total_epochs="${EPOCHS}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.logger='[console]' \
  trainer.n_gpus_per_node="${NPROC}" \
  "$@"

echo "[run_sft] done. HF checkpoint under ${OUTPUT}/**/huggingface"
