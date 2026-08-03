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
# 默认不做验证（null）：Qwen3.5 的 4 行 position_ids 只有 dynamic-bsz 打包路径
# 处理正确，验证走的静态路径会踩同一个 rope 崩溃；test_freq 本来就是 -1。
VAL="${SCALESEEK_SFT_VAL:-null}"
OUTPUT="${SCALESEEK_SFT_OUTPUT:-.smoke/sft_ckpt}"
EPOCHS="${SCALESEEK_SFT_EPOCHS:-3}"
# 8192 而非 4096：2026-07-31 生产数据实测 p95≈4.6k / max≈6.3k token，
# 4096 + truncation=right 会把近半数轨迹的最终答案回合切掉。
MAXLEN="${SCALESEEK_SFT_MAXLEN:-8192}"
# 每 GPU 的动态打包 token 预算（详见下方 use_dynamic_bsz 注释）
MAXTOK="${SCALESEEK_SFT_MAXTOK:-16384}"
BSZ="${SCALESEEK_SFT_BSZ:-32}"
NPROC="${NPROC:-1}"
# 序列并行度。grepseek 跑通 Qwen3.5-9B 用的是 SP=NPROC；须整除 NPROC。
ULYSSES_SP="${ULYSSES_SP:-${NPROC}}"

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
#
# use_dynamic_bsz=true is LOAD-BEARING for Qwen3.5, not a throughput knob: the
# model's interleaved RoPE needs the dataset's 4-row position_ids
# (text + 3 vision rows, built whenever the processor is Qwen2VL-family), and
# only the dynamic-bsz token-packing path collates that shape correctly. The
# static path stacks (bs, 4, seq) and feeds the rope 8 rows where it expects 3:
#   RuntimeError: The size of tensor a (3) must match the size of tensor b (8)
# (jobs 18885490; grepseek's own run_sft.sh — same trainer, same student —
# runs dynamic-bsz + offloads + SP=NPROC, mirrored here).
"${PYBIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN}" \
  data.val_files="${VAL}" \
  data.messages_key=messages \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=true \
  data.max_token_len_per_gpu="${MAXTOK}" \
  data.train_batch_size="${BSZ}" \
  data.micro_batch_size_per_gpu=1 \
  data.max_length="${MAXLEN}" \
  data.truncation=right \
  data.ignore_input_ids_mismatch=true \
  model.path="${BASE}" \
  model.trust_remote_code=true \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  model.enable_activation_offload=true \
  engine.use_torch_compile=false \
  engine.param_offload=true \
  engine.optimizer_offload=true \
  engine.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
  optim.lr="${SCALESEEK_SFT_LR:-5e-6}" \
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
