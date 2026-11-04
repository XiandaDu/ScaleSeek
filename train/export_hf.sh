#!/bin/bash
# =====================================================================================
# A GRPO checkpoint -> a pinned local model the eval harness loads like any HF snapshot:
#     $HF_HUB_CACHE/models--local--scaleseek-<run>-step<N>/snapshots/<revision>/
# <revision> = sha1 over the files' sha256s (40 hex, like an HF commit), so a method yaml
# pins exactly these bytes.  Run inside a CPU job:
#     srun ... bash train/export_hf.sh <run> <step>
# =====================================================================================
set -euo pipefail
RUN="$1"; STEP="$2"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup_env.sh >/dev/null && deactivate
bash setup_env.sh --job-train
R="$SCALESEEK_ROOT"; CKPT="$R/train_runs/$RUN/global_step_$STEP/actor"
TMP="$SLURM_TMPDIR/export"; rm -rf "$TMP"
PYTHONPATH="$R/repos/grepseek/verl" "$SLURM_TMPDIR/scaleseek_env/train/bin/python" -m verl.model_merger merge \
    --backend fsdp --local_dir "$CKPT" --target_dir "$TMP"
REV="$(cd "$TMP" && sha256sum $(ls | sort) | sha1sum | cut -c1-40)"
REPO="$R/hf/hub/models--local--scaleseek-$RUN-step$STEP"
mkdir -p "$REPO/snapshots" "$REPO/refs"
rm -rf "$REPO/snapshots/$REV" && cp -r "$TMP" "$REPO/snapshots/$REV" && echo "$REV" > "$REPO/refs/main"
echo "generator: {model: local/scaleseek-$RUN-step$STEP, revision: $REV}"
ls -la "$REPO/snapshots/$REV"
