#!/usr/bin/env bash
#SBATCH -J p2_grepseek_recover
#SBATCH --account=def-amirhk_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
#
# Recover the completed-but-misfiled GrepSeek run (job 7291, 2026-08-02).
#
# 7291 evaluated all 14,267 questions over 2d17h and then died with "no grepseek
# output jsonl under results/phase2/grepseek_raw". The eval had in fact
# succeeded: run_official_baseline.py launches the harness with cwd=<official
# checkout>, so the relative --out_dir resolved inside the official repo. The
# predictions are here, complete and with 14,267 unique ids:
#
#   $OFFICIAL_ROOT/grepseek/results/phase2/grepseek_raw/test/predictions.jsonl
#
# The path bug itself is fixed in p2_grepseek.sbatch (RAW is now absolute), so
# this script exists only to rescue the already-computed 7291 output. It is CPU
# only -- no GPU, no model -- but it parses a 156MB jsonl, which is exactly the
# kind of work that must not run on a jump host.
#
#   sbatch sbatch/p2_grepseek_recover.sbatch
set -euo pipefail
source /home/a32du/ScaleSeek/sbatch/common.sh
# advance_lane must be in the trap: this script sits in queue_laneA, so without
# it the lane stops here and everything behind it (p2_dci, p2_agentir_encode)
# never gets submitted.
trap 'code=$?; cleanup_tmpdir; advance_lane; exit $code' EXIT

SRC=$OFFICIAL_ROOT/grepseek/results/phase2/grepseek_raw/test/predictions.jsonl
OUT=$REPO/$RESULTS_DIR/${DS}_grepseek.jsonl

[ -f "$SRC" ] || { echo "FATAL: recovered predictions not found at $SRC"; exit 4; }
echo "== source =="
wc -l "$SRC"

echo "== normalize -> $OUT =="
"$PY" "$REPO/scripts/normalize_official_results.py" --method grepseek \
  --input "$SRC" \
  --run-record "$REPO/$RESULTS_DIR/${DS}_grepseek.run.json" \
  --dataset-manifest "$DS_MANIFEST" --dataset-jsonl "$DS_NORM" \
  --id-prefix "${DS}_" \
  --output "$OUT"

sane "$OUT"

echo "== metrics =="
"$PY" "$REPO/scripts/compute_metrics.py" --results "$OUT" \
  --dataset "$DS" --agent grepseek \
  --out "${OUT%.jsonl}.metrics.json"

# Park the raw output where p2_grepseek would have put it, so a later re-run
# finds it and skips instead of burning another 2d17h.
mkdir -p "$REPO/results/phase2/grepseek_raw/test"
cp -n "$SRC" "$REPO/results/phase2/grepseek_raw/test/predictions.jsonl" || true

echo "ALL_DONE ${DS}_grepseek (recovered from 7291)"
