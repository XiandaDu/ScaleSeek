#!/usr/bin/env bash
#SBATCH -J p2_official_recover
#SBATCH --account=def-amirhk_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
#
# Score an official-harness run whose evaluation finished but whose
# post-processing died. No model, no GPU -- normalize + metrics only.
#
#   sbatch -J p2_recover_dci    --export=ALL,METHOD=dci    sbatch/p2_official_recover.sbatch
#   sbatch -J p2_recover_drdci  --export=ALL,METHOD=dr_dci sbatch/p2_official_recover.sbatch
#
# Why this keeps being needed: the harnesses run for hours-to-days and then hand
# off to a few lines of shell that have now failed three separate ways --
# a relative --out_dir writing into the official checkout (grepseek 7291, 2d17h),
# `find | head` raising SIGPIPE under pipefail (dci 7340 / dr_dci 7341, exit 141),
# and an id set compared against the wrong reference file. Each time the
# expensive part had already succeeded. Recovering is minutes; re-running is days.
set -euo pipefail
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'code=$?; cleanup_tmpdir; advance_lane; exit $code' EXIT

METHOD=${METHOD:?export METHOD=dci|dr_dci|rise|grepseek}
RAW=$REPO/$RESULTS_DIR/${METHOD}_raw
OUT=$REPO/$RESULTS_DIR/${DS}_${METHOD}.jsonl
RUNREC=$REPO/$RESULTS_DIR/${DS}_${METHOD}.run.json
CAP=$DATA/dcilite_datasets/${DS}_cap1500.jsonl

# The capped harnesses are scored against the subset; grepseek against the split.
case "$METHOD" in
  # The Pi harness family names the id "query_id"; grepseek emits "id".
  dci|dr_dci|rise) REF=$CAP; EXPECT=(--expect-ids "$DATA/dcilite_datasets/${DS}_cap1500.manifest.json"); EXTRA=(--id-field query_id --prediction-field final_text --finish-field run_status --tool-calls-field tool_metrics.call_count);;
  grepseek)        REF=$DS_NORM; EXPECT=(); EXTRA=(--id-prefix "${DS}_");;
  *) echo "FATAL: unknown METHOD=$METHOD"; exit 2;;
esac

[ -f "$RUNREC" ] || { echo "FATAL: no run record at $RUNREC"; exit 4; }
if [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name '*.jsonl' -print -quit); fi
[ -n "${INPUT:-}" ] && [ -f "$INPUT" ] || { echo "FATAL: no raw jsonl under $RAW"; exit 4; }

echo "== $METHOD: $INPUT ($(wc -l < "$INPUT") rows) vs $REF ($(wc -l < "$REF") rows) =="
"$PY" "$REPO/scripts/normalize_official_results.py" --method "$METHOD" \
  --input "$INPUT" --run-record "$RUNREC" \
  --dataset-manifest "$DS_MANIFEST" --dataset-jsonl "$REF" \
  "${EXTRA[@]}" --output "$OUT"

sane "$OUT"
"$PY" "$REPO/scripts/compute_metrics.py" --results "$OUT" \
  --dataset "$DS" --agent "$METHOD" "${EXPECT[@]}" \
  --out "${OUT%.jsonl}.metrics.json"
echo "ALL_DONE ${DS}_$METHOD (recovered)"
