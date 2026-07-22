#!/usr/bin/env bash
#SBATCH -J p0_dci_corpus
#SBATCH -p rali
#SBATCH -c 16
#SBATCH --mem=12G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# DCI prerequisite: expand Wiki-18 into sharded per-document .txt files with
# the official expand_wiki18_jsonl_to_files.py (plain DCI greps files, not a
# monolithic jsonl). Four resumable range workers; requeue-safe.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; cleanup_tmpdir; advance_lane; exit $code' EXIT
OUT=$DATA/dci_wiki_corpus
mkdir -p "$OUT"
TOTAL=21015324
Q=5253831   # TOTAL/4 exactly

cd "$OFFICIAL_ROOT/dr_dci"
PIDS=()
for i in 0 1 2 3; do
  uv run python scripts/qa/expand_wiki18_jsonl_to_files.py \
    --input "$CORPUS_FILE" --output-dir "$OUT" \
    --start-offset $((i * Q)) --limit $Q \
    --resume --progress-every 500000 \
    > "$REPO/logs/dci_expand_$i.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
tail -2 "$REPO"/logs/dci_expand_*.log
[ "$FAIL" = 0 ] || { echo "FATAL: at least one expand range failed"; exit 1; }
echo "ALL_P0_DCI_CORPUS_DONE"
