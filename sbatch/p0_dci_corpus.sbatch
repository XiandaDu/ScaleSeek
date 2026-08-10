#!/usr/bin/env bash
#SBATCH -J p0_dci_corpus
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# DCI prerequisite: expand Wiki-18 into per-document .txt files with the
# official expand_wiki18_jsonl_to_files.py (plain DCI greps files, not a
# monolithic jsonl).
#
# Rorqual twist: /scratch has a 1M-FILE quota and this expansion is 21,015,324
# files. Alliance's sanctioned pattern for many-small-file datasets is exactly
# what we do here: expand into node-local $SLURM_TMPDIR, then archive to ONE
# tar.zst on /scratch. p2_dci re-extracts it into its own $SLURM_TMPDIR at job
# start (NVMe also greps far faster than Lustre ever did on RALI).
# Not resumable across requeues (node-local staging dies with the job), but on
# NVMe the whole expansion is hours, not the 4 days it was on RALI.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT
ARCHIVE=$DATA/dci_wiki_corpus.tar.zst
[ -f "$ARCHIVE" ] && { echo "archive already exists: $ARCHIVE"; exit 0; }
OUT=$TMPDIR/dci_wiki_corpus
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
tail -n 2 "$REPO"/logs/dci_expand_*.log
[ "$FAIL" = 0 ] || { echo "FATAL: at least one expand range failed"; exit 1; }

echo "== archive to /scratch (single file, quota-safe) =="
command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
tar -C "$TMPDIR" -cf "$ARCHIVE.partial" -I 'zstd -T8 -3' dci_wiki_corpus
mv "$ARCHIVE.partial" "$ARCHIVE"
ls -lh "$ARCHIVE"
echo "ALL_P0_DCI_CORPUS_DONE"
