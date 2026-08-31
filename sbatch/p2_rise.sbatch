#!/usr/bin/env bash
#SBATCH -J p2_rise
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# RISE: official run_rise.py at the frozen commit over the structured Wiki
# article corpus. Index and filename map are built with the official repo
# scripts (bm25s, deliberate k1=1.2 project override recorded in YAML). The
# index/map CLIs are --help-guarded: a flag mismatch exits 4 for interactive
# fixing instead of running with wrong parameters.
#
# Rorqual corpus staging (1M-file /scratch quota): the full 3.2M-article
# structured corpus is assembled in node-local $SLURM_TMPDIR each run —
# raw article tar extracted, p1_rise_toc's persistent candidate TOCs overlaid,
# then the official no-LLM empty-TOC passthrough fills in the rest. The
# retrieval boundary therefore stays the FULL corpus. The harness runs with a
# node-local out-root (its workspace materialization hardlinks from the corpus,
# which requires one filesystem); results are synced to /scratch continuously
# and the complete raw tree is archived at the end.
source /home/a32du/ScaleSeek/sbatch/common.sh

RAW=$REPO/$RESULTS_DIR/rise_raw          # scratch-side synced copy
RAW_LOCAL=$TMPDIR/rise_raw               # node-local live out-root
sync_raw() {
  mkdir -p "$RAW"
  rsync -a --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' \
    "$RAW_LOCAL/" "$RAW/" 2>/dev/null || true
}
trap 'code=$?; stop_vllm; sync_raw; cleanup_tmpdir; advance_lane; exit $code' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))

ARCHIVE=$DATA/rise_wiki_articles.tar.zst
TOC_DIR=$DATA/rise_toc_structured
META=$DATA/rise_wiki_articles_meta
CAND=$META/candidates_${DS}_cap${CAP:-1500}.txt
[ "${CAP:-1500}" = 1500 ] && [ ! -f "$CAND" ] && [ -f "$META/candidates_${DS}.txt" ] && CAND=$META/candidates_${DS}.txt
DOCS=$TMPDIR/rise_wiki_articles_structured
RISE_INDEX=$DATA/rise_bm25_index_wiki_${DS}
MAP=$RISE_INDEX/filename_docid_map.json
mkdir -p "$RESULTS_DIR" "$RAW" "$RAW_LOCAL"
[ -f "$TOC_DIR/audit.jsonl" ] || { echo "FATAL: candidate TOCs missing (run p1_rise_toc)"; exit 4; }
[ -f "$ARCHIVE" ] || { echo "FATAL: $ARCHIVE missing (run p0_rise_articles)"; exit 4; }
[ -f "$CAND" ] || { echo "FATAL: $CAND missing (run p1_rise_toc)"; exit 4; }

start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B
waitp "$PORT" 3600
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export SECTION_MODEL=Qwen/Qwen3.5-9B
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

echo "== stage structured corpus on node-local =="
command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
ART=$TMPDIR/rise_wiki_articles
[ -d "$ART" ] || time tar -C "$TMPDIR" -xf "$ARCHIVE" -I 'zstd -d -T8'
mkdir -p "$DOCS"
echo "-- overlay candidate TOCs from $TOC_DIR"
time cp -a "$TOC_DIR/." "$DOCS/"
echo "-- official empty-TOC passthrough for the remaining articles (no LLM)"
time "$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$DOCS" \
  --candidates "$CAND" --workers 96
N_DOCS=$(ls "$DOCS" | grep -c '\.md$' || true)
echo "structured corpus files: $N_DOCS"

cd "$OFFICIAL_ROOT/rise"
if [ ! -d "$RISE_INDEX" ] || [ -z "$(ls -A "$RISE_INDEX" 2>/dev/null)" ]; then
  uv run python scripts/build_bm25_index.py --help > "$REPO/logs/rise_bm25_help.txt" 2>&1 || exit 4
  uv run python scripts/build_bm25_index.py --corpus "$DOCS" \
    --out "$RISE_INDEX" --k1 1.2 --b 0.75 || {
      echo "FATAL: bm25s index build failed; check flags vs logs/rise_bm25_help.txt"; exit 4; }
fi
if [ ! -f "$MAP" ]; then
  uv run python scripts/build_filename_docid_map.py --help > "$REPO/logs/rise_map_help.txt" 2>&1 || exit 4
  uv run python scripts/build_filename_docid_map.py --corpus "$DOCS" --out "$MAP" || {
      echo "FATAL: filename map build failed; check flags vs logs/rise_map_help.txt"; exit 4; }
fi
cd "$REPO"

# run_official_baseline's rise gate checks the mini-dev row/id counts against a
# manifest with count/unique_ids. The dataset-level manifest describes the FULL
# split, but the sanctioned scope here is the <=1500 subset, so the gate
# manifest is derived from the subset manifest's fingerprinted subset_n. The
# real ID-set check happens in compute_metrics --expect-ids below.
GATE_MANIFEST=$TMPDIR/rise_gate_manifest.json
"$PY" - "$DATA/dcilite_datasets/${DS}_cap${CAP:-1500}.manifest.json" "$GATE_MANIFEST" <<'PYEOF'
import json, sys
sub = json.load(open(sys.argv[1]))
json.dump({"count": sub["subset_n"], "unique_ids": sub["subset_n"],
           "derived_from_subset_manifest": sys.argv[1],
           "subset_ids_sha256": sub["subset_ids_sha256"]},
          open(sys.argv[2], "w"), indent=2)
PYEOF

echo "== continuous scratch sync (results only; workspaces archived at end) =="
( while sleep 1800; do sync_raw; done ) & SYNC_PID=$!

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
"$PY" scripts/run_official_baseline.py rise --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --dataset-manifest "$GATE_MANIFEST" \
  --record $RESULTS_DIR/${DS}_rise.run.json -- \
  --model Qwen/Qwen3.5-9B \
  --mini-dev "$DATA/dcilite_datasets/${DS}_cap${CAP:-1500}.jsonl" \
  --bm25-k 1000 --bm25-top-n-preview 10 --max-turns 100 \
  --bash-truncate-chars 4000 --read-default-limit 2000 \
  --structured-docs --index-dir "$RISE_INDEX" \
  --filename-map "$MAP" --bc-plus-docs "$DOCS" \
  --concurrency "${CONC:-8}" \
  --out-root "$RAW_LOCAL"
stop_vllm
kill "$SYNC_PID" 2>/dev/null || true
sync_raw
echo "== archive full raw tree (incl. workspaces) to scratch =="
tar -C "$TMPDIR" -cf "$RAW/rise_raw_full.tar.zst.partial" -I 'zstd -T8 -3' rise_raw \
  && mv "$RAW/rise_raw_full.tar.zst.partial" "$RAW/rise_raw_full.tar.zst" || true

echo "== normalize + metrics =="
# Pick the aggregate file explicitly, and never pipe find into head.
# `find ... | head -1` sends SIGPIPE upstream when head closes; under
# `set -o pipefail` that exits 141 and kills the job AFTER the evaluation has
# already finished. dci (7340) and dr_dci (7341) both died exactly this way on
# 2026-08-08 with complete 1,500-row outputs on disk. It was also non-deterministic:
# find's first hit can be a per-question events.jsonl rather than the aggregate.
# Read the COMPLETE node-local tree, never the scratch copy: the background
# rsync can capture results.jsonl mid-append and the truncated line kills
# normalize (job 18753890). $RAW remains the fallback for recover-style reruns.
if [ -f "$RAW_LOCAL/results.jsonl" ]; then INPUT=$RAW_LOCAL/results.jsonl
elif [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name '*.jsonl' -printf '%T@ %p\n' | sort -rn | awk 'NR==1{sub(/^[^ ]* /,"");print}'); fi
"$PY" scripts/normalize_official_results.py --method rise \
  --input "$INPUT" \
  --run-record $RESULTS_DIR/${DS}_rise.run.json \
  --dataset-manifest "$DS_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/${DS}_cap${CAP:-1500}.jsonl" \
  --id-field query_id \
  --output $RESULTS_DIR/${DS}_rise.jsonl
sane $RESULTS_DIR/${DS}_rise.jsonl
"$PY" scripts/compute_metrics.py --results $RESULTS_DIR/${DS}_rise.jsonl \
  --dataset "$DS" --agent rise --expect-ids "$DATA/dcilite_datasets/${DS}_cap${CAP:-1500}.manifest.json" \
  --out $RESULTS_DIR/${DS}_rise.metrics.json
echo "ALL_DONE ${DS}_rise"
