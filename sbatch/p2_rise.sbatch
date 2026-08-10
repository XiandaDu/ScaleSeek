#!/usr/bin/env bash
#SBATCH -J p2_rise
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 RISE: official run_rise.py at the frozen commit over the structured
# Wiki article corpus. Index and filename map are built with the official repo
# scripts (bm25s, deliberate k1=1.2 project override recorded in YAML). The
# index/map CLIs are --help-guarded: a flag mismatch exits 4 for interactive
# fixing instead of running with wrong parameters.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
assert_gpus_usable || exit 4
PORT=$((10000 + SLURM_JOB_ID % 20000))
DOCS=$DATA/rise_wiki_articles_structured
RISE_INDEX=$DATA/rise_bm25_index_wiki
MAP=$DATA/rise_bm25_index_wiki/filename_docid_map.json
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/results/phase2/rise_raw
mkdir -p results/phase2 "$RAW"
assert_raw_not_poisoned "$RAW" || exit 4
[ -f "$DOCS/audit.jsonl" ] || { echo "FATAL: structured corpus missing (run p1_rise_toc)"; exit 4; }

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

start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B
waitp "$PORT" 3600
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
"$PY" scripts/run_official_baseline.py rise --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --dataset-manifest "$POPQA_MANIFEST" \
  --record results/phase2/popqa_rise.run.json -- \
  --model Qwen/Qwen3.5-9B \
  --mini-dev "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --bm25-k 1000 --bm25-top-n-preview 10 --max-turns 100 \
  --bash-truncate-chars 4000 --read-default-limit 2000 \
  --structured-docs --index-dir "$RISE_INDEX" \
  --filename-map "$MAP" --bc-plus-docs "$DOCS" \
  --concurrency "${CONC:-8}" \
  --out-root "$RAW"
stop_vllm

echo "== normalize + metrics =="
# Pick the aggregate file explicitly, and never pipe find into head.
# `find ... | head -1` sends SIGPIPE upstream when head closes; under
# `set -o pipefail` that exits 141 and kills the job AFTER the evaluation has
# already finished. dci (7340) and dr_dci (7341) both died exactly this way on
# 2026-08-08 with complete 1,500-row outputs on disk. It was also non-deterministic:
# find's first hit can be a per-question events.jsonl rather than the aggregate.
if [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name '*.jsonl' -printf '%T@ %p\n' | sort -rn | awk 'NR==1{sub(/^[^ ]* /,"");print}'); fi
"$PY" scripts/normalize_official_results.py --method rise \
  --input "$INPUT" \
  --run-record results/phase2/popqa_rise.run.json \
  --dataset-manifest "$POPQA_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --id-field query_id \
  --output results/phase2/popqa_rise.jsonl
sane results/phase2/popqa_rise.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_rise.jsonl \
  --dataset popqa --agent rise --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_rise.metrics.json
echo "ALL_DONE popqa_rise"
