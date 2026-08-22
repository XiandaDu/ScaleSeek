#!/usr/bin/env bash
#SBATCH -J p2_rise
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH --exclude=octal35
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

# The official tooling consumes a (docid, text) parquet -- feeding it the .md
# directory dies in pyarrow open_input_file (7502, 5m51s) -- and the repo's
# build_filename_docid_map.py is a documentation STUB (SystemExit(__doc__))
# that tells the corpus exporter to emit the map. Our exporter counterpart
# packages the TOC'd tree into both artifacts; official build_bm25_index.py
# then builds the index unchanged (project k1=1.2 override recorded in YAML).
PARQUET=$DATA/rise_bm25_index_wiki/wiki_articles_structured.parquet
cd "$OFFICIAL_ROOT/rise"
if [ ! -f "$PARQUET" ] || [ ! -f "$MAP" ]; then
  uv run python "$REPO/scripts/build_rise_wiki_parquet.py" \
    --docs-dir "$DOCS" --parquet-out "$PARQUET" --map-out "$MAP" || {
      echo "FATAL: corpus parquet/map packaging failed"; exit 4; }
fi
if [ ! -d "$RISE_INDEX/bm25_index" ]; then
  uv run python scripts/build_bm25_index.py --corpus "$PARQUET" \
    --out "$RISE_INDEX" --k1 1.2 --b 0.75 || {
      echo "FATAL: bm25s index build failed"; exit 4; }
fi
cd "$REPO"

# Tool-parser flags: the official RISE runner drives search/bash/read through
# the OpenAI tools API with tool_choice auto; without them every request 400s
# ('"auto" tool choice requires --enable-auto-tool-choice...') and 7513
# "finished" 1500 queries in 1s each with zero model calls. Fourth member of
# the same family (DCI e7bdb03, DR-DCI, agentir a829a45).
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
waitp "$PORT" 10800   # 3h: glacier NFS contention starved 1h loads (7503/7505/7504)
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
# The subset manifest, not $POPQA_MANIFEST: RISE runs the sanctioned capped
# file, so the launcher must hold --mini-dev against subset_n/ids_sha256
# (7512 died holding the 1,500-row file against the 14,267 full-split count).
"$PY" scripts/run_official_baseline.py rise --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --dataset-manifest "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
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
# The RISE runner writes one JSON per query under _per_query/ (there is no
# aggregate results.jsonl; 7513's find over '*.jsonl' fed normalize an empty
# path). Point normalize at that directory alone -- $RAW also holds
# _summary.json and _traces/*/single.json that must not be swept in. The
# answer format is the official 'Exact Answer:' block (same as DR-DCI);
# gold_answer carries make_dcilite_datasets' ' / ' join.
"$PY" scripts/normalize_official_results.py --method rise \
  --input "$RAW/_per_query" \
  --run-record results/phase2/popqa_rise.run.json \
  --dataset-manifest "$POPQA_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --id-field query_id \
  --prediction-field final_text --answer-regex '^\**Exact Answer\**:\s*(.+?)\s*$' \
  --gold-field gold_answer --gold-split ' / ' \
  --finish-field terminated_by --turns-field n_turns \
  --output results/phase2/popqa_rise.jsonl
sane results/phase2/popqa_rise.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_rise.jsonl \
  --dataset popqa --agent rise --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_rise.metrics.json
echo "ALL_DONE popqa_rise"
