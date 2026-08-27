#!/usr/bin/env bash
#SBATCH -J p2_dci
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-2 DCI: official DCI-Agent-Lite harness (Pi, 300 turns, L3) over the
# materialized Wiki-18 corpus with Qwen3.5-9B served via vLLM.
# TMPDIR redirection in common.sh is load-bearing here: Pi writes every bash
# tool stdout to os.tmpdir(), which on octal30 is tmpfs (RAM).
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
export PATH=/scratch/a32du/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/$RESULTS_DIR/dci_raw
mkdir -p "$RESULTS_DIR" "$RAW"

# Rorqual: the 21M-file corpus lives as ONE tar.zst on /scratch (1M-file
# quota); extract it into node-local $SLURM_TMPDIR at job start. NVMe extraction
# is ~1-2h and greps run far faster than Lustre on RALI afterwards.
DCI_CORPUS_ARCHIVE=$DATA/dci_wiki_corpus.tar.zst
DCI_CORPUS=$TMPDIR/dci_wiki_corpus
[ -f "$DCI_CORPUS_ARCHIVE" ] || {
  echo "FATAL: $DCI_CORPUS_ARCHIVE missing; p0_dci_corpus must run first"; exit 4; }
if [ ! -d "$DCI_CORPUS" ]; then
  echo "== extracting DCI corpus to $DCI_CORPUS =="
  command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
  time tar -C "$TMPDIR" -xf "$DCI_CORPUS_ARCHIVE" -I 'zstd -d -T8'
fi
# Pi rejects tool_choice:"auto" unless the server advertises tool calling;
# without these it fails with `400 status code (no body)`
# (dr_dci assets/docs/setup.md, and BASELINE_SOURCES.md already noted it).
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 1 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
waitp "$PORT" 3600

# The Pi agent ignores OPENAI_BASE_URL; it resolves providers from
# $PI_CODING_AGENT_DIR/models.json. pi_vllm_provider writes that file.
pi_vllm_provider "$PORT" Qwen/Qwen3.5-9B
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"   # for the Python-side tools
export OPENAI_API_KEY=dummy

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
"$PY" scripts/run_official_baseline.py dci --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --record $RESULTS_DIR/${DS}_dci.run.json -- \
  --model Qwen/Qwen3.5-9B --provider vllm \
  --agent-dir "$PI_CODING_AGENT_DIR" \
  --max-turns 300 --runtime-context-level level3 \
  --tools read,bash --max-concurrency "${CONC:-2}" \
  --judge-model Qwen/Qwen3.5-9B \
  --dataset "$DATA/dcilite_datasets/${DS}_cap1500.jsonl" \
  --corpus-dir "$DCI_CORPUS" \
  --output-root "$RAW"
stop_vllm

echo "== normalize + metrics =="
# Pick the aggregate file explicitly, and never pipe find into head.
# `find ... | head -1` sends SIGPIPE upstream when head closes; under
# `set -o pipefail` that exits 141 and kills the job AFTER the evaluation has
# already finished. dci (7340) and dr_dci (7341) both died exactly this way on
# 2026-08-08 with complete 1,500-row outputs on disk. It was also non-deterministic:
# find's first hit can be a per-question events.jsonl rather than the aggregate.
if [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name "*.jsonl" -print -quit); fi
"$PY" scripts/normalize_official_results.py --method dci \
  --input "$INPUT" \
  --run-record $RESULTS_DIR/${DS}_dci.run.json \
  --dataset-manifest "$DS_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/${DS}_cap1500.jsonl" \
  --id-field query_id \
  --prediction-field final_text --finish-field run_status \
  --tool-calls-field tool_metrics.call_count \
  --output $RESULTS_DIR/${DS}_dci.jsonl
sane $RESULTS_DIR/${DS}_dci.jsonl
"$PY" scripts/compute_metrics.py --results $RESULTS_DIR/${DS}_dci.jsonl \
  --dataset "$DS" --agent dci --expect-ids "$DATA/dcilite_datasets/${DS}_cap1500.manifest.json" \
  --out $RESULTS_DIR/${DS}_dci.metrics.json
echo "ALL_DONE ${DS}_dci"
