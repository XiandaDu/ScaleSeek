#!/usr/bin/env bash
#SBATCH -J p2_dci
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 DCI: official DCI-Agent-Lite harness (Pi, 300 turns, L3) over the
# materialized Wiki-18 corpus with Qwen3.5-9B served via vLLM.
# TMPDIR redirection in common.sh is load-bearing here: Pi writes every bash
# tool stdout to os.tmpdir(), which on octal30 is tmpfs (RAM).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
export PATH=/data/rech/mofengra/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
RAW=results/phase2/dci_raw
mkdir -p results/phase2 "$RAW"

[ -d "$DATA/dci_wiki_corpus" ] && [ -n "$(ls -A "$DATA/dci_wiki_corpus" 2>/dev/null)" ] || {
  echo "FATAL: expanded DCI corpus missing; p0_dci_corpus must run first"; exit 4; }
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B --reasoning-parser qwen3
waitp "$PORT" 3600

export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
"$PY" scripts/run_official_baseline.py dci --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --record results/phase2/popqa_dci.run.json -- \
  --model Qwen/Qwen3.5-9B --provider openai \
  --max-turns 300 --runtime-context-level level3 \
  --tools read,bash --max-concurrency "${CONC:-2}" \
  --judge-model Qwen/Qwen3.5-9B \
  --dataset "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --corpus-dir "$DATA/dci_wiki_corpus" \
  --output-root "$RAW"
stop_vllm

echo "== normalize + metrics =="
INPUT=$(find "$RAW" -name "*.jsonl" -newer results/phase2/popqa_dci.run.json | head -1)
[ -n "$INPUT" ] || INPUT=$(find "$RAW" -name "*.jsonl" | head -1)
"$PY" scripts/normalize_official_results.py --method dci \
  --input "$INPUT" \
  --run-record results/phase2/popqa_dci.run.json \
  --dataset-manifest "$POPQA_MANIFEST" --dataset-jsonl "$POPQA_NORM" \
  --output results/phase2/popqa_dci.jsonl
sane results/phase2/popqa_dci.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dci.jsonl \
  --dataset popqa --agent dci --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_dci.metrics.json
echo "ALL_DONE popqa_dci"
