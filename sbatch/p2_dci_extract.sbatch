#!/usr/bin/env bash
#SBATCH -J p2_dci_extract
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 06:00:00
#SBATCH --exclude=octal35,octal40
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Post-hoc scoring adapter for DCI: the official benchmark prompt requests no
# answer format (free prose; the official metric is an LLM judge), so strict
# EM over the prose is 0.0 by construction while 53.3% of reports contain a
# gold answer. Extract the span each report commits to (frozen generator,
# temp 0, thinking off, extractor never sees golds -- prompts/
# dci_answer_extraction.py) and score with the SAME EM/F1 pipeline as every
# other method. popqa_dci.metrics.json (strict) is left intact;
# popqa_dci_extracted.metrics.json is the extraction-scored sibling.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; exit $code' EXIT
assert_gpus_usable || exit 4
PORT=$((10000 + SLURM_JOB_ID % 20000))
[ -s results/phase2/popqa_dci.jsonl ] || {
  echo "FATAL: normalized DCI rows missing"; exit 4; }

start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3
waitp "$PORT" 10800

"$PY" scripts/extract_dci_answers.py \
  --input results/phase2/popqa_dci.jsonl \
  --output results/phase2/popqa_dci_extracted.jsonl \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model Qwen/Qwen3.5-9B --revision "$GEN_REV" \
  --concurrency 48
stop_vllm

sane results/phase2/popqa_dci_extracted.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dci_extracted.jsonl \
  --dataset popqa --agent dci \
  --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_dci_extracted.metrics.json
echo "ALL_DONE popqa_dci_extracted"
