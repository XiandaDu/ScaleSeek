#!/usr/bin/env bash
#SBATCH -J p2_grepseek
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 GrepSeek: official harness at the frozen commit against the served
# official GRPO checkpoint, full PopQA, then normalize + offline metrics.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/results/phase2/grepseek_raw
mkdir -p results/phase2 "$RAW"

start_vllm "$GREPSEEK_MODEL" "$GREPSEEK_REV" "$PORT" 2 "$GREPSEEK_MODEL" --reasoning-parser qwen3
waitp "$PORT" 3600

"$PY" scripts/run_official_baseline.py grepseek --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GREPSEEK_REV" \
  --record results/phase2/popqa_grepseek.run.json -- \
  --model "$GREPSEEK_MODEL" --tokenizer "$GREPSEEK_MODEL" \
  --base_url "http://127.0.0.1:$PORT/v1" \
  --max_assistant_turns 6 --max_tokens_per_turn 0 --tool_max_tokens 2048 \
  --temperature 0.6 --top_p 1.0 --parallel 16 \
  --input "$POPQA_RAW" --corpus_dir "$CORPUS_DIR" \
  --out_dir "$RAW"
stop_vllm

echo "== normalize + metrics =="
INPUT=$(find "$RAW" -name '*.jsonl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$INPUT" ] || { echo "FATAL: no grepseek output jsonl under $RAW"; exit 4; }
"$PY" scripts/normalize_official_results.py --method grepseek \
  --input "$INPUT" \
  --run-record results/phase2/popqa_grepseek.run.json \
  --dataset-manifest "$POPQA_MANIFEST" --dataset-jsonl "$POPQA_NORM" \
  --output results/phase2/popqa_grepseek.jsonl
sane results/phase2/popqa_grepseek.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_grepseek.jsonl \
  --dataset popqa --agent grepseek --out results/phase2/popqa_grepseek.metrics.json
echo "ALL_DONE popqa_grepseek"
