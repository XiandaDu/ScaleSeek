#!/usr/bin/env bash
#SBATCH -J p2_grepseek
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-2 GrepSeek: official harness at the frozen commit against the served
# official GRPO checkpoint, full $DS split, then normalize + offline metrics.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/$RESULTS_DIR/grepseek_raw
mkdir -p "$RESULTS_DIR" "$RAW"

start_vllm "$GREPSEEK_MODEL" "$GREPSEEK_REV" "$PORT" 2 "$GREPSEEK_MODEL" --reasoning-parser qwen3
waitp "$PORT" 3600

"$PY" scripts/run_official_baseline.py grepseek --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GREPSEEK_REV" \
  --record $RESULTS_DIR/${DS}_grepseek.run.json -- \
  --model "$GREPSEEK_MODEL" --tokenizer "$GREPSEEK_MODEL" \
  --base_url "http://127.0.0.1:$PORT/v1" \
  --max_assistant_turns 6 --max_tokens_per_turn 0 --tool_max_tokens 2048 \
  --cache_dir "$HF_HUB_CACHE" \
  --temperature 0.6 --top_p 1.0 --parallel 16 \
  --input "$DS_RAW" --corpus_dir "$CORPUS_DIR" \
  --out_dir "$RAW"
stop_vllm

echo "== normalize + metrics =="
INPUT=$(find "$RAW" -name '*.jsonl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$INPUT" ] || { echo "FATAL: no grepseek output jsonl under $RAW"; exit 4; }
# The harness appends from --parallel 16 workers; on Lustre the final lines can
# still be in flight when the process exits (job 18753889 died in normalize on
# a torn line that was intact a minute later). Wait until the file is fully
# valid JSON before normalizing, up to 5 minutes.
for attempt in 1 2 3 4 5; do
  if "$PY" - "$INPUT" <<'VEOF'
import json, sys
[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
VEOF
  then break; fi
  echo "[grepseek] output not yet fully flushed (attempt $attempt); waiting 60s"
  sleep 60
  [ "$attempt" = 5 ] && { echo "FATAL: $INPUT still torn after 5 minutes"; exit 4; }
done
"$PY" scripts/normalize_official_results.py --method grepseek \
  --input "$INPUT" \
  --run-record $RESULTS_DIR/${DS}_grepseek.run.json \
  --dataset-manifest "$DS_MANIFEST" --dataset-jsonl "$DS_NORM" \
  --id-prefix "${DS}_" \
  --output $RESULTS_DIR/${DS}_grepseek.jsonl
sane $RESULTS_DIR/${DS}_grepseek.jsonl
"$PY" scripts/compute_metrics.py --results $RESULTS_DIR/${DS}_grepseek.jsonl \
  --dataset "$DS" --agent grepseek --out $RESULTS_DIR/${DS}_grepseek.metrics.json
echo "ALL_DONE ${DS}_grepseek"
