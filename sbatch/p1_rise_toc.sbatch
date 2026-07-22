#!/usr/bin/env bash
#SBATCH -J p1_rise_toc
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# RISE prerequisite: structure every reconstructed Wiki-18 article with the
# official RISE sectioning core (SECTION_PROMPT/validate/build_final imported
# unchanged) using vLLM-served Qwen3.5-9B for throughput. Resumable: already
# structured articles are skipped, so requeues lose nothing.
# max-model-len 49152: the official call requests max_tokens=32768, so the
# server must admit prompt+32768; very long articles beyond that are recorded
# as failures in the audit rather than crashing the run.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
ART=$DATA/rise_wiki_articles
OUT=$DATA/rise_wiki_articles_structured
[ -f "$ART/passage_article_mapping.jsonl" ] || {
  echo "FATAL: article corpus missing; p0_rise_articles must finish first"; exit 4; }

VLLM_MAX_LEN=49152 start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B
waitp "$PORT" 3600
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export SECTION_MODEL=Qwen/Qwen3.5-9B
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

echo "== probe: 500 articles for rate + anchor validation =="
"$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$OUT" \
  --workers 96 --limit 500
echo "== full pass (resumes past the probe) =="
"$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$OUT" \
  --workers 96
stop_vllm
echo "ALL_P1_RISE_TOC_DONE"
