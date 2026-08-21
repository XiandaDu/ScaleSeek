#!/usr/bin/env bash
#SBATCH -J p1_rise_toc
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH --exclude=octal35
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# RISE prerequisite: structure every reconstructed Wiki-18 article with the
# official RISE sectioning core (SECTION_PROMPT/validate/build_final imported
# unchanged) using vLLM-served Qwen3.5-9B for throughput. Resumable: already
# structured articles are skipped, so requeues lose nothing.
# max-model-len 131072 (was 49152): the official call requests
# max_tokens=32768, so the prompt budget is max-model-len minus that. At
# 49152 the budget was 16k tokens and 4/500 probe articles 400'd on length
# (7414); at 131072 the budget is ~98k, beyond the official truncate_for_llm
# cap, so length failures disappear. KV for 131k fits L40S tp=2 beside 9B
# weights.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
assert_gpus_usable || exit 4
PORT=$((10000 + SLURM_JOB_ID % 20000))
ART=$DATA/rise_wiki_articles
OUT=$DATA/rise_wiki_articles_structured
[ -f "$ART/passage_article_mapping.jsonl" ] || {
  echo "FATAL: article corpus missing; p0_rise_articles must finish first"; exit 4; }

VLLM_MAX_LEN=131072 start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B
waitp "$PORT" 3600
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export SECTION_MODEL=Qwen/Qwen3.5-9B
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

# Sectioning the whole article corpus is ~3.2M LLM calls = 24-72 days on this
# server, so only articles reachable from the evaluated questions get an
# LLM-built TOC. Every other article is still emitted (empty TOC), so the BM25
# corpus stays the full 3.2M and RISE's retrieval boundary is unchanged --
# structuring a subset must never become evaluating against a smaller corpus.
CAND=$ART/candidates.txt
QSET=$DATA/dcilite_datasets/popqa_cap1500.jsonl
# Fail on the real cause. 7292 died here because this file did not exist yet
# (p0_assets generates it), but the handler below blamed the cost gate and told
# the operator to lower RISE_CAND_TOPK -- which would not have helped at all.
[ -f "$QSET" ] || {
  echo "FATAL: $QSET missing. Generate it first:"
  echo "  \"\$PY\" scripts/make_dcilite_datasets.py --datasets popqa --cap 1500 \\"
  echo "      --out-dir \"\$DATA/dcilite_datasets\""
  exit 4; }
if [ ! -f "$CAND" ]; then
  echo "== select candidate articles reachable from the evaluated questions =="
  "$PY" scripts/build_rise_candidate_set.py \
    --questions "$QSET" \
    --index-dir "$BM25_INDEX_DIR" \
    --mapping "$ART/passage_article_mapping.jsonl" \
    --top-k "${RISE_CAND_TOPK:-100}" --out "$CAND" \
    --max-articles "${RISE_MAX_CAND:-250000}" || {
      echo "FATAL: candidate selection refused (see the projected hours above)."
      echo "       Lower RISE_CAND_TOPK, or raise RISE_MAX_CAND only after"
      echo "       deciding that much sectioning time is acceptable."
      exit 4; }
fi

echo "== probe: 500 candidate articles for rate + anchor validation =="
"$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$OUT" \
  --candidates "$CAND" --workers 96 --limit 500
echo "== full pass: 12 sharded processes (GIL -> processes, see --shard help) =="
NPROC=12
PIDS=()
for k in $(seq 0 $((NPROC - 1))); do
  "$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$OUT" \
    --candidates "$CAND" --workers 8 --shard "$k/$NPROC" \
    --audit "$OUT/audit.shard${k}of${NPROC}.jsonl" \
    > "$REPO/logs/rise_toc_shard_${SLURM_JOB_ID}_$k.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
# tail -n 1, NOT the obsolete `tail -1`: GNU tail rejects the short form with
# multiple file operands ("option used in invalid context"), and under set -e
# that one cosmetic line FAILED job 7422 *after* all 12 shards had finished a
# 3-day sectioning pass -- skipping the audit merge below.
tail -n 1 "$REPO"/logs/rise_toc_shard_${SLURM_JOB_ID}_*.log
[ "$FAIL" = 0 ] || { echo "FATAL: at least one TOC shard failed its gate"; exit 1; }
# Merge shard audits into the single audit.jsonl that p2_rise gates on.
for k in $(seq 0 $((NPROC - 1))); do
  cat "$OUT/audit.shard${k}of${NPROC}.jsonl" >> "$OUT/audit.jsonl"
done
stop_vllm
echo "ALL_P1_RISE_TOC_DONE"
