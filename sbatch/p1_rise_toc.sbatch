#!/usr/bin/env bash
#SBATCH -J p1_rise_toc
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# RISE prerequisite: structure candidate Wiki-18 articles with the official
# RISE sectioning core (SECTION_PROMPT/validate/build_final imported unchanged)
# using vLLM-served Qwen3.5-9B. Resumable: already-structured articles are
# skipped, so requeues lose nothing.
# max-model-len 49152: the official call requests max_tokens=32768, so the
# server must admit prompt+32768; very long articles beyond that are recorded
# as failures in the audit rather than crashing the run.
#
# Rorqual split of responsibilities (1M-file /scratch quota):
#   * HERE: LLM-TOC ONLY the per-dataset candidate set (<=250k files), written
#     to the persistent $DATA/rise_toc_structured so requeues resume.
#   * p2_rise: extracts the raw article tar into $SLURM_TMPDIR, overlays these
#     candidate TOCs, and runs the fast no-LLM empty-TOC passthrough for the
#     other ~3M articles there, keeping the BM25 corpus the full 3.2M without
#     ever placing 3.2M files on /scratch.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
ARCHIVE=$DATA/rise_wiki_articles.tar.zst
META=$DATA/rise_wiki_articles_meta
TOC_DIR=$DATA/rise_toc_structured
[ -f "$META/passage_article_mapping.jsonl" ] || {
  echo "FATAL: article metadata missing; p0_rise_articles must finish first"; exit 4; }
[ -f "$ARCHIVE" ] || { echo "FATAL: $ARCHIVE missing; p0_rise_articles must finish first"; exit 4; }
mkdir -p "$TOC_DIR"

echo "== extract raw articles to node-local =="
ART=$TMPDIR/rise_wiki_articles
if [ ! -d "$ART" ]; then
  command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
  time tar -C "$TMPDIR" -xf "$ARCHIVE" -I 'zstd -d -T8'
fi
cp -n "$META/passage_article_mapping.jsonl" "$ART/" 2>/dev/null || true

VLLM_MAX_LEN=49152 start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B
waitp "$PORT" 3600
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export SECTION_MODEL=Qwen/Qwen3.5-9B
unset XIAOMI_API_KEY DEEPSEEK_API_KEY 2>/dev/null || true

# Sectioning the whole article corpus is ~3.2M LLM calls = 24-72 days, so only
# articles reachable from the evaluated questions get an LLM-built TOC (the
# empty-TOC passthrough for the rest happens inside p2_rise; the retrieval
# boundary stays the full corpus).
CAND=$META/candidates_${DS}.txt
QSET=$DATA/dcilite_datasets/${DS}_cap1500.jsonl
# Fail on the real cause. 7292 died here because this file did not exist yet
# (p0_assets generates it), but the handler below blamed the cost gate and told
# the operator to lower RISE_CAND_TOPK -- which would not have helped at all.
[ -f "$QSET" ] || {
  echo "FATAL: $QSET missing. Generate it first:"
  echo "  \"\$PY\" scripts/make_dcilite_datasets.py --datasets \"\$DS\" --cap 1500 \\"
  echo "      --out-dir \"\$DATA/dcilite_datasets\""
  exit 4; }
if [ ! -f "$CAND" ]; then
  echo "== select candidate articles reachable from the evaluated questions =="
  "$PY" scripts/build_rise_candidate_set.py \
    --questions "$QSET" \
    --index-dir "$BM25_INDEX_DIR" \
    --mapping "$META/passage_article_mapping.jsonl" \
    --top-k "${RISE_CAND_TOPK:-100}" --out "$CAND" \
    --max-articles "${RISE_MAX_CAND:-250000}" || {
      echo "FATAL: candidate selection refused (see the projected hours above)."
      echo "       Lower RISE_CAND_TOPK, or raise RISE_MAX_CAND only after"
      echo "       deciding that much sectioning time is acceptable."
      exit 4; }
fi

echo "== probe: 500 candidate articles for rate + anchor validation =="
"$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$TOC_DIR" \
  --candidates "$CAND" --workers 96 --limit 500
echo "== full candidate pass (LLM TOC; no passthrough here — see p2_rise) =="
"$PY" scripts/build_wiki_toc.py --articles-dir "$ART" --out-dir "$TOC_DIR" \
  --candidates "$CAND" --workers 96 --limit "$(wc -l < "$CAND")"
stop_vllm
echo "ALL_P1_RISE_TOC_DONE"
