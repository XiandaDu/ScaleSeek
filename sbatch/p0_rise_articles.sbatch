#!/usr/bin/env bash
#SBATCH -J p0_rise_articles
#SBATCH -p rali
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# RISE prerequisite: deterministic Wiki-18 passage -> article reconstruction.
# Produces the article corpus + passage/article mapping and reports the article
# count that sizes the TOC-generation job (the Phase-2 RISE gating decision).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; cleanup_tmpdir; advance_lane; exit $code' EXIT

df -h /var/tmp | tail -1
OUT=$DATA/rise_wiki_articles
mkdir -p "$OUT"
"$PY" scripts/build_wiki_article_corpus.py \
  --corpus "$CORPUS_FILE" \
  --corpus-manifest "$CORPUS_MANIFEST" \
  --out "$OUT" \
  --db "$TMPDIR/rise_group.db"

echo "== article count =="
ls "$OUT" | grep -c '\.md$' || true
wc -l "$OUT/passage_article_mapping.jsonl"
echo "ALL_P0_RISE_ARTICLES_DONE"
