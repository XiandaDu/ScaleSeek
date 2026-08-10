#!/usr/bin/env bash
#SBATCH -J p0_rise_articles
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# RISE prerequisite: deterministic Wiki-18 passage -> article reconstruction.
# Produces the article corpus + passage/article mapping and reports the article
# count that sizes the TOC-generation job.
#
# Rorqual twist: ~3.2M article files vs the 1M-file /scratch quota. Articles are
# built in node-local $SLURM_TMPDIR and archived to ONE tar.zst on /scratch;
# the two small metadata files (mapping + manifest) stay unpacked in
# $DATA/rise_wiki_articles_meta for the candidate builder and gates.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

ARCHIVE=$DATA/rise_wiki_articles.tar.zst
META=$DATA/rise_wiki_articles_meta
[ -f "$ARCHIVE" ] && [ -f "$META/passage_article_mapping.jsonl" ] && {
  echo "archive + meta already exist"; exit 0; }

OUT=$TMPDIR/rise_wiki_articles
mkdir -p "$OUT" "$META"
"$PY" scripts/build_wiki_article_corpus.py \
  --corpus "$CORPUS_FILE" \
  --corpus-manifest "$CORPUS_MANIFEST" \
  --out "$OUT" \
  --db "$TMPDIR/rise_group.db"

echo "== article count =="
ls "$OUT" | grep -c '\.md$' || true
wc -l "$OUT/passage_article_mapping.jsonl"

echo "== persist metadata + archive articles =="
cp "$OUT/passage_article_mapping.jsonl" "$META/"
[ -f "$OUT/article_manifest.json" ] && cp "$OUT/article_manifest.json" "$META/"
command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
tar -C "$TMPDIR" -cf "$ARCHIVE.partial" -I 'zstd -T4 -3' rise_wiki_articles
mv "$ARCHIVE.partial" "$ARCHIVE"
ls -lh "$ARCHIVE"
echo "ALL_P0_RISE_ARTICLES_DONE"
