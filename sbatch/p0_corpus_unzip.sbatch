#!/usr/bin/env bash
#SBATCH -J p0_corpus_unzip
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 6:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Rorqual-only: materialize $CORPUS_FILE from the pinned FlashRAG wiki18_100w.zip
# already prefetched into $HF_HOME on a login node. The zip holds exactly one
# member (wiki18_100w.jsonl, pyserini JsonCollection {"id","contents"}), which
# IS the project corpus file; no conversion happens here, so doc_ids and text
# stay byte-identical to the pinned upstream revision. Then build the corpus
# manifest that every index build validates against.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

ZIP=$HF_HUB_CACHE/datasets--RUC-NLPIR--FlashRAG_datasets/snapshots/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/retrieval-corpus/wiki18_100w.zip
[ -f "$ZIP" ] || { echo "FATAL: $ZIP missing (prefetch on a login node)"; exit 4; }
mkdir -p "$CORPUS_DIR"

if [ ! -f "$CORPUS_FILE" ]; then
  unzip -p "$ZIP" wiki18_100w.jsonl > "$CORPUS_FILE.partial"
  mv "$CORPUS_FILE.partial" "$CORPUS_FILE"
fi
wc -l "$CORPUS_FILE"
LINES=$(wc -l < "$CORPUS_FILE")
[ "$LINES" -eq 21015324 ] || { echo "FATAL: expected 21015324 lines, got $LINES"; exit 4; }

if [ ! -f "$CORPUS_MANIFEST" ]; then
  "$PY" scripts/build_corpus_manifest.py --corpus "$CORPUS_FILE" --out "$CORPUS_MANIFEST"
fi
cat "$CORPUS_MANIFEST"
echo "ALL_P0_CORPUS_UNZIP_DONE"
