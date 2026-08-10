#!/usr/bin/env bash
#SBATCH -J p0_bm25_index
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Rorqual: build the frozen Lucene BM25 index over the full Wiki-18 corpus
# (pyserini, k1/b are query-time parameters; the index itself is parameter-free).
# On RALI this predated the sbatch tree; on Rorqual it is a first-class job so
# the whole asset chain rebuilds from an empty /scratch.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

command -v java >/dev/null 2>&1 || module load java/21
java -version 2>&1 | head -1
# The java module's JAVA_TOOL_OPTIONS default is -Xmx2g; a 21M-doc Lucene build
# with 16 workers GC-thrashes and dies on that. The job has 64G.
export JAVA_TOOL_OPTIONS="-Xmx48g"

[ -f "$CORPUS_MANIFEST" ] || { echo "FATAL: corpus manifest missing (run p0_corpus_unzip)"; exit 4; }

# Pyserini indexes EVERY file in --input. $CORPUS_DIR also holds
# corpus_manifest.json (no "id" field), which crashed job 18743949 with
# "totalFiles = 2 is not equal to completedTaskCount = 1". Feed the indexer a
# staging dir holding only a symlink to the corpus jsonl.
BM25_INPUT=$TMPDIR/bm25_input
mkdir -p "$BM25_INPUT"
ln -sf "$CORPUS_FILE" "$BM25_INPUT/wiki_corpus.jsonl"

if [ ! -f "$BM25_INDEX_DIR/index_manifest.json" ]; then
  rm -rf "${BM25_INDEX_DIR:?}/index"   # drop any partial build from a failed run
  "$PY" scripts/build_bm25_index.py \
    --corpus-dir "$BM25_INPUT" \
    --index-dir "$BM25_INDEX_DIR" \
    --threads 16 \
    --corpus-manifest "$CORPUS_MANIFEST"
fi
[ -f "$BM25_INDEX_DIR/index_manifest.json" ] || \
  "$PY" scripts/backfill_index_manifest.py --backend bm25 --index-dir "$BM25_INDEX_DIR/index" \
    --corpus-manifest "$CORPUS_MANIFEST" --out "$BM25_INDEX_DIR/index_manifest.json"
"$PY" scripts/validate_index_manifests.py --corpus-manifest "$CORPUS_MANIFEST" \
  --index-manifest "$BM25_INDEX_DIR/index_manifest.json"
echo "ALL_P0_BM25_INDEX_DONE"
