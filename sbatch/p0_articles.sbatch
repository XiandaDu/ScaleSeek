#!/usr/bin/env bash
# 构建 article 级 wiki-18 语料 + BM25 索引（长文档训练 regime 的基建）。纯 CPU。
#
#   sbatch sbatch/p0_articles.sbatch
#
# 动机（2026-07-30，用户要求）：训练分布必须包含长文档 regime，让参数自适应
# 能力有非零支持（"效果可以不好，但你不能没有"），且不动任何评测保留集
# （BCP/BRIGHT/PopQA/TriviaQA/bamboogle）。article 级 wiki-18 = 同内容、同训练
# 问题集、文档长度方差巨大 —— b 有长度可归一，k1 有重复词频可饱和
# （Pi-Serini 在 ~2k token 文档上的发现适用的正是这种形态）。
#
# 产物（幂等，已存在则跳过）：
#   $DATA/wiki_articles_corpus/wiki_corpus.jsonl     article 级语料
#   $DATA/wiki_articles_corpus/article_stats.json    长度分布
#   $DATA/bm25_articles_index/index                  Lucene 索引
#SBATCH --account=def-amirhk_cpu
#SBATCH --job-name=ss_articles
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

export SCALESEEK_JVM_HEAP="${SCALESEEK_JVM_HEAP:-24g}"   # Lucene 建索引需要
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

ART_DIR=$DATA/wiki_articles_corpus
ART_INDEX=$DATA/bm25_articles_index

echo "== 1/3 article 语料 =="
if [ -s "$ART_DIR/wiki_corpus.jsonl" ]; then
  echo "  已存在 — 跳过"
else
  "$PY" scripts/build_wiki_article_jsonl.py \
      --corpus "$CORPUS_FILE" --out "$ART_DIR" --db "$TMPDIR/articles.sqlite"
fi
[ -s "$ART_DIR/wiki_corpus.jsonl" ] || { echo "FATAL: 语料构建失败"; exit 1; }
cat "$ART_DIR/article_stats.json"

echo "== 2/3 语料 manifest =="
if [ -s "$ART_DIR/corpus_manifest.json" ]; then
  echo "  已存在 — 跳过"
else
  "$PY" scripts/build_corpus_manifest.py \
      --corpus "$ART_DIR/wiki_corpus.jsonl" --out "$ART_DIR/corpus_manifest.json"
fi

echo "== 3/3 BM25 索引 =="
if [ -d "$ART_INDEX/index" ]; then
  echo "  已存在 — 跳过"
else
  "$PY" scripts/build_bm25_index.py --corpus-dir "$ART_DIR" \
      --index-dir "$ART_INDEX" --threads "${SLURM_CPUS_PER_TASK:-16}" \
      --corpus-manifest "$ART_DIR/corpus_manifest.json"
fi
[ -d "$ART_INDEX/index" ] || { echo "FATAL: 索引构建失败"; exit 1; }
du -sh "$ART_INDEX/index"

echo "== 完成 =="
echo "长文档 SFT 探测: SFT_LIMIT=50 SFT_PARAM_POLICY=search SFT_TAG=articles \\"
echo "  SFT_INDEX_DIR=$ART_INDEX SFT_CORPUS_FILE=$ART_DIR/wiki_corpus.jsonl \\"
echo "  sbatch --time=3:00:00 sbatch/sft_data.sbatch"
