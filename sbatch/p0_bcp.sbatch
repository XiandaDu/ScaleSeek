#!/usr/bin/env bash
# BrowseComp-Plus 基建：下载语料/queries/qrels + 建 BM25 索引。纯 CPU。
#
#   sbatch sbatch/p0_bcp.sbatch
#
# BCP 是长文档 regime（100,195 篇，平均 5179 词 / 32k 字符），与 wiki-18 的
# 102 词定宽切块完全不同 —— 这正是 Pi-Serini 测出 k1/b 调参值 +18% 准确率、
# 而我们在 wiki-18 上测出 k1/b 是死杠杆的那个分界。
#
# 产物（幂等）：
#   $DATASETS/browsecomp_plus/{corpus,queries,qrels.json,doclen.json}
#   $DATA/bcp_bm25_index/index
#SBATCH --account=def-amirhk_cpu
#SBATCH --job-name=ss_bcp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

export SCALESEEK_JVM_HEAP="${SCALESEEK_JVM_HEAP:-24g}"   # 建索引需要大堆
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

BCP_DIR=$DATASETS/browsecomp_plus
BCP_INDEX=$DATA/bcp_bm25_index

if [ -d "$BCP_INDEX/index" ] && [ -s "$BCP_DIR/queries.jsonl" ]; then
  echo "== BCP 已就绪 — 跳过"
else
  echo "== 下载 BCP 语料/queries/qrels 并建索引 =="
  echo "   注：脚本会先打印 HF 原始 schema；字段名不符时用 --doc-id-field 等标志调整"
  "$PY" scripts/build_browsecomp_plus.py --out "$BCP_DIR" --bm25-index "$BCP_INDEX"
fi

[ -d "$BCP_INDEX/index" ] || { echo "FATAL: BCP 索引构建失败"; exit 1; }
echo "== 完成 =="
du -sh "$BCP_DIR" "$BCP_INDEX/index"
wc -l "$BCP_DIR/queries.jsonl" 2>/dev/null
echo
echo "下一步（eval 时把检索器指向 BCP 索引）："
echo "  BM25_INDEX_DIR=$BCP_INDEX \$PY -m eval.run_eval --dataset browsecomp_plus \\"
echo "      --agent scaleseek --retriever bm25 --model <SFT ckpt> --output <out.jsonl>"
