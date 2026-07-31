#!/usr/bin/env bash
# SFT/RL 的数据前置条件，一次跑完。纯 CPU，不占 GPU 配额。
#
#   sbatch sbatch/p0_data_prep.sbatch
#
# 四步，每步幂等（已存在则跳过）：
#   1. wiki-18 语料      PeterJinGo/wiki-18-corpus -> $CORPUS_DIR/wiki_corpus.jsonl (~14 GB)
#   2. QA 数据集         scripts/download_data.py -> $DATASETS/
#   3. BM25 Lucene 索引  $CORPUS_DIR -> $BM25_INDEX_DIR/index (~15 GB, 20-40 min)
#   4. RL 训练/验证集    scripts/prepare_rl_data.py -> $RL_DATA_DIR/{train,dev}.jsonl
#
# 语料来源与 GrepSeek 论文一致（其 README 记录 "verified byte-identical to the
# paper's"），下载器直接复用 grepseek 仓库里的 sft/data_generation/download_corpus.py，
# 它处理了"发布的 .gz 实际是 gzip 过的 tar"这个坑。
#
# 用 CPU 账号：这一整条链没有任何一步需要 GPU，用 _gpu 账号会白占 GPU 配额。
# 内存给 128G 是为了 Lucene —— setup_env.sh 会把 JVM 堆设成 mem/4（这里 32G），
# 21M passage 建索引需要这个量级（Alliance 默认的 -Xmx2g 会 OOM）。
#SBATCH --account=def-amirhk_cpu
#SBATCH --job-name=ss_data_prep
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

# 建索引是唯一真正需要大 JVM 堆的步骤（检索走 mmap，见 setup_env.sh 的说明）。
export SCALESEEK_JVM_HEAP="${SCALESEEK_JVM_HEAP:-32g}"
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

echo "JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS   (Lucene 建索引堆)"

# ── 1. wiki-18 语料 ──────────────────────────────────────────────────────────
if [ -s "$CORPUS_FILE" ]; then
  echo "== 1/4 语料已存在: $CORPUS_FILE ($(du -h "$CORPUS_FILE" | cut -f1)) — 跳过"
else
  echo "== 1/4 下载 wiki-18 语料 -> $CORPUS_DIR"
  "$PY" "$GREPSEEK_ROOT/sft/data_generation/download_corpus.py" --dest "$CORPUS_DIR"
fi
[ -s "$CORPUS_FILE" ] || { echo "FATAL: 语料下载失败"; exit 1; }
echo "   passage 数: $(wc -l < "$CORPUS_FILE")"

# ── 2. QA 数据集 ─────────────────────────────────────────────────────────────
echo "== 2/4 下载 QA 数据集 -> $DATASETS"
"$PY" scripts/download_data.py || { echo "FATAL: 数据集下载失败"; exit 1; }
"$PY" scripts/download_data.py --verify

# ── 2b. PopQA 归一化 + manifest ──────────────────────────────────────────────
# $POPQA_NORM 是 sft_data.sbatch 的默认输入，但 download_data.py 不产出它。
# 这一步原本在 p0_assets.sbatch（Phase-1 baseline 脚本）里，训练链也需要，
# 所以搬过来。断言 14267 是 TASK.md 的 Phase gate 要求。
if [ -s "$POPQA_NORM" ]; then
  echo "== 2b/5 PopQA normalized 已存在 — 跳过"
else
  echo "== 2b/5 生成 PopQA canonical normalized + manifest"
  "$PY" - <<'PYEOF'
import json, os
from eval.datasets import load_dataset, validate_complete_dataset
rows = load_dataset("popqa")
manifest = validate_complete_dataset("popqa", rows)
base = os.path.join(os.environ["DATASETS"], "popqa")
with open(os.path.join(base, "test.normalized.jsonl"), "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n")
with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
assert manifest["count"] == manifest["unique_ids"] == 14267, manifest
print("popqa normalized ok:", manifest["count"], "rows")
PYEOF
fi

# ── 2c. 语料 manifest ────────────────────────────────────────────────────────
# 没有它，build_bm25_index.py 会警告 "index is not eligible for formal
# evaluation"。冒烟无所谓，正式评测必需。
if [ -s "$CORPUS_MANIFEST" ]; then
  echo "== 2c/5 语料 manifest 已存在 — 跳过"
else
  echo "== 2c/5 生成语料 manifest（要遍历 21M passage，较慢）"
  "$PY" scripts/build_corpus_manifest.py --corpus "$CORPUS_FILE" --out "$CORPUS_MANIFEST"
fi

# ── 3. BM25 索引 ─────────────────────────────────────────────────────────────
if [ -d "$BM25_INDEX_DIR/index" ]; then
  echo "== 3/4 BM25 索引已存在 — 跳过"
else
  echo "== 3/4 建 BM25 Lucene 索引（21M passage，预计 20-40 min）"
  "$PY" scripts/build_bm25_index.py --threads "${SLURM_CPUS_PER_TASK:-16}"
fi
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: 索引构建失败"; exit 1; }
du -sh "$BM25_INDEX_DIR/index"

# ── 4. RL 数据 ───────────────────────────────────────────────────────────────
if [ -s "$RL_DATA_DIR/train.jsonl" ]; then
  echo "== 4/4 RL 数据已存在 — 跳过"
else
  echo "== 4/4 准备 RL 训练/验证集 -> $RL_DATA_DIR"
  "$PY" scripts/prepare_rl_data.py --out_dir "$RL_DATA_DIR"
fi
wc -l "$RL_DATA_DIR"/train.jsonl "$RL_DATA_DIR"/dev.jsonl 2>&1

echo
echo "== 全部就绪 =="
echo "下一步: bash sbatch/submit_training.sh"
