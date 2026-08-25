#!/usr/bin/env bash
# 为 BRIGHT 的 12 个任务各建一个 BM25 索引。纯 CPU。
#
#   sbatch sbatch/p0_bright.sbatch
#
# 按任务隔离是 BRIGHT 的协议要求：每个任务有自己的文档池，query 的
# golden_doc_ids 只存在于该池内。合并成一个索引会跨领域泄漏、且候选池膨胀
# 约 10 倍，数字无法与任何已发表的 BRIGHT 结果比较。
#
# 产物（幂等）：$DATA/bright_indexes/<task>/{corpus,index,*_manifest.json}
#SBATCH --account=def-amirhk_cpu
#SBATCH --job-name=ss_bright_idx
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

export SCALESEEK_JVM_HEAP="${SCALESEEK_JVM_HEAP:-16g}"
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT

BRIGHT_INDEX_ROOT=$DATA/bright_indexes
"$PY" scripts/build_bright_indexes.py \
    --bright-dir "$DATASETS/bright" \
    --out-root "$BRIGHT_INDEX_ROOT" \
    --threads "${SLURM_CPUS_PER_TASK:-16}"

echo "== 完成 =="
du -sh "$BRIGHT_INDEX_ROOT" 2>/dev/null
ls "$BRIGHT_INDEX_ROOT" | wc -l | xargs echo "  任务数:"
echo
echo "评测: BRIGHT_INDEX_ROOT=$BRIGHT_INDEX_ROOT sbatch sbatch/eval_bright.sbatch"
