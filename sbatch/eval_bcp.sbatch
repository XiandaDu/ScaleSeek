#!/usr/bin/env bash
# 用 SFT checkpoint 在 BrowseComp-Plus 上评测（vLLM 服务 + 真实 BCP BM25 索引）。
#
#   sbatch sbatch/eval_bcp.sbatch                       # 默认 60 题探测
#   EVAL_N=830 sbatch --time=12:00:00 sbatch/eval_bcp.sbatch   # 全量（需先补 manifest）
#
# 覆盖项：
#   EVAL_MODEL   HF checkpoint 路径   [最新 SFT checkpoint，运行时解析]
#   EVAL_N       题数                 [60]
#   EVAL_CONC    并发                 [4]
#   EVAL_TAG     输出文件后缀         [sft]
#
# 为什么不加 --full-eval：BCP 索引还没有 index_manifest.json（正式评测的契约要求
# 它绑定 corpus manifest）。这是探测性评测，数字不得进结果表 —— 要出正式数字，
# 先用 build_corpus_manifest.py + backfill_index_manifest.py 给 BCP 索引补 manifest。
#
# 预期：学生只在 wiki-18 的 102 词短段落上冷启动过，grep 仅出现在 23% 训练轨迹、
# read_doc 为零；BCP 是平均 5.2k 词的长网页深研究题。分数大概率难看 —— 这个数字
# 的价值是量化"训练分布缺长文档 regime"，并给 RL 之后留下对照基线。
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_eval_bcp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
require_full_gpus 1
trap 'stop_vllm; cleanup_tmpdir' EXIT

# --- 把检索器指向 BCP 索引（BM25Retriever 无参构造时读这个变量）---
export BM25_INDEX_DIR=$DATA/bcp_bm25_index
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: BCP 索引不存在 —— 先跑 sbatch/p0_bcp.sbatch"; exit 1; }

# --- 解析 SFT checkpoint ---
MODEL="${EVAL_MODEL:-}"
if [ -z "$MODEL" ]; then
  MODEL=$(find "$SFT_CKPT_DIR" -type d -name huggingface -printf '%T@ %p\n' 2>/dev/null \
          | sort -n | tail -1 | cut -d' ' -f2-)
fi
[ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || { echo "FATAL: 找不到 SFT checkpoint"; exit 1; }

N="${EVAL_N:-60}"
TAG="${EVAL_TAG:-sft}"
OUT=$RESULTS/bcp_scaleseek_${TAG}_${SLURM_JOB_ID}.jsonl
mkdir -p "$RESULTS"

echo "[eval_bcp] model=$MODEL"
echo "[eval_bcp] index=$BM25_INDEX_DIR  n=$N  out=$OUT"

# --- vLLM 服务本地 checkpoint ---
# 不用 common.sh 的 start_vllm：它固定传 --revision "$rev"，而本地路径没有
# HF revision，传空串会让 vLLM 报错。common.sh 是 baseline 共用的，不动它。
VLLM_LOG=$REPO/logs/vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
echo "[vllm] serving $MODEL on :$LLM_PORT as '$LLM_MODEL' (log: $VLLM_LOG)"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype bfloat16 \
    --host 127.0.0.1 --port "$LLM_PORT" --tensor-parallel-size 1 \
    --served-model-name "$LLM_MODEL" \
    --max-model-len "${VLLM_MAX_LEN:-32768}" \
    --gpu-memory-utilization "${VLLM_MEM_UTIL:-0.90}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
waitp "$LLM_PORT" 2400

"$PY" -m eval.run_eval \
    --dataset browsecomp_plus \
    --agent scaleseek \
    --retriever bm25 \
    --model "$LLM_MODEL" \
    -n "$N" \
    --concurrency "${EVAL_CONC:-4}" \
    --output "$OUT"

stop_vllm

echo "[eval_bcp] 完成 -> $OUT"
"$PY" scripts/compute_metrics.py --results "$OUT" \
    --bcp-qrels "$DATASETS/browsecomp_plus/qrels.json" \
    --bcp-doclen "$DATASETS/browsecomp_plus/doclen.json" 2>&1 | tail -20
