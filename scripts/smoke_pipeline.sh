#!/usr/bin/env bash
# 端到端跑通 SFT 数据 -> parquet -> SFT 训练 的最小闭环，用于验证环境而非产出结果。
#
#   bash scripts/smoke_pipeline.sh              # 需要 vLLM 已在 $LLM_PORT 上服务 teacher
#   SMOKE_TEACHER=hf:Qwen/Qwen3-1.7B bash scripts/smoke_pipeline.sh   # 不用 vLLM
#
# 产物全部写在 $DATA/smoke 下，按 TASK.md 的规定：冒烟数字绝不进结果表。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source setup_env.sh >/dev/null

SMOKE=$DATA/smoke
QUESTIONS=$SMOKE/questions.jsonl
INDEX=$SMOKE/bm25_index
TRAJ=$SMOKE/trajectories.jsonl
PARQUET=$SMOKE/sft_train.parquet
CKPT_OUT=$SMOKE/sft_ckpt
LIMIT="${SMOKE_LIMIT:-4}"
TEACHER="${SMOKE_TEACHER:-openai:http://127.0.0.1:$LLM_PORT/v1}"

[ -f "$QUESTIONS" ] || { echo "FATAL: 缺冒烟问题集 —— 先跑 \$PY scripts/make_smoke_corpus.py --out-dir $SMOKE --build-index"; exit 1; }
[ -d "$INDEX/index" ] || { echo "FATAL: 缺冒烟索引 $INDEX/index"; exit 1; }

echo "════ 1/3 生成 teacher 轨迹 (limit=$LIMIT, teacher=$TEACHER) ════"
"$PY" scripts/generate_sft_data.py \
    --questions "$QUESTIONS" --out "$TRAJ" \
    --teacher "$TEACHER" ${TEACHER:+--teacher-model teacher} \
    --index-dir "$INDEX" --limit "$LIMIT" \
    --param-policy heuristic --no-quality-judge
echo "  轨迹行数: $(wc -l < "$TRAJ")"
[ -s "$TRAJ" ] || { echo "FATAL: 轨迹为空，pipeline 未跑通"; exit 1; }

echo "════ 2/3 转 verl parquet ════"
"$PY" -m train.sft_dataset --in "$TRAJ" --out "$PARQUET"
ls -lh "$PARQUET"

echo "════ 3/3 SFT 训练（1 epoch，最小 batch，只验证能跑）════"
rm -rf "$CKPT_OUT"
SCALESEEK_SFT_BASE=Qwen/Qwen3-0.6B \
SCALESEEK_SFT_TRAIN="$PARQUET" \
SCALESEEK_SFT_OUTPUT="$CKPT_OUT" \
SCALESEEK_SFT_EPOCHS=1 \
SCALESEEK_SFT_BSZ=2 \
SCALESEEK_SFT_MAXLEN=2048 \
NPROC=1 PYBIN="$PY" \
bash scripts/run_sft.sh \
    data.micro_batch_size_per_gpu=1 \
    trainer.experiment_name=smoke \
    trainer.total_training_steps=2

echo
echo "════ 冒烟完成 ════"
find "$CKPT_OUT" -name "*.safetensors" -o -name "config.json" 2>/dev/null | head -5
echo "（按 TASK.md：冒烟结果不得进入任何结果表）"
