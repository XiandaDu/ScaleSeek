#!/usr/bin/env bash
# v3 全量重跑：删掉 v1/v2 的 RL 产物（SFT checkpoint 与所有数据集保留），从 SFT v2 冷启动，
# 用 grpo_trainer_v3.yaml（简单加法 reward）跑 FULL_STEPS 步。
#
#   source setup_env.sh
#   CONFIRM_DELETE=yes setsid nohup bash scripts/phase3_v3.sh > logs/phase3_v3.log 2>&1 < /dev/null &
#
# 覆盖项：FULL_STEPS[300] TRAIN_FILE[$DATA/rl_data_v2/pool.jsonl] GPU_WAIT_H[72]
#         CONFIRM_DELETE=yes 才会真的删除旧 run（否则只是列出来）
set -uo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"; mkdir -p logs
STATUS=logs/phase3_status.txt
mark() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
TRAIN_FILE="${TRAIN_FILE:-$DATA/rl_data_v2/pool.jsonl}"
FULL_STEPS="${FULL_STEPS:-300}"
SFT=$(find "$SFT_CKPT_DIR/v2" -type d -name huggingface 2>/dev/null | sort | tail -1 || true)
[ -n "$SFT" ] && [ -f "$SFT/config.json" ] || { mark "FATAL: SFT v2 checkpoint 不存在"; exit 1; }

OLD_RUNS="grpo_9b_from190 smoke_from190 pilot_v2 v2_4gpu_from40 v2_4gpu_from115 v2_4gpu_from120"
mark "[v3] 旧 RL run: $(du -sh "$RL_CKPT_DIR" 2>/dev/null | cut -f1)"
if [ "${CONFIRM_DELETE:-no}" = "yes" ]; then
  for r in $OLD_RUNS; do [ -d "$RL_CKPT_DIR/$r" ] && { mark "[v3] 删除 $RL_CKPT_DIR/$r ($(du -sh "$RL_CKPT_DIR/$r" | cut -f1))"; rm -rf "$RL_CKPT_DIR/$r"; }; done
  mark "[v3] 删除后剩余: $(du -sh "$RL_CKPT_DIR" 2>/dev/null | cut -f1)"
else
  mark "[v3] CONFIRM_DELETE!=yes —— 不删除，仅继续训练"
fi

export SCALESEEK_REF_PATH="$SFT"          # KL 参考永远是 SFT v2
mark "[v3] start: model=$SFT train=$TRAIN_FILE ($(wc -l < "$TRAIN_FILE") 题) steps=$FULL_STEPS"
RL_CONFIG="${RL_CONFIG:-grpo_trainer_v3_1}" CUR_RUN=__none__ OVERALL_START=0 FULL_STEPS="$FULL_STEPS" \
SCALESEEK_V3_MODEL="$SFT" TRAIN_OVERRIDE="$TRAIN_FILE" \
  bash scripts/phase2_elastic.sh >> logs/phase3_elastic.log 2>&1
mark "[v3] elastic exited rc=$?"
