#!/usr/bin/env bash
# 把 SFT 数据 -> SFT 训练 -> RL 三个阶段串成作业链一次性提交。
#
#   bash sbatch/submit_training.sh              # 全链
#   bash sbatch/submit_training.sh --from sft   # 跳过数据构建
#   bash sbatch/submit_training.sh --from rl    # 只跑 RL
#   bash sbatch/submit_training.sh --smoke      # 小规模冒烟（限 200 条 + 1 epoch）
#
# 用 afterok 依赖：上一阶段失败则后续不启动（和续跑用的 afterany 相反 ——
# 这里前一阶段的产物是后一阶段的输入，失败了继续跑没有意义）。
set -euo pipefail

cd /home/a32du/ScaleSeek

FROM=data
SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2;;
    --smoke) SMOKE=1; shift;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

EXPORTS="ALL"
if [ "$SMOKE" = "1" ]; then
  EXPORTS="ALL,SFT_LIMIT=200,SCALESEEK_SFT_EPOCHS=1"
  echo "[submit] 冒烟模式：200 条轨迹 / 1 epoch"
fi

# 墙钟覆盖。sbatch 头部的 --time 是保守默认（sft_data 12h）；SFT_LIMIT 调大时
# 必须一起调墙钟，否则作业撞墙 -> afterok 把后面两级一并取消，且轨迹生成不可
# 续跑，超时前的产出全部作废。按实测 8.84 s/题 + ~17 min vLLM 启动 + 20% 余量
# 自查：SFT_DATA_TIME 秒数 × 0.8 − 1000 应 > SFT_LIMIT × 8.84。
SFT_DATA_TIME="${SFT_DATA_TIME:-}"   # 例: 24:00:00
if [ -n "${SFT_LIMIT:-}" ] && [ -n "$SFT_DATA_TIME" ]; then
  _secs=$(echo "$SFT_DATA_TIME" | awk -F: '{print $1*3600+$2*60+$3}')
  _need=$(( SFT_LIMIT * 884 / 100 + 1000 ))
  if [ $(( _secs * 8 / 10 )) -lt "$_need" ]; then
    echo "FATAL: SFT_LIMIT=$SFT_LIMIT 需要约 $_need s（含启动），但 SFT_DATA_TIME=$SFT_DATA_TIME"
    echo "       的 80% 只有 $(( _secs * 8 / 10 )) s。调大墙钟或调小 SFT_LIMIT。"
    exit 1
  fi
fi

DEP=""
JID_DATA=""; JID_SFT=""

if [ "$FROM" = "data" ]; then
  # shellcheck disable=SC2086
  JID_DATA=$(sbatch --parsable ${SFT_DATA_TIME:+--time=$SFT_DATA_TIME} \
             --export="$EXPORTS" sbatch/sft_data.sbatch)
  echo "[submit] sft_data   -> $JID_DATA${SFT_DATA_TIME:+  (time=$SFT_DATA_TIME)}"
  DEP="--dependency=afterok:$JID_DATA"
fi

if [ "$FROM" = "data" ] || [ "$FROM" = "sft" ]; then
  # shellcheck disable=SC2086
  JID_SFT=$(sbatch --parsable $DEP --export="$EXPORTS" sbatch/sft_train.sbatch)
  echo "[submit] sft_train  -> $JID_SFT"
  DEP="--dependency=afterok:$JID_SFT"
fi

# RL 需要 SFT 的 HF checkpoint 路径，而它在提交时还不存在。
# rl_train.sbatch 会在运行时从 $SFT_CKPT_DIR 解析出最新的
# global_step_*/huggingface（见该文件的「SFT checkpoint 交接」一节）。
# shellcheck disable=SC2086
JID_RL=$(sbatch --parsable $DEP --export="$EXPORTS" sbatch/rl_train.sbatch)
echo "[submit] rl_train   -> $JID_RL"

echo
echo "查看队列: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.10l %R'"
echo "看日志:   tail -f logs/ss_*-<jobid>.out"
