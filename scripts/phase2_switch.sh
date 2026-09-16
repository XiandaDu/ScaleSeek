#!/usr/bin/env bash
# 通用切段：等 <run> 存到本地 step <at>，优雅停，然后用当前 RL_CONFIG 从 overall=<base>+<at> 续跑。
#   source setup_env.sh; RL_CONFIG=grpo_trainer_v2_1 setsid nohup bash scripts/phase2_switch.sh v2_4gpu_from115 5 115 > logs/phase2_switch.log 2>&1 < /dev/null &
set -u
: "${REPO:?}"; cd "$REPO"
RUN=$1; AT=$2; BASE=$3; SRC=$RL_CKPT_DIR/$RUN
STATUS=logs/phase2_status.txt
mark() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
mark "[switch] waiting for $RUN to reach local step $AT"
while [ "$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)" -lt "$AT" ]; do
  pgrep -f "^bash scripts/phase2_elastic.sh" >/dev/null || { mark "[switch] elastic gone; switching now"; break; }
  sleep 60
done
LOCAL=$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
mark "[switch] stopping $RUN at local step $LOCAL (overall $((BASE + LOCAL)))"
bash scripts/gpu_pause.sh >> logs/phase2_switch.log 2>&1; sleep 20
mark "[switch] relaunching elastic (${RL_CONFIG:-grpo_trainer_v2_1}) from overall $((BASE + LOCAL))"
RL_CONFIG="${RL_CONFIG:-grpo_trainer_v2_1}" CUR_RUN="$RUN" OVERALL_START=$((BASE + LOCAL)) FULL_STEPS="${FULL_STEPS:-300}" \
  setsid nohup bash scripts/phase2_elastic.sh >> logs/phase2_elastic.log 2>&1 < /dev/null &
disown; mark "[switch] done"
