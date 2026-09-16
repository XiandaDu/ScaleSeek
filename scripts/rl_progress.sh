#!/usr/bin/env bash
# 从 run_rl_local.sh 的日志里摘出每步的关键指标（步号、reward、EM、format、时长）。
#   bash scripts/rl_progress.sh [run 名=grpo_9b_from190] [最近 N 步=20]
RUN="${1:-grpo_9b_from190}"; N="${2:-20}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ls -t logs/rl_${RUN}_attempt*.log 2>/dev/null | head -1 | xargs -r echo "log:"
cat logs/rl_${RUN}_attempt*.log 2>/dev/null | sed -E "s/^.*\) //" | grep -a "^step:[0-9]" | tail -n "$N" | \
  sed -E 's/ - / /g' | awk '{
    step=$1; r=""; em=""; fmt=""; t=""; ws=""; calls=""; val="";
    for(i=1;i<=NF;i++){
      if($i ~ /^critic\/score\/mean:/) r=$i;
      if($i ~ /^reward\/em:/) em=$i;
      if($i ~ /^reward\/format_pass:/) fmt=$i;
      if($i ~ /^timing_s\/step:/) t=$i;
      if($i ~ /^metrics\/workspace_size:/) ws=$i;
      if($i ~ /^metrics\/n_bm25_calls:/) calls=$i;
      if($i ~ /^val-core\/.*reward\/em/ || $i ~ /^val\/test_score/) val=$i;
    }
    print step, r, em, fmt, ws, calls, t, val }'
echo "latest checkpoint: $(cat ${RL_CKPT_DIR:-/data/xianda/checkpoints/scaleseek/rl}/$RUN/latest_checkpointed_iteration.txt 2>/dev/null || echo none)"
