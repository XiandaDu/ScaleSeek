#!/usr/bin/env bash
# 优雅地停掉本机的 RL 续跑（先杀外层 launcher，免得它 resume 重启；再停 verl/Ray）。
#   bash scripts/rl_stop.sh              # 立刻停
#   bash scripts/rl_stop.sh "2026-09-05 14:00"   # 到该时刻（本机时区，date -d 可解析）再停，建议 nohup 挂后台
# 续跑：同样的 run_rl_local.sh 命令再跑一次即可（resume_mode=auto 从最近的 global_step_* 接上；
# 未存盘的步会丢，save_freq 默认 10）。
set -u
if [ -n "${1:-}" ]; then
  target=$(date -d "$1" +%s) || { echo "无法解析时间: $1"; exit 1; }
  now=$(date +%s); wait_s=$(( target - now ))
  echo "[rl_stop] 现在 $(date -u '+%F %T %Z')，将在 $(date -u -d @"$target" '+%F %T %Z')（${wait_s}s 后）停止"
  [ "$wait_s" -gt 0 ] && sleep "$wait_s"
fi
echo "[rl_stop] $(date -u '+%F %T %Z') 开始停止"
pkill -f "^bash scripts/run_rl_local.sh" && echo "  launcher 已杀"
pkill -f "^bash scripts/run_rl.sh"
pkill -TERM -f "verl.trainer.main_ppo" && echo "  main_ppo 已发 SIGTERM"
sleep 30
pkill -KILL -f "verl.trainer.main_ppo" 2>/dev/null
pkill -f "^ray::" 2>/dev/null; pkill -f "raylet|gcs_server|ray/_private/workers" 2>/dev/null
sleep 10
pkill -KILL -f "^ray::" 2>/dev/null
echo "[rl_stop] GPU 状态："; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[rl_stop] 最近 checkpoint: $(cat ${RL_CKPT_DIR:-/data/xianda/checkpoints/scaleseek/rl}/${RL_RUN_NAME:-grpo_9b_from190}/latest_checkpointed_iteration.txt 2>/dev/null || echo none)"
