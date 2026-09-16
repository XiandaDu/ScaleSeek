#!/usr/bin/env bash
#   bash scripts/gpu_resume.sh [到时刻再恢复，如 "2026-09-06 00:00"]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; source setup_env.sh >/dev/null
if [ -n "${1:-}" ]; then
  t=$(date -d "$1" +%s); w=$(( t - $(date +%s) )); echo "[resume] 将在 $(date -u -d @"$t" '+%F %T %Z') 恢复（${w}s 后）"; [ "$w" -gt 0 ] && sleep "$w"
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
[ "$busy" -eq 0 ] || echo "[resume] warn: $busy 张 GPU 仍被占用"
echo "[resume] $(date -u '+%F %T %Z') 重启 phase2_chain3"
nohup bash scripts/phase2_chain3.sh >> logs/phase2_chain3.log 2>&1 &
