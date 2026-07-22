#!/usr/bin/env bash
# 一键查看 phase-1/2 链状态。任何机器上直接: bash sbatch/status.sh
cd /data/rech/mofengra/ScaleSeek 2>/dev/null || cd "$(dirname "$0")/.."

echo "== SLURM 队列 =="
squeue -u "$USER" -o "%.9i %.24j %.9T %.11M %.12r"

echo; echo "== 各 lane 剩余作业数 =="
for f in sbatch/queue_lane*.txt; do
  printf '  %-28s %s 个待跑\n' "$(basename "$f")" "$(grep -c . "$f")"
done

echo; echo "== 已完成的 phase-2 指标 =="
ls results/phase2/*.metrics.json 2>/dev/null | while read -r m; do
  python3 - "$m" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
name = sys.argv[1].split("/")[-1].replace(".metrics.json", "")
flat = {k: v for k, v in d.items() if isinstance(v, (int, float, str)) and k not in ("results",)}
keep = {k: flat[k] for k in list(flat)[:8]}
print(f"  {name:40s} {keep}")
EOF
done
[ -z "$(ls results/phase2/*.metrics.json 2>/dev/null)" ] && echo "  (还没有完成的 run)"

echo; echo "== 今天结束的作业 =="
sacct -X -S today -u "$USER" -o JobID%9,JobName%24,State%12,Elapsed%11,ExitCode 2>/dev/null | tail -15

echo; echo "== 日志中的 FATAL / SANE-FAIL =="
found=0
for f in $(ls -t logs/*.out 2>/dev/null | head -12); do
  if grep -qE "FATAL|SANE-FAIL" "$f"; then
    found=1; echo "  -- $f"; grep -E "FATAL|SANE-FAIL" "$f" | tail -2 | sed 's/^/     /'
  fi
done
[ "$found" = 0 ] && echo "  (无)"
