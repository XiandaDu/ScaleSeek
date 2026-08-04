#!/usr/bin/env bash
# 一键查看 phase-1/2 链状态。任何机器上直接: bash sbatch/status.sh
cd /data/rech/mofengra/ScaleSeek 2>/dev/null || cd "$(dirname "$0")/.."

# Loudest thing first: a stalled lane is silent otherwise and has twice parked
# the pipeline for days (2026-07-28 laneA ~2d, 2026-08-02 laneA again).
shopt -s nullglob
_stalls=(logs/.lane_stalled_*)
if [ ${#_stalls[@]} -gt 0 ]; then
  echo "########################################################"
  echo "##  ⛔ LANE STALLED — 没人踢的话这条线不会自己恢复  ##"
  echo "########################################################"
  for s in "${_stalls[@]}"; do echo "-- $s"; sed 's/^/   /' "$s"; done
  echo "########################################################"
  echo
fi
shopt -u nullglob

echo "== SLURM 队列 =="
squeue -u "$USER" -o "%.9i %.24j %.9T %.11M %.12r"

echo; echo "== 各 lane 剩余作业数 =="
for f in sbatch/queue_lane*.txt; do
  printf '  %-28s %s 个待跑\n' "$(basename "$f")" "$(grep -c . "$f")"
done

echo; echo "== 打包作业内各 cell 进度（p2_pack）=="
shopt -s nullglob
packlogs=(logs/cell_*.log)
if [ ${#packlogs[@]} -eq 0 ]; then
  echo "  (无打包作业在跑)"
else
  for c in "${packlogs[@]}"; do
    name=$(basename "$c" .log); name=${name#cell_}
    # On --resume the counter runs over the REMAINING examples (e.g. [6020/7922]),
    # not the full split, so also report rows actually on disk -- otherwise a
    # healthy resumed cell reads as stuck.
    last=$(grep -oE "\[ *[0-9]+/[0-9]+\]" "$c" 2>/dev/null | tail -1)
    out=$(grep -oE "results/phase2/popqa_[a-z0-9_]+\.jsonl" "$c" 2>/dev/null | head -1)
    rows=""; [ -n "$out" ] && [ -f "$out" ] && rows="盘上$(wc -l < "$out")行"
    gate=$(grep -oE "GATE-(PASS|FAIL)" "$c" 2>/dev/null | tail -1)
    done_=$(grep -c "ALL_DONE" "$c" 2>/dev/null)
    st="running"; [ "${done_:-0}" -gt 0 ] && st="DONE"
    grep -q "FATAL" "$c" 2>/dev/null && st="FATAL"
    printf '  %-40s %-8s %-11s %-16s %s\n' "$name" "$st" "${gate:-—}" "${last:-启动中}" "$rows"
  done
fi
shopt -u nullglob

echo; echo "== 已完成的 phase-2 指标 =="
ls results/phase2/*.metrics.json 2>/dev/null | while read -r m; do
  python3 - "$m" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
name = sys.argv[1].split("/")[-1].replace(".metrics.json", "")
a, l = d.get("answer_quality", {}), d.get("latency", {})
ci = a.get("em_ci95") or {}
band = f" ±{(ci['hi']-ci['lo'])/2:.3f}" if ci.get("hi") is not None else ""
print(f"  {name:34s} n={d.get('n')}  EM={a.get('em', 0):.4f}{band}  "
      f"F1={a.get('f1', 0):.4f}  ans={l.get('answer_rate', 0):.3f}  "
      f"conc={l.get('concurrency')}  wall={l.get('wall_s_per_example', '?')}s/ex")
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
