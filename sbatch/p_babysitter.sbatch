#!/usr/bin/env bash
#SBATCH -J p_babysitter
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# No-GPU watchdog so the phase-3 DAG self-heals with nobody logged in.
# Every 6h it re-runs the idempotent orchestrator (submit_phase3.sh reuses
# queued jobs by name, prunes stages whose outputs exist, and resubmits ONLY
# broken chain segments) and writes a human-readable digest to
# logs/BABYSITTER_DIGEST.txt. The 6h cadence caps how much a systematically
# crashing job can be resubmitted (~4x/day) while still healing transient
# failures (node death, OOM after a fix, dependency cascades) within hours.
# Compute nodes can run sbatch/squeue; no internet is needed for any of this.
set -uo pipefail
cd /home/a32du/ScaleSeek
export DS=${DS:-triviaqa}
export PHASE=${PHASE:-phase3}

while true; do
  {
    echo "===== babysitter cycle $(date '+%F %T') ====="
    DS=$DS PHASE=$PHASE bash sbatch/submit_phase3.sh 2>&1 | grep -E "submit|reuse|skipping" || true
    echo "--- completed metrics ---"
    for m in results/$PHASE/*.metrics.json; do
      [ -f "$m" ] || continue
      python3 - "$m" <<'PYEOF' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1])); a=d.get("answer_quality",{})
print(f"  {sys.argv[1].split('/')[-1].replace('.metrics.json',''):34s} n={d.get('n')} {d.get('eval_scope','?')} EM={a.get('em',0):.4f} F1={a.get('f1',0):.4f}")
PYEOF
    done
    echo "--- queue ---"
    squeue -u "$USER" -o "%.9i %.24j %.9T %.11M %.12r" 2>/dev/null || true
    echo "--- failures in the last 12h ---"
    sacct -X -u "$USER" -S "$(date -d '12 hours ago' '+%FT%T')" \
      -o JobID%9,JobName%24,State%14,Elapsed%11 -n 2>/dev/null \
      | grep -E "p[0-9_]" | grep -E "FAILED|OUT_OF_MEM|TIMEOUT|NODE_FAIL" || echo "  (none)"
  } > logs/BABYSITTER_DIGEST.txt.new 2>&1
  mv logs/BABYSITTER_DIGEST.txt.new logs/BABYSITTER_DIGEST.txt
  sleep 21600
done
