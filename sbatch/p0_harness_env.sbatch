#!/usr/bin/env bash
#SBATCH -J p0_harness_env
#SBATCH --account=def-amirhk_cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-1 wave 0 (CPU): runtime environments for the official external harnesses
# (uv envs, node toolchain, pi-mono agent packages). Per-repo status is recorded;
# the job fails only if the harnesses needed by scheduled Phase-2 jobs are broken.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT
STATUS=$REPO/logs/harness_env_status.txt
: > "$STATUS"

echo "== uv =="
command -v uv >/dev/null 2>&1 || pip install --quiet uv
uv --version

echo "== node =="
NODE_ROOT=/scratch/a32du/tools/node
if ! command -v node >/dev/null 2>&1; then
  if [ ! -x "$NODE_ROOT/bin/node" ]; then
    mkdir -p "$NODE_ROOT"
    curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
      | tar -xJ --strip-components=1 -C "$NODE_ROOT"
  fi
  export PATH=$NODE_ROOT/bin:$PATH
fi
node --version; npm --version

sync_repo() {  # sync_repo <name> [setup]
  local name=$1 setup=${2:-}
  echo "== $name =="
  (
    set -e
    cd "$OFFICIAL_ROOT/$name"
    uv sync 2>&1 | tail -3
    if [ -n "$setup" ] && [ -f setup.sh ]; then
      bash setup.sh 2>&1 | tail -10
    fi
    if [ -d pi-mono/packages/coding-agent ]; then
      ( cd pi-mono/packages/coding-agent && npm install 2>&1 | tail -3 )
    fi
  ) && echo "$name OK" >> "$STATUS" || echo "$name FAILED" >> "$STATUS"
}

sync_repo dci_agent_lite setup
sync_repo dr_dci setup
sync_repo rise
sync_repo agentir

cat "$STATUS"
grep -q "dci_agent_lite OK" "$STATUS" || { echo "FATAL: dci_agent_lite env broken"; exit 1; }
echo "ALL_P0_HARNESS_DONE"
