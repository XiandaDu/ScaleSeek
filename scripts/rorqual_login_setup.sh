#!/usr/bin/env bash
# Rorqual LOGIN-NODE provisioning (cluster-packing-and-decile-scope-Rorqual).
#
# On RALI this work lived in sbatch/p0_harness_env.sbatch, but Rorqual compute
# nodes have NO internet, so everything that downloads must happen here:
#   1. frozen official-baseline checkouts (bootstrap_official_repos.py)
#   2. node toolchain for the Pi agent (dci / dr_dci)
#   3. uv + per-repo uv-synced envs + pi-mono npm packages
# Jobs then consume these read-only with UV_OFFLINE=1 / HF_HUB_OFFLINE=1
# (exported by setup_env.sh whenever SLURM_JOB_ID is set).
#
# Downloads and git clones are I/O-bound and are sanctioned login-node work;
# nothing here starts model inference or long CPU jobs.
set -euo pipefail
case "$(hostname)" in
  rorqual*) ;;
  *) echo "this script is for a rorqual LOGIN node"; exit 1;;
esac

source "$HOME/ScaleSeek/setup_env.sh"
export OFFICIAL_ROOT=${OFFICIAL_ROOT:-/scratch/a32du/official-baselines}
STATUS=$REPO/logs/harness_env_status.txt
mkdir -p "$REPO/logs" "$OFFICIAL_ROOT"
: > "$STATUS"

echo "== official repo checkouts (commit-pinned) =="
python scripts/bootstrap_official_repos.py --root "$OFFICIAL_ROOT"

echo "== node =="
NODE_ROOT=/scratch/a32du/tools/node
if [ ! -x "$NODE_ROOT/bin/node" ]; then
  mkdir -p "$NODE_ROOT"
  curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
    | tar -xJ --strip-components=1 -C "$NODE_ROOT"
fi
export PATH=$NODE_ROOT/bin:$PATH
node --version; npm --version

echo "== uv =="
command -v uv >/dev/null 2>&1 || pip install --quiet uv
uv --version

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
grep -q "FAILED" "$STATUS" && exit 1
echo "ALL_RORQUAL_LOGIN_SETUP_DONE"
