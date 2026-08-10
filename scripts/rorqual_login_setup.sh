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

echo "== refs/main pins for offline by-name model loads =="
# Models are prefetched at pinned SHAs, but several official harnesses load
# tokenizers/models BY NAME (no revision). Offline resolution of "main" needs
# refs/main in the cache, which `hf download --revision <sha>` does not write.
# Pin main == the frozen revision (job 18752420 died on exactly this).
HUB=${HF_HUB_CACHE:-/scratch/a32du/data/hf_cache/hub}
while read -r repo sha; do
  d=$HUB/$repo
  [ -d "$d/snapshots/$sha" ] || { echo "warn: snapshot missing for $repo@$sha"; continue; }
  mkdir -p "$d/refs"; printf '%s' "$sha" > "$d/refs/main"
done <<'REFEOF'
models--Qwen--Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3 395b18f1fecee52f1b51fb22f898c220f0a08ec3
models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3 d65f11c88d3c129a01466f0c154aaad7d9b09225
models--alireza7--GrepSeek-Qwen3.5-9B-GRPO a79563970cfdd2ced3cc5fde481737d0ebea6fa4
models--Tevatron--AgentIR-4B e31abb637caa227c4b7d04176a24ecff1bcb10f4
models--intfloat--e5-base-v2 f52bf8ec8c7124536f0efb74aca902b2995e5bcd
models--Qwen--Qwen3-Embedding-4B 5cf2132abc99cad020ac570b19d031efec650f2b
REFEOF

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
