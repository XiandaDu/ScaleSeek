#!/usr/bin/env bash
# =====================================================================================
# ScaleSeek environment -- one file.
#
#   source setup_env.sh               activate.
#                                     * inside a SLURM job: the environment is BUILT on
#                                       node-local $SLURM_TMPDIR, once per job, offline,
#                                       from the hash-pinned wheelhouse, so no import ever
#                                       touches /scratch;
#                                     * on a login node: the shared $SCALESEEK_ROOT/envs/main
#                                       (used only for downloads: run.py prefetch).
#   bash setup_env.sh --install       LOGIN node: uv, CPython, wheelhouse, login env, repos
#   bash setup_env.sh --build-wheels  inside a job, once: wheels for the sdist-only packages
#   bash setup_env.sh --job-agentir   inside a job: build the AgentIR env (run.py calls it)
#   bash setup_env.sh --relock        re-resolve env/main.lock.txt from env/main.in
#
# Why node-local: on Narval /scratch (Lustre) opening a small file can take seconds when
# the metadata servers are loaded, and a Python process opens thousands (one
# `import transformers` measured 931 s).  Big sequential reads are fast, so the wheels
# live on /scratch and are unpacked onto local NVMe per job (~2 min).
#
# Under $SCALESEEK_ROOT (default /scratch/$USER/scaleseek_v2):
#   tools/uv/uv, tools/pipenv          pinned uv binary (sha256-checked); pip for downloads
#   downloads/python-mirror/           pinned CPython tarball (uv checks its sha256)
#   wheelhouse/{main,agentir}          every locked wheel, hash-checked at install
#   wheelhouse/{main,agentir}-extra    wheels of packages PyPI ships only as sdist (built
#                                      once by --build-wheels; sha256 in env/*_extra.txt)
#   wheelhouse/src                     those sdists / sources
#   envs/main                          login-node env;   hf/  HF_HOME
# Alliance traps: its python stack injects PYTHONPATH (a manylinux override that rejects
# manylinux wheels) and PIP_CONFIG_FILE (its own wheelhouse): both are unset, and we never
# load the `python` module.
# =====================================================================================

_SS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCALESEEK_REPO="$_SS_REPO"
export SCALESEEK_ROOT="${SCALESEEK_ROOT:-/scratch/$USER/scaleseek_v2}"

_SS_UV_VERSION="0.12.18"
_SS_UV_SHA256="89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16"
_SS_PYTHON="3.12.12"
_SS_PYTHON_TARBALL="20260211/cpython-3.12.12+20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
_SS_PYTHON_SHA256="1dbaa624a09e15afe7efbdac08d42993135a68db8d34f986ef6977a6d77bdc3c"
_SS_JAVA_MODULE="java/21.0.1"
# Locked packages that PyPI ships only as sdists (main env).  They are left out of the
# hashed sync and installed from wheels built once by --build-wheels.
_SS_MAIN_SDIST="pyserini"
# AgentIR extras (already excluded from env/agentir.lock.txt by its export):
_SS_FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
_SS_TEVATRON_COMMIT="6555f12c844c64f020f5762096405a7ce5148b77"   # pinned by repos/AgentIR/uv.lock
_SS_JIEBA_REQ="jieba==0.42.1 --hash=sha256:055ca12f62674fafed09427f176506079bc135638a14e23e25be909131928db2"

_ss_uv() { env -u PYTHONPATH -u PIP_CONFIG_FILE "$SCALESEEK_ROOT/tools/uv/uv" "$@"; }
_ss_pip() { env -u PYTHONPATH -u PIP_CONFIG_FILE "$SCALESEEK_ROOT/tools/pipenv/bin/python" -m pip "$@"; }
_ss_fail() { echo "setup_env.sh: ERROR: $*" >&2; return 1; }
_ss_local_root() { echo "$SLURM_TMPDIR/scaleseek_env"; }

_ss_common_env() {
  unset PYTHONPATH PIP_CONFIG_FILE PYTHONHOME
  export UV_PYTHON_PREFERENCE=only-managed UV_NO_PROGRESS=1 UV_HTTP_TIMEOUT=300
  export HF_HOME="$SCALESEEK_ROOT/hf" HF_HUB_CACHE="$SCALESEEK_ROOT/hf/hub"
  # Offline by default everywhere.  `run.py prefetch` turns the network on for itself.
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  # FlashInfer's sampler JIT-compiles with nvcc, which the runtime does not ship.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
  export PATH="$SCALESEEK_ROOT/tools/uv:$PATH"
}

_ss_modules() {
  command -v module >/dev/null 2>&1 || return 0
  module load StdEnv/2023 >/dev/null 2>&1 || true
  module load "$_SS_JAVA_MODULE" >/dev/null 2>&1 || _ss_fail "cannot load $_SS_JAVA_MODULE (pyserini needs Java 21)"
}

# ------------------------------------------------------------------------- lock helpers
_ss_strip_lock() {  # <lock> <names...>: the lock without those package blocks
  local lock="$1"; shift
  awk -v drop=" $* " 'BEGIN{skip=0}
    /^[A-Za-z0-9_.-]+==/ { split($1, a, "=="); skip = index(drop, " " a[1] " ") > 0 }
    /^#/ { skip = 0 }
    !skip { print }' "$lock"
}

_ss_lock_entry() {  # <lock> <name>: "name==ver --hash=..." for one package
  awk -v n="$2" '/^[A-Za-z0-9_.-]+==/{p = (index($1, n "==") == 1)} p && /^    --hash/ {h = h " " $1}
    p && /^[A-Za-z0-9_.-]+==/ {s = $1} END {print s h}' "$1"
}

_ss_install_extras() {  # <venv> <extras list> <dir>: wheels by name + sha256, no deps
  local venv="$1" list="$2" dir="$3" f want got wheels=()
  while read -r f want; do
    case "$f" in ''|\#*) continue ;; esac
    got="$(sha256sum "$dir/$f" 2>/dev/null | cut -d' ' -f1)"
    [ "$got" = "$want" ] || { _ss_fail "$dir/$f: sha256 '$got' does not match $(basename "$list")"; return 1; }
    wheels+=("$dir/$f")
  done < "$list"
  [ ${#wheels[@]} -eq 0 ] && return 0
  UV_CACHE_DIR="$(_ss_local_root)/uv_cache" UV_LINK_MODE=copy _ss_uv pip install -q \
      --python "$venv/bin/python" --offline --no-index --no-deps "${wheels[@]}"
}

# ------------------------------------------------------------------------- job-local envs
_ss_local_python() {  # offline CPython install from the pinned local mirror
  local L; L="$(_ss_local_root)"
  UV_PYTHON_INSTALL_DIR="$L/python" \
  UV_PYTHON_INSTALL_MIRROR="file://$SCALESEEK_ROOT/downloads/python-mirror" \
  UV_CACHE_DIR="$L/uv_cache" _ss_uv python install --no-bin "$_SS_PYTHON" >/dev/null 2>&1 \
    || _ss_fail "offline CPython $_SS_PYTHON install failed (mirror under downloads/python-mirror)"
}

_ss_local_env() {  # <name> <lock> <sdist names> : builds $SLURM_TMPDIR/scaleseek_env/<name>
  local name="$1" lock="$2" sdist="$3" L venv want fd t0=$SECONDS
  L="$(_ss_local_root)"; venv="$L/$name"
  want="$(cat "$lock" "$_SS_REPO/env/${name}_extra.txt" | sha256sum | cut -d' ' -f1)"
  [ -f "$venv/.scaleseek_ok" ] && [ "$(cat "$venv/.scaleseek_ok")" = "$want" ] && return 0
  mkdir -p "$L"
  # Several job steps may source this at once: one builds, the others wait.
  exec {fd}>"$L/.$name.lock"; flock "$fd"
  if [ -f "$venv/.scaleseek_ok" ] && [ "$(cat "$venv/.scaleseek_ok")" = "$want" ]; then flock -u "$fd"; return 0; fi
  {
    _ss_local_python &&
    rm -rf "$venv" &&
    UV_PYTHON_INSTALL_DIR="$L/python" _ss_uv venv -q --python "$_SS_PYTHON" "$venv" &&
    _ss_strip_lock "$lock" $sdist > "$L/$name.sync.txt" &&
    UV_CACHE_DIR="$L/uv_cache" UV_LINK_MODE=copy _ss_uv pip sync -q --python "$venv/bin/python" \
        --offline --no-index --find-links "$SCALESEEK_ROOT/wheelhouse/$name" --require-hashes "$L/$name.sync.txt" &&
    _ss_install_extras "$venv" "$_SS_REPO/env/${name}_extra.txt" "$SCALESEEK_ROOT/wheelhouse/$name-extra" &&
    echo "$want" > "$venv/.scaleseek_ok"
  } || { flock -u "$fd"; _ss_fail "could not build the job-local $name env"; return 1; }
  flock -u "$fd"
  echo "setup_env.sh: built job-local env $name in $((SECONDS - t0)) s ($venv)" >&2
}

# ------------------------------------------------------------------------------ activate
_ss_activate() {
  _ss_modules || return 1
  _ss_common_env
  local venv
  if [ -n "${SLURM_TMPDIR:-}" ]; then
    _ss_local_env main "$_SS_REPO/env/main.lock.txt" "$_SS_MAIN_SDIST" || return 1
    venv="$(_ss_local_root)/main"
    export SCALESEEK_AGENTIR_PYTHON="$(_ss_local_root)/agentir/bin/python"
  else
    venv="$SCALESEEK_ROOT/envs/main"
    [ -x "$venv/bin/python" ] || { _ss_fail "no login env $venv (bash setup_env.sh --install)"; return 1; }
    export SCALESEEK_AGENTIR_PYTHON="(only inside a SLURM job)"
  fi
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
}

# ------------------------------------------------------------------------------ install
_ss_install_uv() {
  local dst="$SCALESEEK_ROOT/tools/uv" tmp
  if [ -x "$dst/uv" ] && "$dst/uv" --version | grep -q " $_SS_UV_VERSION "; then return 0; fi
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/uv.tar.gz" \
    "https://github.com/astral-sh/uv/releases/download/$_SS_UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" || return 1
  echo "$_SS_UV_SHA256  $tmp/uv.tar.gz" | sha256sum -c - >/dev/null || { _ss_fail "uv sha256 mismatch"; return 1; }
  mkdir -p "$dst" && tar -xzf "$tmp/uv.tar.gz" -C "$dst" --strip-components=1 && rm -rf "$tmp"
}

_ss_install_python_mirror() {
  local f="$SCALESEEK_ROOT/downloads/python-mirror/$_SS_PYTHON_TARBALL"
  if [ ! -f "$f" ]; then
    mkdir -p "$(dirname "$f")"
    curl -fsSL --retry 5 -o "$f" \
      "https://releases.astral.sh/github/python-build-standalone/releases/download/${_SS_PYTHON_TARBALL/+/%2B}" || return 1
  fi
  echo "$_SS_PYTHON_SHA256  $f" | sha256sum -c - >/dev/null || { _ss_fail "CPython tarball sha256"; return 1; }
  ln -sf "$(basename "$f")" "$(dirname "$f")/$(basename "${f/+/%2B}")"   # uv may ask for %2B
}

_ss_pip_download() {  # <requirements> <dest>: wheels only, hash-checked, compute-node tags
  local plat=() v
  for v in $(seq 34 -1 17); do plat+=(--platform "manylinux_2_${v}_x86_64"); done
  plat+=(--platform manylinux2014_x86_64 --platform manylinux2010_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64)
  mkdir -p "$2"
  _ss_pip download --no-deps --only-binary=:all: --implementation cp --python-version 3.12 \
      --abi cp312 --abi abi3 --abi none "${plat[@]}" --timeout 300 --retries 10 -d "$2" -r "$1"
}

_ss_install() {
  [ -z "${SLURM_JOB_ID:-}" ] || { _ss_fail "inside a job: compute nodes have no internet; use a login node"; return 1; }
  set -e
  unset PYTHONPATH PIP_CONFIG_FILE PYTHONHOME
  export UV_CACHE_DIR="$SCALESEEK_ROOT/.uv_cache" UV_PYTHON_INSTALL_DIR="$SCALESEEK_ROOT/tools/python"
  export UV_PYTHON_PREFERENCE=only-managed UV_HTTP_TIMEOUT=300 UV_LINK_MODE=hardlink
  local R="$SCALESEEK_ROOT" tmp; tmp="$(mktemp -d)"
  mkdir -p "$R"/{tools,envs,hf,repos,data,indexes,runs,smoke,downloads} "$R"/wheelhouse/{main,agentir,main-extra,agentir-extra,src}
  echo "--- uv + CPython"
  _ss_install_uv; _ss_install_python_mirror; _ss_uv python install --no-bin "$_SS_PYTHON"
  [ -x "$R/tools/pipenv/bin/python" ] || _ss_uv venv --python "$_SS_PYTHON" "$R/tools/pipenv"
  _ss_uv pip install -q --python "$R/tools/pipenv/bin/python" pip==25.2
  echo "--- wheelhouse/main";    _ss_strip_lock "$_SS_REPO/env/main.lock.txt" $_SS_MAIN_SDIST > "$tmp/main.txt"
  _ss_pip_download "$tmp/main.txt" "$R/wheelhouse/main"
  echo "--- wheelhouse/agentir"; _ss_pip_download "$_SS_REPO/env/agentir.lock.txt" "$R/wheelhouse/agentir"
  echo "--- sources of the sdist-only packages (built in a job by --build-wheels)"
  for n in $_SS_MAIN_SDIST; do _ss_lock_entry "$_SS_REPO/env/main.lock.txt" "$n"; done > "$tmp/sdist.txt"
  echo "$_SS_JIEBA_REQ" >> "$tmp/sdist.txt"
  _ss_pip download --no-deps --no-binary :all: --timeout 300 --retries 10 -d "$R/wheelhouse/src" -r "$tmp/sdist.txt"
  [ -d "$R/wheelhouse/src/tevatron/.git" ] || { git init -q "$R/wheelhouse/src/tevatron";
    git -C "$R/wheelhouse/src/tevatron" remote add origin https://github.com/texttron/tevatron.git; }
  git -C "$R/wheelhouse/src/tevatron" fetch -q --depth 1 origin "$_SS_TEVATRON_COMMIT"
  git -C "$R/wheelhouse/src/tevatron" -c advice.detachedHead=false checkout -q FETCH_HEAD
  local fa; fa="$R/wheelhouse/agentir-extra/$(basename "${_SS_FLASH_ATTN_URL//%2B/+}")"
  [ -f "$fa" ] || curl -fsSL --retry 5 -o "$fa" "$_SS_FLASH_ATTN_URL"
  echo "--- login env envs/main (downloads only; jobs build their own)"
  [ -x "$R/envs/main/bin/python" ] || _ss_uv venv --python "$_SS_PYTHON" "$R/envs/main"
  _ss_uv pip sync --python "$R/envs/main/bin/python" --offline --no-index \
      --find-links "$R/wheelhouse/main" --require-hashes "$tmp/main.txt"
  rm -rf "$tmp"
  set +e
  _ss_activate || return 1
  echo "--- official repositories (pinned commits) + pyserini catalogue"
  HF_HUB_OFFLINE=0 python "$_SS_REPO/run.py" prefetch --only repos
  echo
  echo "Next:  python run.py prefetch                         (login node: models, datasets)"
  echo "       srun ... bash setup_env.sh --build-wheels      (once; record the printed sha256s)"
}

_ss_build_wheels() {  # inside a job, once: wheels for the locked sdist-only packages
  [ -n "${SLURM_TMPDIR:-}" ] || { _ss_fail "--build-wheels runs inside a job"; return 1; }
  _ss_modules; _ss_common_env
  local L="$(_ss_local_root)" R="$SCALESEEK_ROOT" d
  _ss_local_python || return 1
  rm -rf "$L/build" "$L/sdist" && mkdir -p "$L/sdist"
  UV_PYTHON_INSTALL_DIR="$L/python" _ss_uv venv -q --python "$_SS_PYTHON" "$L/build" || return 1
  UV_CACHE_DIR="$L/uv_cache" _ss_uv pip install -q --python "$L/build/bin/python" --offline --no-index \
      --find-links "$R/wheelhouse/main" setuptools || return 1
  for d in pyserini-2.4.0 jieba-0.42.1; do tar -xzf "$R/wheelhouse/src/$d.tar.gz" -C "$L/sdist"; done
  _ss_build_one() {
    UV_CACHE_DIR="$L/uv_cache" _ss_uv build -q --wheel --offline --no-build-isolation \
        --python "$L/build/bin/python" --out-dir "$2" "$1" || _ss_fail "wheel build $1"; }
  _ss_build_one "$L/sdist/pyserini-2.4.0" "$R/wheelhouse/main-extra" || return 1
  _ss_build_one "$L/sdist/jieba-0.42.1" "$R/wheelhouse/agentir-extra" || return 1
  _ss_build_one "$R/wheelhouse/src/tevatron" "$R/wheelhouse/agentir-extra" || return 1
  echo "Record in env/main_extra.txt:";    (cd "$R/wheelhouse/main-extra" && sha256sum pyserini-*.whl | awk '{print $2, $1}')
  echo "Record in env/agentir_extra.txt:"; (cd "$R/wheelhouse/agentir-extra" && sha256sum tevatron-*.whl jieba-*.whl | awk '{print $2, $1}')
}

_ss_relock() {
  unset PYTHONPATH PIP_CONFIG_FILE
  local tmp; tmp="$(mktemp)"
  _ss_uv pip compile "$_SS_REPO/env/main.in" --python-version 3.12 \
      --python-platform x86_64-manylinux_2_34 --generate-hashes --no-header --no-annotate -o "$tmp" || return 1
  { echo "# Full lock of envs/main (generated from env/main.in; hashes are checked at install)."
    echo "# Regenerate: bash setup_env.sh --relock"; cat "$tmp"; } > "$_SS_REPO/env/main.lock.txt"
  rm -f "$tmp"
}

# ------------------------------------------------------------------------------ dispatch
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  _ss_activate
else
  set -uo pipefail
  case "${1:-}" in
    --install)      _ss_install ;;
    --build-wheels) _ss_build_wheels ;;
    --job-agentir)  _ss_modules && _ss_common_env &&
                    _ss_local_env agentir "$_SS_REPO/env/agentir.lock.txt" "" ;;
    --relock)       _ss_relock ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
  esac
fi
