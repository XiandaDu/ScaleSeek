#!/usr/bin/env bash
#SBATCH -J p2_dr_dci
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 DR-DCI (Wiki-18 official E5 pull route). The official
# searchr1_wiki18_dci_server maps FAISS position -> corpus line; our frozen E5
# index is provably in corpus order (ordered_doc_ids_sha256 matches the corpus
# manifest), and its query prefix/pooling/normalization match the server's
# encoder exactly, so the manifest-pinned project E5 index backs the official
# /retrieve + /document service (recorded in the run record).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
assert_gpus_usable || exit 4
export PATH=/data/rech/mofengra/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
PULL_PORT=$((PORT + 1))
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/results/phase2/dr_dci_raw
mkdir -p results/phase2 "$RAW"
assert_raw_not_poisoned "$RAW" || exit 4

echo "== official wiki18 E5 retriever service (project E5 index) =="
SVC_LOG=$REPO/logs/dr_dci_retriever_$SLURM_JOB_ID.log
# Interpreter, deliberately: the official script runs under OUR env, not `uv run`.
# The dr_dci checkout's uv environment has no faiss, so `uv run` fails at
# `import faiss` (job 7331, 2026-08-06). Adding faiss there would edit uv.lock in
# the pinned checkout and trip the commit gate.
#
# This is a tool process, not the method: it is the official
# searchr1_wiki18_dci_server.py, unmodified, serving OUR frozen E5 index over
# HTTP. Same script, same index, same FAISS query path -- only the interpreter
# differs, which cannot change retrieval semantics. The Pi agent harness itself
# still runs under the official env via run_official_baseline.py.
( cd "$OFFICIAL_ROOT/dr_dci" && "$PY" tools/dense_retriever/searchr1_wiki18_dci_server.py \
    --index-path "$E5_INDEX_DIR/index.faiss" \
    --corpus-path "$CORPUS_FILE" \
    --model-name-or-path intfloat/e5-base-v2 \
    --encoder-device cpu \
    --host 127.0.0.1 --port "$PULL_PORT" \
    --max-top-k 1000 > "$SVC_LOG" 2>&1 ) &
SVC_WAIT=0
until curl -sf "http://127.0.0.1:$PULL_PORT/health" >/dev/null 2>&1; do
  SVC_WAIT=$((SVC_WAIT+10))
  [ "$SVC_WAIT" -ge 1800 ] && { echo "FATAL: retriever service not healthy; see $SVC_LOG"; exit 4; }
  sleep 10
done
echo "service healthy after ${SVC_WAIT}s; probe /retrieve:"
curl -sf -X POST "http://127.0.0.1:$PULL_PORT/retrieve" \
  -H 'Content-Type: application/json' -d '{"query": "capital of France", "top_k": 3}' \
  | head -c 500 || { echo "FATAL: /retrieve probe failed"; exit 4; }
echo

# Pi rejects tool_choice:"auto" unless the server advertises tool calling;
# without these it fails with `400 status code (no body)`
# (dr_dci assets/docs/setup.md, and BASELINE_SOURCES.md already noted it).
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
waitp "$PORT" 3600

# The Pi agent ignores OPENAI_BASE_URL; it resolves providers from
# $PI_CODING_AGENT_DIR/models.json. pi_vllm_provider writes that file.
pi_vllm_provider "$PORT" Qwen/Qwen3.5-9B "$OFFICIAL_ROOT/dr_dci"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"   # for the Python-side tools
export OPENAI_API_KEY=dummy
export DCI_PULL_BASE_URL="http://127.0.0.1:$PULL_PORT/retrieve"
export DCI_PULL_DOCUMENT_BASE_URL="http://127.0.0.1:$PULL_PORT/document"
mkdir -p "$OFFICIAL_ROOT/dr_dci/corpus/wiki18_empty"

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
# Build the invocation once so the smoke question and the full run are
# provably the same command apart from --dataset/--output-root.
HARNESS=("$PY" scripts/run_official_baseline.py dr_dci --repo-root "$OFFICIAL_ROOT"   --full-eval --model-revision "$GEN_REV"   --record results/phase2/popqa_dr_dci.run.json --   --model Qwen/Qwen3.5-9B --provider vllm   --max-turns 300 --runtime-context-level level3   --tools read,bash,pull   --pull-view-mode hardlink --pull-terminal-tools --pull-backend local   --pull-base-url "$DCI_PULL_BASE_URL"   --pull-document-base-url "$DCI_PULL_DOCUMENT_BASE_URL"   --pull-layout root --pull-prompt-mode rank_aware   --pull-materialization-mode root_flat_disclosed   --pull-min-top-k 300 --pull-max-top-k 600 --pull-max-queries 10   --pull-preview-mode ranked --pull-preview-limit 20   --max-concurrency "${CONC:-4}" --skip-judge   --pi-thinking-level high --node-max-old-space-size-mb 8192 --corpus-dir "$OFFICIAL_ROOT/dr_dci/corpus/wiki18_empty")
pi_smoke_gate "$DATA/dcilite_datasets/popqa_cap1500.jsonl" "$RAW.smoke" "${HARNESS[@]}" || exit 5
"${HARNESS[@]}" \
  --dataset "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --output-root "$RAW"
stop_vllm

echo "== normalize + metrics =="
# Pick the aggregate file explicitly, and never pipe find into head.
# `find ... | head -1` sends SIGPIPE upstream when head closes; under
# `set -o pipefail` that exits 141 and kills the job AFTER the evaluation has
# already finished. dci (7340) and dr_dci (7341) both died exactly this way on
# 2026-08-08 with complete 1,500-row outputs on disk. It was also non-deterministic:
# find's first hit can be a per-question events.jsonl rather than the aggregate.
if [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name "*.jsonl" -print -quit); fi
"$PY" scripts/normalize_official_results.py --method dr_dci \
  --input "$INPUT" \
  --run-record results/phase2/popqa_dr_dci.run.json \
  --dataset-manifest "$POPQA_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --id-field query_id \
  --output results/phase2/popqa_dr_dci.jsonl
sane results/phase2/popqa_dr_dci.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dr_dci.jsonl \
  --dataset popqa --agent dr_dci --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_dr_dci.metrics.json
echo "ALL_DONE popqa_dr_dci"
