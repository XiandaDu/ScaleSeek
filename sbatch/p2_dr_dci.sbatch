#!/usr/bin/env bash
#SBATCH -J p2_dr_dci
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-2 DR-DCI (Wiki-18 official E5 pull route). The official
# searchr1_wiki18_dci_server maps FAISS position -> corpus line; our frozen E5
# index is provably in corpus order (ordered_doc_ids_sha256 matches the corpus
# manifest), and its query prefix/pooling/normalization match the server's
# encoder exactly, so the manifest-pinned project E5 index backs the official
# /retrieve + /document service (recorded in the run record).
source /home/a32du/ScaleSeek/sbatch/common.sh
# Rorqual: root_flat pull materialization can write up to ~0.9M workspace files
# for 1,500 questions — too close to the 1M-file /scratch quota. The harness
# therefore runs with a NODE-LOCAL out-root; result files sync to /scratch every
# 30 min and the full raw tree (incl. workspaces) is archived at the end, which
# also preserves the official pull artifacts TASK.md requires.
RAW=$REPO/$RESULTS_DIR/dr_dci_raw          # scratch-side synced copy
RAW_LOCAL=$TMPDIR/dr_dci_raw               # node-local live out-root
sync_raw() {
  mkdir -p "$RAW"
  rsync -a --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' \
    "$RAW_LOCAL/" "$RAW/" 2>/dev/null || true
}
trap 'code=$?; stop_vllm; sync_raw; cleanup_tmpdir; advance_lane; exit $code' EXIT
export PATH=/scratch/a32du/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
PULL_PORT=$((PORT + 1))
# out-root stays ABSOLUTE: run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
mkdir -p "$RESULTS_DIR" "$RAW" "$RAW_LOCAL"

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
pi_vllm_provider "$PORT" Qwen/Qwen3.5-9B
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
( while sleep 1800; do sync_raw; done ) & SYNC_PID=$!
"$PY" scripts/run_official_baseline.py dr_dci --repo-root "$OFFICIAL_ROOT" \
  --full-eval --model-revision "$GEN_REV" \
  --record $RESULTS_DIR/${DS}_dr_dci.run.json -- \
  --model Qwen/Qwen3.5-9B --provider vllm \
  --agent-dir "$PI_CODING_AGENT_DIR" \
  --max-turns 300 --runtime-context-level level3 \
  --tools read,bash,pull \
  --pull-view-mode hardlink --pull-terminal-tools --pull-backend local \
  --pull-base-url "$DCI_PULL_BASE_URL" \
  --pull-document-base-url "$DCI_PULL_DOCUMENT_BASE_URL" \
  --pull-layout root --pull-prompt-mode rank_aware \
  --pull-materialization-mode root_flat_disclosed \
  --pull-min-top-k 300 --pull-max-top-k 600 --pull-max-queries 10 \
  --pull-preview-mode ranked --pull-preview-limit 20 \
  --max-concurrency "${CONC:-4}" --skip-judge \
  --pi-thinking-level high --node-max-old-space-size-mb 8192 \
  --dataset "$DATA/dcilite_datasets/${DS}_cap1500.jsonl" \
  --corpus-dir "$OFFICIAL_ROOT/dr_dci/corpus/wiki18_empty" \
  --output-root "$RAW_LOCAL"
stop_vllm
kill "$SYNC_PID" 2>/dev/null || true
sync_raw
echo "== archive full raw tree (incl. workspaces) to scratch =="
command -v zstd >/dev/null 2>&1 || module load zstd 2>/dev/null || true
tar -C "$TMPDIR" -cf "$RAW/dr_dci_raw_full.tar.zst.partial" -I 'zstd -T8 -3' dr_dci_raw \
  && mv "$RAW/dr_dci_raw_full.tar.zst.partial" "$RAW/dr_dci_raw_full.tar.zst" || true

echo "== normalize + metrics =="
# Pick the aggregate file explicitly, and never pipe find into head.
# `find ... | head -1` sends SIGPIPE upstream when head closes; under
# `set -o pipefail` that exits 141 and kills the job AFTER the evaluation has
# already finished. dci (7340) and dr_dci (7341) both died exactly this way on
# 2026-08-08 with complete 1,500-row outputs on disk. It was also non-deterministic:
# find's first hit can be a per-question events.jsonl rather than the aggregate.
# Read the COMPLETE node-local tree, never the scratch copy: the background
# rsync can capture results.jsonl mid-append and the truncated line kills
# normalize (job 18753890). $RAW remains the fallback for recover-style reruns.
if [ -f "$RAW_LOCAL/results.jsonl" ]; then INPUT=$RAW_LOCAL/results.jsonl
elif [ -f "$RAW/results.jsonl" ]; then INPUT=$RAW/results.jsonl
else INPUT=$(find "$RAW" -name "*.jsonl" -print -quit); fi
"$PY" scripts/normalize_official_results.py --method dr_dci \
  --input "$INPUT" \
  --run-record $RESULTS_DIR/${DS}_dr_dci.run.json \
  --dataset-manifest "$DS_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/${DS}_cap1500.jsonl" \
  --id-field query_id \
  --output $RESULTS_DIR/${DS}_dr_dci.jsonl
sane $RESULTS_DIR/${DS}_dr_dci.jsonl
"$PY" scripts/compute_metrics.py --results $RESULTS_DIR/${DS}_dr_dci.jsonl \
  --dataset "$DS" --agent dr_dci --expect-ids "$DATA/dcilite_datasets/${DS}_cap1500.manifest.json" \
  --out $RESULTS_DIR/${DS}_dr_dci.metrics.json
echo "ALL_DONE ${DS}_dr_dci"
