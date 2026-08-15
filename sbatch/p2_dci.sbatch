#!/usr/bin/env bash
#SBATCH -J p2_dci
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH --exclude=octal25,octal30,octal31,octal35
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 DCI: official DCI-Agent-Lite harness (Pi, 300 turns, L3) over the
# materialized Wiki-18 corpus with Qwen3.5-9B served via vLLM.
# TMPDIR redirection in common.sh is load-bearing here: Pi writes every bash
# tool stdout to os.tmpdir(), which on octal30 is tmpfs (RAM).
#
# --mem=300G, NOT the 56G a 9B server needs: the corpus is staged into node
# tmpfs below (~125G: 21M files x 4K page + dentry/inode slab, unreclaimable),
# and cgroup v2 charges tmpfs pages to the job that wrote them. 200G was NOT
# enough on top of that: vLLM host RSS (~35G) + 8 node agents (8G heap each,
# buffering 100s-of-MB event streams) live at the ceiling, and the kernel then
# fails socket-buffer allocation mid-RPC -- 7417 decayed from 6.8 to 2.0
# completions/h with 11% of rows dead on `write ENOBUFS` node crashes and
# ~51.5KB "Unterminated string" truncations (the same ENOBUFS-as-JSON-
# truncation disguise first seen on octal30's tmpfs /tmp).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
LOCAL_CORPUS=/dev/shm/mofengra_dci_wiki_corpus
cleanup_local_corpus() { rm -rf "$LOCAL_CORPUS" 2>/dev/null || true; }
trap 'code=$?; stop_vllm; cleanup_tmpdir; cleanup_local_corpus; advance_lane; exit $code' EXIT
assert_gpus_usable || exit 4
export PATH=/data/rech/mofengra/tools/node/bin:$PATH
PORT=$((10000 + SLURM_JOB_ID % 20000))
# ABSOLUTE. run_official_baseline.py launches the harness with
# cwd=<official checkout> (subprocess.run(..., cwd=repo)), so a relative
# --out_dir/--output-root lands inside the OFFICIAL repo, not ours, and the
# post-run `find $RAW` then reports 'no output jsonl' after the whole eval
# already ran. That silently cost grepseek 2d17h of GPU on 2026-08-02.
RAW=$REPO/results/phase2/dci_raw
mkdir -p results/phase2 "$RAW"
assert_raw_not_poisoned "$RAW" || exit 4

[ -d "$DATA/dci_wiki_corpus" ] && [ -n "$(ls -A "$DATA/dci_wiki_corpus" 2>/dev/null)" ] || {
  echo "FATAL: expanded DCI corpus missing; p0_dci_corpus must run first"; exit 4; }

# --- Stage the corpus into node-local tmpfs ---------------------------------
# The 21,015,324-file official layout is unusable over NFS: close-to-open
# consistency makes every open() a GETATTR round-trip, so ONE bare rg scan
# takes ~50min warm and 3h30m cold (7376 measured both). At 1-3 scans per
# question x 1500 questions that is months, and a sustained metadata storm on
# the shared filer. Rebuilding the tree in tmpfs with the SAME official
# expand script costs ~15min once and makes scans RAM-speed. Same files, same
# layout, same tools -- storage locality is environment, not method.
# /dev/shm, NOT /tmp: octal41 mounts /tmp with nr_inodes=1048576, and the tree
# needs ~42M inodes (21M leaf dirs + 21M files) -- 7381 died ENOSPC at shard
# 00193 with 252G of bytes still free. /dev/shm has no inode cap beyond the
# default (66M on octal41, probe job 7382).
SHM=$(dirname "$LOCAL_CORPUS")
[ "$(stat -f -c %T "$SHM")" = tmpfs ] || {
  echo "FATAL: $SHM on $(hostname) is not tmpfs; local corpus staging needs it."; exit 4; }
SHM_AVAIL_G=$(df -BG --output=avail "$SHM" | tail -1 | tr -dc 0-9)
SHM_IFREE=$(df --output=iavail "$SHM" | tail -1 | tr -dc 0-9)
[ "${SHM_AVAIL_G:-0}" -ge 160 ] || {
  echo "FATAL: $SHM has only ${SHM_AVAIL_G}G free; the staged corpus needs ~130G."; exit 4; }
[ "${SHM_IFREE:-0}" -ge 45000000 ] || {
  echo "FATAL: $SHM has only ${SHM_IFREE} free inodes; the tree needs ~42M."
  echo "       (/tmp-style nr_inodes caps kill the staging: 7381.)"; exit 4; }
EXPECT_DOCS=21015324
Q=5253831   # EXPECT_DOCS/4 exactly, same split as p0_dci_corpus
echo "== staging corpus into $LOCAL_CORPUS (official expand script, 4 workers) =="
mkdir -p "$LOCAL_CORPUS"
( cd "$OFFICIAL_ROOT/dr_dci"
  PIDS=()
  for i in 0 1 2 3; do
    uv run python scripts/qa/expand_wiki18_jsonl_to_files.py \
      --input "$CORPUS_FILE" --output-dir "$LOCAL_CORPUS" \
      --start-offset $((i * Q)) --limit $Q \
      --resume --progress-every 2000000 \
      > "$REPO/logs/dci_local_expand_${SLURM_JOB_ID}_$i.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  exit $FAIL ) || {
    echo "FATAL: local corpus staging failed; see logs/dci_local_expand_${SLURM_JOB_ID}_*.log"
    tail -3 "$REPO"/logs/dci_local_expand_${SLURM_JOB_ID}_*.log; exit 4; }
# Top-level shard count is a cheap structural check (full find over 21M tmpfs
# files would burn minutes for little extra assurance; the expand script's own
# exit codes cover per-range completeness).
SHARDS=$(ls "$LOCAL_CORPUS" | wc -l)
[ "$SHARDS" -ge 21016 ] || {
  echo "FATAL: staged corpus has $SHARDS top-level shards, expected >=21016"; exit 4; }
echo "== staged: $SHARDS shards in $LOCAL_CORPUS =="
# Pi rejects tool_choice:"auto" unless the server advertises tool calling;
# without these it fails with `400 status code (no body)`.
# The parser must match the MODEL's chat template, not the harness docs:
# Qwen3.5's template emits XML function calls (<tool_call><function=...>) and its
# model card (README "Tool Call") prescribes `--tool-call-parser qwen3_coder`.
# The dr_dci setup.md said `hermes`, but that was written for the JSON
# <tool_call> format of the Qwen3-era generator the authors used. With hermes,
# vLLM consumed every tool call and emitted no tool_calls delta, so Pi saw
# stop-reason "stop" with content "\n\n" and recorded a "completed" run with an
# empty final_text at turn 1 -- the whole 7340/7341/7351/7354/7355/7359/7360
# blank-answer family.
# 131072 window: 32k dies mid-trajectory on long agent runs (7377 request 46,
# see p2_dr_dci.sbatch); exported so pi_vllm_provider declares the same
# contextWindow in models.json.
export VLLM_MAX_LEN=131072
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
waitp "$PORT" 3600

# The Pi agent ignores OPENAI_BASE_URL; it resolves providers from
# $PI_CODING_AGENT_DIR/models.json. pi_vllm_provider writes that file.
pi_vllm_provider "$PORT" Qwen/Qwen3.5-9B "$OFFICIAL_ROOT/dci_agent_lite"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"   # for the Python-side tools
export OPENAI_API_KEY=dummy

# Scope: the dataset file below IS the sanctioned <=1500-question subset (TASK.md
# "评测规模上限"). `--full-eval` here is the official-harness gate -- it forbids
# any further --limit/-n/--offset and pins the frozen model/revision -- so the
# two together mean "run the entire capped file, unsliced". compute_metrics then
# verifies the produced id set against the subset manifest (--expect-ids).
# Build the invocation once so the smoke question and the full run are
# provably the same command apart from --dataset/--output-root.
HARNESS=("$PY" scripts/run_official_baseline.py dci --repo-root "$OFFICIAL_ROOT"   --full-eval --model-revision "$GEN_REV"   --record results/phase2/popqa_dci.run.json --   --model Qwen/Qwen3.5-9B --provider vllm   --max-turns 300 --runtime-context-level level3   --tools read,bash --max-concurrency "${CONC:-8}"   --node-max-old-space-size-mb 8192   --judge-model Qwen/Qwen3.5-9B --corpus-dir "$LOCAL_CORPUS")
# CONC default 8, not 2: DCI trajectories are heavy-tailed (median 82s but
# mean 1750s, max 17h over the first 50 questions of 7385) -- at conc 2 one
# monster question idles half the run, 27 days projected. 8 lanes drown the
# tail (~4 days). node-max-old-space matches the dr_dci scripts; two of the
# first 50 questions died with "RPC process exited unexpectedly" node crashes.
# The official prompt tells the model to use rg; without it on PATH the agent
# falls back to unbounded `grep -r` over 14GB (7361: 4h49m on one question).
command -v rg >/dev/null || { echo "FATAL: rg not on PATH (see common.sh tools/bin)"; exit 4; }
pi_smoke_gate "$DATA/dcilite_datasets/popqa_cap1500.jsonl" "$RAW.smoke" "${HARNESS[@]}" || exit 5

# Resume seal: the outer script only skips a completed question dir if its
# JUDGE succeeded, and our judge can never succeed (hardcoded api.openai.com,
# always 401). Unsealed, every resume relaunches completed dirs, the inner
# launcher exits 2 ("Cannot resume completed run"), and good answers become
# error rows -- 7351's poison, and 45/50 of 7385's rows were headed the same
# way. Synthesize a judge verdict (is_correct=false, never read by our popqa
# EM/F1) so judge_result_succeeded() passes; the real 401 record is kept as
# eval_result.judgefail.json. Result-format adaptation only; idempotent.
"$PY" - "$RAW" <<'SEALEOF'
import json, pathlib, sys
raw = pathlib.Path(sys.argv[1])
sealed = 0
for qd in (p for p in raw.iterdir() if p.is_dir()) if raw.is_dir() else []:
    try:
        state = json.loads((qd / "state.json").read_text())
        row = json.loads((qd / "result.json").read_text())
    except (OSError, ValueError):
        continue
    if state.get("status") != "completed" or not (row.get("final_text") or "").strip():
        continue
    ev = qd / "eval_result.json"
    try:
        j = json.loads(ev.read_text())
    except (OSError, ValueError):
        j = None
    if isinstance(j, dict) and not j.get("error") and isinstance(j.get("is_correct"), bool):
        continue
    if isinstance(j, dict):
        (qd / "eval_result.judgefail.json").write_text(json.dumps(j, indent=2))
    ev.write_text(json.dumps({
        "judge_model": "offline-seal",
        "judged_at": "1970-01-01T00:00:00+00:00",
        "judge_status": "completed",
        "is_correct": False,
        "normalized_prediction": None,
        "reason": "synthetic seal so resume skips this completed dir; real judge "
                  "record in eval_result.judgefail.json; popqa scoring ignores judges",
        "error": None, "attempt_count": 0, "usage": {},
        "cost_estimate_usd": {"input_cost": 0.0, "cached_input_cost": 0.0,
                              "output_cost": 0.0, "total_cost": 0.0},
    }, indent=2))
    sealed += 1
print(f"[resume-seal] sealed {sealed} completed dirs against judge-failure re-runs")
SEALEOF

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
# Same field mapping as p2_dr_dci (7383's all-null lesson). No answer-regex:
# the official DCI benchmark prompt requests no answer format, so final_text
# is free prose -- the official metric for it is an LLM judge, and strict
# EM against prose is a known caveat to raise in the phase-2 report, not
# something to paper over with an invented extractor.
"$PY" scripts/normalize_official_results.py --method dci \
  --input "$INPUT" \
  --run-record results/phase2/popqa_dci.run.json \
  --dataset-manifest "$POPQA_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --id-field query_id \
  --prediction-field final_text \
  --gold-field gold_answer --gold-split ' / ' \
  --finish-field run_status --turns-field turn_count \
  --tool-calls-field tool_metrics.call_count \
  --output results/phase2/popqa_dci.jsonl
sane results/phase2/popqa_dci.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dci.jsonl \
  --dataset popqa --agent dci --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_dci.metrics.json
echo "ALL_DONE popqa_dci"
