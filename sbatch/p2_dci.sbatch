#!/usr/bin/env bash
#SBATCH -J p2_dci
#SBATCH -p rali
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=56G
#SBATCH -t 14-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 DCI: official DCI-Agent-Lite harness (Pi, 300 turns, L3) over the
# materialized Wiki-18 corpus with Qwen3.5-9B served via vLLM.
# TMPDIR redirection in common.sh is load-bearing here: Pi writes every bash
# tool stdout to os.tmpdir(), which on octal30 is tmpfs (RAM).
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'code=$?; stop_vllm; cleanup_tmpdir; advance_lane; exit $code' EXIT
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
HARNESS=("$PY" scripts/run_official_baseline.py dci --repo-root "$OFFICIAL_ROOT"   --full-eval --model-revision "$GEN_REV"   --record results/phase2/popqa_dci.run.json --   --model Qwen/Qwen3.5-9B --provider vllm   --max-turns 300 --runtime-context-level level3   --tools read,bash --max-concurrency "${CONC:-2}"   --judge-model Qwen/Qwen3.5-9B --corpus-dir "$DATA/dci_wiki_corpus")
# The official prompt tells the model to use rg; without it on PATH the agent
# falls back to unbounded `grep -r` over 14GB (7361: 4h49m on one question).
command -v rg >/dev/null || { echo "FATAL: rg not on PATH (see common.sh tools/bin)"; exit 4; }
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
"$PY" scripts/normalize_official_results.py --method dci \
  --input "$INPUT" \
  --run-record results/phase2/popqa_dci.run.json \
  --dataset-manifest "$POPQA_MANIFEST" \
  --dataset-jsonl "$DATA/dcilite_datasets/popqa_cap1500.jsonl" \
  --id-field query_id \
  --output results/phase2/popqa_dci.jsonl
sane results/phase2/popqa_dci.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_dci.jsonl \
  --dataset popqa --agent dci --expect-ids "$DATA/dcilite_datasets/popqa_cap1500.manifest.json" \
  --out results/phase2/popqa_dci.metrics.json
echo "ALL_DONE popqa_dci"
