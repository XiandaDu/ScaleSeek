#!/usr/bin/env bash
#SBATCH -J p2_agentir
#SBATCH -p rali
# octal41 (503G) and NOTHING smaller: even 16x12.75G resharded pieces die on
# a 257G node -- faiss IndexFlat.add() grows a std::vector, and the doubling
# realloc needs old+new address space transiently (7530: std::bad_alloc at
# piece 13/16, ~154G loaded). 215G of flat codes under a ~250G ceiling is
# arithmetically impossible with an add()-only loader. Eligible for BOTH
# 500G nodes, first to free wins: octal40's silent-killer era (7494/7522/
# 7523, still reproducible on 08-23 probe 7533) appears over -- a foreign
# job has survived 14h+ on it on 08-30 -- and assert_gpus_usable plus the
# text-lookup validity gate turn any relapse into a minutes-scale failure.
# L40S 48G serve the 9B at tp=1: vLLM on GPU0, AgentIR-4B encoder on GPU1.
#SBATCH --exclude=octal25,octal30,octal31,octal35
#SBATCH --gres=gpu:2
#SBATCH -c 24
#SBATCH --mem=450G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# Phase-2 AgentIR: official reasoning-aware OSS agent loop. vLLM Qwen3.5-9B on
# GPUs 0-1; the AgentIR-4B query encoder takes GPU 2. The flat FAISS index over
# 21M x 2560 float32 embeddings = ~215G resident. 230G survived 7416 only
# because its broken tool parser meant vLLM never actually served an agent
# loop; under real load (7511) the server frontend + 8 threads x 100-iteration
# conversations pushed past the cgroup at t+4.5h with zero results written
# (OOM kill). 250G is the practical ceiling of octal31's 257000MB RealMemory;
# CONC=4 halves the client-side conversation buffers as the second lever.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap 'stop_vllm; cleanup_tmpdir' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))
SNAP=$HF_HUB_CACHE/models--Tevatron--AgentIR-4B/snapshots/e31abb637caa227c4b7d04176a24ecff1bcb10f4
# ABSOLUTE: run_official_baseline runs the harness with cwd=<official
# checkout>, so a relative --output-dir lands inside the agentir repo. 7416's
# 14,267 outputs went to $OFFICIAL_ROOT/agentir/results/... and normalize then
# read the empty project-side dir (same trap as grepseek 2026-08-02).
OUT=$REPO/results/phase2/agentir_raw
mkdir -p results/phase2 "$OUT"

N=$(ls "$DATA"/agentir_official_index/resharded/corpus.reshard.*.pkl 2>/dev/null | wc -l)
[ "$N" -eq 16 ] || {
  echo "FATAL: need 16 resharded pkl pieces, found $N; run p2_agentir_reshard first"; exit 4; }

export VLLM_GPUS=0
# Tool parser flags are load-bearing: the oss_client drives the agent loop
# through the OpenAI tools API, and without them vLLM returns the model's
# <tool_call> XML as plain text -- the search is never executed and every
# query "completes" after one turn with retrieved_docids=[] (7416: 714/714
# sampled rows, 9h of garbage). Same root cause as DCI/DR-DCI (e7bdb03).
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 1 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
# 10800, not 3600: weight loading shares the glacier NFS with whatever else
# is running -- 7503/7505 both starved past the 1h mark (0/4 shards at t+65min)
# while p2_rise packaged 3.2M files on the same filer. Generous timeout costs
# nothing when the filer is quiet.
waitp "$PORT" 10800

cd "$REPO"
# OMP explicitly wide for the CLIENT only: its faiss flat-IP searches scan the
# 215G index per query (10+ searches/query) and 7531 crawled at 73s/query with
# the client stuck near ~3 cores while vLLM sat idle -- search, not the LLM,
# is the bottleneck. Not exported globally: vLLM manages its own thread pools.
OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_official_baseline.py agentir \
  --repo-root "$OFFICIAL_ROOT" --full-eval --model-revision "$GEN_REV" \
  --record results/phase2/popqa_agentir.run.json -- \
  --model Qwen/Qwen3.5-9B --model-url "http://127.0.0.1:$PORT/v1" \
  --searcher-type faiss --index-path "$DATA/agentir_official_index/resharded/corpus.reshard.*.pkl" \
  --dataset-name json --data-files "$DATA/agentir_official_index/corpus_tevatron.jsonl" \
  --model-name "$SNAP" --reasoning-aware --normalize \
  --k 5 --max-iterations 100 \
  --query "$DATA/agentir_queries/popqa_full.tsv" \
  --QA-jsonl "$DATA/agentir_queries/popqa_full_qa.jsonl" \
  --num-threads "${CONC:-8}" \
  --output-dir "$OUT" &
HARNESS_PID=$!
# Validity gate on the FIRST trajectories: 7571 ran three days of blind
# retrieval because the searcher's docid->text lookup defaulted to the
# BrowseComp-Plus corpus (--dataset-name default) against Wiki-18 docids --
# 93% of trajectories carried "Text not found" in every snippet and the agent
# answered from parametric knowledge. The explicit json/data-files args above
# fix it; this gate makes any recurrence cost minutes, not days.
(
  while kill -0 "$HARNESS_PID" 2>/dev/null; do
    early=$(ls "$OUT"/run_*.json 2>/dev/null | head -25)
    n=$(printf '%s\n' "$early" | grep -c . || true)
    if [ "${n:-0}" -ge 25 ]; then
      bad=$(grep -l "Text not found" $early 2>/dev/null | wc -l)
      if [ "$bad" -gt 5 ]; then
        echo "FATAL: $bad/25 early trajectories hit 'Text not found' -- the"
        echo "       docid->text lookup is broken again; killing the run now"
        kill "$HARNESS_PID" 2>/dev/null
      fi
      break
    fi
    sleep 60
  done
) &
GATE_PID=$!
HR=0; wait "$HARNESS_PID" || HR=$?
kill "$GATE_PID" 2>/dev/null || true
[ "$HR" -eq 0 ] || { echo "FATAL: harness exited $HR"; exit 1; }
stop_vllm

echo "== normalize + metrics =="
# The client echoes neither question nor golds and writes one Responses-style
# json per query; the converter joins golds back from the canonical dataset
# and takes the last output_text (7571 died here on default field names:
# "empty or duplicate IDs"). Answer format is the official Exact Answer block,
# same regex as DR-DCI/RISE.
"$PY" scripts/convert_agentir_raw.py --raw-dir "$OUT" \
  --dataset-jsonl "$POPQA_NORM" --output "$OUT/results.jsonl"
"$PY" scripts/normalize_official_results.py --method agentir \
  --input "$OUT/results.jsonl" \
  --run-record results/phase2/popqa_agentir.run.json \
  --dataset-manifest "$POPQA_MANIFEST" --dataset-jsonl "$POPQA_NORM" \
  --id-field query_id \
  --prediction-field final_text --answer-regex '^\**Exact Answer\**:\s*(.+?)\s*$' \
  --finish-field status --turns-field n_turns \
  --output results/phase2/popqa_agentir.jsonl
sane results/phase2/popqa_agentir.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_agentir.jsonl \
  --dataset popqa --agent agentir --out results/phase2/popqa_agentir.metrics.json
echo "ALL_DONE popqa_agentir"
