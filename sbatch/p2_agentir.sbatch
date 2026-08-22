#!/usr/bin/env bash
#SBATCH -J p2_agentir
#SBATCH -p rali
# octal41, and no smaller node can work: the official loader's faiss add()
# duplicates each shard while inserting, so the load PEAK is ~204G(faiss) +
# 51G(numpy shard) = 255G -- physically above the 257G-class octal30/31
# (7511 OOMed at t+4.5h mid-load under the NFS storm; 7518 OOMed in 30min
# warm-cache at the same point). octal41 has 503G, and its 48G L40S run the
# 9B server at tp=1, so two GPUs suffice: vLLM on 0, AgentIR-4B encoder on 1.
#SBATCH -w octal41
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

ls "$DATA"/agentir_official_index/corpus.*.pkl >/dev/null 2>&1 || {
  echo "FATAL: official AgentIR pkl shards missing; run p2_agentir_encode first"; exit 4; }

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
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_official_baseline.py agentir \
  --repo-root "$OFFICIAL_ROOT" --full-eval --model-revision "$GEN_REV" \
  --record results/phase2/popqa_agentir.run.json -- \
  --model Qwen/Qwen3.5-9B --model-url "http://127.0.0.1:$PORT/v1" \
  --searcher-type faiss --index-path "$DATA/agentir_official_index/corpus.*.pkl" \
  --model-name "$SNAP" --reasoning-aware --normalize \
  --k 5 --max-iterations 100 \
  --query "$DATA/agentir_queries/popqa_full.tsv" \
  --QA-jsonl "$DATA/agentir_queries/popqa_full_qa.jsonl" \
  --num-threads "${CONC:-8}" \
  --output-dir "$OUT"
stop_vllm

echo "== normalize + metrics =="
"$PY" scripts/normalize_official_results.py --method agentir \
  --input "$OUT" \
  --run-record results/phase2/popqa_agentir.run.json \
  --dataset-manifest "$POPQA_MANIFEST" --dataset-jsonl "$POPQA_NORM" \
  --output results/phase2/popqa_agentir.jsonl
sane results/phase2/popqa_agentir.jsonl
"$PY" scripts/compute_metrics.py --results results/phase2/popqa_agentir.jsonl \
  --dataset popqa --agent agentir --out results/phase2/popqa_agentir.metrics.json
echo "ALL_DONE popqa_agentir"
