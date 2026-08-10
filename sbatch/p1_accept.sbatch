#!/usr/bin/env bash
#SBATCH -J p1_accept
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH -t 24:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Phase-1 server acceptance (GPU): index same-query stability, real-model
# end-to-end smokes for every locally-run method, judge serving closure, and
# Search-R1 7B/14B + GrepSeek serving closures. Smoke outputs live only in
# results/smoke/ and never enter any results table.
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'stop_vllm; cleanup_tmpdir' EXIT
mkdir -p results/smoke
PORT=$((10000 + SLURM_JOB_ID % 20000))

# Stage markers make reruns cheap: a passed stage is never repeated, so a
# failure in a late stage does not re-pay the earlier model loads.
passed() { [ -f "results/smoke/.stage$1.done" ]; }
mark()   { touch "results/smoke/.stage$1.done"; }

if ! passed A; then
echo "== A. same-query stability: bm25 + e5 =="
"$PY" scripts/probe_retriever_stability.py --backends bm25,e5 --device cpu \
  --out results/smoke/stability_bm25_e5.json
mark A
fi

if ! passed BC; then
echo "== B. serve Qwen3.5-9B and smoke the local agents =="
start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 2 Qwen/Qwen3.5-9B --reasoning-parser qwen3
waitp "$PORT"

smoke() {  # smoke <name> <agent> [extra run_eval args...]
  local name=$1 agent=$2; shift 2
  local out=results/smoke/smoke_${name}.jsonl
  echo "-- smoke $name ($agent $*) -> $out"
  "$PY" -m eval.run_eval --dataset "$DS" -n 8 --agent "$agent" \
    --host 127.0.0.1 --port "$PORT" --model Qwen/Qwen3.5-9B \
    --concurrency 4 --retries 2 --output "$out" "$@"
  sane "$out"
}

smoke direct direct
smoke rag_bm25 rag --retriever bm25 --retriever-device cpu
smoke rag_e5 rag --retriever e5 --retriever-device cpu
smoke scaleseek scaleseek --tokenizer Qwen/Qwen3.5-9B
smoke search_o1_bm25 search_o1 --retriever bm25 --retriever-device cpu --tokenizer Qwen/Qwen3.5-9B

echo "== C. BCP judge prompt hash + serving closure =="
"$PY" - "$PORT" <<'PYEOF'
import hashlib, sys
from openai import OpenAI
from eval.browsecomp_plus_judge import BCP_F_JUDGE_PROMPT
digest = hashlib.sha256(BCP_F_JUDGE_PROMPT.encode()).hexdigest()
expected = "c30a12f670d4787d89b211f2ef61bf7512595acde5a124312a59f07ef09ae6ea"
assert digest == expected, f"judge prompt hash drifted: {digest}"
client = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[1]}/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B", temperature=0.0, top_p=1.0, max_tokens=1024,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}])
msg = resp.choices[0].message
out = msg.content or getattr(msg, "reasoning_content", None) or ""
assert out.strip(), "judge closure: empty response"
print("judge serving closure OK:", out.strip()[:80])
PYEOF
stop_vllm
mark BC
fi

if ! passed D; then
echo "== D. Search-R1 7B smoke =="
start_vllm "$SR1_7B" "$SR1_7B_REV" "$PORT" 2 search-r1-7b
waitp "$PORT"
"$PY" -m eval.run_eval --dataset "$DS" -n 8 --agent search_r1 \
  --retriever bm25 --retriever-device cpu \
  --search-r1-host 127.0.0.1 --search-r1-port "$PORT" \
  --search-r1-model search-r1-7b --search-r1-tokenizer "$SR1_7B" \
  --concurrency 4 --retries 2 --output results/smoke/smoke_search_r1_7b.jsonl
sane results/smoke/smoke_search_r1_7b.jsonl
stop_vllm
mark D
fi

if ! passed E; then
echo "== E. Search-R1 14B serving closure (load + one completion) =="
VLLM_MEM_UTIL=0.93 start_vllm "$SR1_14B" "$SR1_14B_REV" "$PORT" 2 search-r1-14b
waitp "$PORT" 3600
"$PY" - "$PORT" <<'PYEOF'
import sys
from openai import OpenAI
client = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[1]}/v1", api_key="dummy")
resp = client.completions.create(model="search-r1-14b", prompt="Question: capital of France?\n",
                                 max_tokens=32, temperature=0.7, top_p=1.0)
print("search-r1-14b closure OK:", resp.choices[0].text[:80])
PYEOF
stop_vllm
mark E
fi

if ! passed F; then
echo "== F. GrepSeek official-harness smoke =="
start_vllm "$GREPSEEK_MODEL" "$GREPSEEK_REV" "$PORT" 2 "$GREPSEEK_MODEL" --reasoning-parser qwen3
waitp "$PORT"
"$PY" scripts/run_official_baseline.py grepseek --repo-root "$OFFICIAL_ROOT" \
  --record results/smoke/grepseek_smoke.run.json -- \
  --model "$GREPSEEK_MODEL" --tokenizer "$GREPSEEK_MODEL" \
  --base_url "http://127.0.0.1:$PORT/v1" \
  --max_assistant_turns 6 --max_tokens_per_turn 0 --tool_max_tokens 2048 \
  --cache_dir "$HF_HUB_CACHE" \
  --temperature 0.6 --top_p 1.0 --parallel 4 --limit 8 \
  --input "$DS_RAW" --corpus_dir "$CORPUS_DIR" \
  --out_dir results/smoke/grepseek_smoke
stop_vllm
mark F
fi

echo "ALL_P1_ACCEPT_DONE"
# Smoke artifacts are throwaway (acceptance evidence lives in this job log);
# removing them also clears the stage markers for any future fresh acceptance.
rm -rf results/smoke

# Acceptance passed. On Rorqual the matrix is orchestrated by
# sbatch/submit_phase3.sh with plain SLURM dependencies (this cluster has no
# 4-job QOS cap, so the RALI lane machinery is unnecessary); downstream jobs
# hang on afterok:<this job> and start now.
echo "acceptance gate passed for DS=$DS"
mkdir -p "results/$PHASE"; touch "results/$PHASE/.accept_passed_${DS}"
