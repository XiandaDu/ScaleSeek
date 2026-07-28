# Phase-1 reproducibility runbook

All formal runs use the immutable settings in `configs/baselines.yaml` and
record dataset/config/prompt/model identifiers in every result row.
The commands are intended for the experiment server, not this code-authoring
machine, and remain ordered from cheap validation to expensive full runs.

## Required shell initialization

Every command in this runbook must be run from a shell initialized by the
repository setup script. Do this first on each new compute-node shell or tmux
pane; do not replace it with a partial set of hand-written exports:

```bash
cd /data/rech/mofengra/ScaleSeek
source setup_env.sh
```

`source` is required (not `bash setup_env.sh`) so `REPO`, `DATA`, `DATASETS`,
`CORPUS_DIR`, `CORPUS_FILE`, `BM25_INDEX_DIR`, the LLM endpoint variables, and
the `scaleseek` conda environment remain active in the current shell. The setup
script also changes directory to `$REPO`; all relative commands below rely on
that behavior.

The existing corpus-I/O observation remains relevant when choosing server
parallelism: start DCI-Agent-Lite with `--max-concurrency 2` or lower because
concurrent full-corpus grep scans can thrash disk. In the earlier local DCI run,
concurrency 16 caused 44% grep timeouts and reduced EM from 0.274 to 0.179;
GrepSeek's tighter commands tolerated about `--parallel 16` with 1.6% timeouts.
These are operational starting points, not method hyperparameters—record and
adjust them only after measuring the target server.

Project-owned prompts are Python constants under `prompts/`. The stable runtime
identifiers are `prompts.direct:PROMPT`, `prompts.rag:PROMPT`, and
`prompts.scaleseek_prompt:PROMPT` (default: no numeric parameter bias);
`SCALESEEK_PROMPT=scaleseek_prompt_withparams` selects the numeric-hint ablation
only for non-formal runs. Upstream paths shown
for official external harnesses remain inside their pinned repositories.

## Action and search budgets

There is deliberately no artificial universal budget: each method retains its
documented protocol.

| Method | Maximum interaction/search budget |
|---|---|
| Direct, RAG | one generator call; RAG performs one top-3 retrieval |
| Search-R1 | 4 generated actions (`B=4`), each search returns top-3 |
| Search-O1 | 15 continuation turns; at most 5 searches on single-hop or 10 on multi-hop, each top-3 |
| GrepSeek | 6 assistant turns, hence at most 5 tool calls plus a final answer turn |
| DCI-Agent-Lite | 300 turns/calls, L3, command timeout 30 s |
| DR-DCI | 300 turns; at most 10 pull queries; each pull chooses 300–600 candidates and previews 20 |
| RISE | 100 model calls; BM25 K=1000 per subquery and preview 10; one-hour per query |
| AgentIR | official OSS agent: at most 100 iterations and top-5 per search |
| ScaleSeek | 8 assistant turns, hence at most 7 tool calls before an answer turn |

## Environment

`setup_env.sh` owns the core paths. Define only the additional Phase-1 index
and tokenizer locations after sourcing it:

```bash
export E5_INDEX_DIR=$DATA/e5_index
export QWEN3_EMB_INDEX_DIR=$DATA/qwen3_embedding_4b_index
export LLM_TOKENIZER=Qwen/Qwen3.5-9B
```

Serve Qwen3.5-9B for prompt methods, each selected Search-R1 checkpoint on its
own endpoint, and the official GrepSeek checkpoint on its own endpoint. The
served artifact must be the revision recorded in YAML.

## Full local run examples

```bash
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
python -m eval.run_eval --dataset popqa --full-eval --agent direct \
  --model Qwen/Qwen3.5-9B --generator-revision "$REV" \
  --output results/phase2/popqa_direct.jsonl

python -m eval.run_eval --dataset popqa --full-eval --agent rag \
  --retriever bm25 --model Qwen/Qwen3.5-9B --generator-revision "$REV" \
  --output results/phase2/popqa_rag_bm25.jsonl

SR1=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3
SR1_REV=395b18f1fecee52f1b51fb22f898c220f0a08ec3
python -m eval.run_eval --dataset popqa --full-eval --agent search_r1 \
  --retriever e5 --search-r1-model search-r1-7b --search-r1-tokenizer "$SR1" \
  --generator-revision "$SR1_REV" --output results/phase2/popqa_search_r1_7b_e5.jsonl
```

Repeat RAG, Search-R1 and Search-O1 with `bm25`, `e5`, and `qwen3_emb_4b`.
Repeat Search-R1 for both frozen checkpoints. `--retrieval-top-k` is fixed to 3.

## Cluster: pack a whole node per job

The `rali` QOS is `MaxJobsPU=2 / MaxSubmitPU=4` with **no `MaxTRES`** — it caps
the number of jobs, not GPUs. Asking for `--gres=gpu:2` on a 4-GPU node therefore
wastes half of every allocation and serialises a matrix whose cells are fully
independent (distinct output files, no shared state). Use `p2_pack.sbatch` to
hold a node and run one cell per GPU:

```bash
sbatch -J p2_packA1 --export=ALL,LANE=laneA,\
CELLS=rag_bm25:rag:bm25+rag_e5:rag:e5+scaleseek:scaleseek:bm25 sbatch/p2_pack.sbatch
```

- `CELLS` = `outname:agent:ret[:sr1]`, separated by **`+`** (not spaces or
  commas: `--export` splits on commas and the lane queue is word-split).
- Cells per node are decided at runtime from GPU memory: ≥40 GB (L40S, A6000)
  runs Qwen3.5-9B at TP=1 → 4 cells; 24 GB (A5000) needs TP=2 → 2 cells. A
  `qwen3_emb_4b` cell takes one extra GPU for its query encoder.
- Cells that do not fit are **re-submitted automatically** as a follow-up pack
  job, which also inherits `LANE` so the queue advances exactly once.
- A cell whose `.metrics.json` already exists is skipped, so a requeued job
  resumes the remainder.
- Per-cell logs: `logs/cell_<job>_<id>_<name>.log`; `sbatch/status.sh` shows
  each cell's gate verdict and progress.

Packing changes only **how many cells run at once**. Every cell still runs
`--full-eval` over the complete 14,267-row split — no `-n`, no `--offset`, no
subsetting. Never pack a cell that another queued job is already writing: the
skip check fires on `.metrics.json`, which only appears when a run finishes.

## Frozen official repositories

```bash
python scripts/bootstrap_official_repos.py --root /data/official-baselines
```

Examples below show the mandatory invariant flags. Dataset, corpus, endpoint,
output and concurrency arguments must also be supplied for the target machine.

```bash
# GrepSeek
python scripts/run_official_baseline.py grepseek --repo-root /data/official-baselines \
  --full-eval --model-revision a79563970cfdd2ced3cc5fde481737d0ebea6fa4 -- --model alireza7/GrepSeek-Qwen3.5-9B-GRPO \
  --tokenizer alireza7/GrepSeek-Qwen3.5-9B-GRPO --max_assistant_turns 6 \
  --max_tokens_per_turn 0 --tool_max_tokens 2048 --temperature 0.6 --top_p 1.0 \
  --parallel 16 \
  --input "$DATASET_JSONL" --corpus_dir "$GREPSEEK_CORPUS" --out_dir "$OUT"

# DCI
python scripts/run_official_baseline.py dci --repo-root /data/official-baselines \
  --full-eval --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a -- --model Qwen/Qwen3.5-9B --max-turns 300 \
  --runtime-context-level level3 --tools read,bash --max-concurrency 2 --dataset "$DATASET" \
  --corpus-dir "$DCI_CORPUS" --output-root "$OUT"

# DR-DCI (Wiki uses E5 pull; BCP uses the official Qwen3-Embedding-8B service)
python scripts/run_official_baseline.py dr_dci --repo-root /data/official-baselines \
  --full-eval --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a -- --model Qwen/Qwen3.5-9B --max-turns 300 \
  --runtime-context-level level3 --pull-min-top-k 300 --pull-max-top-k 600 \
  --pull-max-queries 10 \
  --pull-preview-limit 20 --pull-materialization-mode root_flat_disclosed \
  --dataset "$DATASET" --output-root "$OUT"

# RISE: build its official bm25s index with the deliberate project k1 override
cd /data/official-baselines/rise
uv run python scripts/build_bm25_index.py --corpus "$RISE_CORPUS" \
  --out "$RISE_INDEX" --k1 1.2 --b 0.75
cd -
python scripts/run_official_baseline.py rise --repo-root /data/official-baselines \
  --full-eval --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a -- --model Qwen/Qwen3.5-9B --bm25-k 1000 \
  --bm25-top-n-preview 10 --max-turns 100 --bash-truncate-chars 4000 \
  --read-default-limit 2000 --structured-docs --index-dir "$RISE_INDEX" \
  --bc-plus-docs "$RISE_DOCS" --out-root "$OUT"

# AgentIR reasoning-aware official loop
python scripts/run_official_baseline.py agentir --repo-root /data/official-baselines \
  --full-eval --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a -- --model Qwen/Qwen3.5-9B --model-url "$VLLM_URL" \
  --searcher-type faiss --index-path "$AGENTIR_INDEX" \
  --model-name "$HF_HOME/hub/models--Tevatron--AgentIR-4B/snapshots/e31abb637caa227c4b7d04176a24ecff1bcb10f4" --reasoning-aware --normalize \
  --k 5 --max-iterations 100 --query "$QUERIES_TSV" --output-dir "$OUT"
```

RISE's main row requires structured TOC documents. Its plain-document mode is
named `RISE-BM25` and is not interchangeable with the main method.

All BrowseComp-Plus agent outputs are judged independently with:

```bash
python scripts/judge_browsecomp_plus.py --input "$AGENT_OUTPUT" \
  --output "$JUDGE_OUTPUT" --base-url "$JUDGE_VLLM_URL" \
  --model Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

## Phase gates

Before Phase 2, run all tests, verify PopQA reports 14,267 unique IDs, validate
all index/doc-ID manifests, and verify a real-model search→tool→answer smoke for
each retained method. Phase 2 must run all 14,267 PopQA examples without any
sampling flag. After reporting Phase-2 results, stop until the user approves
Phase 3.
