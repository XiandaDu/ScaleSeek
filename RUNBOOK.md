# ScaleSeek Eval — RUNBOOK

Everything here is meant to be run **by you** (Claude does not launch model / index
/ eval jobs). Steps are ordered cheap → expensive. All commands assume:

```bash
cd /data/rech/mofengra/ScaleSeek
source setup_env.sh                 # sets REPO, DATA, DATASETS, CORPUS_FILE, BM25_INDEX_DIR, LLM_*
```

**Concurrency (`--concurrency N`)** — vLLM batches N in-flight examples for a big
speedup. Safe/fast for **indexed** agents (bm25_rag, search_o1, direct): use 24–48.
For agents that `grep` the raw 15 GB corpus, parallel scans thrash the disk and hit
the 30 s tool timeout: **dci must run at conc ≤ 2** (at conc 16 it timed out on 44 %
of greps and EM fell 0.274 → 0.179); **grepseek is fine at conc ≈ 16** (trained tight
greps, 1.6 % timeouts). See `reports/metric_support.md §4`.

Servers used (start in tmux, never Ctrl-C the vLLM panes):
- main LLM (Qwen3-4B) → `127.0.0.1:8000`, model name `agent`  (direct/bm25_rag/dci/scaleseek/agentir/dr_dci/search_o1)
- Search-R1 (Qwen2.5-3B) → `:8001`
- GrepSeek (9B) → `:8002`

Extra env for the new steps:
```bash
export TITLE_INDEX_DB=$DATA/corpus_title_index.db     # built in step 2
export GREPSEEK_TOKENIZER=<path-or-hf-id of alireza7/GrepSeek-Qwen3.5-9B-GRPO>  # for step 4
```

---

## 0. Sanity: offline unit tests (no model, ~seconds)
```bash
python -m pytest tests/test_retrieval_metrics.py -q
# or: python tests/test_retrieval_metrics.py
```
Expect all green. Verifies Gold/Qrel R@W, coverage, localization, and the
surfaced-doc extractors on hand-computed fixtures.

## 1. Re-score existing runs offline (no model, ~seconds each)
Confirms the metric path end-to-end on runs you already have:
```bash
for f in results/popqa_grepseek.jsonl results/popqa_dci.jsonl \
         results/popqa_scaleseek_4b.jsonl results/popqa_bm25_rag_4b.jsonl \
         results/popqa_direct_4b.jsonl results/popqa_search_r1.jsonl; do
  python scripts/compute_metrics.py --results "$f" --out "${f%.jsonl}.metrics.json"
done
```
PopQA has no gold docs → you'll get EM/F1 + latency (retrieval_note explains why).

## 2. Build the corpus title index (one-time, heavy: ~15–40 min, sqlite ~1–2 GB)
Needed for title-level **Gold R@W** on HotpotQA / 2Wiki:
```bash
python scripts/compute_metrics.py --build-title-index \
    --corpus-path $CORPUS_FILE --title-index-db $TITLE_INDEX_DB
# smoke test first with --limit 200000 if you want.
```

## 3. GrepSeek — paper-faithful re-run  (needs GREPSEEK_TOKENIZER, server on :8002)
Fixes temp (0.6), 2048-token stdout cap, 6 turns. Re-runs PopQA (and any
previously api_error rows are simply re-run as part of the full pass):
```bash
python -m eval.run_eval --dataset popqa --agent grepseek \
    --grepseek-port 8002 --grepseek-tokenizer $GREPSEEK_TOKENIZER \
    --corpus-path $CORPUS_FILE \
    --output results/popqa_grepseek.jsonl
python scripts/compute_metrics.py --results results/popqa_grepseek.jsonl \
    --out results/popqa_grepseek.metrics.json
```
Check a trajectory: tool `stdout` should end with `[... truncated at 2048 tokens]`
(not `... chars`), confirming the token cap.

## 4. BM25 (k1,b) sweep — 5 configs  (server on :8000, BM25 index built)
Run each config; example on PopQA and HotpotQA (add other datasets as needed):
```bash
for DS in popqa hotpotqa; do
 for KB in "0.9 0.4" "25 1.0" "16 1.0" "1.2 0.75" "1.5 0.75"; do
   set -- $KB; K1=$1; B=$2
   python -m eval.run_eval --dataset $DS --agent bm25_rag \
       --bm25-k1 $K1 --bm25-b $B --bm25-top-k 5 \
       --output results/${DS}_bm25_k1-${K1}_b-${B}.jsonl
 done
done
# then score (HotpotQA also gets Gold R@W):
for f in results/*_bm25_k1-*.jsonl; do
  python scripts/compute_metrics.py --results "$f" \
      --title-index-db $TITLE_INDEX_DB --out "${f%.jsonl}.metrics.json"
done
```
See `reports/metric_support.md §3` for what each config is and why.

## 5. Search-O1 — new baseline  (server on :8000, BM25 index)
```bash
for DS in popqa hotpotqa 2wikimultihopqa musique bamboogle; do
  python -m eval.run_eval --dataset $DS --agent search_o1 \
      --bm25-top-k 5 --max-turns 10 \
      --output results/${DS}_search_o1.jsonl
  python scripts/compute_metrics.py --results results/${DS}_search_o1.jsonl \
      --title-index-db $TITLE_INDEX_DB --out results/${DS}_search_o1.metrics.json
done
```

## 6b. AgentIR on a small-RAM node (current: octal35, 2×A6000, ~32GB avail RAM)
The full 21M SQ8 index (54GB) cannot be loaded at once. Sharded workflow:
```bash
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python   # ALWAYS absolute (bare python = base env)
# build: 6 shards × 3,502,554 via logs/driver_gpu.sh (sequential per GPU, --resume checkpoints)
#   GPU0: shards 0,2,4 (offsets 0/7005108/14010216); GPU1: 1,3,5 (3502554/10507662/17512770)
#   output: /data/rech/mofengra/data/agentir_index_v2/shard{0..5}
# eval: shard-by-shard exact retrieval (one 9GB shard in RAM at a time), then cached reader:
$PY scripts/precompute_agentir_retrieval.py --dataset popqa_full -n 1500 \
    --index-root /data/rech/mofengra/data/agentir_index_v2 --top-k 5 --device cuda \
    --out results/popqa_full_agentir_retrieval.jsonl
$PY -m eval.run_eval --dataset popqa_full --agent agentir_rag --concurrency 32 \
    --agentir-cache results/popqa_full_agentir_retrieval.jsonl \
    --output results/popqa_full_agentir.jsonl
```

## 6. AgentIR (dense) — build index then run  (HEAVY)
Build the FAISS index over 21M passages (long encode; pick an index type by host RAM):
```bash
# hnsw_sq8 (~54 GB RAM) if the node has it; else ivfpq (~1–3 GB RAM)
python scripts/build_agentir_index.py \
    --corpus $CORPUS_FILE --out $DATA/agentir_index \
    --device cuda --batch-size 256 --index-type hnsw_sq8
# smoke: add --max-passages 200000 first.
export AGENTIR_INDEX_DIR=$DATA/agentir_index
```
The encoder now auto-halves a batch on CUDA OOM, so the build won't die on one wide
batch; lower `--batch-size`/`--max-length` if it halves too often. Then:
```bash
for DS in popqa hotpotqa 2wikimultihopqa musique bamboogle; do
  python -m eval.run_eval --dataset $DS --agent agentir_rag \
      --agentir-index-dir $AGENTIR_INDEX_DIR --agentir-device cuda --bm25-top-k 5 \
      --output results/${DS}_agentir.jsonl
  python scripts/compute_metrics.py --results results/${DS}_agentir.jsonl \
      --title-index-db $TITLE_INDEX_DB --out results/${DS}_agentir.metrics.json
done
```
(The old `results/popqa_agentir_rag_4b.jsonl` was the wrong BM25→rerank pipeline —
these runs replace it.)

## 7. DR-DCI — official Pi harness, only the model endpoint swapped
Decision: reproduce DR-DCI unchanged (same prompt + hyperparams), pointing only the
LLM at our vLLM Qwen3-4B. DR-DCI ships as a Pi TS coding-agent, so we run *their*
repo and convert its outputs into our metric pipeline.

```bash
# a) clone + build (uv for python, npm for the bundled pi-mono)
git clone https://github.com/EigenTom/DR-DCI && cd DR-DCI
bash setup.sh                                  # == uv sync; cd pi-mono && npm install && npm run build

# b) point the model endpoint at our vLLM (OpenAI-compatible). In .env:
cp .env.template .env
#   OPENAI_API_KEY=EMPTY
#   OPENAI_BASE_URL=http://127.0.0.1:8000/v1     # confirm the exact var name in their
#                                                #   provider code; VLLM_API_KEY=dummy hints a vLLM path
#   run the eval with:  --provider openai --model agent   (our served model name)

# c) retriever for pull(): start their native BCP retriever (most faithful for BCP)
bash scripts/bcplus_eval/start_qwen3emb8b_retriever.sh   # serves :8002/retrieve

# d) smoke, then full 830-query BCP dynamic-pull run (their exact L3 / 300-turn script)
BCP_LIMIT=1 DCI_RUN_NAME=smoke \
  bash scripts/bcplus_eval/run_full830_dynamic_pull_root_flat_openai_high_l3_300turn_parallel30.sh
DCI_RUN_NAME=drdci_qwen3_4b \
  bash scripts/bcplus_eval/run_full830_dynamic_pull_root_flat_openai_high_l3_300turn_parallel30.sh
# outputs -> outputs/bcplus_eval/drdci_qwen3_4b/
```

Then convert + score in our framework (identical metric defs as every baseline):
```bash
cd /data/rech/mofengra/ScaleSeek
# probe the artifact schema first, then set the field flags it prints:
python scripts/convert_dr_dci_output.py \
    --run-dir <DR-DCI>/outputs/bcplus_eval/drdci_qwen3_4b --schema-probe
python scripts/convert_dr_dci_output.py \
    --run-dir <DR-DCI>/outputs/bcplus_eval/drdci_qwen3_4b \
    --out results/bcp_dr_dci.jsonl \
    --id-field <...> --answer-field <...> --workspace-field <...> --time-field <...>
python scripts/compute_metrics.py --results results/bcp_dr_dci.jsonl \
    --dataset bcp --agent dr_dci \
    --bcp-qrels $DATASETS/browsecomp_plus/qrels.json \
    --bcp-doclen $DATASETS/browsecomp_plus/doclen.json \
    --out results/bcp_dr_dci.metrics.json
```
Note: base model is Qwen3-4B (vs the paper's GPT-5.4-nano) — prompt+harness+hyperparams
are identical; only the model differs, so this is apples-to-apples with our other
Qwen3-4B baselines, not with the paper's absolute numbers.

## 8. BrowseComp-Plus — LAST (separate corpus + qrels + index)
```bash
python scripts/build_browsecomp_plus.py --out $DATASETS/browsecomp_plus \
    --bm25-index $DATA/bcp_bm25_index
# then run any agent against the BCP corpus/index and score with native qrels:
python scripts/compute_metrics.py --results results/bcp_<agent>.jsonl \
    --bcp-qrels $DATASETS/browsecomp_plus/qrels.json \
    --bcp-doclen $DATASETS/browsecomp_plus/doclen.json \
    --out results/bcp_<agent>.metrics.json
```
This is the only dataset where **Qrel R@W**, **Coverage(any/mean/all)**, and
**Localization** are all computed. (Verify the HF schema when you run the build
script — see its header notes.)

---

### Quick reference: what each run supports
- EM/F1 + latency: **every** dataset.
- + Gold R@W (title): **hotpotqa, 2wikimultihopqa** (needs `$TITLE_INDEX_DB`).
- + Qrel R@W + Coverage + Localization: **browsecomp_plus** only.
