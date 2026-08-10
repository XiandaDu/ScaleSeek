# Rorqual runbook (branch: cluster-packing-and-decile-scope-Rorqual)

This branch adapts the frozen ScaleSeek evaluation stack to Compute Canada's
**Rorqual** cluster and parameterizes the phase-2 job tree by dataset (`$DS`)
for phase 3. First target: **phase-3 benchmark #2 = TriviaQA** (11,313 rows;
DCI/DR-DCI/RISE run the sanctioned SHA256-rank cap-1500 subset; IRCoT was
removed in phase 1, so the matrix is 14 local cells + 5 official-harness rows).

Nothing in the frozen protocol changed: models, revisions, prompts, retriever
settings, budgets and the cap-1500 rule are byte-identical to `main`. What
changed is *where things live and how jobs are scheduled*.

## Cluster facts the port encodes

| Fact | Consequence |
|---|---|
| Login nodes `rorqual1..N`; compute `rg*` (4×H100-80GB, 64c, 510G) / `rc*` | `common.sh` refuses login nodes; H100 ⇒ TP=1, 4 cells per pack node |
| Compute nodes have **no internet** | all downloads happen on login (`scripts/rorqual_login_setup.sh`, `hf download`); jobs run with `HF_HUB_OFFLINE=1`, `UV_OFFLINE=1` (set by `setup_env.sh` under SLURM) |
| No conda support; PyPI manylinux binaries blocked by the module python | venv `~/scaleseek_env` from the CC wheelhouse. Exact pins held for vllm 0.23.0 / torch 2.11.0 / transformers 5.12.1; every deviation is in `requirements.rorqual.txt` + `~/requirements.rorqual.changes` |
| `/scratch`: 20 TB but **1M-file quota** | DCI's 21M-file corpus and RISE's 3.2M articles are expanded in node-local `$SLURM_TMPDIR` and persisted as single `tar.zst`; `p2_dci`/`p2_rise` re-extract at job start; DR-DCI and RISE run with node-local out-roots synced back every 30 min and archived at the end |
| QOS has no 2-running/4-queued cap | the RALI lane machinery is retired; `sbatch/submit_phase3.sh` submits one idempotent SLURM dependency DAG |
| Max wall 7 days | all 14-day RALI limits capped to 7d (jobs are resumable/requeueable) |

## Fresh-cluster bring-up (login node)

```bash
# 1. venv (wheelhouse; see requirements.rorqual.txt)
bash ~/build_venv4.sh              # or follow requirements.rorqual.changes
# 2. models + data at pinned revisions -> /scratch/a32du/data/hf_cache
bash ~/download_models.sh && bash ~/download_corpus.sh
# 3. official repos, uv envs, node toolchain
bash scripts/rorqual_login_setup.sh
# 4. submit the whole phase-3 DAG for triviaqa
DS=triviaqa bash sbatch/submit_phase3.sh
# 5. watch
bash sbatch/status.sh
```

## DAG (submit_phase3.sh)

```
p0_corpus_unzip ─┬─ p0_bm25_index ──┬─ p0_assets ── p1_accept ─┬─ p3_packA..E (14 local cells)
                 ├─ p0_e5_index ────┘                          ├─ p3_grepseek
                 ├─ p1_qwen3emb_index ─────────── (packs D,E)  ├─ p3_dr_dci
                 ├─ p2_agentir_encode ─────────────────────────── p3_agentir
                 ├─ p0_dci_corpus ─────────────────────────────── p3_dci
                 └─ p0_rise_articles ── p1_rise_toc ───────────── p3_rise
```

Every job skips work whose outputs exist, so re-running `submit_phase3.sh`
after any failure only re-queues the missing pieces. Results land in
`results/phase3/triviaqa_*.jsonl` + `.metrics.json`; capped rows are verified
against `triviaqa_cap1500.manifest.json` via `--expect-ids`.

## Deliberate deviations recorded for the report

- Serving/eval stack: vllm 0.23.0 / torch 2.11.0 / transformers 5.12.1 exactly
  as phase 2; faiss-cpu 1.14.3→1.12.0, numpy 2.3.5→2.3.3, pyserini 1.6.0 from
  sdist, pip CUDA stack replaced by cluster CUDA (full list:
  `requirements.rorqual.changes`). Hardware differs (H100 vs A5000/L40S).
- RISE `--full-eval` gate manifest is derived from the cap-1500 subset manifest
  (`subset_n`); the RALI script passed the full-split manifest, which can never
  match a 1,500-row mini-dev — that gate had not been exercised there.
- RISE structured corpus: identical content, but the empty-TOC passthrough for
  non-candidate articles is regenerated node-locally inside `p2_rise` instead
  of being stored (file-count quota); candidate TOCs are persistent in
  `$DATA/rise_toc_structured` and the retrieval boundary remains the full corpus.
