# ScaleSeek

ScaleSeek trains and evaluates retrieval agents that construct a bounded BM25
workspace and then use `grep_workspace` / `read_doc` for fine-grained search.

## Phase-1 source of truth

- `TASK.md`: gated three-phase experiment plan.
- `configs/baselines.yaml`: frozen method/model/retrieval parameters.
- `configs/official_repos.yaml`: official repository commits.
- `BASELINE_SOURCES.md`: paper, prompt and harness provenance.
- `RUNBOOK.md`: executable commands and action/search budgets.
- `cleanup_manifest.json`: removed approximations and obsolete launch scripts.
- `prompts/*.py`: project-authored or project-modified prompt constants.
- `eval/*`: verbatim third-party prompts beside their corresponding evaluator.

Prompt-based generators and the BrowseComp-Plus judge use
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Search-R1 uses only
the frozen 7B and 14B v0.3 checkpoints; GrepSeek and AgentIR use their official
specialized checkpoints.

## Local runner

The local runner contains only methods whose loop is correctly implemented in
this repository:

```bash
python -m eval.run_eval --help
```

It supports `direct`, `rag`, `search_r1`, `search_o1`, and `scaleseek`.
RAG, Search-R1 and Search-O1 accept `--retriever bm25|e5|qwen3_emb_4b`;
their comparable local retrieval returns top-3. Fixed BM25 uses k1=1.2, b=0.75.

Direct, RAG and ScaleSeek prompts are imported from `prompts.direct:PROMPT`,
`prompts.rag:PROMPT` and `prompts.scaleseek_prompt:PROMPT`. Training and
evaluation resolve the same canonical ScaleSeek constant. Prompt provenance
hashes cover the exact Python string sent to the model, without newline/file
normalization ambiguity.

`--full-eval` forbids `--n` and `--offset`, verifies known complete-split counts,
requires the frozen generator revision, and checks that final output IDs exactly
equal the dataset manifest. In particular, canonical `popqa` means all 14,267
test examples; there are no `popqa_full` or long-tail aliases.

## Official external harnesses

GrepSeek, DCI, DR-DCI, RISE and AgentIR are intentionally not reimplemented locally.
Bootstrap their pinned repositories and launch through the commit gate:

```bash
python scripts/bootstrap_official_repos.py --root /data/official-baselines
python scripts/run_official_baseline.py --help
```

The launcher passes argv directly without a shell, verifies the checkout commit,
and rejects method parameters that violate the frozen Phase-1 profile.

## Indexes

On the experiment server, initialize the repository environment before running
any command:

```bash
cd /data/rech/mofengra/ScaleSeek
source setup_env.sh
```

```bash
python scripts/build_corpus_manifest.py --corpus "$CORPUS_FILE" --out "$DATA/indexes/wiki18-corpus.json"
python scripts/build_bm25_index.py --corpus-dir "$CORPUS_DIR" --index-dir "$BM25_INDEX_DIR" --corpus-manifest "$DATA/indexes/wiki18-corpus.json"
python scripts/build_e5_index.py --backend e5 --corpus "$CORPUS_FILE" --out "$E5_INDEX_DIR" --corpus-manifest "$DATA/indexes/wiki18-corpus.json"
python scripts/build_qwen3_embedding_index.py --corpus "$CORPUS_FILE" --out "$QWEN3_EMB_INDEX_DIR" --corpus-manifest "$DATA/indexes/wiki18-corpus.json"
```

The dense implementations use the frozen model revisions, E5 query/passage
prefixes and masked-mean pooling, and Qwen3's official query instruction,
left-padding-compatible last-token pooling and L2 normalization.

## Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/scaleseek-pycache python -m pytest -q
python scripts/verify_official_prompts.py --repo-root /data/official-baselines
```

Phase 2 may begin only after Phase 1 passes. Phase 2 evaluates the complete
PopQA test split and must stop for user review before any Phase-3 experiment.
