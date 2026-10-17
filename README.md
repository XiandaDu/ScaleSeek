# ScaleSeek baselines

Re-runs, from scratch, the baseline matrix of [BASELINES.md](BASELINES.md): Direct, RAG,
Search-R1 (7B, 14B), Search-o1, GrepSeek and AgentIR on seven FlashRAG QA sets and
BrowseComp-Plus, with three retrievers (bm25, e5, qwen3_emb_4b), 15 runs per dataset.

```
same subset + same corpus + same decoding profile
        ==>  the only things that vary are the method and (where stated) the retriever
```

## Layout

```
run.py                      the only entry point (python run.py --help)
setup_env.sh                source it to activate; bash setup_env.sh --install to build
submit.sh                   sbatch wrapper for every stage (account in cluster.env)
configs/base.yaml           shared: datasets, corpora, retrievers, decoding profiles, seed
configs/methods/*.yaml      one file per method: model + revision, server, prompts (sha256)
prompts/                    project-defined prompts; prompts/official/ = verbatim copies
scaleseek/                  the package (config, data, corpus, retrieval/, methods/, ...)
slurm/                      job.sh (one command), array.sh (one shard per task)
env/                        locks: main.in + main.lock.txt, agentir.lock.txt, *_extra.txt
docs/deviations.md          everything that differs from an official implementation
tests/                      CPU tests: configs, byte diffs, sampling, metrics, method loops
```

Two config files define a run, and nothing else: `configs/base.yaml` and one
`configs/methods/<method>.yaml`. Unknown keys and missing keys are both errors; the code
has no defaults for experimental parameters. Prompts are frozen by sha256 in the yaml;
official prompts are byte-diffed against the pinned checkout before every run.

## Setup (once, on a LOGIN node: compute nodes have no internet)

```bash
cp cluster.env.example cluster.env          # account, asset root
bash setup_env.sh --install                 # uv, CPython, wheelhouse, login env, repos
source setup_env.sh
python run.py prefetch                      # pinned models, datasets, corpora (I/O only)
# once, in any job: build the two AgentIR deps that have no PyPI wheel, then record
# the printed sha256 lines in env/agentir_extra.txt
srun --account=def-amirhk --time=1:00:00 --mem=8G bash setup_env.sh --build-wheels
```

Everything lives under `$SCALESEEK_ROOT` (default `/scratch/$USER/scaleseek_v2`).
Inside a job, `source setup_env.sh` builds the Python environment on node-local
`$SLURM_TMPDIR` from the hash-pinned wheelhouse (offline, ~2 min, once per job): Lustre
serves big files fast but small-file opens can take seconds, and a Python process opens
thousands. The AgentIR environment is built the same way on first use.
Mind the 60-day scratch purge: `runs/` holds the only copy of the trajectories.

## Pipeline

```bash
bash submit.sh prepare                                   # preflight + subsets + corpora
bash submit.sh index-bm25       --corpus wiki18
bash submit.sh encode           --retriever e5 --corpus wiki18 --shards 8
bash submit.sh build-hnsw       --retriever e5 --corpus wiki18 --shards 8
bash submit.sh encode           --retriever qwen3_emb_4b --corpus wiki18 --shards 64
bash submit.sh build-hnsw       --retriever qwen3_emb_4b --corpus wiki18 --shards 64
bash submit.sh encode-agentir   --corpus wiki18 --shards 64
bash submit.sh finalize-agentir --corpus wiki18 --shards 64
bash submit.sh check-retrievers --corpus wiki18          # doc-id alignment + determinism
bash submit.sh eval-all --dataset popqa                  # the 15 runs
bash submit.sh score    --dataset popqa                  # metrics.json + REPORT.md
# BrowseComp-Plus: the same with --corpus bcp / --dataset browsecomp_plus, then
bash submit.sh judge    --dataset browsecomp_plus
```

`--after <jobid>` chains a stage on another; `--dry-run` prints the sbatch command.
`--tasks 13,15,22` (with `--shards N`) re-runs only those shards of an array stage.
A run resumes where it stopped (records are appended one by one); it refuses to resume
under a different config id or harness commit; formal runs need committed code.
Scoring refuses a run whose ids are not exactly the subset manifest's.

Order (§2/§5): **PopQA first**, report, and only then the other seven datasets.

## Smoke test

One allocation, every method on 4 questions against mini corpora built from the real
ones (`smoke` in base.yaml), the BCP judge, and a smoke report under `$SCALESEEK_ROOT/smoke`
(never reported):

```bash
salloc --account=def-amirhk --gpus-per-node=a100:4 --cpus-per-task=48 --mem=0 --time=8:00:00   # 4 GPUs: the Qwen3-32B judge is TP=4
source setup_env.sh
python run.py smoke-prepare && python run.py smoke
```

## Outputs (per run: `runs/<dataset>/<method>[__<retriever>]/`)

`provenance.json` (config id, harness commit, subset manifest hash, model revision, corpus
and index manifests, prompt hashes, byte-diff results, official commit, server command,
software versions, seeds), `sessions.jsonl` (host, job, harness commit, wall time per session),
`trajectories.jsonl` (one record per question with the full trajectory), `metrics.json`,
and for official harnesses their raw output under `official/`.

## Decisions on record

See docs/deviations.md: AgentIR on the official OSS client with QUERY_TEMPLATE; BCP
documents shown to agents cut to 512 tokens (Qwen3-0.6B); Direct/RAG on BCP capped at
10,000 tokens per response; GrepSeek on BCP as the official harness is; presence_penalty
2.0 added to every generation request of every method (Qwen3.5 thinking loops); BCP judge =
Qwen3-32B (TP=4, the official open-source judge) instead of Qwen3.5-9B; judge verdicts per
judge config; jobs run from a git worktree of the submitted commit.
