# ScaleSeek SFT cold-start pipeline

Builds GrepSeek-style cold-start supervised trajectories for the ScaleSeek
two-stage retrieval agent, then fine-tunes a small student on them. The SFT
checkpoint warm-starts RL (`scripts/run_rl.sh` via `SCALESEEK_MODEL_PATH`).

## What each piece is

| File | Role |
|---|---|
| `prompts/sft_prompts.py` | Tutor/Planner prompt suite (decompose, backward tool-trace, judge, bridge, planner, tutor-edit, quality-judge). Used verbatim. |
| `train/sft/teacher.py` | Teacher client: `hf:` (local transformers) / `openai:` (vLLM endpoint) / `FakeTeacher` (tests). |
| `train/sft/coldstart.py` | The pipeline: backward answer→trace discovery + forward reconstruction into the `<think>/<tool_call>/<answer>` format, grounded in real BM25 retrievals. |
| `scripts/make_smoke_corpus.py` | Builds a tiny local corpus + Lucene index from `train/sft/smoke_fixtures.json` so the whole chain runs without the 21M-doc wiki-18 index. |
| `scripts/generate_sft_data.py` | CLI driver → trajectory JSONL (messages + assistant-only loss mask + status/meta). |
| `train/sft_dataset.py` | Trajectory JSONL → verl multi-turn parquet, or token-level `input_ids`/`labels` for a local trainer. Loss on assistant tokens only. |
| `scripts/run_sft.sh` | **Cluster path**: verl SFT trainer (`verl.trainer.sft_trainer`), matches the RL framework. |
| `scripts/run_sft_local.py` | **Local smoke path**: minimal single-GPU transformers trainer, identical HF checkpoint, no verl dependency stack. |
| `scripts/run_scaleseek_smoke_eval.py` | Runs the SFT'd model through the real `eval.agent.run_agent` loop + BM25 env (no vLLM). |
| `tests/test_sft_generation.py` | No-GPU contract tests (fake teacher + fake retriever). |

## Local smoke (single GPU, tiny corpus)

```bash
python scripts/make_smoke_corpus.py --out-dir .smoke --build-index
python scripts/generate_sft_data.py \
    --questions .smoke/questions.jsonl --index-dir .smoke/bm25_index \
    --teacher hf:Qwen/Qwen3-4B --student-tokenizer Qwen/Qwen3-1.7B \
    --out .smoke/sft_trajectories.jsonl
python scripts/run_sft_local.py --base Qwen/Qwen3-1.7B \
    --trajectories .smoke/sft_trajectories.jsonl --out .smoke/sft_ckpt/huggingface --epochs 3
python scripts/run_scaleseek_smoke_eval.py --model .smoke/sft_ckpt/huggingface \
    --index-dir .smoke/bm25_index --questions .smoke/questions.jsonl --limit 8
```

## Cluster (verl SFT + full corpus)

Generate with a vLLM teacher endpoint and the full BM25 index, export the verl
parquet, then train with verl:

```bash
python scripts/generate_sft_data.py --questions $DATA/rl_data/train.jsonl \
    --index-dir $BM25_INDEX_DIR --teacher openai:http://127.0.0.1:8000/v1 \
    --out $DATA/sft/trajectories.jsonl
python -m train.sft_dataset --in $DATA/sft/trajectories.jsonl --out $DATA/sft/train.parquet
SCALESEEK_SFT_TRAIN=$DATA/sft/train.parquet SCALESEEK_SFT_OUTPUT=$CKPT/sft NPROC=4 bash scripts/run_sft.sh
```

The SFT checkpoint then feeds RL:

```bash
export SCALESEEK_MODEL_PATH=$CKPT/sft/**/huggingface
bash scripts/run_rl.sh
```

## Notes / caveats

- **Smoke fixtures are not evaluation data.** `train/sft/smoke_fixtures.json` exists
  only to exercise the pipeline; its numbers must never enter a result table (TASK.md).
- **A ≤4B teacher over-decomposes** simple single-hop questions into 2–3 hops. The
  trajectories stay well-formed and answer-correct, but for research-grade data use a
  stronger teacher (e.g. the frozen Qwen3.5-9B) via `--teacher openai:...`.
- **Local vs cluster SFT framework.** verl (PyPI 0.8.0) pins a transformers/numpy
  combination that conflicts with this desktop's generation/eval env, so the local
  smoke uses `run_sft_local.py`. Both emit the same HF checkpoint; the cluster uses the
  vendored, version-matched verl via `run_sft.sh`.
- **Answer-leak rule** is enforced both in the prompt and programmatically: a
  `bm25_retrieve` query never contains the expected answer; `grep_workspace` may.
