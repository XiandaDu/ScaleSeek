# ScaleSeek

Training adaptive retrieval agents for scalable Direct Corpus Interaction (DCI).

## Overview

Full-corpus DCI (grep/shell search over 21M Wikipedia passages) becomes inefficient at scale: latency grows, noise increases, and accuracy degrades. ScaleSeek makes DCI scalable by first running BM25 to construct a small bounded **workspace**, then letting the agent perform fine-grained search only within that workspace.

```
Question
   ↓
Adaptive BM25 Retrieval  (query, top_k, k1, b, mode — all agent-controlled)
   ↓
Bounded Workspace        (merge or replace across turns)
   ↓
grep_workspace / read_doc
   ↓
Answer
```

The agent learns to control every BM25 decision (query, parameters, workspace mode) rather than using a fixed retrieval pipeline.

---

## Project Structure

```
prompts/
    system_prompt.txt       ← clean system prompt used in RL training and final inference
    scalaseek_prompt.txt    ← eval-only version: adds Search Strategy + Parameter Guidance
    bm25_rag.txt            ← single-retrieve-then-answer baseline
    direct.txt              ← no-retrieval baseline
    sft_prompts.py          ← Tutor/Planner prompt suite for cold-start SFT data generation

eval/
    agent.py                ← ScaleSeek prompt agent loop + bm25_rag / direct baselines
    run_eval.py             ← unified eval runner (all agents)
    bm25_retriever.py       ← Pyserini BM25 wrapper (21M passage Lucene index)
    datasets.py             ← dataset loaders (PopQA, HotpotQA, 2Wiki, MuSiQue, ...)
    metrics.py              ← EM / F1
    shell_tool.py           ← safe shell executor (rg/grep) for DCI and GrepSeek agents
    dci_agent.py            ← prompt-based DCI baseline (grep on full corpus, no BM25)
    grepseek_agent.py       ← GrepSeek trained model wrapper (separate vLLM port)
    search_r1_agent.py      ← Search-R1 baseline (Qwen2.5-3B GRPO, separate vLLM port)
    agentir_retriever.py    ← AgentIR-4B full-corpus FAISS dense retrieval

sft/                            ← (future) SFT pipeline code

train/
    environment.py          ← per-rollout ScaleSeekEnv (workspace + BM25 singleton)
    agent_loop.py           ← verl-compatible agent loop
    reward.py               ← EM reward
    dataset.py              ← RL dataset wrapper

scripts/
    build_agentir_index.py  ← encode all 21M passages → FAISS index (run once, ~12–24h GPU)
    build_bm25_index.py     ← build Pyserini Lucene index from wiki_corpus.jsonl
    prepare_rl_data.py
    precompute_agentir.py   ← deprecated; use build_agentir_index.py instead
```

---

## Prompt Design

### Two prompt files, two purposes

**`prompts/system_prompt.txt`** is the canonical system prompt. It describes the three tools and the output format. This is what the RL-trained agent sees at inference time. It contains no strategy hints — the model is expected to learn retrieval strategy from training.

**`prompts/scalaseek_prompt.txt`** is a superset used only for the prompt-based eval agent (Stage 1, before RL training). It adds a five-step Search Strategy section and BM25 Parameter Guidance to compensate for the fact that the base model has no training signal for workspace management.

The split exists because hints that improve a zero-shot model hurt a trained one (the trained model should discover its own policy, not be constrained by a fixed strategy written into the prompt).

### Tool design

The agent has three tools that form a two-stage hierarchy:

| Tool | Stage | Purpose |
|---|---|---|
| `bm25_retrieve(query, top_k, k1, b, mode)` | Coarse | Load passages from corpus into workspace |
| `grep_workspace(pattern, case_insensitive)` | Fine | Search within workspace passages |
| `read_doc(doc_id)` | Fine | Read a specific passage in full |

`bm25_retrieve` exposes BM25 parameters `k1` (term-frequency saturation) and `b` (length normalization) as agent-controllable arguments, along with `mode` (`"replace"` to start fresh, `"merge"` to accumulate). Controlling these is ScaleSeek's core research contribution — baseline agents fix k1=1.5, b=0.75 and never merge.

### Output format

All tool calls and answers use a shared format (same as GrepSeek, enabling joint training):

```
<tool_call>
{"name": "bm25_retrieve", "arguments": {"query": "...", "top_k": 5, "k1": 1.5, "b": 0.75, "mode": "replace"}}
</tool_call>

<answer>
concise answer
</answer>
```

---

## SFT Data Generation (`sft/data_generation/utils/prompts.py`)

Cold-start SFT trajectories are synthesised using a Tutor/Planner pipeline adapted from GrepSeek. Two LLM roles collaborate to build clean training trajectories without human annotation:

**Tutor** sees the gold answer and works backward. It decomposes the question, discovers which BM25 queries retrieve the right passages, and edits the planner's reasoning to remove information leakage.

**Planner** sees only the question and the tool outputs accumulated so far. It generates the next (reasoning, tool\_call) step forward, as if solving the problem blind.

### Pipeline phases

**Phase A — Decomposition** (`DECOMPOSE_PROMPT`): Tutor splits the multi-hop question into 1–3 ordered single-hop sub-questions.

**Phase B — Backward tool-trace discovery**: For each sub-question, the Tutor finds a `bm25_retrieve` call (and optional `grep_workspace`) that surfaces the answer passage.

- `BACKWARD_TOOL_SYSTEM` / `BACKWARD_TOOL_USER_INITIAL`: Tutor proposes an initial tool trace.
- `BACKWARD_TOOL_USER_REFINE`: Tutor refines the query if the judge rejects it.
- `JUDGE_PROMPT`: Verifies the BM25 result contains a passage confirming the expected answer.
- `BRIDGE_EXTRACT_PROMPT`: Extracts the intermediate bridge entity from a retrieved passage.

The critical constraint is the **BM25 ANSWER-LEAK RULE**: the `bm25_retrieve` query must not contain the expected answer string. The answer must emerge from the retrieved passage. `grep_workspace` may reference the answer — it runs after BM25 and is used to verify the correct passage landed in the workspace.

**Phase C — Forward planner generation**: Planner (`PLANNER_SYSTEM` / `PLANNER_USER`) generates draft (reasoning, tool\_call) pairs forward from the question alone. Tutor (`TUTOR_EDIT_PROMPT`) edits the planner's reasoning to align with the known-correct tool call without introducing facts the planner could not yet know.

**Phase D — Final answer** (`FINAL_ANSWER_USER`): Planner synthesizes the answer from the accumulated trace.

**Phase E — Quality filter** (`QUALITY_JUDGE_PROMPT`): Post-hoc judge checks the assembled trajectory for four failure modes:

| Check | Failure condition |
|---|---|
| 1 | Reasoning names a fact not yet observed in the trace |
| 2 | `bm25_retrieve` query contains the answer string (answer-leak) |
| 3 | Reasoning says "I have the answer" but action is a tool call |
| 4 | Final answer is not supported by any prior tool output |

Note: `grep_workspace` patterns that reference the answer are **not** a Check 2 failure.

### Relation to GrepSeek

The Tutor/Planner structure and quality-judge design are directly adapted from GrepSeek. The key difference is the tool set: GrepSeek's backward phase discovers a single shell command (`rg` / `grep` on `corpus.jsonl`); ScaleSeek's backward phase discovers a `bm25_retrieve` call with explicit parameter choices, optionally followed by `grep_workspace`. Prompts with no tool-specific content (`DECOMPOSE_PROMPT`, `BRIDGE_EXTRACT_PROMPT`, `PLANNER_USER`, `FINAL_ANSWER_USER`) are used unchanged.

---

## Baselines

| Agent | Paper | Model | Retrieval |
|---|---|---|---|
| `direct` | — | Qwen3-4B | none |
| `bm25_rag` | — | Qwen3-4B | BM25 top-k (fixed, single call) |
| `dci` | [Beyond Semantic Similarity](https://arxiv.org/abs/2605.05242) | Qwen3-4B (prompt) | grep on full corpus, no BM25 |
| `agentir_rag` | [AgentIR](https://huggingface.co/Tevatron/AgentIR-4B) | AgentIR-4B (embedding) + Qwen3-4B reader | full-corpus FAISS dense retrieval |
| `search_r1` | [Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Qwen2.5-3B (GRPO) | adaptive BM25 via `<search>` tags |
| `grepseek` | [GrepSeek](https://arxiv.org/abs/2605.29307) | GrepSeek checkpoint (trained) | grep on full corpus, training-based |
| `scalaseek` | this work | Qwen3-4B (prompt / RL) | adaptive BM25 + bounded workspace DCI |

**Design axis:** `direct` → `bm25_rag` → `dci`/`grepseek` → `scalaseek` progresses from no retrieval, through fixed retrieval, through full-corpus DCI, to bounded-workspace DCI with learned parameters. `agentir_rag` is the dense-retrieval counterpart to `bm25_rag`. `search_r1` is the trained adaptive-retrieval counterpart without workspace management.

## Eval

```bash
source setup_env.sh

# ScaleSeek prompt agent
python -m eval.run_eval --dataset hotpotqa --agent scalaseek \
    --output results/hotpotqa_scalaseek.jsonl

# BM25-RAG baseline
python -m eval.run_eval --dataset hotpotqa --agent bm25_rag \
    --output results/hotpotqa_bm25_rag.jsonl

# DCI baseline (Beyond Semantic Similarity — grep on full corpus, no BM25)
python -m eval.run_eval --dataset hotpotqa --agent dci \
    --corpus-path /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \
    --output results/hotpotqa_dci.jsonl

# AgentIR dense retrieval (build FAISS index once, then eval)
python scripts/build_agentir_index.py \
    --corpus /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \
    --out /data/rech/mofengra/data/agentir_index --device cuda
python -m eval.run_eval --dataset hotpotqa --agent agentir_rag \
    --agentir-index-dir /data/rech/mofengra/data/agentir_index \
    --output results/hotpotqa_agentir.jsonl

# GrepSeek trained model (separate vLLM on port 8002)
python -m eval.run_eval --dataset hotpotqa --agent grepseek \
    --grepseek-port 8002 \
    --corpus-path /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \
    --output results/hotpotqa_grepseek.jsonl

# Search-R1 (separate vLLM on port 8001)
python -m eval.run_eval --dataset hotpotqa --agent search_r1 \
    --search-r1-port 8001 --output results/hotpotqa_search_r1.jsonl
```
