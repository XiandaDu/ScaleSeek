# Reproducibility question: PopQA F1 — official inference code + public checkpoint gives ~0.415, paper reports 0.486

Hi, and thank you for releasing the full training + inference stack and the
`alireza7/GrepSeek-Qwen3.5-9B-GRPO` checkpoint — the DCI idea is great and the
code is unusually complete.

We are building a baseline comparison and tried to reproduce the **PopQA** number.
Running **your own `inference/run.py` unchanged** with the **public checkpoint**, we
get **F1 ≈ 0.415**, whereas the paper reports **PopQA F1 0.486** (Table 1, the GrepSeek
row — DCI grepping the raw corpus, no external retriever). We suspect the gap comes
from something we had to **assume** because it is not
fully specified in the repo/paper, and we would love your guidance on which knob it
is. Details below.

## What we ran

| item | value |
|---|---|
| repo commit | `1f6ea58` |
| checkpoint | `alireza7/GrepSeek-Qwen3.5-9B-GRPO` (bf16, Qwen3.5-9B base) |
| serving | vLLM, `--reasoning-parser qwen3 --max-model-len 32768` (TP=2 on 2×A5000, `--disable-custom-all-reduce`, `NCCL_P2P_DISABLE=1`) |
| corpus | Search-R1 Wiki-18 DPR passages (`wiki_18_corpus`, ~14GB, `GREPSEEK_CORPUS_ROOT`) |
| dataset | `RUC-NLPIR/FlashRAG_datasets` → `popqa/test`, via your `inference/load_dataset.py` |
| scoring | your `inference/scoring.py` (`normalize_answer` + `f1_score`) |
| sampling | temperature **0.6**, top_p 1.0 |
| loop | **6** turns (5 tool + 1 answer) |
| tool stdout cap | **2048 tokens** (model tokenizer) |
| per-turn generation | **uncapped** |

## The observation

On an **identical 1,000-example PopQA subset**, same server, we compared your code
against our own re-implementation as a sanity check:

| harness | EM | F1 |
|---|---|---|
| `grepseek/inference/run.py` (unchanged) | 0.3664 | **0.4154** |
| our re-implementation | 0.3654 | 0.4187 |

The two agree to **0.001 EM / 0.003 F1** (74% per-example exact agreement, the rest
explained by temperature-0.6 sampling), so we are fairly confident the **agent loop
is faithful** — i.e. the ~7 F1-point gap to the paper's 0.486 is **not** an
implementation bug on our side, but comes from the **evaluation protocol / assets**.
(On a separate random 1,500-example subset we get F1 0.387, i.e. our numbers
sit in the 0.387–0.419 range depending on the subset — always well below 0.486.)

## Assumptions we had to make (candidate causes)

These are the points the paper/repo did not fully pin down, so we picked a value —
any of them could be the source of the gap:

1. **Evaluation set / sampling.** We scored a **subset** (1,000 / 1,500), not the full
   14,267-example `popqa/test`. Is the paper's 0.486 on the **full** test set? A 7-point
   gap is much larger than our sampling noise (95% CI ≈ ±2.3 F1 at n=1,500), but we want
   to rule this out — did you subsample, and if so how? (Our subset is a **uniform
   random sample of 1,500** from the 14,267-example test set, so it should be
   representative; a 7-F1 gap is well beyond its sampling noise.)
2. **Inference temperature.** We used **0.6** (the value we found in the configs). Is the
   **evaluation** run at 0.6, or greedy (temp 0), or best-of-N / multiple samples averaged?
   A single temp-0.6 sample is a plausible few-point penalty vs greedy or vote@k.
3. **Number of turns / search budget.** We capped at **6** turns. Is the eval-time budget
   larger?
4. **Tool stdout cap.** We cap tool output at **2048 tokens** with the model's own
   tokenizer (matching `--tool_max_tokens 2048` in SFT data generation). Is the
   **inference-time** cap the same, or larger (character-based, or a different token
   budget)?
5. **Per-turn generation cap.** We left it **uncapped** (a fixed cap truncated the
   `<think>` block and produced parse errors). What is the official eval setting?
6. **Corpus snapshot / chunking.** We used the Search-R1 Wiki-18 DPR passages
   (~100-word chunks, 14GB). DCI greps raw text, so the exact **chunking / dedup /
   ordering** of the corpus changes which lines `rg` returns. Is this the exact corpus
   snapshot used for Table 1, or a differently preprocessed Wikipedia dump?
7. **Checkpoint identity.** Is the public `alireza7/GrepSeek-Qwen3.5-9B-GRPO` the **exact**
   checkpoint behind the Table 1 numbers, or an earlier/updated merge?
8. **ripgrep version / flags.** Which `rg` version and default flags (literal `-F`, case
   sensitivity, context lines) does the released harness assume? These affect recall on a
   grep-based agent.
9. **Answer extraction & error handling.** We parse the final answer from the `<answer>`
   tag and terminate after 2 consecutive parse errors; on context overflow we clamp+retry.
   Does your eval recover differently (e.g. retries, re-prompts)?

## Questions

- Which of the above is the main driver of the 0.486 vs ~0.415 gap on PopQA?
- Could you share the **exact eval command / config** used for Table 1 (temperature,
  turns, tool cap, sample size, corpus path)?
- Is the released checkpoint the one used for the reported numbers?

We are happy to re-run on the **full 14,267 set** with your exact config and post the
result here, and to share our full logs / per-example traces if useful. Thanks again
for the great work and for any pointers!
