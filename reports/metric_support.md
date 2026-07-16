# ScaleSeek Eval — Metric Support & Definitions

Which metric each dataset can support (given the labels we actually have on disk),
the exact definitions with paper sources, and the BM25 sweep design. Generated for
the stage-1 baseline comparison.

## 0. 评测标准集（用户 2026-07-13 拍板）

**popqa_full = 从 FlashRAG popqa/test（14,267）随机抽的固定 1500 样本**
（`$DATASETS/popqa_full/test.jsonl`，Jul-06 由 popqa/preprocess.py 用 `random` 抽，
id 均匀散布全 0-14265、均值 7286≈期望 7133 → 确系均匀随机；**exact seed 不可考**），
正式定为 stage-1 主对比标准集。抽样先例：IRCoT（ACL 2023）每集 500 随机、DR-DCI 每集 50。
n=1500 时 EM/F1 95% 置信带约 **±2.3 点**——同集内**方法间排序**严格可比；对**论文绝对值
对标**（GrepSeek/Search-R1 用全 14,267）7 点差距远超此带。同集所有 agent 跑同一 1500，横比有效。
`popqa_longtail`（Self-RAG 的 1,399 长尾子集，pop≤99）为前项目遗留资产，非标准集。

## 1. Dataset × metric support

| Dataset | Local file | Gold labels on disk | EM / F1 | Gold R@W / coverage | Qrel R@W | Localization | Latency |
|---|---|---|:--:|:--:|:--:|:--:|:--:|
| NQ | `nq/test.jsonl` | answers only | ✅ | ❌ | ❌ | ❌ | ✅ |
| TriviaQA | `trivialqa/trivialqa_test.jsonl` | answers only | ✅ | ❌ | ❌ | ❌ | ✅ |
| PopQA | `popqa/popqa_longtail.jsonl` | answers (+`ctxs`,`s_wiki_title`) | ✅ | ⚠️ subject-title only | ❌ | ❌ | ✅ |
| **HotpotQA** | `hotpotqa/hotpot_dev.jsonl` (+distractor json) | 2 gold passages → **titles** | ✅ | ✅ (title-level) | ≡ Gold R@W | ⚠️ optional | ✅ |
| **2Wiki** | `2wiki/2wiki_dev.jsonl` | `supporting_facts.title` (+10 ctx paras) | ✅ | ✅ (title-level) | ≡ Gold R@W | ⚠️ optional | ✅ |
| MuSiQue | `musique/dev.jsonl` | **answers only** (support paras stripped) | ✅ | ❌ | ❌ | ❌ | ✅ |
| Bamboogle | `bamboogle/test.jsonl` | **answers only** (none exist) | ✅ | ❌ | ❌ | ❌ | ✅ |
| **BrowseComp-Plus** | *(download step)* | native **gold + evidence + distractor** qrels | ✅¹ | ✅ (doc-id) | ✅ (doc-id) | ✅ | ✅ |

**Corrections to the initial assumption.** The claim that *HotpotQA / 2Wiki /
MuSiQue / Bamboogle* all support gold+qrel recall holds **only for HotpotQA and
2Wiki** with our current files:
- **MuSiQue** — our `dev.jsonl` was pre-stripped to answers only (the original
  MuSiQue *does* ship `paragraph_support_idx`; re-download if gold recall is wanted).
- **Bamboogle** — 125 Google-searchable questions with **no gold documents by
  construction**; EM/F1 only.
- **"qrel recall" ≠ "gold recall"** only where an *evidence* set distinct from the
  *gold* set exists — that is **BrowseComp-Plus only**. HotpotQA/2Wiki have a single
  gold set, so their Qrel R@W is identical to Gold R@W.
- **BrowseComp-Plus supports everything**, but ships its **own ~100K-doc corpus +
  qrels**, so it needs a *separate* BM25 index (not wiki-18). Wired last.
- ¹ BCP answers are free-form → the papers score with an **LLM judge**; our EM/F1
  will under-report on BCP. (LLM-judge scoring is out of scope unless requested.)

Granularity note: wiki-18 gold labels are **title-level** (a gold title counts as
recalled if the agent surfaced ≥1 corpus passage of that article). BrowseComp-Plus
qrels are **doc-id level**.

**Pipeline validation (smoke50, 2026-07-06)** — the matrix above is empirically
confirmed: hotpotqa (GoldR@W .470 / cov_any .780) and 2wiki (.285 / .540) produce
title-level retrieval metrics with 50/50 qrels defined; nq/triviaqa/musique/bamboogle
correctly degrade to EM/F1-only with an explanatory note; **BrowseComp-Plus** built
end-to-end (100,195 docs; 830 queries canary-decrypted, gold docids verified against
the corpus; separate BM25 index) and produces **all five metric families** —
EM/F1, Gold R@W, Qrel R@W, coverage(any/mean/all), localization. Absolute smoke
numbers on BCP are near zero by design: real BCP runs need BCP-adapted prompts,
large turn budgets (papers use 300), an LLM judge, and a stronger backbone.

## 2. Metric definitions (with sources)

Answer quality lives in `eval/metrics.py`; the rest in `eval/retrieval_metrics.py`.

**Answer quality.** EM = exact match after normalization; F1 = max token-overlap
F1 over gold answers (NQ/TriviaQA convention).

**Workspace recall — DR-DCI** ([2606.14885](https://arxiv.org/abs/2606.14885) §4.1):
`Gold R@W = |W_T ∩ G(q)| / |G(q)|`, `Qrel R@W = |W_T ∩ R(q)| / |R(q)|`, where
`W_T` is the agent's final materialized workspace, `G(q)` the gold docs, `R(q)` the
qrel/evidence docs. For non-workspace agents (dci/grepseek) `W_T` = the set of
corpus docs the agent surfaced in its tool outputs.

**Coverage — DCI paper** ([2605.05242](https://arxiv.org/abs/2605.05242) Eq. 1),
`M` = surfaced gold docs: `coverage_any=1[|M|≥1]`, `coverage_mean=|M|/|D*|`
(= gold recall), `coverage_all=1[|M|=|D*|]`.

**Localization — DCI paper** (Eqs. 2–5), fixed-width `c_seg` char segments:
`ν(x)=max(1,⌈x/c_seg⌉)`, `ψ(a;b)=max(1−log a/log b,0)` for `b>1` (`ψ(a;1)=1`),
`seg-score(d,d*)=ψ(ν(ℓ_snippet);ν(|d*|))`, best per surfaced gold doc, averaged over
`M`. Measures whether the agent narrowed to a small evidence span inside a reached
document (BrowseComp-Plus, where full doc lengths are available). `c_seg` default
500 chars; confirm against DCI-Agent-Lite §A.3 at run time.

**Latency / efficiency.** sec/query, LLM time, tool time, #tool calls, #bm25 calls,
and max-turns / api-error / parse-error rates (from the saved records).

## 3. BM25 (k1, b) sweep — the five configurations

Requested: ≥3 configs. We run **five**, chosen to span the two regimes that matter
and to reproduce the grid points from *Rethinking Agentic Search with Pi-Serini*
([2605.10848](https://arxiv.org/abs/2605.10848)).

| # | (k1, b) | Source / rationale |
|---|---|---|
| 1 | **0.9 / 0.4** | Anserini/Lucene **default** — Pi-Serini Fig. 3 ×-mark; standard for short passages. |
| 2 | **25 / 1.0** | Pi-Serini **long-document tuned** setting (their main BCP config). |
| 3 | **16 / 1.0** | Pi-Serini **best grid-search** point on BCP. |
| 4 | **1.2 / 0.75** | Classic Robertson / Elasticsearch default. |
| 5 | **1.5 / 0.75** | Our previous ScaleSeek default (for continuity with earlier runs). |

Why these five: configs 2–3 were tuned on **BrowseComp-Plus's long documents**
(median ~2k tokens); our wiki-18 corpus is **short ~100-word DPR passages**, so we
*expect* 0.9/0.4 and the 1.x/0.75 family to win on wiki-18 and 25/1 & 16/1 to
underperform — reproducing Pi-Serini's own observation that Anserini's default is
"tuned for the shorter-document regime." Running all five makes that contrast
explicit instead of assuming it. Retrieval depth follows Pi-Serini's finding that
depth drives surfaced recall; we sweep `--bm25-top-k` separately if needed.

## 4. Reproduction-fidelity notes

- **GrepSeek — reproduction VALIDATED against the official harness (head-to-head).**
  Serve the 9B checkpoint with `--tensor-parallel-size 2 --disable-custom-all-reduce
  --reasoning-parser qwen3 --max-model-len 32768` (A5000s: NCCL_P2P_DISABLE=1). Run with
  temp **0.6**, **6** turns, tool stdout capped at **2048 tokens** (model tokenizer).
  On an **identical 1000-example** PopQA set, same server:
  | harness | EM | F1 |
  |---|---|---|
  | ours (`eval/grepseek_agent.py`) | 0.3654 | 0.4187 |
  | official (`grepseek/inference/run.py`, unchanged) | 0.3664 | 0.4154 |
  Match to **0.001 EM / 0.003 F1** (74% per-example agreement; the rest is temp-0.6
  sampling variance). **Our implementation == the official harness.**
- **The paper's PopQA F1 0.486 is NOT reproducible** with the public checkpoint +
  our wiki-18 corpus: the *official* code itself gets only F1 0.416 here. The ~7-F1
  residual is inherent to the checkpoint/corpus/eval口径 (candidates: paper used the
  full 14267-set which is higher-popularity/easier, a different corpus snapshot, or a
  different checkpoint), **not our implementation**.
- Fidelity knobs found along the way (all real, all matched to `grepseek/inference/`):
  tool stdout cap 2048 tokens is *separate* from the per-turn generation cap (a fixed
  2048 generation cap truncates long `<think>` → parse_errors; official default is
  2048 but with the reasoning parser the `<think>` is split into `reasoning_content`);
  the server needs `--reasoning-parser qwen3` so `<think>` turns are well-formed; TP=2
  needs custom-all-reduce disabled on non-NVLink GPUs.
- **Raw-corpus grep agents must run at LOW concurrency (measured, important).**
  `--concurrency` gives a ~10× speedup for indexed agents (bm25_rag, search_o1) and
  is safe for them. But for agents that `grep` the raw 15 GB corpus, parallel scans
  thrash the disk and hit the 30 s tool timeout: at conc=16, **dci timed out on 44 %
  of tool calls** (981/1399 examples) → the 4B saw empty results, abstained, and
  churned (max_turns 67 → 155), tanking EM **0.274 → 0.179**. **GrepSeek tolerated
  conc=16 fine (1.6 % timeouts)** because the trained model writes tight
  `rg -F "phrase" | head` greps that finish fast. Rule: run **dci at conc ≤ 2**
  (or raise `tool_timeout`); grepseek is OK at conc≈16; bm25/search_o1 any concurrency.
- **DCI truncation cap** is kept at **8000 chars** (`run_dci(tool_max_chars=8000)`).
  The DCI paper's L3 is 20K (calibrated for GPT-5.4-nano); under our conc-16 runs 20K
  was a minor *secondary* drag on top of the timeout artifact (0.179 → 0.155). Pass
  20000 only with a strong model + low concurrency.
- **`clean_answer` `<think>`-leak fix**: real bug, retained, but its effect on the
  dci PopQA score is negligible (measured offline on the valid conc-1 run:
  EM 0.2738 → 0.2752, +2 examples) — the leaked predictions are verbose sentences
  that stay non-exact after tag-stripping. So the reported **dci = 0.274** stands.
- **Model substitution for prompt-based baselines.** DCI, DR-DCI, Search-O1 in the
  papers use frontier models (GPT-5.4-nano etc.); we cannot serve those locally, so
  they run on our **Qwen3-4B**, identical to our other prompt-based baselines. This
  makes them apples-to-apples with `scaleseek` (same model/harness) but **not**
  directly comparable to the papers' absolute numbers.
- **DR-DCI harness caveat.** The official DR-DCI ([repo](https://github.com/EigenTom/DR-DCI))
  is built on the **Pi TS coding-agent** harness (bash/read tools), not a
  single copyable prompt. See the runbook / open decision on how to reproduce it.

## Sources
- [Beyond Semantic Similarity (DCI), 2605.05242](https://arxiv.org/abs/2605.05242)
- [DR-DCI, 2606.14885](https://arxiv.org/abs/2606.14885) · [repo](https://github.com/EigenTom/DR-DCI)
- [Rethinking Agentic Search with Pi-Serini, 2605.10848](https://arxiv.org/abs/2605.10848)
- [GrepSeek, 2605.29307](https://arxiv.org/abs/2605.29307)
- [Search-o1 repo](https://github.com/sunnynexus/Search-o1)
