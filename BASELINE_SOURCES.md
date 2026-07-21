# Baseline source ledger

This ledger is the source of truth for Phase 1. Runtime configuration lives in
`configs/baselines.yaml`; repository pins live in `configs/official_repos.yaml`.
No implementation may claim to be official or verbatim unless it is checked
against the pinned source below.

## Global substitutions

- Prompt-based generators and the BrowseComp-Plus judge use
  `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Fixed BM25 baselines use `k1=1.2, b=0.75` by project decision.
- Ordinary retrieval returns top-3. Large-workspace methods retain their native
  candidate sizes: DR-DCI 300–600 and RISE 1000 per sub-query.
- Wiki QA uses one Wiki-18 content manifest. RISE uses an article-level view of
  the same content because its navigable-object interface is document based.

## Direct and RAG

GrepSeek reports Direct and RAG comparisons, but its public repository does not
publish the exact prompts used for every table baseline. The local prompts are
therefore project-defined as `prompts.direct:PROMPT` and `prompts.rag:PROMPT`,
and hash-pinned over the exact Python strings sent to the model; they must never
be described as official GrepSeek prompts. All three RAG retrievers share the
same prompt, passage rendering, top-3 and context budget.

## Search-R1

- Paper: <https://arxiv.org/abs/2503.09516>
- Code: <https://github.com/PeterGriffinJin/Search-R1>, commit
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`
- Inference source: `infer.py`
- Models:
  - `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3@395b18f1fecee52f1b51fb22f898c220f0a08ec3`
  - `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3@d65f11c88d3c129a01466f0c154aaad7d9b09225`
- Preserved protocol: tokenizer chat template, exact user prompt (including the
  upstream `as your want` typo), raw continuation, stop on `</search>`, top-3
  injection, `B=4`, 1024 new tokens, temperature 0.7.
- Project extension: BM25/E5/Qwen3-Embedding-4B retriever matrix.

## Search-O1

- Paper: <https://arxiv.org/abs/2501.05366>
- Code: <https://github.com/RUC-NLPIR/Search-o1>, commit
  `c76a700fb2a948039ada577c03b3be958aa57282`
- Prompt source: `scripts/prompts.py`
- Loop source: `scripts/run_search_o1.py`
- Single-hop datasets use the complete single-QA instruction; multi-hop datasets
  use the complete multi-QA instruction. Qwen3.5 uses the non-QwQ task branch.
- Preserved protocol: one continuous completion, query stop token,
  Reason-in-Documents call, inline result injection, final boxed-answer parsing,
  search limits 5/10 and max turn 15.
- Project substitutions: Bing becomes a local top-3 retriever; both main and RiD
  calls use Qwen3.5-9B.

## GrepSeek

- Paper: <https://arxiv.org/abs/2605.29307>
- Code: <https://github.com/alirezasalemi7/grepseek>, commit
  `1f6ea58372defe774213e22c7650b7fd1b842ab8`
- Model: `alireza7/GrepSeek-Qwen3.5-9B-GRPO@a79563970cfdd2ced3cc5fde481737d0ebea6fa4`
- Sources: `inference/agent.py`, `inference/tools.py`
- Six assistant turns, temperature 0.6, top-p 1, no fixed per-turn generation
  cap, 2048-token tool output using the checkpoint tokenizer.

## DCI

- Paper: <https://arxiv.org/abs/2605.05242>
- Code: <https://github.com/DCI-Agent/DCI-Agent-Lite>, commit
  `271f37e71f053bf0c99c05ce6d2fb53b841d922e`
- Sources: `prompts/system_prompt.txt` and
  `scripts/bcplus_eval/run_bcplus_eval.py`
- The official Pi/DCI-Agent-Lite runner is mandatory. Local adapters may only
  convert datasets and results. Qwen3.5-9B replaces GPT-5.4-nano; 300 calls,
  level3 context management and a 30-second command timeout are retained.

## DR-DCI

- Paper: <https://arxiv.org/abs/2606.14885>
- Code: <https://github.com/EigenTom/DR-DCI>, commit
  `0d0410f3c2b98fb33145adc250a09fded028cd3c`
- Official Pi harness and task scripts are mandatory.
- Wiki-18 retains the official E5 pull route. BrowseComp-Plus retains the
  official `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
  index. There is no retriever sweep.
- Retained settings: 300 calls, level3, 300–600 candidates per pull, at most 10
  pull queries, ranked top-20 preview, root-flat disclosed materialization and
  30-second command timeout.

## RISE

- Paper: <https://arxiv.org/abs/2606.06880>
- Code: <https://github.com/texttron/RISE>, commit
  `32672b59ead8381decae7d412f05bb53d399946a`
- Source: `scripts/run_rise.py` and `src/rise/`
- Retained settings: BM25 boundary, K=1000 per sub-query, top-10 preview,
  monotonic file workspace, structured TOC documents, 100 model calls, one-hour
  query cap, 60-second subprocess cap, 4000-character bash output and 2000-line
  reads.
- Deliberate difference: official bm25s uses 1.5/0.75; this project uses the
  globally fixed 1.2/0.75.
- BCP TOC asset:
  `Tevatron/browsecomp-plus-md-toc-gpt5.4-nano@9ae6db9bb5c006cf77b82d6835e3df43a6774d6f`.
- Independent BCP judging uses the verbatim Appendix-F prompt in
  `scripts/judge_browsecomp_plus.py`, Qwen3.5-9B, temperature 0 and top-p 1.

## AgentIR

- Paper: <https://arxiv.org/abs/2603.04384>
- Code: <https://github.com/texttron/AgentIR>, commit
  `9626bdc7ab90649608d483e66febe13c6eebab2c`
- Model: `Tevatron/AgentIR-4B@e31abb637caa227c4b7d04176a24ecff1bcb10f4`
- The query representation must be `Reasoning: ...\n\nQuery: ...` under the
  official instruction prefix. Encoding the raw question alone is only a dense
  RAG ablation and may not be labeled AgentIR.
- The official OSS agent keeps `k=5` and at most 100 iterations. This is not
  changed to top-3 because it is AgentIR's native interactive configuration,
  not the GrepSeek-style Search-R1/Search-O1 retriever swap.

## Removed: IRCoT approximation

The old local module was a zero-shot mechanism approximation. The official
IRCoT repository requires dataset-specific demonstrations and has no PopQA
configuration. Phase 1 therefore removes this row rather than inventing a
PopQA adapter and labeling it IRCoT.

## ScaleSeek

ScaleSeek is the project method, not an external reproduction. Its prompt,
tool schema and environment protocol are versioned locally. Evaluation and RL
both import `prompts.scaleseek_prompt:PROMPT`; the no-parameter ablation imports
`prompts.scaleseek_prompt_noparams:PROMPT`. Qwen3.5-9B is the generator; its
BM25 fallback is 1.2/0.75. Adaptive k1/b choices remain a method feature and
must be logged per call.
