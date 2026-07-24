# Prompt modules

This directory contains only project-authored or project-modified prompts.
Baseline runtime code imports them through `eval.prompts.load()`; SFT code
imports its adapted templates from `prompts.sft_prompts`.

- `prompts.direct:PROMPT`: Direct baseline.
- `prompts.rag:PROMPT`: backend-neutral RAG reader.
- `prompts.scaleseek_prompt:PROMPT`: canonical ScaleSeek evaluation/RL prompt —
  default, explains the BM25 knobs without anchoring them to numeric values.
- `prompts.scaleseek_prompt_withparams:PROMPT`: ablation that adds numeric
  parameter hints (top_k/k1/b examples).
- `prompts.sft_prompts`: GrepSeek-derived templates modified for ScaleSeek's
  BM25/workspace tools. The complete SFT prompt suite is intentionally kept as
  one unit here, including the few unchanged upstream fragments.

`configs/prompt_snapshots.yaml` hashes the actual constant values passed to the
models. Verbatim third-party prompts live in the corresponding `eval` module:
Search-R1 and Search-O1 in their agent modules, the BCP judge in
`eval.browsecomp_plus_judge`. Prompts consumed only by pinned official external
harnesses remain in those upstream checkouts and are not copied locally. SFT is
the explicit exception to the placement split: its full suite stays together in
`prompts.sft_prompts`.
