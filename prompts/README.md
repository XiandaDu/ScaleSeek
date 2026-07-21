# Prompt modules

This directory contains only project-authored or project-modified prompts.
Baseline runtime code imports them through `eval.prompts.load()`; SFT code
imports its adapted templates from `prompts.sft_prompts`.

- `prompts.direct:PROMPT`: Direct baseline.
- `prompts.rag:PROMPT`: backend-neutral RAG reader.
- `prompts.scaleseek_prompt:PROMPT`: canonical ScaleSeek evaluation/RL prompt.
- `prompts.scaleseek_prompt_noparams:PROMPT`: parameter-guidance ablation.
- `prompts.sft_prompts`: GrepSeek-derived templates modified for ScaleSeek's
  BM25/workspace tools. Its two unchanged upstream templates are imported from
  `eval.grepseek_sft_prompts` rather than duplicated here.

`configs/prompt_snapshots.yaml` hashes the actual constant values passed to the
models. Verbatim third-party prompts live in the corresponding `eval` module:
Search-R1 and Search-O1 in their agent modules, the BCP judge in
`eval.browsecomp_plus_judge`, and unchanged GrepSeek SFT fragments in
`eval.grepseek_sft_prompts`. Prompts consumed only by pinned official external
harnesses remain in those upstream checkouts and are not copied locally.
