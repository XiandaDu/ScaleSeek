# Prompt modules

All project-owned prompts are Python constants. Runtime code must import them
through `eval.prompts.load()` or directly from their module; it must not read
prompt text from the filesystem.

- `prompts.direct:PROMPT`: Direct baseline.
- `prompts.rag:PROMPT`: backend-neutral RAG reader.
- `prompts.scaleseek_prompt:PROMPT`: canonical ScaleSeek evaluation/RL prompt.
- `prompts.scaleseek_prompt_noparams:PROMPT`: parameter-guidance ablation.
- `prompts.system_prompt:PROMPT`: legacy SFT provenance prompt.
- `prompts.sft_prompts`: tutor/planner templates used to generate SFT data.

`configs/prompt_snapshots.yaml` hashes the actual constant values passed to the
models. Files named in official-source documentation, such as DCI-Agent-Lite's
`prompts/system_prompt.txt`, live in pinned upstream repositories and are not
local runtime prompt files.
