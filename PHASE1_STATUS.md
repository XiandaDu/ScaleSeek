# Phase 1 status

Status: **Phase 1 implementation complete; final execution tests are assigned to the experiment server**.

## Completed in this workspace

- Frozen model, dataset, retriever and official repository revisions.
- Canonical full PopQA pin and preflight: 14,267 rows, 14,267 unique IDs,
  raw SHA256 `f976372fce3d7fe01b070357040391f10bd2710eeaa7c40d21c6574ffcad6cb8`,
  normalized SHA256 `56269abde328a259e4e38a186941cfd755ab0d72d7fbd7a1e8801a8ea781bd42`.
- Backend-neutral BM25/E5/Qwen3-Embedding-4B retrieval and index/corpus manifests.
- Search-R1 7B/14B protocol, B=4 and top-3; official prompt byte diff.
- Search-O1 exact single/multi/RiD prompts, tokenizer template, 5/10 search
  limits, 15 turns, top-3 and official sampling/RiD settings.
- Official commit-gated launchers for GrepSeek, DCI-Agent-Lite, DR-DCI,
  RISE and AgentIR. Local approximations were removed.
- Independent Qwen3.5-9B BrowseComp-Plus Appendix-F judge.
- Full-eval sampling guards, retries, strict resume provenance, exact ID-set
  validation, method fingerprints and normalized external-result adapter.
- Project-authored/modified prompts live under `prompts/`; byte-identical
  evaluation prompts live beside the corresponding evaluator under `eval/`.
  The complete SFT suite stays together under `prompts/sft_prompts.py`.
  Training, evaluation, snapshots and provenance consume the same constants.
- Deterministic Wiki passage-to-article reconstruction for the RISE structured
  corpus pipeline.
- Cleanup of obsolete 4B/3B/1500 launch scripts and mislabeled baseline code;
  historical results were preserved as legacy-only.

## Verification performed

- `20 passed` offline tests (prompt snapshots, prompt-placement contracts,
  shared train/eval prompt registry,
  parsers, fake loops, pooling,
  full-data guards, resume, retrieval merge and harness invariants).
- Official Search-R1/Search-O1 prompt byte diff passed at frozen commits.
- All five official-harness launch profiles passed commit-gated dry runs.
- Tiny corpus → corpus manifest → article reconstruction → aligned index
  manifest smoke passed.
- Python compilation, YAML/JSON parsing and `git diff --check` passed.

## Server-side execution handoff

By design, this desktop is only the code-authoring environment. The experiment
server must execute the following final acceptance runs:

1. Same-query retrieval against the three real full-corpus indexes.
2. Real Qwen3.5-9B and Search-R1 7B/14B search→tool→answer smoke runs.
3. Official GrepSeek/DCI/DR-DCI/RISE/AgentIR one-example model smokes.
4. Building and validating the full Wiki article TOC asset for RISE.
5. Deleting obsolete checkpoints at resolved server paths; no such path exists
   in this workspace, so `cleanup_manifest.json` records them as unresolved.

These are execution tests, not unfinished local implementation. Phase 2 starts
only after they pass on the server.
