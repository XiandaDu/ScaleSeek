"""ScaleSeek SFT cold-start data generation.

This package builds GrepSeek-style cold-start supervised trajectories for the
ScaleSeek two-stage retrieval agent, using the prompt suite in
``prompts/sft_prompts.py`` and the same tool environment as evaluation
(``eval.agent``). See ``train/sft/coldstart.py`` for the pipeline and
``scripts/generate_sft_data.py`` for the CLI.
"""
