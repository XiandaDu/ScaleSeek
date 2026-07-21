"""Verbatim GrepSeek SFT templates reused without modification.

Source: alirezasalemi7/grepseek@1f6ea58372defe774213e22c7650b7fd1b842ab8,
``sft/data_generation/utils/prompts.py``. ScaleSeek-specific adaptations remain
in ``prompts/sft_prompts.py``.
"""

PLANNER_USER = """Question: {question}

Trace so far:
{history}

Produce the next step."""

FINAL_ANSWER_USER = """Question: {question}

Trace so far:
{history}

You now have enough information. Produce a brief reasoning paragraph synthesizing the answer from the trace, then output exactly:

<answer>
the final answer (concise — just a name, date, or short noun phrase)
</answer>"""
