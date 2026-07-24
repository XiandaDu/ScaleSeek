"""Registry for versioned Python prompt constants."""
from __future__ import annotations

from prompts.direct import PROMPT as DIRECT_PROMPT
from prompts.rag import PROMPT as RAG_PROMPT
from prompts.scaleseek_prompt import PROMPT as SCALESEEK_PROMPT
from prompts.scaleseek_prompt_withparams import PROMPT as SCALESEEK_WITHPARAMS_PROMPT

_PROMPTS = {
    "direct": DIRECT_PROMPT,
    "rag": RAG_PROMPT,
    # Default is the no-parameter-bias prompt; the numeric-hint version is an ablation.
    "scaleseek_prompt": SCALESEEK_PROMPT,
    "scaleseek_prompt_withparams": SCALESEEK_WITHPARAMS_PROMPT,
}
_ALIASES = {
    # Older training code used this logical name even though no matching
    # scaleseek.txt ever existed. Keep the logical alias, not a file fallback.
    "scaleseek": "scaleseek_prompt",
}


def load(name: str) -> str:
    """Return a prompt constant by its stable logical name."""
    resolved = _ALIASES.get(name, name)
    try:
        return _PROMPTS[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROMPTS))
        raise KeyError(f"Unknown prompt {name!r}; choose one of: {choices}") from exc
