"""Registry for versioned Python prompt constants."""
from __future__ import annotations

from prompts.direct import PROMPT as DIRECT_PROMPT
from prompts.rag import PROMPT as RAG_PROMPT
from prompts.scaleseek_prompt import PROMPT as SCALESEEK_PROMPT
from prompts.scaleseek_prompt_noparams import PROMPT as SCALESEEK_NOPARAMS_PROMPT
from prompts.system_prompt import PROMPT as LEGACY_SYSTEM_PROMPT

_PROMPTS = {
    "direct": DIRECT_PROMPT,
    "rag": RAG_PROMPT,
    "scaleseek_prompt": SCALESEEK_PROMPT,
    "scaleseek_prompt_noparams": SCALESEEK_NOPARAMS_PROMPT,
    "system_prompt": LEGACY_SYSTEM_PROMPT,
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
