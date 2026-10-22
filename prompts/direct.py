"""Direct prompt -- PROJECT-DEFINED.

The GrepSeek paper reports a Direct row but publishes no prompt, so this one is ours.  It
shares its wording with the RAG prompt (prompts/rag.py) so the two differ only by the
documents block.  Frozen by the sha256 in configs/methods/direct.yaml.
"""

DIRECT_USER_TEMPLATE = (
    "Answer the given question. Give the final answer inside <answer> and </answer>, "
    "as a short phrase without explanation, for example <answer> Beijing </answer>.\n\n"
    "Question: {question}"
)
