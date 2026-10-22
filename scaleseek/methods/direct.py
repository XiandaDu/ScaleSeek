"""Direct: one thinking-mode call, no retrieval (prompt: prompts/direct.py, project-defined)."""

from __future__ import annotations

from .base import Method, answer_after_think


class Direct(Method):
    def run_example(self, ex: dict, rec: dict) -> None:
        prompt = self.chat_prompt(self.ctx.prompts["user"].format(question=ex["question"]))
        out = self.generate(rec, prompt, cap=self.profile["single_call_max_tokens"])
        rec["n_turns"] = 1
        if out is None:
            return
        finish_answer(rec, out)


def finish_answer(rec: dict, out: dict) -> None:
    """Shared by Direct and RAG: parse <answer> after </think>."""
    answer, why = answer_after_think(out["text"])
    rec["prediction"] = answer
    if answer is not None:
        rec["finish_reason"] = "answer"
    elif out["finish_reason"] == "length":
        # Ran out of tokens: the single-call cap (BCP) or the context window itself.
        rec["finish_reason"] = "max_tokens_cap" if out["capped"] else "context_exhausted"
    else:
        rec["finish_reason"] = "no_answer"
        rec["errors"]["parse"] += 1
        rec["error"] = why
