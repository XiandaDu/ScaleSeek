"""RAG: retrieve top-k once with the question, one reader call (prompts/rag.py,
project-defined).  Passages are the corpus text, whole; the same template for every
retriever."""

from __future__ import annotations

from ..corpus import split_title
from .base import Method
from .direct import finish_answer


class RAG(Method):
    def run_example(self, ex: dict, rec: dict) -> None:
        hits = self.search(rec, ex["question"])
        if hits is None:
            rec["finish_reason"] = "tool_error"
            return
        rule = self.ctx.title_rule
        docs = ""
        for i, h in enumerate(hits, 1):
            title, text = split_title(h["contents"], rule)
            docs += self.ctx.prompts["passage"].format(index=i, title=title, text=text)
        user = self.ctx.prompts["user"].format(documents=docs, question=ex["question"])
        out = self.generate(rec, self.chat_prompt(user), cap=self.profile["single_call_max_tokens"])
        rec["n_turns"] = 1
        if out is None:
            return
        finish_answer(rec, out)
