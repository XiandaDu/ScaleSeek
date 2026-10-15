"""Search-o1: the official scripts/run_search_o1.py loop, one question at a time.

Byte-diffed copies (prompts/official/search_o1.py): the instruction functions, the task
instruction, Reason-in-Documents, extract_answer, extract_between, the four tags and the
inline f-string messages.  The control flow below re-implements run_search_o1.py main()
line for line for ONE sequence (the official script batches many sequences per vLLM call;
batching does not change any single sequence's prompts):

  turn loop (<= max_assistant_turns): generate with stop=[END_SEARCH_QUERY, eos_token];
    query = extract_between(this turn's text); if a query is present and the output ends
    with END_SEARCH_QUERY:
      * search_count < limit and query is new -> retrieve, build the truncated previous
        reasoning (official algorithm), run Reason-in-Documents, append
        RESULT_MESSAGE(extract_answer(rid, 'infogen'));
      * search_count >= limit -> LIMIT_MESSAGE;  repeated query -> REPEAT_MESSAGE;
    otherwise the sequence is finished.
  answer = extract_answer(output, mode='qa')  (last \\boxed{...}).

Changed (docs/deviations.md): the answer is read only from the text after the last
</think>, as for Direct/RAG.  Local-corpus adaptation: Bing results become top-k corpus documents
shown as {"id", "title", "context"}; with no search snippet the official fallback
`context = raw_context[:max_doc_len * 2]` applies.
"""

from __future__ import annotations

import json
import re

from ..corpus import split_title
from .base import Method


def fill(template: str, **values) -> str:
    """Fill an official f-string template whose placeholders are the official
    expressions, e.g. fill(STEP_LINE, **{'i + 1': 3, 'step': s})."""
    keys = sorted(values, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape("{" + k + "}") for k in keys))
    # One pass, so text inserted for one placeholder is never re-scanned for another.
    return pattern.sub(lambda m: str(values[m.group(0)[1:-1]]), template)


def truncated_prev_reasoning(output: str, P: dict) -> str:
    """run_search_o1.py lines 562-584, verbatim logic."""
    BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT = P["begin_search_query"], P["begin_search_result"]
    all_reasoning_steps = output
    all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

    truncated = ""
    for i, step in enumerate(all_reasoning_steps):
        truncated += fill(P["step_line"], **{"i + 1": i + 1, "step": step})

    prev_steps = truncated.split('\n\n')
    if len(prev_steps) <= 5:
        truncated = '\n\n'.join(prev_steps)
    else:
        truncated = ''
        for i, step in enumerate(prev_steps):
            if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                truncated += step + '\n\n'
            else:
                if truncated[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                    truncated += '...\n\n'
    return truncated.strip('\n')


class SearchO1(Method):
    def run_example(self, ex: dict, rec: dict) -> None:
        P = self.ctx.prompts
        tok = self.ctx.tokenizer
        tags = {k: P[k] for k in ("begin_search_query", "end_search_query",
                                  "begin_search_result", "end_search_result")}
        END_SEARCH_QUERY = tags["end_search_query"]
        branch = self.spec.dataset_cfg["search_o1_instruction"]
        limit = self.profile["max_assistant_turns"] - 1        # MAX_SEARCH_LIMIT (§4)
        max_doc_len = self.spec.method["search_o1"]["max_doc_len"]
        instruction = P[branch](limit)
        user_prompt = P["task"](ex["question"])          # non-QwQ branch (model_name=None)
        prompt = self.chat_prompt(instruction + user_prompt)
        output = ""
        search_count, executed = 0, set()
        finished = False
        for _ in range(self.profile["max_assistant_turns"]):
            out = self.generate(rec, prompt, stop=[END_SEARCH_QUERY, tok.eos_token],
                                include_stop_str_in_output=True)
            if out is None:
                break
            rec["n_turns"] += 1
            text = out["text"]
            prompt += text
            output += text
            if out["finish_reason"] == "length":
                rec["finish_reason"] = "context_exhausted"
                break
            search_query = P["extract_between"](text, tags["begin_search_query"], END_SEARCH_QUERY)
            if search_query and output.rstrip().endswith(END_SEARCH_QUERY):
                if search_count < limit and search_query not in executed:
                    append = self._search_and_reason(rec, ex, output, search_query, max_doc_len, P, tags)
                    if append is None:          # context window full during RiD
                        break
                    search_count += 1
                    executed.add(search_query)
                elif search_count >= limit:
                    append = fill(P["limit_message"], **{"BEGIN_SEARCH_RESULT": tags["begin_search_result"],
                                                         "END_SEARCH_RESULT": tags["end_search_result"]})
                else:
                    append = fill(P["repeat_message"], **{"BEGIN_SEARCH_RESULT": tags["begin_search_result"],
                                                          "END_SEARCH_RESULT": tags["end_search_result"]})
                prompt += append
                output += append
            else:
                finished = True
                break
        if finished:
            rec["finish_reason"] = "answer"
        elif rec["finish_reason"] is None:
            rec["finish_reason"] = "max_turns"
        rec["trajectory"].append({"type": "final_output", "text": output})
        # As for Direct/RAG (Qwen3.5 thinking mode), only the text after the last </think>
        # counts; the official last-\boxed{} rule (extract_answer, mode="qa") is applied to it.
        after_think = output.rsplit("</think>", 1)[1] if "</think>" in output else ""
        rec["prediction"] = P["extract_answer"](after_think, mode="qa") or None
        if rec["finish_reason"] == "answer" and rec["prediction"] is None:
            rec["finish_reason"] = "no_answer"
            rec["errors"]["parse"] += 1
            rec["error"] = "no \\boxed{} after </think>"

    def _search_and_reason(self, rec, ex, output, query, max_doc_len, P, tags):
        hits = self.search(rec, query) or []
        formatted_documents = ""
        for i, h in enumerate(hits):
            title, text = split_title(h["contents"], self.ctx.title_rule)
            doc_info = {"id": i + 1, "title": title, "context": text[:max_doc_len * 2]}
            formatted_documents += fill(P["web_page_header"], **{"i + 1": i + 1})
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        rid_user = P["reason_in_documents"](truncated_prev_reasoning(output, P), query, formatted_documents)
        rid = self.generate(rec, self.chat_prompt(rid_user))
        if rid is None:
            return None
        rec["trajectory"][-1]["role"] = "reason_in_documents"
        analysis = P["extract_answer"](rid["text"], mode="infogen")
        return fill(P["result_message"], **{"BEGIN_SEARCH_RESULT": tags["begin_search_result"],
                                            "analysis": analysis,
                                            "END_SEARCH_RESULT": tags["end_search_result"]})
