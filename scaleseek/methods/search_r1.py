"""Search-R1: the official infer.py loop (PeterGriffinJin/Search-R1 @ pinned commit).

Reproduced from infer.py, with the copied pieces byte-diffed (prompts/official/search_r1.py):
  * question.strip(); append '?' if it does not end with one;
  * PROMPT_TEMPLATE through the checkpoint's own chat template (one user message; the
    template inserts whatever it inserts by itself -- no extra system message from us);
  * generate until a STOP_SEQUENCES match or an EOS_TOKEN_IDS token;
  * EOS -> done.  Otherwise query = get_query(<whole decoded sequence>), results =
    _passages2string(top-k) ('' when there is no query), and
    prompt += SEARCH_TEMPLATE.format(output_text, search_results), then continue.
  * the answer is the last <answer>...</answer> of the generated text (infer.py's
    evaluation, qa_em.extract_solution, takes the last match after the prompt's example).

Changed for BASELINES.md §4 (docs/deviations.md): at most `max_assistant_turns`
generation calls (infer.py loops until EOS); no max_new_tokens=1024 cap (the window's
remainder instead); temperature 0.6 / top_p 1.0 instead of 0.7; top-k from the yaml.
vLLM matches stop strings on text, not token-id tails, so when "</search>" and a following
newline share one token the newline is not kept (infer.py would keep it).
"""

from __future__ import annotations

import re

from .base import Method

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class SearchR1(Method):
    def run_example(self, ex: dict, rec: dict) -> None:
        P = self.ctx.prompts
        question = ex["question"].strip()
        if question[-1] != "?":
            question += "?"
        prompt = P["user"].format(question=question)
        tok = self.ctx.tokenizer
        if tok.chat_template:
            prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, tokenize=False)
        generated = ""
        for _ in range(self.profile["max_assistant_turns"]):
            out = self.generate(rec, prompt, stop=list(P["stop_sequences"]),
                                stop_token_ids=list(P["eos_token_ids"]),
                                include_stop_str_in_output=True)
            if out is None:
                break
            rec["n_turns"] += 1
            text = out["text"]
            if out["finish_reason"] == "length":
                generated += text
                prompt += text
                rec["finish_reason"] = "context_exhausted"
                break
            if not isinstance(out["stop_reason"], str):   # an EOS id (int) or the model's eos (None)
                generated += text
                prompt += text
                rec["finish_reason"] = "answer"           # refined below if no <answer>
                break
            tmp_query = P["get_query"](prompt + text)
            if tmp_query:
                hits = self.search(rec, tmp_query)
                search_results = P["passages2string"](
                    [{"document": {"contents": h["contents"]}} for h in hits]) if hits else ""
            else:
                search_results = ""
            search_text = P["search_template"].format(output_text=text, search_results=search_results)
            prompt += search_text
            generated += search_text
        else:
            rec["finish_reason"] = "max_turns"
        rec["trajectory"].append({"type": "final_sequence", "text": prompt})
        found = _ANSWER.findall(generated)
        rec["prediction"] = found[-1].strip() if found else None
        if rec["finish_reason"] == "answer" and rec["prediction"] is None:
            rec["finish_reason"] = "no_answer"
            rec["errors"]["parse"] += 1
            rec["error"] = "stopped without <answer>"
