"""ScaleSeek (ours): the agent of scaleseek/agent.py, run exactly as RL trains it.

verl's ToolAgentLoop (train/scaleseek_rl/agent_loop.py) is reproduced on the text level:
the system + user messages go through the generator's chat template with the tool
schemas; each turn generates at most agent.MAX_TURN_TOKENS; a turn ending in EOS gets
the EOS token appended, then the tool message rendered by the chat template (minus the
template's implicit system prefix, as verl strips it); the first call of a turn runs; at
most profile.max_assistant_turns turns, so the last turn's call is never run (verl:
max_user_turns = turns - 1).  Decoding is the shared profile.

`find` goes to the exact-search service, `search` to the run's dense retriever, `answer`
ends the episode.  The prediction is the `answer` call, else the last informative
current_answer (agent.draft); `answered` in the record says which (only an `answer` call
is scored under the shared rule, metrics.final_answer).
"""

from __future__ import annotations

from .. import agent
from ..retrieval.service import RetrievalError
from .base import Method


class ScaleSeek(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        tok = ctx.tokenizer
        one = tok.apply_chat_template([{"role": "user", "content": ""}], tokenize=False)
        two = tok.apply_chat_template([{"role": "user", "content": ""}] * 2, tokenize=False)
        self.implicit_system = one[: len(one) - (len(two) - len(one))]
        self.eos = tok.eos_token

    def tool_turn(self, texts: list[str]) -> str:
        """The turn's tool messages as verl appends them: Qwen3.5's template refuses a
        conversation without a user message, so verl (utils/chat_template.py) renders them
        after a dummy empty user turn and cuts that turn off again."""
        tok = self.ctx.tokenizer
        dummy = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        full = tok.apply_chat_template(dummy + [{"role": "tool", "content": t} for t in texts],
                                       add_generation_prompt=True, tokenize=False)
        pre = tok.apply_chat_template(dummy, add_generation_prompt=False, tokenize=False)
        return full[len(pre):][len(self.implicit_system):]

    def call(self, rec: dict, name: str, args: dict) -> str:
        try:
            if err := agent.call_error(name):
                raise agent.ToolCallError(err)
            if name == "find":
                res = self.ctx.exact.find(agent.find_query(args), agent.FIND_TOP_K)
                text = agent.render_find(args, res)
                hits = res["hits"]
            else:
                hits = self.ctx.retrieval.search(agent.search_query(args), agent.SEARCH_TOP_K)
                text = agent.render_search(args, hits)
        except (agent.ToolCallError, RetrievalError) as exc:
            if "unreachable" in str(exc):
                raise
            rec["errors"]["tool"] += 1
            rec["trajectory"].append({"type": name, "args": args, "error": str(exc)})
            return f"Error: {exc}"
        rec["n_retrievals"] += 1
        rec["trajectory"].append({"type": name, "args": args, "hits": [h["id"] for h in hits]})
        return text

    def run_example(self, ex: dict, rec: dict) -> None:
        messages = [{"role": "system", "content": agent.SYSTEM_PROMPT},
                    {"role": "user", "content": agent.user_message(ex["question"])}]
        prompt = self.ctx.tokenizer.apply_chat_template(messages, tools=agent.TOOLS,
                                                        add_generation_prompt=True, tokenize=False)
        calls, answer = [], None
        turns = self.profile["max_assistant_turns"]
        for turn in range(turns):
            out = self.generate(rec, prompt, cap=agent.MAX_TURN_TOKENS)
            if out is None:
                break                                        # context_exhausted (marked)
            rec["n_turns"] += 1
            prompt += out["text"]
            if out["finish_reason"] == "length":
                rec["finish_reason"] = "turn_cut" if out["capped"] else "context_exhausted"
                break
            prompt += self.eos
            parsed = agent.parse_tool_calls(out["text"])
            if not parsed:
                rec["finish_reason"] = "no_answer"
                break
            if parsed[0][0] == agent.ANSWER_TOOL:
                answer = agent.final_answer(out["text"])
                rec["finish_reason"] = "answer" if answer is not None else "no_answer"
                break
            batch = parsed[: agent.MAX_PARALLEL_CALLS]          # verl: max_parallel_calls
            rec["errors"]["parse"] += len(parsed) > agent.MAX_PARALLEL_CALLS
            calls += [{"tool": n, **a} for n, a in batch if agent.call_error(n) is None]
            if turn == turns - 1:
                rec["finish_reason"] = "max_turns"
                break
            prompt += self.tool_turn([self.call(rec, n, a) for n, a in batch])
        rec["trajectory"].append({"type": "final_sequence", "text": prompt})
        rec["answered"] = answer is not None
        rec["prediction"] = answer if answer is not None else agent.draft(calls)
