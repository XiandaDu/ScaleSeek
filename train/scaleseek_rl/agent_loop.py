"""verl agent loop of the ScaleSeek agent: verl's ToolAgentLoop (native Qwen3.5 tool
calling, tool schemas rendered by the chat template) plus

  - a generation cap of agent.MAX_TURN_TOKENS per assistant turn;
  - an `answer` call ends the episode (nothing is executed);
  - the protocol variants of config scaleseek.variants, as in evaluation (methods/scaleseek.py):
      budget   the turn's last tool response ends with agent.budget_note
      commit   the last turn: free reasoning, then the harness writes the opening of the
               `answer` call (agent.COMMIT, no loss) and the model fills in the answer
  - the outcome fields the reward reads, in extra_fields:
      answer      the `answer` call's answer, or None
      draft       last informative current_answer (agent.draft)
      valid       format gate: no malformed call, at most MAX_PARALLEL_CALLS per turn, no turn cut
                  by the per-turn cap, and the trajectory ends in a non-empty `answer`
                  or out of budget (turns or context)
"""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentLoopOutput, AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from scaleseek import agent


@register("scaleseek_agent")
class ScaleSeekAgentLoop(ToolAgentLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variants = set(self.config.scaleseek.variants)
        assert self.variants <= agent.VARIANTS, self.variants
        self.commit_ids = self.tokenizer.encode(agent.COMMIT, add_special_tokens=False)     # starts with </think>

    async def process_vision_info(self, messages: list[dict]) -> dict:
        return {}       # text only (Qwen3.5 ships an image processor; GrepSeek skips it the same way)

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any],
                                       ignore_termination: bool = False) -> AgentState:
        start, cap = len(agent_data.prompt_ids), agent.MAX_TURN_TOKENS
        if "commit" in self.variants and agent_data.assistant_turns == self.max_assistant_turns - 1:
            fill = agent.COMMIT_TOKENS + 16         # + the closing tags, which the harness writes in evaluation
            room = self.prompt_length + self.response_length - start - len(self.commit_ids) - fill
            if room <= 0:
                return AgentState.TERMINATED        # the context is full: the episode ends without an answer
            end = [self.commit_ids[0], self.tokenizer.eos_token_id]
            think = await self.server_manager.generate(
                request_id=agent_data.request_id, prompt_ids=agent_data.prompt_ids,
                sampling_params={**sampling_params, "max_tokens": min(cap, room), "stop_token_ids": end})
            ids = [t for t in think.token_ids if t not in end]
            agent_data.prompt_ids += ids + self.commit_ids
            agent_data.response_mask += [1] * len(ids) + [0] * len(self.commit_ids)
            cap = fill
        state = await super()._handle_generating_state(agent_data, {**sampling_params, "max_tokens": cap}, ignore_termination)
        x = agent_data.extra_fields
        x["last_text"] = self.tokenizer.decode(agent_data.prompt_ids[start:], skip_special_tokens=False)
        x["n_cut"] = x.get("n_cut", 0) + (len(agent_data.response_ids) >= cap)
        # (agent_data.tool_calls is only refreshed when the loop goes on to run tools)
        if state == AgentState.PROCESSING_TOOLS:
            x["n_multi"] = x.get("n_multi", 0) + (len(agent_data.tool_calls) > agent.MAX_PARALLEL_CALLS)
            if agent_data.tool_calls[0].name == agent.ANSWER_TOOL:
                return AgentState.TERMINATED          # the answer call ends the episode; nothing runs
        return state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        if err := agent.call_error(tool_call.name):     # same message as evaluation (methods/scaleseek.py)
            agent_data.extra_fields["n_invalid"] = agent_data.extra_fields.get("n_invalid", 0) + 1
            response, reward, res = ToolResponse(text=f"Error: {err}"), 0.0, {}
        else:
            response, reward, res = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if "budget" in self.variants and tool_call is agent_data.tool_calls[: self.max_parallel_calls][-1]:
            # assistant_turns counts this turn: search turns left = turns - 1 (answering) - this one
            note = agent.budget_note(self.max_assistant_turns - 1 - agent_data.assistant_turns)
            response = ToolResponse(text=(response.text or "") + note)
        return response, reward, res

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        out = await super().run(sampling_params, **kwargs)
        x = out.extra_fields
        calls = x.get("calls", [])
        last = x.get("last_text", "")
        parsed = agent.parse_tool_calls(last)
        answer = agent.final_answer(last)
        # out of budget: the last turn was a find/search call (turn cap, or no room left
        # for its response) or it was cut by the context limit
        by_budget = ((bool(parsed) and parsed[0][0] != agent.ANSWER_TOOL)
                     or len(out.response_ids) >= self.response_length)
        x.update(answer=answer, draft=agent.draft(calls), n_calls=len(calls),
                 n_find=sum(c["tool"] == "find" for c in calls), n_search=sum(c["tool"] == "search" for c in calls),
                 valid=(x.get("n_invalid", 0) == 0 and x.get("n_multi", 0) == 0 and x.get("n_cut", 0) == 0
                        and (answer is not None or by_budget)),
                 response_tokens=len(out.response_ids))
        x.pop("calls", None)       # keep extra_fields small; the rollout dump has the text
        return out
