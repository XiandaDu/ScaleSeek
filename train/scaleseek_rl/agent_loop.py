"""verl agent loop of the ScaleSeek agent: verl's ToolAgentLoop (native Qwen3.5 tool
calling, tool schemas rendered by the chat template) plus

  - a generation cap of agent.MAX_TURN_TOKENS per assistant turn;
  - the outcome fields the reward reads, in extra_fields:
      answer      <answer> of the last turn, or None
      draft       last informative current_answer (agent.draft)
      valid       format gate: no malformed call, at most one call per turn, no turn cut
                  by the per-turn cap, and the trajectory ends in an <answer> or out of
                  budget (turns or context)
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

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any],
                                       ignore_termination: bool = False) -> AgentState:
        state = await super()._handle_generating_state(
            agent_data, {**sampling_params, "max_tokens": agent.MAX_TURN_TOKENS}, ignore_termination)
        x = agent_data.extra_fields
        x["last_text"] = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        x["n_cut"] = x.get("n_cut", 0) + (len(agent_data.response_ids) >= agent.MAX_TURN_TOKENS)
        # (agent_data.tool_calls is only refreshed when the loop goes on to run tools)
        if state == AgentState.PROCESSING_TOOLS:
            x["n_multi"] = x.get("n_multi", 0) + (len(agent_data.tool_calls) > 1)
        return state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        if tool_call.name not in self.tools:       # same message as evaluation (methods/scaleseek.py)
            agent_data.extra_fields["n_invalid"] = agent_data.extra_fields.get("n_invalid", 0) + 1
            return ToolResponse(text=f"Error: unknown tool {tool_call.name!r}"), 0.0, {}
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        out = await super().run(sampling_params, **kwargs)
        x = out.extra_fields
        calls = x.get("calls", [])
        last = x.get("last_text", "")
        answer = agent.extract_answer(last)
        # out of budget: the last turn was a call (turn cap, or no room left for its
        # response) or it was cut by the context limit
        by_budget = answer is None and ("<tool_call>" in last or len(out.response_ids) >= self.response_length)
        x.update(answer=answer, draft=agent.draft(calls), n_calls=len(calls),
                 n_find=sum(c["tool"] == "find" for c in calls), n_search=sum(c["tool"] == "search" for c in calls),
                 valid=(x.get("n_invalid", 0) == 0 and x.get("n_multi", 0) == 0 and x.get("n_cut", 0) == 0
                        and (answer is not None or by_budget)),
                 response_tokens=len(out.response_ids))
        x.pop("calls", None)       # keep extra_fields small; the rollout dump has the text
        return out
