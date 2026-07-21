"""ScaleSeek VERL agent loop.

Subclasses GrepSeek's ToolAgentLoop pattern with two key changes:

1. Tool dispatch is handled INSIDE the agent loop (not via verl's tool registry)
   because workspace state must persist across turns within a rollout. All three
   ScaleSeek tools (bm25_retrieve, grep_workspace, read_doc) share one Workspace
   stored in agent_data.extra_fields["env"].

2. The BM25Retriever is a process-level singleton (loaded once in __init__ via
   train.environment.get_retriever) because loading a 21M-doc Lucene index per
   rollout would be prohibitive.

Format contract (same as eval/agent.py):
    assistant: <think>...</think><tool_call>{...}</tool_call>
               OR
               <think>...</think><answer>...</answer>
    tool response is serialized JSON appended as {"role": "tool", "content": ...}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .environment import ScaleSeekEnv, TOOL_SCHEMAS, DEFAULT_MAX_RESPONSE_TOKENS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of verl: only available when running under the RL trainer.
# The eval pipeline imports this file's sibling modules, not this one.
# ---------------------------------------------------------------------------

try:
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentData,
        AgentLoopOutput,
        AgentState,
        ToolAgentLoop,
    )
    from verl.experimental.agent_loop.agent_loop import register
    from verl.utils.rollout_trace import rollout_trace_op
    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _parse_tool_call(text: str) -> tuple[str | None, dict | None, str | None]:
    """Return (tool_name, args, error). Mirrors eval/agent.parse_assistant logic."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, None, "no <tool_call> block"
    try:
        obj = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        return None, None, f"JSON parse error: {e}"
    name = obj.get("name")
    if not name:
        return None, None, "tool_call missing 'name'"
    args = obj.get("arguments") or {}
    if not isinstance(args, dict):
        return None, None, "'arguments' must be a dict"
    return name, args, None


def _parse_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _has_answer(text: str) -> bool:
    return bool(_ANSWER_RE.search(text))


if _VERL_AVAILABLE:

    @register("scaleseek_agent")
    class ScaleSeekAgentLoop(ToolAgentLoop):
        """Agent loop for ScaleSeek BM25+workspace RL training.

        Key difference from GrepSeekAgentLoop: three tools with stateful workspace.
        We bypass verl's tool registry and dispatch directly from _handle_processing_tools_state.
        """

        def __init__(self, *args, **kwargs):
            # Pass no tools to super — we handle dispatch ourselves.
            super().__init__(*args, **kwargs)
            # Tool spec is embedded in prompts.scaleseek_prompt.PROMPT.
            # We do NOT register verl tools; dispatch is via _handle_processing_tools_state.
            # The retriever is loaded lazily on first rollout via get_retriever().
            from .environment import get_retriever as _warm_retriever
            # Warm up retriever in __init__ so first rollout doesn't pay the load cost.
            _warm_retriever()
            # Per-response token budget for tool outputs (whole passages dropped to
            # fit). Counted with self.tokenizer, matching the eval prompt agent so
            # train/inference tool outputs are byte-aligned. Override via env var.
            self._max_tool_response_tokens = int(
                os.environ.get("SCALESEEK_MAX_TOOL_RESPONSE_TOKENS", DEFAULT_MAX_RESPONSE_TOKENS)
            )

        # ------------------------------------------------------------------
        # Prevent verl from injecting a second tool spec into the chat template.
        # The system prompt (scaleseek.txt) already contains the full tool spec.
        # ------------------------------------------------------------------

        async def _handle_pending_state(
            self, agent_data: AgentData, sampling_params: dict[str, Any]
        ) -> AgentState:
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=[],  # tool spec already in system prompt text
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
            agent_data.prompt_ids = prompt_ids
            return AgentState.GENERATING

        # ------------------------------------------------------------------
        # Tool dispatch: bypass verl's registry, use ScaleSeekEnv directly.
        # ------------------------------------------------------------------

        async def _handle_processing_tools_state(
            self, agent_data: AgentData
        ) -> AgentState:
            """Execute tool calls against per-rollout ScaleSeekEnv."""
            env: ScaleSeekEnv = agent_data.extra_fields.setdefault(
                "env",
                ScaleSeekEnv(
                    tokenizer=self.tokenizer,
                    max_response_tokens=self._max_tool_response_tokens,
                ),
            )

            # Get the decoded text of the last assistant turn.
            response_text = self.tokenizer.decode(
                agent_data.response_ids, skip_special_tokens=True
            )

            # Check for a direct answer (agent decided not to use a tool this turn).
            if _has_answer(response_text):
                return AgentState.TERMINATED

            tool_name, args, err = _parse_tool_call(response_text)

            if err:
                logger.debug("ScaleSeekAgentLoop: tool parse error: %s", err)
                err_payload = json.dumps({"error": f"format error: {err}"}, ensure_ascii=False)
                await self._append_tool_response(agent_data, err_payload)
                agent_data.extra_fields["invalid_tool_request"] = True
                agent_data.extra_fields["invalid_tool_request_reason"] = "malformed_tool_call"
                return AgentState.TERMINATED

            result = env.execute(tool_name, args)

            # Track workspace stats for reward computation.
            agent_data.extra_fields["workspace_size"] = env.workspace_size
            agent_data.extra_fields.setdefault("n_bm25_calls", 0)
            if tool_name == "bm25_retrieve":
                agent_data.extra_fields["n_bm25_calls"] += 1

            payload = json.dumps(result, ensure_ascii=False)
            await self._append_tool_response(agent_data, payload)
            return AgentState.GENERATING

        async def _append_tool_response(self, agent_data: AgentData, content: str) -> None:
            """Tokenize a tool response and append it to prompt_ids / response_mask."""
            tool_msg = [{"role": "tool", "content": content}]
            ids = await self.apply_chat_template(tool_msg, remove_system_prompt=True)
            max_len = self.max_tool_response_length
            if max_len and len(ids) > max_len:
                ids = ids[:max_len]
            agent_data.prompt_ids += ids
            agent_data.response_mask += [0] * len(ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(ids)

        # ------------------------------------------------------------------
        # run(): record trajectory stats for reward function.
        # ------------------------------------------------------------------

        @rollout_trace_op
        async def run(
            self, sampling_params: dict[str, Any], **kwargs: Any
        ) -> AgentLoopOutput:
            output: AgentLoopOutput = await super().run(sampling_params, **kwargs)
            n_tokens = len(output.response_ids)
            output.extra_fields["response_token_count"] = n_tokens
            output.extra_fields["hit_max_context"] = n_tokens >= self.response_length
            output.extra_fields.setdefault("invalid_tool_request", False)
            output.extra_fields.setdefault("invalid_tool_request_reason", "")
            output.extra_fields.setdefault("n_bm25_calls", 0)
            output.extra_fields.setdefault("workspace_size", 0)
            return output

else:
    # Allow importing this module without verl (e.g., unit tests, linters).
    class ScaleSeekAgentLoop:  # type: ignore[no-redef]
        """Stub: verl not installed. Import works but class is not usable."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ScaleSeekAgentLoop requires verl. "
                "Install from grepseek/verl or the verl upstream repo."
            )
