"""verl tools `find` and `search`: thin HTTP clients of the retrieval services the job
starts (scaleseek/retrieval/service.py), rendering exactly as scaleseek/agent.py does at
evaluation time.  Each call's arguments are appended to agent_data.extra_fields["calls"]
(drafts and the format gate read them); malformed calls count in ["n_invalid"]."""

from __future__ import annotations

import asyncio
from typing import Any

import requests
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from scaleseek import agent

_SCHEMAS = {t["function"]["name"]: t for t in agent.TOOLS}


class _ServiceTool(BaseTool):
    name_: str

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, OpenAIFunctionToolSchema.model_validate(_SCHEMAS[self.name_]))
        self.url = config["url"].rstrip("/")
        self.session = requests.Session()

    def _post(self, path: str, query: str, top_k: int) -> dict:
        r = self.session.post(self.url + path, json={"query": query, "top_k": top_k}, timeout=120)
        if r.status_code != 200:
            raise agent.ToolCallError(r.json().get("error", r.text[:300]))
        return r.json()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        extra = kwargs["agent_data"].extra_fields
        extra.setdefault("calls", []).append({"tool": self.name, **parameters})
        try:
            text = await asyncio.get_running_loop().run_in_executor(None, self._run, parameters)
        except agent.ToolCallError as e:
            extra["n_invalid"] = extra.get("n_invalid", 0) + 1
            text = f"Error: {e}"
        return ToolResponse(text=text), 0.0, {}


class FindTool(_ServiceTool):
    name_ = "find"

    def _run(self, args: dict) -> str:
        return agent.render_find(args, self._post("/find", agent.find_query(args), agent.FIND_TOP_K))


class SearchTool(_ServiceTool):
    name_ = "search"

    def _run(self, args: dict) -> str:
        return agent.render_search(args, self._post("/search", agent.search_query(args), agent.SEARCH_TOP_K)["hits"])


class AnswerTool(BaseTool):
    """Registered only so the chat template shows its schema: the agent loop ends the
    episode on an `answer` call before any tool runs."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, OpenAIFunctionToolSchema.model_validate(_SCHEMAS[agent.ANSWER_TOOL]))
