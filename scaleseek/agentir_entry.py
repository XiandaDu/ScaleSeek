"""AgentIR entry point -- runs INSIDE the job-local AgentIR environment.

    <agentir python> scaleseek/agentir_entry.py <settings.json> -- <official oss_client args>

Standalone on purpose: it imports nothing from the scaleseek package (which lives in the
other environment).  It runs the OFFICIAL `evaluation/search_agent/oss_client.py` main()
of the pinned checkout -- official QUERY_TEMPLATE, official Responses-API tool loop,
official reasoning-aware AgentIR search tool, official run_*.json output -- after
overriding exactly these points (configs/methods/agentir.yaml, docs/deviations.md):

  1. Every `client.responses.create` call carries the profile temperature / top_p /
     presence_penalty (the official client sends none, i.e. server defaults) and
     max_output_tokens=None (official 20000 per response): no per-turn cap, as for every
     method.  presence_penalty goes in extra_body (not a Responses-API SDK argument; vLLM
     reads it from the request body).
  2. At most `max_turns` model calls per question.  The official loop does not count a
     call whose output ends in a reasoning item (an "upstream vLLM bug" workaround), so it
     could loop forever; past the budget the call fails, which the official loop counts as
     an iteration, and the question ends after --max-iterations (= max_turns).
  3. One search per model call: the official loop runs every tool call of a response;
     calls after the first are dropped (the AgentIR README turns parallel search off for a
     fair comparison with the BrowseComp-Plus baselines; every other method searches once
     per turn).
  4. The client's per-request timeout and retries are `request_timeout_s` / `max_retries`,
     as for every other method (the SDK default 600 s / 2 cut long BCP turns short).
  5. Our per-call log (ok / failure kind / usage / dropped calls) is written next to the
     official run_*.json (`sidecars/<query_id>.json`), with the wall time of the question.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

TOOL_CALL_TYPES = ("function_call", "mcp_call", "custom_tool_call")   # as the official loop


def drop_parallel_calls(response) -> int:
    """One search per model call, as for every other method: tool calls after the first
    one of a response are removed before the official loop sees them.  Returns how many."""
    calls = [o for o in response.output if o.type in TOOL_CALL_TYPES]
    extra = {id(o) for o in calls[1:]}
    response.output = [o for o in response.output if id(o) not in extra]
    return len(extra)


def main() -> None:
    settings = json.loads(Path(sys.argv[1]).read_text())
    assert sys.argv[2] == "--"
    repo = settings["repo"]
    sys.path.insert(0, repo)
    os.chdir(repo)

    import openai
    import evaluation.search_agent.oss_client as oss_client

    calls = threading.local()
    sidecar_dir = Path(settings["sidecar_dir"])
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    real_openai = openai.OpenAI

    def make_client(*args, **kwargs):
        kwargs.update(timeout=settings["request_timeout_s"], max_retries=settings["max_retries"])
        client = real_openai(*args, **kwargs)
        official_create = client.responses.create

        def create(**request):
            log = calls.__dict__.setdefault("log", [])
            if len(log) >= settings["max_turns"]:
                log.append({"ok": False, "kind": "turn_budget"})
                raise RuntimeError("assistant-turn budget exhausted")
            request = dict(request, temperature=settings["temperature"], top_p=settings["top_p"],
                           max_output_tokens=None,
                           extra_body={**request.get("extra_body", {}),
                                       "presence_penalty": settings["presence_penalty"]})
            try:
                response = official_create(**request)
            except openai.BadRequestError as exc:
                text = str(exc)
                kind = "context_overflow" if ("context length" in text or "maximum context" in text) else "bad_request"
                log.append({"ok": False, "kind": kind, "error": text[:500]})
                raise
            except Exception as exc:
                log.append({"ok": False, "kind": "transient", "error": f"{type(exc).__name__}: {exc}"[:500]})
                raise
            dropped = drop_parallel_calls(response)
            usage = getattr(response, "usage", None)
            log.append({"ok": True, "status": getattr(response, "status", None),
                        "dropped_parallel_calls": dropped,
                        "usage": usage.model_dump() if usage is not None else None})
            return response

        client.responses.create = create
        return client

    oss_client.openai.OpenAI = make_client

    official_run = oss_client.run_conversation_with_tools

    def run_conversation_with_tools(client, initial_request, tool_handler, max_iterations=100, **kw):
        calls.log = []
        calls.t0 = time.time()
        return official_run(client, initial_request, tool_handler, max_iterations, **kw)

    oss_client.run_conversation_with_tools = run_conversation_with_tools

    official_persist = oss_client._persist_response

    def persist_response(out_dir, initial_request, messages, all_turns, status, *, query_id=None, args=None):
        official_persist(out_dir, initial_request, messages, all_turns, status, query_id=query_id, args=args)
        side = {"query_id": query_id, "status": status, "calls": getattr(calls, "log", []),
                "seconds": round(time.time() - getattr(calls, "t0", time.time()), 3)}
        (sidecar_dir / f"{query_id}.json").write_text(json.dumps(side, ensure_ascii=False, indent=1))

    oss_client._persist_response = persist_response

    sys.argv = ["evaluation/search_agent/oss_client.py"] + sys.argv[3:]
    oss_client.main()


if __name__ == "__main__":
    main()
