"""Per-rollout environment: workspace + BM25 retriever state for ScaleSeek training.

The agent has three tools:
    bm25_retrieve(query, top_k, k1, b, mode)  — fetch passages, write to workspace
    grep_workspace(pattern, case_insensitive)  — regex search within workspace
    read_doc(doc_id)                           — read one passage from workspace

The BM25Retriever is an expensive singleton (loads a 21M-doc Lucene index).
Each rollout gets its own fresh Workspace. The ScaleSeekAgentLoop stores the
workspace in agent_data.extra_fields["workspace"] so it persists across turns.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from eval.agent import Workspace, execute_tool, DEFAULT_MAX_RESPONSE_TOKENS
from eval.bm25_retriever import BM25Retriever

# ---------------------------------------------------------------------------
# Tool schemas (injected into system prompt, NOT via verl's native tool call)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bm25_retrieve",
            "description": (
                "Run a BM25 search against the full Wikipedia corpus and load the top-k "
                "passages into the workspace. Use this to coarsely identify relevant documents "
                "before doing fine-grained search with grep_workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "BM25 search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to retrieve (1–50). Default 5.",
                    },
                    "k1": {
                        "type": "number",
                        "description": (
                            "BM25 k1 term-frequency saturation (0.5–3.0). "
                            "Higher → rewards exact-term repetition more. "
                            "Default 1.2. Use ~2.0 for rare named entities."
                        ),
                    },
                    "b": {
                        "type": "number",
                        "description": (
                            "BM25 b length normalization (0.0–1.0). "
                            "Higher → penalizes long documents more. "
                            "Default 0.75. Use ~0.9 for long natural-language queries."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "merge"],
                        "description": (
                            "'replace' (default): discard existing workspace, load fresh results. "
                            "'merge': add new results to workspace without removing existing docs."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_workspace",
            "description": (
                "Regex search across all passages currently in the workspace. "
                "Use after bm25_retrieve to locate the specific passage containing the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex pattern to match against passage text.",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Whether to ignore case (default true).",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": (
                "Read the full text of a specific document from the workspace by its doc_id. "
                "doc_id values appear in bm25_retrieve and grep_workspace results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from a previous bm25_retrieve or grep_workspace result.",
                    },
                },
                "required": ["doc_id"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Shared BM25 retriever singleton (one per process; lazy-loaded)
# ---------------------------------------------------------------------------

_RETRIEVER: Optional[BM25Retriever] = None


def get_retriever() -> BM25Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = BM25Retriever()
    return _RETRIEVER


# ---------------------------------------------------------------------------
# Per-rollout environment
# ---------------------------------------------------------------------------

class ScaleSeekEnv:
    """One instance per rollout. Owns a fresh Workspace and a ref to the shared retriever.

    `tokenizer` (the model tokenizer, supplied by ScaleSeekAgentLoop) and
    `max_response_tokens` bound each tool response by token count, identically to
    the eval-time prompt agent — keeping train/inference tool outputs aligned.
    """

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        max_response_tokens: Optional[int] = DEFAULT_MAX_RESPONSE_TOKENS,
    ) -> None:
        self.workspace = Workspace()
        self.tokenizer = tokenizer
        self.max_response_tokens = max_response_tokens
        # Union of every passage that ever entered the workspace (mode="replace"
        # discards earlier hits from the workspace but not from this record), so
        # the retrieval-process reward (eval/evidence.py) can ask "did the policy
        # ever pull the supporting document", independent of the final answer.
        self.seen_docs: dict[str, str] = {}
        self.n_calls: dict[str, int] = {"bm25_retrieve": 0, "grep_workspace": 0, "read_doc": 0}
        # Passages grep_workspace actually SHOWED the model (post response-budget), with
        # the pattern that surfaced them. The v3 reward pays for "your grep pulled up a
        # passage containing the answer", so it must score what the model saw, not what
        # the workspace happened to hold.
        self.grep_shown: list[tuple[str, str]] = []   # (pattern, rendered passage)

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch bm25_retrieve / grep_workspace / read_doc to eval/agent.execute_tool."""
        result = execute_tool(
            tool_name, arguments, self.workspace, get_retriever(),
            tokenizer=self.tokenizer, max_response_tokens=self.max_response_tokens,
        )
        if tool_name in self.n_calls and "error" not in result:
            self.n_calls[tool_name] += 1
        if tool_name == "bm25_retrieve":
            for d in self.workspace.docs:
                self.seen_docs.setdefault(d["doc_id"], d.get("text", ""))
        elif tool_name == "grep_workspace" and "error" not in result:
            pattern = str(arguments.get("pattern", ""))
            for rendered in result.get("results", []) or []:
                self.grep_shown.append((pattern, str(rendered)))
        return result

    @property
    def workspace_size(self) -> int:
        return self.workspace.size

    def stats(self) -> dict:
        """Workspace stats for reward computation and logging."""
        return {
            "workspace_size": self.workspace.size,
            "doc_ids": [d["doc_id"] for d in self.workspace.docs],
            "n_seen_docs": len(self.seen_docs),
            "n_grep_shown": len(self.grep_shown),
            **{f"n_{k}": v for k, v in self.n_calls.items()},
        }

    def grep_hits(self, phrases: list[str]) -> dict:
        """Did a substantive grep pattern surface a passage containing one of `phrases`?

        Returns {"grep_hit": 0/1, "grep_hit_broad": 0/1} — the second ignores the
        catch-all-pattern guard, for logging how often the guard bites.
        """
        from eval.evidence import any_phrase_in, pattern_is_substantive
        hit = broad = 0.0
        for pattern, rendered in self.grep_shown:
            if any_phrase_in(rendered, phrases):
                broad = 1.0
                if pattern_is_substantive(pattern):
                    hit = 1.0
                    break
        return {"grep_hit": hit, "grep_hit_broad": broad}
