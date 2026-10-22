"""Shared machinery for the methods implemented in this repository.

Every method turns one example {"id", "question", "golden_answers"} into one RECORD:

    id, question, golden_answers
    prediction          final answer string, or None when the method produced none
    finish_reason       answer | no_answer | max_turns | context_exhausted |
                        parse_error | api_error | tool_error | exception
    n_turns             assistant generation calls (BASELINES.md §4 "assistant turns")
    n_retrievals        retrieval / tool calls actually executed
    errors              {"parse": n, "tool": n, "api": n}
    error               last error message, if any
    wall_time_s         wall clock for this example
    trajectory          every LLM call and every retrieval, in order, verbatim

Records are appended to <run_dir>/trajectories.jsonl one by one (fsync), so a killed
job resumes where it stopped.  A dead vLLM server or retrieval service aborts the run
(ServerDied) instead of producing a burst of failure records.
"""

from __future__ import annotations

import re
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..llm import LLMRequestError, ServerDied
from ..retrieval.service import RetrievalError
from ..utils import JsonlAppender, iter_jsonl, log


def new_record(ex: dict) -> dict:
    return {"id": ex["id"], "question": ex["question"], "golden_answers": ex["golden_answers"],
            "prediction": None, "finish_reason": None, "n_turns": 0, "n_retrievals": 0,
            "errors": {"parse": 0, "tool": 0, "api": 0}, "error": None,
            "wall_time_s": None, "trajectory": []}


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def answer_after_think(text: str) -> tuple[str | None, str]:
    """Answer of a Qwen3.5 thinking-mode completion.  The chat template opens <think> in
    the prompt, so the completion is "...</think>...".  Only text AFTER the last </think>
    counts; returns (answer or None, reason)."""
    if "</think>" not in text:
        return None, "thinking never closed"
    tail = text.rsplit("</think>", 1)[1]
    found = _ANSWER.findall(tail)
    if not found:
        return None, "no <answer> after </think>"
    return found[-1].strip(), "ok"


class Method:
    """Subclasses implement `run_example`; `run_one` adds timing and error accounting."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.spec = ctx.spec
        self.profile = ctx.spec.profile

    def run_example(self, ex: dict, rec: dict) -> None:
        raise NotImplementedError

    def run_one(self, ex: dict) -> dict:
        rec = new_record(ex)
        t0 = time.time()
        try:
            self.run_example(ex, rec)
        except ServerDied:
            raise
        except LLMRequestError as exc:
            rec["errors"]["api"] += 1
            rec["finish_reason"] = "api_error"
            rec["error"] = str(exc)
        except RetrievalError as exc:
            rec["errors"]["tool"] += 1
            rec["finish_reason"] = "tool_error"
            rec["error"] = str(exc)
        except Exception as exc:  # recorded, counted, never silently dropped
            rec["finish_reason"] = "exception"
            rec["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        rec["wall_time_s"] = round(time.time() - t0, 3)
        return rec

    # -------------------------------------------------------------- shared helpers
    def generate(self, rec: dict, prompt: str, cap: int | None = None, **kw) -> dict | None:
        """One completion.  max_tokens = what is left of the context window, or `cap` if
        that is smaller (only the profile's single_call_max_tokens, for Direct/RAG).
        Returns None (and marks the record) when the window is already full."""
        room = self.ctx.llm.room(prompt)
        if room <= 0:
            rec["finish_reason"] = "context_exhausted"
            rec["trajectory"].append({"type": "llm", "skipped": "no room left in the context window"})
            return None
        max_tokens = min(room, cap) if cap else room
        out = self.ctx.llm.complete(prompt, temperature=self.profile["temperature"],
                                    top_p=self.profile["top_p"],
                                    presence_penalty=self.profile["presence_penalty"],
                                    max_tokens=max_tokens, **kw)
        out["capped"] = max_tokens < room
        rec["trajectory"].append({"type": "llm", "max_tokens": max_tokens, "room": room, **out})
        return out

    def search(self, rec: dict, query: str) -> list[dict] | None:
        """Top-k hits, or None on a (recorded) retrieval error."""
        k = self.spec.method["retrieval"]["top_k"]
        try:
            hits = self.ctx.retrieval.search(query, k)
        except RetrievalError as exc:
            if "unreachable" in str(exc):
                raise ServerDied(str(exc)) from exc
            rec["errors"]["tool"] += 1
            rec["error"] = str(exc)
            rec["trajectory"].append({"type": "search", "query": query, "error": str(exc)})
            return None
        rec["n_retrievals"] += 1
        rec["trajectory"].append({"type": "search", "query": query,
                                  "hits": [{"id": h["id"], "score": h["score"]} for h in hits]})
        return hits

    def chat_prompt(self, user: str) -> str:
        """The generator's OWN chat template, one user message, generation prompt on."""
        return self.ctx.tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                                      tokenize=False, add_generation_prompt=True)


def run_examples(method: Method, examples: list[dict], out_path: Path, concurrency: int) -> dict:
    done = {r["id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
    todo = [ex for ex in examples if ex["id"] not in done]
    log(f"{len(done)} done, {len(todo)} to run (concurrency {concurrency})")
    writer = JsonlAppender(out_path)
    t0 = time.time()
    n = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {pool.submit(method.run_one, ex) for ex in todo}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    exc = fut.exception()
                    if exc is not None:
                        for p in pending:
                            p.cancel()
                        raise exc
                    writer.write(fut.result())
                    n += 1
                    if n % 25 == 0 or not pending:
                        rate = n / max(time.time() - t0, 1e-6)
                        log(f"  {n}/{len(todo)} ({rate * 60:.1f}/min)")
    finally:
        writer.close()
    return {"ran": n, "skipped_done": len(done), "seconds": round(time.time() - t0, 1)}
