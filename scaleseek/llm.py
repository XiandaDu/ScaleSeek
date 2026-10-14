"""vLLM server lifecycle and a small OpenAI-compatible client.

Decoding policy (BASELINES.md §4): the client sends temperature and top_p and NOTHING else
(no top_k, penalties or thinking budget), and the server runs with
`--generation-config vllm` so a checkpoint's generation_config.json cannot inject its own
sampling defaults (Qwen checkpoints ship top_k=20 etc.).  There is no per-turn cap:
`max_tokens` is always "whatever is left of max_model_len", computed with the server's
own tokenizer (/tokenize) so the count is exact.

Errors: transient failures (connection, timeout, HTTP 5xx) are retried with back-off.  An
HTTP 4xx is not retried (it is deterministic).  If retries run out and the server is no
longer healthy, ServerDied aborts the whole run so the unfinished questions are re-run on
resume instead of being recorded as failures.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from .utils import PipelineError, log


class ServerDied(PipelineError):
    pass


class LLMRequestError(RuntimeError):
    """A request failed for a reason that retrying will not fix (recorded as api_error)."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMServer:
    def __init__(self, model_path: Path, server_cfg: dict, max_model_len: int, log_path: Path,
                 gpus: list[str]):
        self.model_path = model_path
        self.cfg = server_cfg
        self.max_model_len = max_model_len
        self.log_path = log_path
        self.gpus = gpus
        self.port = free_port()
        self.served_name = server_cfg["served_model_name"]
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        c = self.cfg
        cmd = ["vllm", "serve", str(self.model_path), "--host", "127.0.0.1",
               "--port", str(self.port), "--served-model-name", self.served_name,
               "--tensor-parallel-size", str(c["tensor_parallel_size"]),
               "--max-model-len", str(self.max_model_len),
               "--max-num-seqs", str(c["max_num_seqs"]),
               "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
               "--seed", str(c["seed"]), "--generation-config", "vllm"]
        if c["reasoning_parser"]:
            cmd += ["--reasoning-parser", c["reasoning_parser"]]
        if c["enable_auto_tool_choice"]:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", c["tool_call_parser"]]
        if c["language_model_only"]:
            cmd += ["--language-model-only"]
        if c["trust_remote_code"]:
            cmd += ["--trust-remote-code"]
        return cmd

    def start(self, timeout_s: float) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(self.gpus[:self.cfg["tensor_parallel_size"]]))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"vllm: {' '.join(self.command())}  (log {self.log_path})")
        self._logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.command(), stdout=self._logf, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise PipelineError(f"vLLM exited with {self.proc.returncode}; see {self.log_path}")
            if self.healthy():
                log(f"vllm ready after {time.time() - t0:.0f}s on :{self.port}")
                return
            time.sleep(5)
        self.stop()
        raise PipelineError(f"vLLM not ready after {timeout_s}s; see {self.log_path}")

    def healthy(self) -> bool:
        if self.proc is not None and self.proc.poll() is not None:
            return False
        try:
            return requests.get(self.base_url + "/health", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if self.proc:
            self._logf.close()


class LLMClient:
    def __init__(self, server: VLLMServer, client_cfg: dict):
        self.server = server
        self.model = server.served_name
        self.cfg = client_cfg
        self.session = requests.Session()

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict) -> dict:
        last = None
        for attempt in range(self.cfg["max_retries"] + 1):
            try:
                r = self.session.post(self.server.base_url + path, json=payload,
                                      timeout=self.cfg["request_timeout_s"])
                if r.status_code == 200:
                    return r.json()
                if 400 <= r.status_code < 500:
                    raise LLMRequestError(f"HTTP {r.status_code}: {r.text[:500]}")
                last = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            if not self.server.healthy():
                raise ServerDied(f"vLLM server is down ({last}); see {self.server.log_path}")
            if attempt < self.cfg["max_retries"]:
                time.sleep(self.cfg["retry_backoff_s"] * (2 ** attempt))
        raise LLMRequestError(f"gave up after {self.cfg['max_retries']} retries: {last}")

    # ------------------------------------------------------------------ endpoints
    def count_tokens(self, prompt: str) -> int:
        return int(self._post("/tokenize", {"model": self.model, "prompt": prompt})["count"])

    def room(self, prompt: str) -> int:
        """Tokens left in the context window after `prompt` (the no-cap max_tokens)."""
        return self.server.max_model_len - self.count_tokens(prompt)

    def complete(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int,
                 stop: list[str] | None = None, stop_token_ids: list[int] | None = None,
                 include_stop_str_in_output: bool = False) -> dict:
        payload = {"model": self.model, "prompt": prompt, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens, "n": 1,
                   "skip_special_tokens": True,
                   "include_stop_str_in_output": include_stop_str_in_output}
        if stop:
            payload["stop"] = stop
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids
        out = self._post("/v1/completions", payload)
        ch = out["choices"][0]
        return {"text": ch["text"], "finish_reason": ch["finish_reason"],
                "stop_reason": ch.get("stop_reason"),
                "prompt_tokens": out["usage"]["prompt_tokens"],
                "completion_tokens": out["usage"]["completion_tokens"]}

    def chat(self, messages: list[dict], *, temperature: float, top_p: float, max_tokens: int,
             chat_template_kwargs: dict | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens, "n": 1}
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        out = self._post("/v1/chat/completions", payload)
        ch = out["choices"][0]
        msg = ch["message"]
        return {"content": msg.get("content") or "",
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                "finish_reason": ch["finish_reason"],
                "prompt_tokens": out["usage"]["prompt_tokens"],
                "completion_tokens": out["usage"]["completion_tokens"]}
