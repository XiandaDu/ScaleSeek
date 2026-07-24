"""Teacher-model client for SFT cold-start generation.

The cold-start pipeline (``train/sft/coldstart.py``) drives a *teacher* model
through the Tutor/Planner prompts in ``prompts/sft_prompts.py``. Two production
backends and one test backend implement the same tiny interface:

    complete(messages: list[{"role","content"}], **gen) -> str

Backends
--------
- ``TransformersTeacher``  local HF model (default for the single-GPU smoke).
      Qwen3 "thinking" is disabled for Tutor calls so structured JSON/XML output
      is easy to parse; enable it per-call with ``enable_thinking=True``.
- ``OpenAITeacher``        OpenAI-compatible endpoint (e.g. a vLLM server), for
      scaling generation on the cluster. Mirrors eval.agent's client usage.
- ``FakeTeacher``          deterministic scripted responses for unit tests
      (no GPU, no network).

``build_teacher(spec)`` constructs one from a string spec:
    "hf:Qwen/Qwen3-4B"                  -> TransformersTeacher
    "openai:http://127.0.0.1:8000/v1"   -> OpenAITeacher (model via `model=`)
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Protocol


class Teacher(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        enable_thinking: bool = False,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Local transformers backend
# ---------------------------------------------------------------------------

class TransformersTeacher:
    """Local HF causal-LM teacher. Loads model + tokenizer once."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.max_model_len = max_model_len

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        enable_thinking: bool = False,
    ) -> str:
        torch = self._torch
        # Qwen3 accepts enable_thinking; other templates ignore the kwarg.
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_model_len,
        ).to(self.device)

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return _apply_stop(text, stop)


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint backend (vLLM etc.)
# ---------------------------------------------------------------------------

class OpenAITeacher:
    """OpenAI-compatible chat client (e.g. a vLLM server on the cluster)."""

    def __init__(self, base_url: str, model: str, api_key: str = "dummy") -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
        self.model = model

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        enable_thinking: bool = False,
    ) -> str:
        extra_body = {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            stop=stop or None, extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "") if resp.choices else ""


# ---------------------------------------------------------------------------
# Fake backend for unit tests
# ---------------------------------------------------------------------------

class FakeTeacher:
    """Deterministic teacher for tests.

    ``responder`` maps (messages) -> str. A convenience form accepts a list of
    (substring, response) rules matched against the last user message.
    """

    def __init__(
        self,
        responder: Callable[[list[dict]], str] | list[tuple[str, str]],
    ) -> None:
        if isinstance(responder, list):
            rules = responder

            def _match(messages: list[dict]) -> str:
                last = messages[-1]["content"] if messages else ""
                for needle, out in rules:
                    if needle in last:
                        return out
                return ""

            self._fn = _match
        else:
            self._fn = responder
        self.calls: list[list[dict]] = []

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        enable_thinking: bool = False,
    ) -> str:
        self.calls.append(messages)
        return _apply_stop(self._fn(messages), stop)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_stop(text: str, stop: Optional[list[str]]) -> str:
    if not stop:
        return text
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def build_teacher(spec: str, *, model: Optional[str] = None, **kwargs: Any) -> Teacher:
    """Construct a teacher from a ``backend:target`` spec.

    "hf:Qwen/Qwen3-4B"                 -> TransformersTeacher(model_name)
    "openai:http://127.0.0.1:8000/v1"  -> OpenAITeacher(base_url, model)
    """
    if ":" not in spec:
        raise ValueError(f"teacher spec must be 'backend:target', got {spec!r}")
    backend, target = spec.split(":", 1)
    backend = backend.lower()
    if backend in ("hf", "transformers"):
        return TransformersTeacher(target, **kwargs)
    if backend == "openai":
        m = model or os.environ.get("LLM_MODEL", "agent")
        return OpenAITeacher(target, m)
    raise ValueError(f"unknown teacher backend {backend!r} (use 'hf' or 'openai')")
