#!/usr/bin/env python3
"""Run the ScaleSeek prompt agent on the smoke set with a LOCAL transformers model.

Closes the smoke loop: loads a (base or SFT-tuned) Qwen3 checkpoint, wraps it in a
minimal OpenAI-compatible client, and drives ``eval.agent.run_agent`` — the real
multi-turn ScaleSeek loop — against the real BM25 tool environment. No vLLM server
needed, so it runs on a single GPU.

    python scripts/run_scaleseek_smoke_eval.py \
        --model .smoke/sft_ckpt/huggingface \
        --index-dir .smoke/bm25_index \
        --questions .smoke/questions.jsonl --limit 6

Compare --model Qwen/Qwen3-1.7B (base) vs the SFT checkpoint to see the effect.
This is a smoke harness; its numbers must never enter a result table (TASK.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.agent import run_agent
from eval.bm25_retriever import BM25Retriever
from eval.metrics import exact_match, f1 as compute_f1


class TransformersChatClient:
    """Minimal ``client.chat.completions.create`` over a local HF model.

    Returns an object shaped like the OpenAI SDK response so eval.agent's
    ``_chat_completion`` consumes it unchanged. Generates with enable_thinking=False
    so the model emits its own ``<think>`` exactly as in the SFT training format.
    """

    def __init__(self, model_name: str, *, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        td = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=td, trust_remote_code=True).to(self.device).eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, temperature=0.0, top_p=1.0,
                max_tokens=1024, stop=None, extra_body=None, **_):
        torch = self._torch
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=12288).to(self.device)
        do_sample = (temperature or 0.0) > 0.0
        kw = dict(max_new_tokens=max_tokens or 1024, do_sample=do_sample,
                  pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if do_sample:
            kw.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if stop:
            for s in stop:
                i = text.find(s)
                if i != -1:
                    text = text[:i]
        msg = SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _load_questions(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model dir or hub id (base or SFT checkpoint)")
    ap.add_argument("--index-dir", default=os.environ.get("BM25_INDEX_DIR"))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.index_dir:
        sys.exit("Set --index-dir or $BM25_INDEX_DIR")
    os.environ["BM25_INDEX_DIR"] = args.index_dir

    retriever = BM25Retriever(index_dir=args.index_dir)
    client = TransformersChatClient(args.model)
    questions = _load_questions(args.questions, args.limit)
    print(f"[smoke_eval] model={args.model} | {len(questions)} questions | index={args.index_dir}\n")

    n_em = 0
    f1_sum = 0.0
    records = []
    for i, ex in enumerate(questions):
        rec = run_agent(ex, client=client, model=args.model, retriever=retriever,
                        max_turns=args.max_turns, max_tokens=args.max_tokens,
                        temperature=0.0, tokenizer=client.tokenizer)
        golds = ex.get("golden_answers", [])
        pred = rec.prediction or ""
        em = 1.0 if (golds and exact_match(pred, golds)) else 0.0
        f1v = compute_f1(pred, golds) if golds else 0.0
        n_em += int(em)
        f1_sum += f1v
        print(f"  [{i+1}/{len(questions)}] {ex.get('id')}: pred={pred!r} gold={golds} "
              f"EM={em:.0f} F1={f1v:.2f} turns={rec.n_turns} tools={rec.n_tool_calls} "
              f"finish={rec.finish_reason}")
        records.append({**rec.to_dict(), "em": em, "f1": f1v})

    n = len(questions)
    print(f"\n[smoke_eval] EM={n_em}/{n} ({100*n_em/max(n,1):.0f}%)  meanF1={f1_sum/max(n,1):.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[smoke_eval] wrote records -> {args.out}")


if __name__ == "__main__":
    main()
