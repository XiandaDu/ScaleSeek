"""AgentIR-4B reranker: BM25 top-N → AgentIR-4B rerank → top-k → LLM answer.

AgentIR-4B (Tevatron/AgentIR-4B) is a dense embedding model (not a generative
LLM). It encodes (instruction + query) and documents to compute similarity.

We use it as a reranker on top of BM25 — this avoids building a full 21M-doc
dense index (which would take hours) and lets us run it alongside vLLM without
GPU conflicts by loading on CPU.

Pipeline per query:
    1. BM25 retrieve top-N candidates (default 50)
    2. AgentIR-4B encode query + all N docs → rerank by dot-product
    3. Keep top-k passages (default 5)
    4. Pass to generative LLM (Qwen3-4B on port 8000) with bm25_rag prompt

Ref: https://huggingface.co/Tevatron/AgentIR-4B
"""
from __future__ import annotations

import time
from typing import Any

import torch

from .bm25_retriever import BM25Retriever
from .agent import AgentRecord, _chat_completion
from . import prompts

_AGENTIR_MODEL_ID = "Tevatron/AgentIR-4B"

# Instruction prefix used during query encoding (from model card)
_QUERY_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating "
    "the user's reasoning\nQuery: "
)

_MODEL = None
_TOKENIZER = None


def _load_agentir(device: str = "cpu"):
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoModel, AutoTokenizer
        print(f"[AgentIR] loading {_AGENTIR_MODEL_ID} on {device} ...")
        _TOKENIZER = AutoTokenizer.from_pretrained(_AGENTIR_MODEL_ID)
        _MODEL = AutoModel.from_pretrained(
            _AGENTIR_MODEL_ID,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        _MODEL.eval()
        print("[AgentIR] loaded.")
    return _MODEL, _TOKENIZER


def _embed(texts: list[str], *, is_query: bool, device: str = "cpu",
           batch_size: int = 16) -> torch.Tensor:
    model, tokenizer = _load_agentir(device)
    all_reps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        if is_query:
            batch = [_QUERY_PREFIX + t for t in batch]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc, return_dict=True).last_hidden_state
            reps = hidden[:, -1]  # last token pooling (Tevatron convention)
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        all_reps.append(reps.cpu())
    return torch.cat(all_reps, dim=0)


def rerank(query: str, hits: list[dict], top_k: int,
           device: str = "cpu") -> list[dict]:
    """Rerank BM25 hits with AgentIR-4B and return top_k."""
    if not hits:
        return hits
    doc_texts = [h.get("text", "")[:1024] for h in hits]
    q_vec = _embed([query], is_query=True, device=device)[0]
    d_vecs = _embed(doc_texts, is_query=False, device=device)
    scores = (d_vecs @ q_vec).tolist()
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[:top_k]]


def run_agentir_rag(
    example: dict,
    *,
    client: Any,            # openai client → Qwen3-4B vLLM (port 8000)
    model: str,
    retriever: BM25Retriever,
    bm25_top_n: int = 50,  # BM25 candidates fed to AgentIR reranker
    top_k: int = 5,         # passages kept after reranking
    agentir_device: str = "cpu",
    precomputed_passages: Optional[list[dict]] = None,  # from cache, skips reranking
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> AgentRecord:
    """BM25 top-N → AgentIR-4B rerank → top-k → LLM answer."""
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    # 1+2. Use precomputed passages if available, otherwise run BM25 + rerank
    t_tool = time.perf_counter()
    if precomputed_passages is not None:
        hits = precomputed_passages
    else:
        hits = retriever.retrieve(question, top_k=bm25_top_n)
        record.n_bm25_calls = 1
        hits = rerank(question, hits, top_k=top_k, device=agentir_device)
    record.tool_time_s = time.perf_counter() - t_tool
    record.n_tool_calls = 1
    record.final_workspace_size = len(hits)

    # 3. LLM answer with reranked passages
    passages_text = "\n\n".join(
        f"[Passage {i + 1}]\n{h['text'][:1500]}" for i, h in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": prompts.load("bm25_rag")},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages_text}"},
    ]

    t_llm = time.perf_counter()
    text, err = _chat_completion(
        client, model=model, messages=messages,
        temperature=temperature, top_p=1.0, max_tokens=max_tokens,
    )
    record.llm_time_s = time.perf_counter() - t_llm
    record.total_time_s = time.perf_counter() - t_start

    if err:
        record.finish_reason = "api_error"
        record.error = err
        return record

    from .agent import parse_assistant
    parsed = parse_assistant(text or "")
    record.prediction = parsed.answer
    record.finish_reason = "answer" if parsed.answer else "parse_error"
    record.n_turns = 1
    return record
