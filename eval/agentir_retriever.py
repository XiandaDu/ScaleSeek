"""AgentIR-4B dense retriever: FAISS full-corpus index → LLM answer.

AgentIR-4B (Tevatron/AgentIR-4B) is a dense embedding model. Retrieval is ANN
search over a FAISS index built offline from all 21M corpus passages.

Build the FAISS index once (takes ~12–24h on GPU):
    python scripts/build_agentir_index.py \\
        --corpus /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \\
        --out $AGENTIR_INDEX_DIR

Then set $AGENTIR_INDEX_DIR and run eval:
    python -m eval.run_eval --agent agentir_rag --agentir-index-dir $AGENTIR_INDEX_DIR

This replaces the previous BM25→rerank pipeline. AgentIR is used for
primary retrieval (no BM25), which is correct per the paper.

Ref: https://huggingface.co/Tevatron/AgentIR-4B
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .agent import AgentRecord, _chat_completion, parse_assistant
from . import prompts

_AGENTIR_MODEL_ID = "Tevatron/AgentIR-4B"
_QUERY_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating "
    "the user's reasoning\nQuery: "
)

# Module-level singletons (loaded once per process)
_EMBED_MODEL = None
_EMBED_TOKENIZER = None
_FAISS_INDEX = None
_DOC_IDS: Optional[list[str]] = None
_DOC_TEXTS: Optional[list[str]] = None
_LOADED_INDEX_DIR: Optional[Path] = None


def _load_embed_model(device: str = "cpu"):
    global _EMBED_MODEL, _EMBED_TOKENIZER
    if _EMBED_MODEL is None:
        from transformers import AutoModel, AutoTokenizer
        print(f"[AgentIR] loading {_AGENTIR_MODEL_ID} on {device} ...")
        _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(_AGENTIR_MODEL_ID)
        _EMBED_MODEL = AutoModel.from_pretrained(
            _AGENTIR_MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        _EMBED_MODEL.eval()
        print("[AgentIR] embed model loaded.")
    return _EMBED_MODEL, _EMBED_TOKENIZER


def _load_index(index_dir: Path) -> None:
    global _FAISS_INDEX, _DOC_IDS, _DOC_TEXTS, _LOADED_INDEX_DIR
    if _LOADED_INDEX_DIR == index_dir:
        return
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Install faiss:  conda install -c pytorch faiss-gpu  (or faiss-cpu)"
        )
    idx_path = index_dir / "index.faiss"
    ids_path = index_dir / "doc_ids.txt"
    texts_path = index_dir / "doc_texts.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {idx_path}.\n"
            "Build it: python scripts/build_agentir_index.py "
            "--corpus /path/to/wiki_corpus.jsonl --out <dir>"
        )
    print(f"[AgentIR] loading FAISS index from {idx_path} ...")
    _FAISS_INDEX = faiss.read_index(str(idx_path))
    _DOC_IDS = ids_path.read_text().splitlines() if ids_path.exists() else []
    _DOC_TEXTS = None
    if texts_path.exists():
        _DOC_TEXTS = []
        with open(texts_path) as f:
            for line in f:
                try:
                    _DOC_TEXTS.append(json.loads(line).get("text", ""))
                except Exception:
                    _DOC_TEXTS.append("")
    _LOADED_INDEX_DIR = index_dir
    print(f"[AgentIR] index ready: {_FAISS_INDEX.ntotal:,} vectors, "
          f"{len(_DOC_IDS):,} doc_ids, texts={'yes' if _DOC_TEXTS else 'no'}")


def _embed_query(query: str, device: str) -> "torch.Tensor":
    model, tokenizer = _load_embed_model(device)
    text = _QUERY_PREFIX + query
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = model(**enc, return_dict=True).last_hidden_state
        rep = hidden[:, -1]
        rep = torch.nn.functional.normalize(rep, p=2, dim=-1)
    return rep.cpu().float()


def retrieve(
    query: str,
    top_k: int,
    *,
    index_dir: Path,
    device: str = "cpu",
) -> list[dict]:
    """Dense retrieval: encode query → FAISS ANN search → top_k passages."""
    import numpy as np
    _load_index(index_dir)
    q_vec = _embed_query(query, device=device).numpy()
    scores, indices = _FAISS_INDEX.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        doc_id = _DOC_IDS[idx] if idx < len(_DOC_IDS) else str(idx)
        text = ""
        if _DOC_TEXTS is not None and idx < len(_DOC_TEXTS):
            text = _DOC_TEXTS[idx]
        results.append({"doc_id": doc_id, "score": float(score), "text": text})
    return results


def run_agentir_rag(
    example: dict,
    *,
    client: Any,
    model: str,
    index_dir: Path,
    top_k: int = 5,
    agentir_device: str = "cpu",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> AgentRecord:
    """AgentIR-4B dense retrieval → LLM reader."""
    ex_id = str(example.get("id", ""))
    question = example.get("question", "")
    golds = list(example.get("golden_answers", []))
    record = AgentRecord(id=ex_id, question=question, gold_answers=golds)
    t_start = time.perf_counter()

    t_tool = time.perf_counter()
    hits = retrieve(question, top_k, index_dir=index_dir, device=agentir_device)
    record.tool_time_s = time.perf_counter() - t_tool
    record.n_tool_calls = 1
    record.final_workspace_size = len(hits)
    record.workspace_doc_ids = [h["doc_id"] for h in hits]
    record.bm25_calls = [{
        "query": question, "k1": None, "b": None, "top_k": top_k,
        "mode": "replace", "doc_ids": record.workspace_doc_ids,
    }]

    passages_text = "\n\n".join(
        f"[Passage {i+1}]\n{h['text'][:1500]}" for i, h in enumerate(hits)
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

    parsed = parse_assistant(text or "")
    record.prediction = parsed.answer
    record.finish_reason = "answer" if parsed.answer else "parse_error"
    record.n_turns = 1
    return record
