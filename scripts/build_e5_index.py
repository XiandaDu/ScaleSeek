#!/usr/bin/env python3
"""Build FAISS dense index over the full 21M-passage corpus using e5-base-v2.

Uses a resumable, checkpointed, OOM-adaptive build loop with the E5 encoder
contract:
  - mean pooling over last_hidden_state (attention-mask weighted), L2-normalized
  - passages embedded with the "passage: " prefix (queries use "query: ")
  - dim 768 -> single sq8_flat shard is ~16 GB; no multi-GPU sharding needed.

This is the retriever Search-R1 was trained with (intfloat/e5-base-v2) — the
index feeds the paper-faithful search_r1 rerun (E5 top-3) and the RAG(E5) row.

Usage:
    python scripts/build_e5_index.py \\
        --corpus /data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl \\
        --out /data/rech/mofengra/data/e5_index \\
        --device cuda --batch-size 1024 --max-length 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_E5_MODEL_ID = "intfloat/e5-base-v2"
_E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
_QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
_QWEN_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
_PASSAGE_PREFIX = "passage: "
_BACKEND = "e5"


def load_model(device: str):
    model_id, revision = ((_E5_MODEL_ID, _E5_REVISION) if _BACKEND == "e5"
                          else (_QWEN_MODEL_ID, _QWEN_REVISION))
    print(f"[{_BACKEND}] loading {model_id}@{revision} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision,
        padding_side="left" if _BACKEND == "qwen3_emb_4b" else "right")
    want = torch.float16 if device != "cpu" else torch.float32
    model = None
    for kw in ({"dtype": want}, {"torch_dtype": want}):
        try:
            model = AutoModel.from_pretrained(
                model_id, revision=revision, attn_implementation="sdpa", **kw).to(device)
            break
        except TypeError:
            continue
    if model is None:
        model = AutoModel.from_pretrained(model_id, revision=revision).to(device).to(want)
    model.eval()
    p = next(model.parameters())
    print(f"[E5] model loaded. dtype={p.dtype}")
    return model, tokenizer


def _encode_once(texts: list[str], model, tokenizer, device: str, max_length: int) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = model(**enc, return_dict=True).last_hidden_state
        if _BACKEND == "e5":
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            reps = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            attention_mask = enc["attention_mask"]
            if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
                reps = hidden[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                reps = hidden[range(hidden.shape[0]), sequence_lengths]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    out = reps.cpu().float().numpy()
    del enc, hidden, reps
    return out


def _trim_file_lines(path, n_keep: int) -> None:
    tmp = str(path) + ".trim"
    kept = 0
    with open(path, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= n_keep:
                break
            fout.write(line)
            kept += 1
    os.replace(tmp, path)


_WORKING_CHUNK: int | None = None


def encode_batch(texts: list[str], model, tokenizer, device: str,
                 max_length: int = 256) -> np.ndarray:
    """OOM-safe encode with iterative retry outside the exception block."""
    import gc
    global _WORKING_CHUNK
    chunk = min(_WORKING_CHUNK or len(texts), len(texts))
    outs: list[np.ndarray] = []
    i = 0
    while i < len(texts):
        n = min(chunk, len(texts) - i)
        oom = False
        try:
            outs.append(_encode_once(texts[i:i + n], model, tokenizer, device, max_length))
        except torch.cuda.OutOfMemoryError:
            oom = True
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
        if oom:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if n == 1:
                raise RuntimeError("OOM even at batch size 1 after cache clear")
            chunk = max(1, n // 2)
            _WORKING_CHUNK = chunk
            print(f"  [oom] chunk -> {chunk}", flush=True)
            continue
        i += n
    return outs[0] if len(outs) == 1 else np.vstack(outs)


def main():
    global _BACKEND
    parser = argparse.ArgumentParser(
        description="Build e5-base-v2 FAISS index over full corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--backend", choices=["e5", "qwen3_emb_4b"], default="e5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Encoder length (default: E5=256, Qwen3=8192)")
    parser.add_argument("--index-type", default="sq8_flat",
                        choices=["flat", "sq8_flat"],
                        help="sq8_flat: 1 byte/dim, exact scan over SQ8 codes "
                             "(21M x 768 = ~16 GB). flat: fp32, ~65 GB.")
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--skip-passages", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=2_000_000)
    parser.add_argument("--corpus-manifest", default=None,
                        help="Manifest from build_corpus_manifest.py (required for formal index)")
    args = parser.parse_args()
    _BACKEND = args.backend
    if args.max_length is None:
        args.max_length = 256 if _BACKEND == "e5" else 8192

    import faiss

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"Corpus not found: {corpus_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Counting passages ...")
    n_total = 0
    with open(corpus_path) as f:
        for i, _ in enumerate(f):
            if i < args.skip_passages:
                continue
            n_total += 1
            if args.max_passages and n_total >= args.max_passages:
                break
    print(f"  {n_total:,} passages to encode "
          f"(skip={args.skip_passages:,}, max={args.max_passages or 'all'}).")

    model, tokenizer = load_model(args.device)

    prefix = _PASSAGE_PREFIX if _BACKEND == "e5" else ""
    dummy = encode_batch([prefix + "hello"], model, tokenizer,
                         args.device, args.max_length)
    dim = dummy.shape[1]
    print(f"  Embedding dim: {dim}")
    del dummy

    doc_ids_path = out_dir / "doc_ids.txt"
    texts_path = out_dir / "doc_texts.jsonl"
    idx_ckpt = out_dir / "index.faiss"
    resume_done = 0
    index = None
    if args.resume and idx_ckpt.exists() and doc_ids_path.exists():
        print(f"[resume] loading checkpoint {idx_ckpt} ...")
        index = faiss.read_index(str(idx_ckpt))
        resume_done = index.ntotal
        _trim_file_lines(doc_ids_path, resume_done)
        _trim_file_lines(texts_path, resume_done)
        print(f"[resume] {resume_done:,} vectors already indexed; continuing after them.")

    if index is not None:
        pass
    elif args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:  # sq8_flat
        index = faiss.IndexScalarQuantizer(
            dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT,
        )

    _mode = "a" if resume_done else "w"
    ids_f = open(doc_ids_path, _mode)
    texts_f = open(texts_path, _mode)

    needs_training = (resume_done == 0) and args.index_type == "sq8_flat"
    sample_target = 262_144
    trained = not needs_training
    training_vecs: list[np.ndarray] = []
    pending_vecs: list = []

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    total_encoded = resume_done
    skip_total = args.skip_passages + resume_done

    with open(corpus_path) as f:
        for line_no, line in enumerate(f):
            if line_no < skip_total:
                continue
            if total_encoded + len(batch_texts) >= n_total:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc_id = str(obj.get("id", total_encoded))
            text = obj.get("contents", "")
            batch_texts.append(text)
            batch_ids.append(doc_id)

            if len(batch_texts) < args.batch_size:
                continue

            vecs = encode_batch([prefix + t for t in batch_texts],
                                model, tokenizer, args.device, args.max_length)

            if not trained:
                training_vecs.append(vecs)
                pending_vecs.append((vecs, batch_texts, batch_ids))
                n_collected = sum(v.shape[0] for v in training_vecs)
                if n_collected >= sample_target:
                    train_arr = np.vstack(training_vecs)
                    print(f"  Training sq8 on {len(train_arr):,} vectors ...")
                    index.train(train_arr)
                    trained = True
                    del training_vecs, train_arr
                    for pv, pt, pi in pending_vecs:
                        index.add(pv)
                        for bid in pi:
                            ids_f.write(bid + "\n")
                        for t in pt:
                            texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    del pending_vecs
                    pending_vecs = []
            else:
                index.add(vecs)
                for bid in batch_ids:
                    ids_f.write(bid + "\n")
                for t in batch_texts:
                    texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

            n_add = len(batch_texts)
            total_encoded += n_add
            batch_texts, batch_ids = [], []
            if (total_encoded // 500_000) != ((total_encoded - n_add) // 500_000):
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  [{total_encoded:,}/{n_total:,}] encoded", flush=True)
            ce = args.checkpoint_every
            if trained and ce and (total_encoded // ce) != ((total_encoded - n_add) // ce):
                ids_f.flush(); texts_f.flush()
                tmp = str(out_dir / "index.faiss.tmp")
                faiss.write_index(index, tmp)
                os.replace(tmp, out_dir / "index.faiss")
                print(f"  [checkpoint] index.faiss written at {index.ntotal:,} vectors", flush=True)

    if batch_texts:
        vecs = encode_batch([prefix + t for t in batch_texts],
                            model, tokenizer, args.device, args.max_length)
        if not trained:
            all_vecs = np.vstack([v for v, _, _ in pending_vecs] + [vecs])
            print(f"  Training sq8 on {len(all_vecs):,} vectors (small corpus)...")
            index.train(all_vecs)
            trained = True
            for pv, pt, pi in pending_vecs:
                index.add(pv)
                for bid in pi:
                    ids_f.write(bid + "\n")
                for t in pt:
                    texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        index.add(vecs)
        for bid in batch_ids:
            ids_f.write(bid + "\n")
        for t in batch_texts:
            texts_f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        total_encoded += len(batch_texts)

    ids_f.close()
    texts_f.close()

    idx_path = out_dir / "index.faiss"
    print(f"  Writing FAISS index to {idx_path} ...")
    faiss.write_index(index, str(idx_path))
    print(f"\nDone. {total_encoded:,} passages indexed.")
    print(f"  index.faiss: {idx_path.stat().st_size / 1e9:.1f} GB")
    ids_hasher = hashlib.sha256()
    with doc_ids_path.open("rb") as ids_stream:
        for chunk in iter(lambda: ids_stream.read(8 * 1024 * 1024), b""):
            ids_hasher.update(chunk)
    ids_digest = ids_hasher.hexdigest()
    corpus_manifest = {}
    if args.corpus_manifest:
        corpus_manifest = json.loads(Path(args.corpus_manifest).read_text())
        if args.max_passages is None and args.skip_passages == 0 \
                and index.ntotal != corpus_manifest["count"]:
            sys.exit("Dense index count does not match corpus manifest")
    model_id, revision = ((_E5_MODEL_ID, _E5_REVISION) if _BACKEND == "e5"
                          else (_QWEN_MODEL_ID, _QWEN_REVISION))
    manifest = {
        "schema_version": 1, "backend": _BACKEND, "model_id": model_id,
        "model_revision": revision, "pooling": ("masked_mean" if _BACKEND == "e5"
                                                   else "last_token"),
        "normalize": True, "max_length": args.max_length,
        "count": int(index.ntotal), "ordered_doc_ids_sha256": ids_digest,
        "corpus_manifest_id": corpus_manifest.get("corpus_manifest_id"),
        "index_type": args.index_type,
        "partial_smoke": args.max_passages is not None or args.skip_passages != 0,
    }
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.corpus_manifest:
        print("  warning: no corpus manifest; this index is not eligible for formal evaluation")


if __name__ == "__main__":
    main()
