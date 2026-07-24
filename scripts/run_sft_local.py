#!/usr/bin/env python3
"""Minimal single-GPU SFT trainer for the ScaleSeek cold-start smoke.

This is the LOCAL fallback for ``scripts/run_sft.sh`` (verl). It fine-tunes a
small Qwen3 student on the cold-start trajectories and writes a plain HF
checkpoint — the same artifact ``scripts/run_rl.sh`` / eval consume — but without
verl's heavy dependency stack (which, on this bleeding-edge desktop, pins a
transformers/numpy combination that conflicts with the generation/eval env).

The loss mask is exact: it reuses ``train.sft_dataset.encode_example``, which
diffs the chat-template prefix per assistant turn so loss falls only on assistant
tokens — the same contract verl's MultiTurnSFTDataset enforces on the cluster.

    python scripts/run_sft_local.py \
        --base Qwen/Qwen3-1.7B \
        --trajectories .smoke/sft_trajectories.jsonl \
        --out .smoke/sft_ckpt/huggingface --epochs 3

Smoke only; its numbers must never enter a result table (TASK.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from train.sft_dataset import load_ok_trajectories, SFTTokenDataset, collate_pad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--optim", choices=["adamw_8bit", "adamw"], default="adamw_8bit",
                    help="adamw_8bit (bitsandbytes) fits full-param 1.7B on 24GB; "
                         "adamw is the fp32 torch optimizer (needs more memory)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reduce allocator fragmentation so the optimizer + activations fit on 24GB.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trajs = load_ok_trajectories(args.trajectories)
    dataset = SFTTokenDataset(trajs, tokenizer, max_len=args.max_len)
    if len(dataset) == 0:
        sys.exit("no trainable examples (need trajectories with status=ok)")
    print(f"[run_sft_local] base={args.base} | {len(dataset)} SFT examples | device={device}")

    def _collate(batch):
        return collate_pad(batch, tokenizer.pad_token_id)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    optim = None
    if args.optim == "adamw_8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
            optim = AdamW8bit(model.parameters(), lr=args.lr)
            print("[run_sft_local] optimizer: bitsandbytes AdamW8bit")
        except Exception as e:
            print(f"[run_sft_local] AdamW8bit unavailable ({e}); falling back to torch AdamW")
    if optim is None:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("[run_sft_local] optimizer: torch AdamW (fp32 states)")
    n_steps = 0
    for epoch in range(args.epochs):
        optim.zero_grad()
        running = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                n_steps += 1
        print(f"[run_sft_local] epoch {epoch+1}/{args.epochs}  mean_loss={running/len(loader):.4f}  opt_steps={n_steps}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"[run_sft_local] saved HF checkpoint -> {out_dir}")


if __name__ == "__main__":
    main()
