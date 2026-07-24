"""SFT data plumbing for ScaleSeek cold-start trajectories.

Consumes the trajectory JSONL written by ``scripts/generate_sft_data.py`` and
produces two shapes:

1. ``export_verl_parquet`` — a parquet with a ``messages`` column (the full
   [system, user, assistant, tool, …] chat), which verl's multi-turn SFT dataset
   masks and tokenizes itself. This is the path used by ``scripts/run_sft.sh``.

2. ``encode_example`` / ``SFTTokenDataset`` — token-level ``input_ids`` + ``labels``
   with loss on assistant tokens ONLY, computed by prefix-diffing the chat
   template per assistant turn. Used by the lightweight local trainer fallback
   and by unit tests. The masking is exact: labels are -100 everywhere except the
   tokens the assistant actually generated.

Only trajectories with ``status == "ok"`` are used for training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ok_trajectories(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("messages"):
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Token-level encoding with assistant-only loss mask
# ---------------------------------------------------------------------------

def encode_example(
    messages: list[dict],
    tokenizer,
    *,
    max_len: int = 8192,
) -> Optional[dict]:
    """Return {"input_ids", "labels", "attention_mask"} for one trajectory.

    labels = input_ids on assistant-generated tokens, IGNORE_INDEX elsewhere.
    Computed by diffing ``apply_chat_template`` prefixes so it matches exactly what
    the model is asked to generate at inference — no hand-rolled tag heuristics.
    """
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    labels = [IGNORE_INDEX] * len(input_ids)

    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # prompt prefix that ends right where this assistant turn's content begins
        prefix = tokenizer.apply_chat_template(
            messages[:k], tokenize=True, add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            messages[: k + 1], tokenize=True, add_generation_prompt=False,
        )
        start = len(prefix)
        end = len(full)
        # guard against template inconsistency
        if start >= end or end > len(input_ids):
            continue
        for i in range(start, end):
            labels[i] = input_ids[i]

    if all(l == IGNORE_INDEX for l in labels):
        return None
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class SFTTokenDataset:
    """torch Dataset of encoded trajectories (lazy torch import)."""

    def __init__(self, trajectories: list[dict], tokenizer, *, max_len: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: list[dict] = []
        for t in trajectories:
            enc = encode_example(t["messages"], tokenizer, max_len=max_len)
            if enc is not None:
                self.examples.append(enc)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch
        e = self.examples[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
        }


def collate_pad(batch: list[dict], pad_token_id: int):
    """Right-pad a batch of {input_ids,labels,attention_mask} tensors."""
    import torch
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        n = b["input_ids"].shape[0]
        pad = max_len - n
        out["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        out["labels"].append(torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# verl multi-turn SFT parquet export
# ---------------------------------------------------------------------------

def export_verl_parquet(in_jsonl: str | Path, out_parquet: str | Path) -> int:
    """Write a parquet with a ``messages`` column for verl's multi-turn SFT dataset.

    Returns the number of rows written. verl computes the assistant loss mask and
    tokenization from the chat template itself, so we only pass the message list.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    trajs = load_ok_trajectories(in_jsonl)
    messages = [t["messages"] for t in trajs]
    ids = [t.get("id", str(i)) for i, t in enumerate(trajs)]
    # enable_thinking=False: our assistant turns already carry a literal <think>…</think>,
    # so verl's per-turn chat template must NOT inject its own Qwen3 thinking scaffold.
    enable_thinking = [False] * len(messages)
    table = pa.table({"messages": messages, "id": ids, "enable_thinking": enable_thinking})
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_parquet))
    return len(messages)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Export cold-start trajectories to a verl SFT parquet.")
    ap.add_argument("--in", dest="in_jsonl", required=True)
    ap.add_argument("--out", dest="out_parquet", required=True)
    args = ap.parse_args()
    n = export_verl_parquet(args.in_jsonl, args.out_parquet)
    print(f"[sft_dataset] wrote {n} SFT rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
