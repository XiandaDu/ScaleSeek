"""SFT rows = agent trajectories exactly as they were generated (train/build_sft.py).

A row is a list of text segments -- the prompt, each assistant turn (with its EOS), each tool
turn -- with a `train` flag.  Segments are tokenized one by one, as verl's agent loop builds
a rollout, and the loss covers the assistant turns only.  (verl's MultiTurnSFTDataset
re-renders every message through the chat template instead, which does not reproduce what
the agent saw: thinking blocks, tool-call whitespace.)
"""

from __future__ import annotations

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class TrajectorySFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        files = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [r for f in files for r in pq.read_table(f).column("segments").to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ids, mask = [], []
        for seg in self.rows[i]:
            t = self.tokenizer.encode(seg["text"], add_special_tokens=False)
            ids += t
            mask += [int(seg["train"])] * len(t)
        if len(ids) > self.max_length:
            raise ValueError(f"row {i}: {len(ids)} tokens > max_length {self.max_length}")
        return {"input_ids": torch.tensor(ids), "position_ids": torch.arange(len(ids)),
                "loss_mask": torch.tensor(mask)}
