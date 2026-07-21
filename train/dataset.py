"""VERL-compatible dataset for ScaleSeek RL training.

Mirrors GrepSeek's GrepSeekDataset but uses the canonical Python prompt constant
(`prompts.scaleseek_prompt.PROMPT`) and the three-tool schema from
train/environment.py.

Each item exposes:
    raw_prompt:   [system, user] chat messages for verl's rollout
    data_source:  qid (for logging)
    extra_info:   {qid, golden_answers, is_validation, ...}
    reward_model: {ground_truth: {qid, golden_answers}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from eval.datasets import load_dataset as load_eval_dataset, ALL_DATASETS

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Return the same canonical ScaleSeek prompt used for evaluation."""
    from eval.prompts import load
    return load("scaleseek_prompt")


class ScaleSeekDataset:
    """Dataset for ScaleSeek GRPO training (torch Dataset when torch is available).

    Reads JSONL files already in ScaleSeek canonical format:
        {"id": str, "question": str, "golden_answers": [str]}

    Or loads from HuggingFace if local files are absent (via eval.datasets.load_dataset).

    Config keys (from Hydra YAML data section):
        train_files:        Path(s) to training JSONL
        val_files:          Path(s) to validation JSONL
        dataset_name:       Dataset name for HF fallback (e.g. "hotpotqa")
        answerable_only:    bool (default True) — skip examples with no gold answers
        debug_mode_train / debug_mode_val: bool
        debug_num_samples:  int
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
        is_train: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # -- Load examples ------------------------------------------------
        if isinstance(data_files, list):
            data_path = data_files[0] if data_files else None
        else:
            data_path = data_files or None

        examples: list[dict] = []
        if data_path and Path(data_path).exists():
            examples = _load_jsonl(data_path)
            logger.info("Loaded %d examples from %s", len(examples), data_path)
        else:
            # Fallback: load from eval.datasets (local cache or HuggingFace).
            dataset_name = config.get("dataset_name", "hotpotqa")
            logger.info(
                "data_files not found (%s); loading via eval.datasets.load_dataset(%r)",
                data_path, dataset_name,
            )
            examples = load_eval_dataset(dataset_name)

        # -- Filter ---------------------------------------------------
        answerable_only = bool(config.get("answerable_only", True))
        if answerable_only:
            before = len(examples)
            examples = [e for e in examples if e.get("golden_answers")]
            logger.info("answerable_only: %d → %d", before, len(examples))

        # -- Debug subset ---------------------------------------------------
        debug_key = "debug_mode_train" if is_train else "debug_mode_val"
        if config.get(debug_key, False):
            n = int(config.get("debug_num_samples", 32))
            examples = examples[:n]
            logger.info("debug_mode: capped at %d examples", n)

        self.examples = examples
        self._system_prompt = _build_system_prompt()
        logger.info(
            "ScaleSeekDataset (%s): %d examples",
            "train" if is_train else "val", len(examples),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        qid = str(ex.get("id", idx))
        question = ex.get("question", "")
        golden_answers = list(ex.get("golden_answers", []))

        raw_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        item = {
            "raw_prompt": raw_prompt,
            "data_source": qid,
            "extra_info": {
                "qid": qid,
                "question": question,
                "golden_answers": golden_answers,
                "is_validation": not self.is_train,
                "sample_index": idx,
            },
            "reward_model": {
                "ground_truth": {
                    "qid": qid,
                    "golden_answers": golden_answers,
                }
            },
        }
        if _TORCH_AVAILABLE:
            import torch
            item["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        item["index"] = idx
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Data prep: build RL train/val splits from eval datasets
# ---------------------------------------------------------------------------

def prepare_rl_splits(
    datasets: list[str],
    out_dir: Path,
    *,
    train_limit: int | None = None,
    val_limit: int | None = 500,
    seed: int = 42,
) -> None:
    """Build train.jsonl + dev.jsonl from ScaleSeek canonical datasets.

    Uses eval.datasets.load_dataset for local-first loading.
    Saves to out_dir/train.jsonl and out_dir/dev.jsonl.
    """
    import random
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for name in datasets:
        split_key, default_split = _DATASET_SPLITS.get(name, (name, "train"))
        # train
        train_ex = load_eval_dataset(name, split="train" if name in _HAS_TRAIN else None)
        train_ex = [e for e in train_ex if e.get("golden_answers")]
        train_rows.extend(train_ex)
        # val
        val_ex = load_eval_dataset(name)   # default split (test/dev)
        val_ex = [e for e in val_ex if e.get("golden_answers")]
        if val_limit:
            val_ex = rng.sample(val_ex, min(val_limit, len(val_ex)))
        val_rows.extend(val_ex)
        logger.info("%s: %d train, %d val", name, len(train_ex), len(val_ex))

    rng.shuffle(train_rows)
    if train_limit:
        train_rows = train_rows[:train_limit]

    _write_jsonl(train_rows, out_dir / "train.jsonl")
    _write_jsonl(val_rows, out_dir / "dev.jsonl")
    logger.info("Wrote %d train, %d val to %s", len(train_rows), len(val_rows), out_dir)


_HAS_TRAIN = {"nq", "triviaqa", "hotpotqa", "2wikimultihopqa", "musique"}
_DATASET_SPLITS: dict[str, tuple[str, str]] = {}


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)
