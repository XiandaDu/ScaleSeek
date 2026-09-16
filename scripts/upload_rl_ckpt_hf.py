#!/usr/bin/env python3
"""Upload a merged RL checkpoint to the user's HF Hub, mirroring the layout of
Allenda/ScaleSeek-Qwen3.5-9B-GRPO-step190 (weights + README.md + DATA_PROVENANCE.md).

    python scripts/upload_rl_ckpt_hf.py --merged $RL_CKPT_DIR/grpo_9b_from190/merged_step_310 \
        --repo Allenda/ScaleSeek-Qwen3.5-9B-GRPO-step500 --overall-step 500 \
        --train $RL_DATA_DIR/train.jsonl --continuation $RL_DATA_DIR/train_from_step190.jsonl

Requires HF_TOKEN (setup_env.sh loads it from .env).
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True, help="HF-format dir from scripts/merge_rl_ckpt.sh")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--overall-step", type=int, required=True)
    ap.add_argument("--train", type=Path, required=True, help="original rl_data/train.jsonl")
    ap.add_argument("--continuation", type=Path, default=None, help="train_from_step190.jsonl if this is a continuation run")
    ap.add_argument("--warm-start", default="Allenda/ScaleSeek-Qwen3.5-9B-GRPO-step190")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    assert (args.merged / "config.json").exists(), args.merged
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    def row(p: Path) -> str:
        n = sum(1 for _ in open(p, encoding="utf-8"))
        return f"| file | `{p.name}` |\n| sha256 | `{sha256_file(p)}` |\n| lines | {n} |\n| bytes | {p.stat().st_size} |"

    prov = ["# Data provenance", "",
            "RL training set used to produce these checkpoints:", "", "| field | value |", "|---|---|", row(args.train), ""]
    if args.continuation:
        prov += [f"Steps 191..{args.overall_step} were trained on the local 4×A100 box as a fresh verl run "
                 f"warm-started from `{args.warm_start}` (only merged weights survived the Nibi run, so the AdamW "
                 "moments were re-initialised). To avoid replaying questions already seen, the continuation used the "
                 "remainder of the same permutation (verl sampler replayed by `scripts/carve_rl_continuation.py`):", "",
                 "| field | value |", "|---|---|", row(args.continuation), ""]
    prov += ["Built by `scripts/prepare_rl_data.py` from the **train** splits of NQ,",
             "HotpotQA, 2WikiMultihopQA and MuSiQue. Evaluation splits are never read into",
             "it, so PopQA / TriviaQA / Bamboogle remain out-of-domain for this policy.", "",
             "Retrieval corpus: wiki-18 (`PeterJinGo/wiki-18-corpus`), 21,015,324 passages."]
    (args.merged / "DATA_PROVENANCE.md").write_text("\n".join(prov) + "\n")
    name = args.repo.split("/")[-1]
    readme = f"""# {name}

GRPO checkpoint at overall step {args.overall_step}, warm-started from the SFT checkpoint
(`Allenda/ScaleSeek-Qwen3.5-9B-SFT`); steps 191+ continued from `{args.warm_start}`.

Base model: `Qwen/Qwen3.5-9B`. See `DATA_PROVENANCE.md` for the
training-set checksums.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("{args.repo}", trust_remote_code=True)
t = AutoTokenizer.from_pretrained("{args.repo}", trust_remote_code=True)
```
"""
    (args.merged / "README.md").write_text(readme)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(args.merged), repo_id=args.repo, repo_type="model",
                      commit_message=f"GRPO overall step {args.overall_step}")
    print("uploaded", args.repo)


if __name__ == "__main__":
    main()
