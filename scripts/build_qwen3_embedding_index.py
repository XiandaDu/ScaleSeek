#!/usr/bin/env python3
"""Build the frozen Qwen3-Embedding-4B FAISS index.

This is a narrow entry point around the shared resumable dense-index builder.
All remaining options (corpus, output, device, batch size, index type) are
forwarded unchanged.
"""
from __future__ import annotations

import sys

from build_e5_index import main


if __name__ == "__main__":
    if "--backend" not in sys.argv:
        sys.argv[1:1] = ["--backend", "qwen3_emb_4b"]
    main()
