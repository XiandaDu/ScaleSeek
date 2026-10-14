"""Retrievers over one materialised corpus (BASELINES.md §3).

    bm25          Pyserini / Lucene BM25, k1/b from base.yaml        (bm25.py)
    e5            intfloat/e5-base-v2, mean pooling, FAISS HNSW       (dense.py)
    qwen3_emb_4b  Qwen/Qwen3-Embedding-4B, last-token pooling, HNSW   (dense.py)
    agentir_4b    official tevatron encoding + official searcher      (agentir.py)

Our methods reach bm25/e5/qwen3_emb_4b through one local HTTP service (service.py), so
the JVM and the GPU query encoder live in exactly one process per run.
"""
