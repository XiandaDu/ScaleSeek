"""Retriever acceptance check (BASELINES.md §5 step 3).

All indexes of a corpus were built from the SAME corpus file (sha256 + doc count in every
index manifest), every hit id exists in that corpus, Lucene's stored ids are the corpus
ids, and the same query run `repeats` times returns identical hits.  Writes
<indexes>/<corpus>/retrieval_check.json and raises if anything fails.
"""

from __future__ import annotations

import json

from ..corpus import DocStore, corpus_dir, load_corpus_manifest
from ..data import load_subset
from ..utils import PipelineError, log, write_json
from .common import index_dir, load_index_manifest
from .dense import release_gpu
from .service import make_searcher


def check_retrievers(base: dict, corpus: str, smoke: bool) -> dict:
    """§5.3: same corpus/doc ids for every index, and repeated queries give identical hits."""

    cman = load_corpus_manifest(base, corpus, smoke)
    chk = base["retrieval_check"]
    # The configured query set when it is on this corpus, else the first dataset that is.
    dataset = (chk["dataset"] if base["datasets"][chk["dataset"]]["corpus"] == corpus
               else next(d for d, c in base["datasets"].items() if c["corpus"] == corpus))
    questions = [r["question"] for r in load_subset(base, dataset)[0][: chk["n_queries"]]]
    store = DocStore(corpus_dir(base, corpus, smoke))
    report = {"corpus": corpus, "corpus_sha256": cman["sha256"], "docs": cman["docs"],
              "dataset": dataset, "retrievers": {}}
    for r in ("bm25", "e5", "qwen3_emb_4b"):
        man = load_index_manifest(base, r, corpus, smoke)      # same corpus sha + doc count
        searcher = make_searcher(base, r, corpus, smoke, "cuda:0")
        runs = [[searcher.search(q, chk["top_k"]) for q in questions] for _ in range(chk["repeats"])]
        stable = all(run == runs[0] for run in runs[1:])
        ids_known = all(d in store for res in runs[0] for d, _ in res)
        report["retrievers"][r] = {"docs": man["docs"], "repeat_identical": stable,
                                   "hit_ids_in_corpus": ids_known,
                                   "sample": [[d for d, _ in res] for res in runs[0][:3]]}
        if r == "bm25":
            n = searcher.searcher.num_docs
            report["retrievers"][r]["lucene_num_docs"] = n
            stored = [searcher.searcher.doc(store.id_of(i)) is not None for i in (0, store.n // 2, store.n - 1)]
            report["retrievers"][r]["lucene_ids_match_corpus"] = all(stored) and n == cman["docs"]
        del searcher
        release_gpu()
    ag = load_index_manifest(base, "agentir_4b", corpus, smoke)
    report["retrievers"]["agentir_4b"] = {"docs": ag["docs"], "corpus_sha256_match": True}
    ok = all(v.get("repeat_identical", True) and v.get("hit_ids_in_corpus", True)
             and v.get("lucene_ids_match_corpus", True) for v in report["retrievers"].values())
    report["ok"] = ok

    write_json(index_dir(base, "bm25", corpus, smoke).parent / "retrieval_check.json", report)
    if not ok:
        raise PipelineError(f"retriever check FAILED: {json.dumps(report['retrievers'])[:800]}")
    log(f"retriever check ok for {corpus}{' (smoke)' if smoke else ''}")
    return report


