"""How much of the BCP corpus does one GrepSeek call show?  BCP documents are one JSON line each,
and a call's stdout is cut at 6,000 tokens: document lengths (from offsets.npy) against the
number of lines a BCP call actually returned.  Writes analysis/failures/bcp_visibility.json."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/a32du/scaleseek_v2")
A = ROOT / "analysis" / "failures"

off = np.load(ROOT / "data" / "corpora" / "bcp" / "offsets.npy")
size = (ROOT / "data" / "corpora" / "bcp" / "corpus.jsonl").stat().st_size
doc_bytes = np.diff(np.append(off, size))
q = {f"p{p}": int(np.percentile(doc_bytes, p)) for p in (10, 25, 50, 75, 90)}
out = {"n_docs": len(doc_bytes), "doc_bytes": q, "mean_bytes": int(doc_bytes.mean()),
       "share_over_24k_bytes": round(float((doc_bytes > 24000).mean()), 3)}   # ~6,000 tokens at ~4 bytes/token

lines, trunc, n = Counter(), 0, 0
for line in open(A / "grepseek_features.jsonl"):
    r = json.loads(line)
    if r["ds"] != "browsecomp_plus":
        continue
    for c in r["calls"]:
        if c["empty"] or c["timed_out"]:
            continue
        n += 1
        trunc += c["truncated"]
        lines[min(c["lines"], 9)] += 1
out["nonempty_calls"] = n
out["truncated"] = trunc
out["lines_per_call"] = dict(sorted(lines.items()))
(A / "bcp_visibility.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
