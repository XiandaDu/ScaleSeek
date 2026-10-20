"""Why did GrepSeek's searches come back empty?  (compute node; Slurm array, one shard per task)

For every Wiki GrepSeek command whose output was empty (not a timeout), re-run its FIRST
pipeline stage against the corpus:
  as_issued   : the first stage exactly as written (with its own flags)
  ci          : the same pattern, case-insensitive
  escaped     : non-ASCII characters written as JSON \\uXXXX escapes (NFC and NFD), case-insensitive
and classify the empty command:
  filtered_out        first stage matched; the later AND-filters removed everything
  case_miss           only the case-insensitive pattern matches
  escape_miss         only the \\u-escaped pattern matches (accented names)
  no_such_string      nothing matches
Usage: SHARDS=n sbatch --array=0-(n-1) ... empty_grep_experiment.py   then   empty_grep_experiment.py merge
Writes analysis/failures/empty_grep_experiment.json.
"""
import json
import os
import shlex
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.hub import _parallel_copy

ROOT = Path("/scratch/a32du/scaleseek_v2")
A = ROOT / "analysis" / "failures"
SRC = ROOT / "data" / "corpora" / "wiki18" / "corpus.jsonl"   # same text as GrepSeek's wiki_corpus.jsonl
CORPUS = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "wiki_corpus.jsonl"   # local copy: repeated full scans
SHARD_DIR = A / "empty_grep_shards"


def first_stage(cmd: str):
    seg = cmd.split("|")[0]
    try:
        toks = shlex.split(seg)
    except ValueError:
        return None
    if not toks or toks[0] != "rg":
        return None
    flags, args = [], []
    for t in toks[1:]:
        (flags if t.startswith("-") else args).append(t)
    pats = [a for a in args if a != "corpus.jsonl"]
    return (flags, pats[0]) if pats else None


def esc(s: str) -> str:
    return json.dumps(s)[1:-1]


def hit(pattern: str, flags) -> bool:
    cmd = ["rg", "--no-config", "-q", "-m", "1", *flags, "--", pattern, str(CORPUS)]
    return subprocess.run(cmd, capture_output=True, timeout=1800).returncode == 0


def classify(item):
    fs = first_stage(item["cmd"])
    if fs is None:
        return {**item, "class": "unparsed"}
    flags, pat = fs
    fl = [f for f in flags if f in ("-F", "-i", "-w") or f.startswith("-") and set(f[1:]) <= set("Fiw")]
    if hit(pat, fl):
        return {**item, "class": "filtered_out" if item["stages"] >= 3 else "first_stage_matches"}
    ci = sorted(set(fl) | {"-i"})
    if "-i" not in "".join(fl) and hit(pat, ci):
        return {**item, "class": "case_miss"}
    if any(ord(c) > 127 for c in pat):
        for form in ("NFC", "NFD"):
            if hit(esc(unicodedata.normalize(form, pat)), ci):
                return {**item, "class": "escape_miss", "form": form}
    return {**item, "class": "no_such_string"}


def wiki_rows():
    for line in open(A / "grepseek_features.jsonl"):
        r = json.loads(line)
        if r["ds"] != "browsecomp_plus":
            yield r


def run_shard(i: int, n: int):
    items = [{"ds": r["ds"], "id": r["id"], "cmd": c["cmd"], "stages": c["stages"], "correct": r["correct"],
              "non_ascii": any(ord(ch) > 127 for ch in c["cmd"])}
             for r in wiki_rows() for c in r["calls"] if c["empty"] and not c["timed_out"] and c["exit"] in (0, 1)]
    if not CORPUS.exists():
        _parallel_copy(SRC, CORPUS)
    SHARD_DIR.mkdir(exist_ok=True)
    with open(SHARD_DIR / f"rows_{i}.jsonl", "w") as f:
        for item in items[i::n]:
            print(json.dumps(classify(item), ensure_ascii=False), file=f, flush=True)


def merge():
    res = [json.loads(l) for p in sorted(SHARD_DIR.glob("rows_*.jsonl")) for l in open(p)]
    out = {"n_empty_commands": len(res), "classes": Counter(r["class"] for r in res),
           "classes_non_ascii": Counter(r["class"] for r in res if r["non_ascii"]),
           "by_dataset": {ds: Counter(r["class"] for r in res if r["ds"] == ds) for ds in sorted({r["ds"] for r in res})},
           "examples": {cl: [r["cmd"] for r in res if r["class"] == cl][:12] for cl in ("case_miss", "escape_miss", "filtered_out")}}
    # all Wiki commands: non-ASCII patterns and how often they come back empty
    allc = [c for r in wiki_rows() for c in r["calls"]]
    na = [c for c in allc if any(ord(ch) > 127 for ch in c["cmd"])]
    out["non_ascii_commands"] = {"n": len(na), "empty": sum(c["empty"] for c in na),
                                 "ascii_n": len(allc) - len(na), "ascii_empty": sum(c["empty"] for c in allc) - sum(c["empty"] for c in na)}
    (A / "empty_grep_experiment.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=dict))
    with open(A / "empty_grep_experiment_rows.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("n_empty_commands", "classes", "classes_non_ascii", "non_ascii_commands", "by_dataset")},
                     indent=1, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge()
    else:
        run_shard(int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SHARDS"]))
