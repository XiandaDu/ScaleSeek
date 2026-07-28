#!/usr/bin/env python3
"""Backfill concurrency / seed provenance into pre-2026-07-28 phase-2 metrics.

`eval/run_eval.py` only started writing `concurrency` and `decode_seed` into each
result row on 2026-07-28. Runs completed before that carry timing columns whose
meaning depends on a batch size recorded nowhere in the artifact — but the launcher
*printed* it, so `logs/p2_<job>.out` still has the ground truth:

    Running agent='search_o1' on 14267 examples (concurrency=16) ...

This recovers that line, writes the value into the `.metrics.json` alongside an
explicit comparability flag, and marks the run as unseeded. It does NOT touch the
per-row JSONL (those rows are the raw run record) and it does NOT recompute any
score — EM/F1 are unaffected by this metadata.

    python scripts/backfill_run_provenance.py --results-dir results/phase2 \
        --logs-dir logs [--apply]

Without --apply it prints what it would change.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_CONC_RE = re.compile(
    r"Running agent='(?P<agent>[a-z0-9_]+)' on (?P<n>\d+) examples "
    r"\(concurrency=(?P<conc>\d+)"
)


_OUT_RE = re.compile(r"results/[a-z0-9_]+/[a-z0-9_]+\.jsonl")
# "Wall time          : 3113.7 min (13.1s/example)" — the launcher already prints
# the honest throughput. sec_per_query divided by this is exactly the concurrency
# factor (search_o1_bm25: 208.95 / 13.1 = 15.9 at concurrency 16), which is why
# sec_per_query must never be quoted as latency.
_WALL_RE = re.compile(r"Wall time\s*:\s*([\d.]+) min \(([\d.]+)s/example\)")


def _scan_logs(logs_dir: Path) -> dict[str, dict]:
    """result-jsonl path -> {concurrency, wall_time_min, wall_s_per_example}.

    Keying on the output path rather than (agent, n): the 7B and 14B Search-R1
    cells both log `agent='search_r1'` with n=14267 but ran at concurrency 8 and 6,
    so an (agent, n) key is ambiguous exactly where it matters. Falls back to the
    job-name stem for runs that died before printing "Results saved ->".
    """
    found: dict[str, dict] = {}
    for log in sorted(logs_dir.glob("p2_*.out")):
        text = log.read_text(errors="replace")
        concs = {int(m.group("conc")) for m in _CONC_RE.finditer(text)}
        if not concs:
            continue
        wall = _WALL_RE.findall(text)
        info = {"concurrency": concs, "log": log.name}
        if len(wall) == 1:
            info["wall_time_min"] = float(wall[0][0])
            info["wall_s_per_example"] = float(wall[0][1])
        keys = set(_OUT_RE.findall(text))
        # Job-name stem, e.g. "p2_search_o1_e5-7261" -> "search_o1_e5". Lets a run
        # that has not yet printed its output path still be identified.
        keys.add("job:" + re.sub(r"-\d+$", "", log.stem).removeprefix("p2_"))
        for k in keys:
            slot = found.setdefault(k, {"concurrency": set()})
            slot["concurrency"] |= concs
            for f in ("wall_time_min", "wall_s_per_example", "log"):
                if f in info:
                    slot.setdefault(f, info[f])
    return found


def _match(metrics: dict, log_hits: dict[str, dict]) -> dict | None:
    """Run facts for this metrics file: exact output path first, job stem second."""
    path_key = str(metrics.get("file", "")).lstrip("./")
    stem = Path(path_key).stem
    for key in (path_key, "job:" + re.sub(r"^[a-z0-9]+_", "", stem)):
        hit = log_hits.get(key)
        if hit and len(hit["concurrency"]) == 1:
            return {**hit, "concurrency": next(iter(hit["concurrency"]))}
    return None  # 0 matches, or a log that ran the file twice -> refuse to guess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase2"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default: dry run)")
    args = ap.parse_args()

    log_hits = _scan_logs(args.logs_dir)
    print(f"parsed concurrency from {len(log_hits)} launcher logs\n")

    for path in sorted(args.results_dir.glob("*.metrics.json")):
        metrics = json.loads(path.read_text())
        lat = metrics.setdefault("latency", {})
        if lat.get("concurrency") is not None:
            print(f"  {path.name}: already has concurrency={lat['concurrency']}, skip")
            continue
        hit = _match(metrics, log_hits)
        if hit is None:
            print(f"  {path.name}: NO unambiguous log match "
                  f"(agent={metrics.get('agent')}, n={metrics.get('n')}) -> left alone")
            continue
        conc = hit["concurrency"]
        lat["concurrency"] = conc
        lat["decode_seed"] = None
        lat["latency_comparable"] = (conc == 1)
        # The launcher's own wall-clock figure: the only defensible efficiency
        # number for a concurrent run, and the one to quote instead of
        # sec_per_query (which is ~concurrency x larger).
        if "wall_s_per_example" in hit:
            lat["wall_time_min"] = hit["wall_time_min"]
            lat["wall_s_per_example"] = hit["wall_s_per_example"]
            spq = lat.get("sec_per_query")
            if spq:
                lat["concurrency_inflation"] = round(spq / hit["wall_s_per_example"], 1)
        lat["latency_note"] = (
            f"backfilled from {hit.get('log', 'launcher log')}: concurrency={conc}. "
            "sec_per_query/llm_time_s/tool_time_s are wall-clock INCLUDING queueing "
            "at that batch load — they are NOT per-query latency and NOT comparable "
            "across agents run at different concurrency. For throughput use "
            "wall_s_per_example; for a concurrency-invariant cost comparison use "
            "n_tool_calls_mean / n_bm25_calls_mean.")
        lat["provenance_backfilled"] = "2026-07-28 scripts/backfill_run_provenance.py"
        extra = (f", wall={hit['wall_s_per_example']}s/ex "
                 f"(sec_per_query inflated {lat.get('concurrency_inflation')}x)"
                 if "wall_s_per_example" in hit else "")
        print(f"  {path.name}: concurrency={conc}, seed=None{extra}")
        if args.apply:
            path.write_text(json.dumps(metrics, indent=1) + "\n")

    print("\n(dry run; pass --apply to write)" if not args.apply else "\nwritten.")


if __name__ == "__main__":
    main()
