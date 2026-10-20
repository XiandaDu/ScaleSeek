"""Per-run metrics.json and the per-dataset report (BASELINES.md §7).

Every number carries n and a 95% interval.  Significance: paired exact McNemar on EM
against the strongest run (highest EM) of the same dataset; differences with p >= alpha
are flagged "n.s." and must not be described as differences between methods.
"""

from __future__ import annotations

from pathlib import Path

from .config import load_method
from .data import load_subset
from .judge import judge_file
from .metrics import check_ids, mcnemar_exact, proportion_ci, run_metrics
from .utils import PipelineError, git_state, iter_jsonl, read_json, repo_root, write_json


def _wall(run_dir: Path) -> tuple[float, int]:
    """Total wall time of completed sessions, and how many sessions were killed before
    they could log their end (their time is not counted; the report says so)."""
    rows = list(iter_jsonl(run_dir / "sessions.jsonl"))
    ends = [s for s in rows if s["event"] == "end"]
    starts = [s for s in rows if s["event"] == "start"]
    return sum(s["seconds"] for s in ends), len(starts) - len(ends)


def score_run(base: dict, dataset: str, run_dir: Path, smoke: bool, judge: str) -> dict:
    records = list(iter_jsonl(run_dir / "trajectories.jsonl"))
    _, manifest = load_subset(base, dataset)
    check_ids(records, manifest["ids"], partial_ok=smoke)
    z = base["reporting"]["ci_z"]
    wall, interrupted = _wall(run_dir)
    m = run_metrics(records, z, wall)
    m["interrupted_sessions"] = interrupted
    prov = read_json(run_dir / "provenance.json")
    m.update({"run": run_dir.name, "dataset": dataset, "config_id": prov["config_id"],
              "harness_commit": prov["harness_commit"], "scored_with": git_state(repo_root())["commit"],
              "subset_ids_sha256": manifest["ids_sha256"], "smoke": smoke})
    if base["datasets"][dataset]["profile"] == "bcp":
        judged = {r["id"]: r for r in iter_jsonl(judge_file(run_dir, judge))}
        if set(judged) != {r["id"] for r in records}:
            raise PipelineError(f"{run_dir.name}: judge covers {len(judged)} of {len(records)} records")
        correct = {q: bool(r["judge_result"].get("correct")) for q, r in judged.items()}
        m["judge"] = {
            "judge_config": judge,
            "accuracy": proportion_ci(sum(correct.values()), len(correct), z),
            "judge_failures": sum(1 for r in judged.values()
                                  if r["called"] and (r["judge_result"].get("parse_error") or r.get("api_error"))),
            "not_judged_no_answer": sum(1 for r in judged.values() if not r["called"]),
            "per_question_correct": correct,
        }
    m["agent_failures"] = sum(v for k, v in m["finish_reasons"].items()
                              if k in ("api_error", "tool_error", "exception", "parse_error", "timeout"))
    write_json(run_dir / "metrics.json", m)
    return m


def _fmt(x: dict) -> str:
    return "—" if x["mean"] is None else f"{100 * x['mean']:.1f} ± {100 * x['ci95']:.1f}"


def dataset_report(base: dict, dataset: str, root: Path, smoke: bool, judge: str) -> str:
    runs = sorted(p for p in (root / dataset).iterdir() if (p / "trajectories.jsonl").exists())
    metrics = {p.name: score_run(base, dataset, p, smoke, judge) for p in runs}
    bcp = base["datasets"][dataset]["profile"] == "bcp"
    key = (lambda m: m["judge"]["per_question_correct"]) if bcp else (lambda m: m["per_question_em"])
    score = (lambda m: m["judge"]["accuracy"]["mean"]) if bcp else (lambda m: m["em"]["mean"])
    best = max(metrics, key=lambda r: score(metrics[r]))
    alpha = base["reporting"]["significance_alpha"]
    lines = [f"# {dataset} — {'SMOKE (not a result)' if smoke else 'results'}", "",
             f"n per run = subset size; 95% normal intervals; McNemar vs strongest run `{best}`.", ""]
    if bcp:
        lines += ["| run | n | judge acc | judge fail | agent fail | avg turns | avg tools | wall h | p vs best |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["| run | n | EM | F1 | answered | avg turns | avg retr. | parse/tool/api err | wall h | p vs best |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        if name == best:
            sig = "—"
        else:
            a, b = key(m), key(metrics[best])
            if smoke:                  # smoke runs may be partial; formal ones never are
                shared = set(a) & set(b)
                a, b = {q: a[q] for q in shared}, {q: b[q] for q in shared}
            t = mcnemar_exact(a, b)
            sig = f"{t['p_value']:.3g}" + ("" if t["p_value"] < alpha else " (n.s.)")
        if bcp:
            lines.append(f"| {name} | {m['n']} | {_fmt(m['judge']['accuracy'])} | {m['judge']['judge_failures']} "
                         f"| {m['agent_failures']} | {m['avg_turns']:.2f} | {m['avg_retrievals']:.2f} "
                         f"| {m['wall_time_s'] / 3600:.2f}{'+' if m['interrupted_sessions'] else ''} | {sig} |")
        else:
            e = m["errors"]
            lines.append(f"| {name} | {m['n']} | {_fmt(m['em'])} | {_fmt(m['f1'])} | {100 * m['answered_rate']:.1f}% "
                         f"| {m['avg_turns']:.2f} | {m['avg_retrievals']:.2f} | {e['parse']}/{e['tool']}/{e['api']} "
                         f"| {m['wall_time_s'] / 3600:.2f}{'+' if m['interrupted_sessions'] else ''} | {sig} |")
    lines += ["", "`+` after a wall time: some session was killed before logging its end, so its "
              "time is missing from the total."]
    lines += ["", "## Finish reasons", ""]
    for name, m in metrics.items():
        lines.append(f"- `{name}`: " + ", ".join(f"{k}={v}" for k, v in sorted(m["finish_reasons"].items())))
    lines += ["", "## Harness commit per run", "",
              "Runs started after a fix use the fixed commit; docs/deviations.md lists what each fix "
              "changed and why the earlier runs are unaffected.", ""]
    for name, m in metrics.items():
        lines.append(f"- `{name}`: {m['harness_commit'][:7]}")
    if not bcp:
        lines += ["", "## Retrievers side by side (EM)", "",
                  "| method | bm25 | e5 | qwen3_emb_4b |", "|---|---:|---:|---:|"]
        for meth in ("rag", "search_r1_7b", "search_r1_14b", "search_o1"):
            cells = [_fmt(metrics[f"{meth}__{r}"]["em"]) if f"{meth}__{r}" in metrics else "—"
                     for r in ("bm25", "e5", "qwen3_emb_4b")]
            lines.append(f"| {load_method(meth, base)['display_name']} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    (root / dataset / "REPORT.md").write_text(report)
    return report
