#!/usr/bin/env python3
"""Structure Wiki-18 articles with the official RISE sectioning core.

This is a data-plumbing adapter only: SECTION_PROMPT, call_llm,
validate_and_locate and build_final are imported unchanged from the pinned
RISE checkout. The paper structured BCP with gpt-5.4-nano (low reasoning);
this project substitutes the frozen Qwen3.5-9B served by vLLM, selected via
SECTION_MODEL and the generic OPENAI_* environment variables that the official
client factory already supports. The audit log keeps every section proposal,
anchor-validation outcome and failure, as TASK.md requires.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rise-repo", type=Path,
                        default=Path(os.environ.get("OFFICIAL_ROOT",
                                                    "/data/rech/mofengra/official-baselines")) / "rise")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None,
                        help="Probe mode: only process this many pending articles")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="File of article filenames (one per line) that get an "
                             "LLM-built TOC. Every OTHER article is still emitted, "
                             "with the official empty-TOC form, so the BM25 corpus "
                             "stays the full 3.2M and the retrieval boundary is "
                             "unchanged. Without this flag the whole corpus is "
                             "sectioned: ~3.2M LLM calls, 24-72 days.")
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.rise_repo / "src"))
    sys.path.insert(0, str(args.rise_repo / "scripts" / "structured"))
    import section_one_doc as S  # noqa: E402  (official core, unchanged)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.audit or (args.out_dir / "audit.jsonl")
    model = os.environ.get("SECTION_MODEL", "")
    if not model:
        sys.exit("SECTION_MODEL must name the frozen generator (Qwen/Qwen3.5-9B)")
    header = {
        "adapter": "scripts/build_wiki_toc.py (file plumbing only)",
        "official_core": "rise scripts/structured/section_one_doc.py",
        "model": model,
        "model_note": "project substitution for paper's gpt-5.4-nano low reasoning",
        "endpoint": os.environ.get("OPENAI_BASE_URL", ""),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    candidates: set[str] | None = None
    if args.candidates:
        candidates = {ln.strip() for ln in
                      args.candidates.read_text(encoding="utf-8").splitlines()
                      if ln.strip()}
        header["candidate_set"] = str(args.candidates)
        header["n_candidates"] = len(candidates)

    pending = []       # LLM-sectioned
    passthrough = []   # emitted with an empty TOC, no model call
    done = 0
    with os.scandir(args.articles_dir) as it:
        for entry in it:
            if not entry.name.endswith(".md"):
                continue
            if (args.out_dir / entry.name).exists():
                done += 1
                continue
            if candidates is not None and entry.name not in candidates:
                passthrough.append(entry.name)
            else:
                pending.append(entry.name)
    pending.sort()
    passthrough.sort()
    if args.limit is not None:
        pending = pending[: args.limit]
        passthrough = []          # probe mode measures the LLM path only
    print(f"[toc] {done:,} already structured; {len(pending):,} to section "
          f"via LLM; {len(passthrough):,} passthrough (empty TOC) "
          f"(workers={args.workers}, model={model})", flush=True)

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "sections": 0, "located": 0, "skipped": 0,
             "passthrough": 0}
    t0 = time.time()

    def work(name: str) -> dict:
        text = (args.articles_dir / name).read_text(encoding="utf-8")
        row: dict = {"file": name}
        try:
            data, usage = S.call_llm(text, model=model)
            sections = data.get("sections") or []
            located, skipped, warnings = S.validate_and_locate(text, sections)
            empty_reason = None
            if not located:
                empty_reason = "parse-error" if data.get("_parse_error") else "llm-zero"
            final = S.build_final(text, located, empty_reason=empty_reason)
            tmp = args.out_dir / (name + ".tmp")
            tmp.write_text(final, encoding="utf-8")
            tmp.rename(args.out_dir / name)
            row.update(ok=True, n_proposed=len(sections), n_located=len(located),
                       n_skipped=len(skipped), warnings=warnings[:5],
                       sections=[s.get("title") for s in sections][:50],
                       parse_error=data.get("_parse_error"), usage=usage)
        except Exception as exc:  # audit failures; never lose the run to one doc
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:400])
        return row

    def passthrough_one(name: str) -> dict:
        """Emit an out-of-candidate article with the official empty-TOC form.

        Uses the same S.build_final the LLM path uses, so every file in the
        structured corpus has one shape; only the TOC is absent. This is what
        keeps the BM25 corpus complete -- omitting these files would silently
        shrink RISE's retrieval space and make the task easier.
        """
        text = (args.articles_dir / name).read_text(encoding="utf-8")
        final = S.build_final(text, [], empty_reason="outside-candidate-set")
        tmp = args.out_dir / (name + ".tmp")
        tmp.write_text(final, encoding="utf-8")
        tmp.rename(args.out_dir / name)
        return {"file": name, "ok": True, "passthrough": True,
                "n_proposed": 0, "n_located": 0, "n_skipped": 0}

    with audit_path.open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(header, ensure_ascii=False) + "\n")
        if passthrough:
            t_pt = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, row in enumerate(pool.map(passthrough_one, passthrough), 1):
                    audit.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["passthrough"] += 1
                    if i % 100_000 == 0:
                        print(f"[toc] passthrough {i:,}/{len(passthrough):,} "
                              f"({i/(time.time()-t_pt):.0f}/s)", flush=True)
            audit.flush()
            print(f"[toc] passthrough done: {len(passthrough):,} in "
                  f"{time.time()-t_pt:.0f}s", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, n): n for n in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                with lock:
                    audit.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if row.get("ok"):
                        stats["ok"] += 1
                        stats["sections"] += row.get("n_proposed", 0)
                        stats["located"] += row.get("n_located", 0)
                        stats["skipped"] += row.get("n_skipped", 0)
                    else:
                        stats["fail"] += 1
                    if i % 1000 == 0:
                        rate = i / (time.time() - t0)
                        eta_h = (len(pending) - i) / rate / 3600 if rate else -1
                        print(f"[toc] {i:,}/{len(pending):,} rate={rate:.1f}/s "
                              f"eta={eta_h:.1f}h fail={stats['fail']}", flush=True)
                    audit.flush()

    proposed = stats["located"] + stats["skipped"]
    print(json.dumps({
        "sectioned_via_llm": stats["ok"] + stats["fail"],
        "ok": stats["ok"], "failed_docs": stats["fail"],
        "passthrough_empty_toc": stats["passthrough"],
        "anchor_validation_rate": round(stats["located"] / proposed, 4) if proposed else None,
        "rate_per_s": round((stats["ok"] + stats["fail"]) / (time.time() - t0), 2),
    }, indent=2), flush=True)
    if stats["fail"] > 0.02 * max(1, stats["ok"] + stats["fail"]):
        sys.exit("TOC build FAILED: >2% document failures; inspect the audit log")


if __name__ == "__main__":
    main()
