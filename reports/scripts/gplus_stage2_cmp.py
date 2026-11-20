"""The three zero-training checks for the next method, on the validation questions (GrepSeek+ = `all`):
  1  re-reading the evidence in a clean context (one rollout, two rollouts pooled)
  2  verify: an answer found in no tool output is checked once
  3  commit when two rollouts agree; otherwise use the pooled re-read / a third rollout
Strict EM, paired exact McNemar; every comparison says what it costs."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/a32du/ScaleSeek")
from scaleseek.metrics import exact_match, final_answer, mcnemar_exact, normalize_answer

P = Path("/scratch/a32du/scaleseek_v2/pilots/grepseek")


def load(name):
    f = P / name / "trajectories.jsonl"
    return {r["id"]: r for r in map(json.loads, open(f))} if f.exists() else None


def ans(r):
    a = final_answer(r)
    return normalize_answer(a) if a else None


def em(pred, runs):          # pred: {id: record whose answer counts}
    return {i: exact_match(final_answer(r), r["golden_answers"]) for i, r in pred.items()}


def show(tag, a, b, cost):
    mc = mcnemar_exact(a, b)
    n = len(a)
    line = f"  {tag:58s} EM {100 * sum(a.values()) / n:5.1f}"
    for src in ("nq", "hotpotqa"):
        sel = [i for i in a if i.startswith(src)]
        line += f"  {src} {100 * sum(a[i] for i in sel) / len(sel):5.1f}"
    print(line + f"   vs GrepSeek+ {100 * (sum(a.values()) - sum(b.values())) / n:+5.1f} (won {mc['only_a']:3d} lost {mc['only_b']:3d} p={mc['p_value']:.3f})   cost: {cost}")


A, B, C = load("all"), load("all_r2"), load("all_r3")
eA, eB = em(A, None), em(B, None)
print(f"GrepSeek+ single rollout: all {100 * sum(eA.values()) / len(eA):.1f}, all_r2 {100 * sum(eB.values()) / len(eB):.1f}"
      + (f", all_r3 {100 * sum(em(C, None).values()) / len(C):.1f}" if C else ""))

print("\n1  re-read in a clean context (reference: the rollout `all`)")
for name, cost in (("reread_single", "1 rollout + 1 answer-only call"), ("reread_pooled", "2 rollouts + 1 answer-only call")):
    X = load(name)
    if X is None:
        continue
    fin = Counter(r["finish_reason"] for r in X.values())
    pred = {i: (X[i] if X[i]["finish_reason"] == "answer" else A[i]) for i in A}      # no answer on the re-read: keep the rollout's
    show(f"{name} (falls back to `all` when it gives no answer)", em(pred, None), eA, cost)
    show(f"{name}, strict (no answer = wrong)", em(X, None), eA, cost)
    same = sum(ans(X[i]) == ans(A[i]) for i in A)
    fix = sum(em(X, None)[i] and not eA[i] for i in A)
    brk = sum(eA[i] and not em(X, None)[i] for i in A)
    print(f"      finish {dict(fin)}; same answer as `all` {100 * same / len(A):.1f}%; wrong->right {fix}, right->wrong {brk}")

print("\n3  two rollouts: commit when they agree")
agree = {i for i in A if ans(A[i]) is not None and ans(A[i]) == ans(B[i])}
print(f"   agree on {100 * len(agree) / len(A):.1f}% of the questions")
X = load("reread_pooled")
if X:
    pred = {i: (A[i] if i in agree else (X[i] if X[i]["finish_reason"] == "answer" else A[i])) for i in A}
    show("agree -> that answer; else pooled re-read", em(pred, None), eA, f"2 rollouts + {100 * (1 - len(agree) / len(A)):.0f}% answer-only calls")
if C:
    def plurality(i):
        votes = Counter(a for a in (ans(A[i]), ans(B[i]), ans(C[i])) if a)
        if not votes:
            return A[i]
        best = votes.most_common(1)[0]
        return next(r for r in (A[i], B[i], C[i]) if ans(r) == best[0]) if best[1] >= 2 else A[i]
    show("plurality of 3 rollouts (tie: the first)", em({i: plurality(i) for i in A}, None), eA, "3 rollouts")
    pred = {i: (A[i] if i in agree else plurality(i)) for i in A}
    show("agree -> that answer; else a third rollout votes", em(pred, None), eA, f"2 rollouts + {100 * (1 - len(agree) / len(A)):.0f}% third rollouts")
    if X:
        def both(i):
            if i in agree:
                return A[i]
            votes = Counter(a for a in (ans(A[i]), ans(B[i]), ans(C[i]), ans(X[i]) if X[i]["finish_reason"] == "answer" else None) if a)
            best = votes.most_common(1)[0] if votes else None
            return next(r for r in (X[i], A[i], B[i], C[i]) if ans(r) == best[0]) if best and best[1] >= 2 else (X[i] if X[i]["finish_reason"] == "answer" else A[i])
        show("agree -> that answer; else vote of 3 rollouts + pooled re-read", em({i: both(i) for i in A}, None), eA, "2 rollouts + 30% (third rollout + answer-only call)")
    oracle = {i: eA[i] or eB[i] or em(C, None)[i] for i in A}
    print(f"   oracle (any of 3 rollouts right): {100 * sum(oracle.values()) / len(A):.1f}")

print("\n2  verify (same turn budget as GrepSeek+)")
for name in ("verify", "verify_r2"):
    V = load(name)
    if V is None:
        continue
    eV = em(V, None)
    show(f"{name} vs all", eV, eA, "1 rollout, <= 6 turns")
    show(f"{name} vs all_r2", eV, eB, "1 rollout, <= 6 turns")
    n_ch = kept = changed = searched = first_ok = final_ok = 0
    for r in V.values():
        msgs = r["trajectory"]["messages"] or []
        idx = [j for j, m in enumerate(msgs) if m["role"] == "tool" and "does not appear in any search result" in json.loads(m["content"]).get("notice", "")]
        if not idx:
            continue
        n_ch += 1
        first = msgs[idx[0] - 1]["content"].split("<answer>")[-1].split("</answer>")[0].strip()
        first_ok += exact_match(first, r["golden_answers"])
        final_ok += exact_match(final_answer(r), r["golden_answers"])
        searched += any(m["role"] == "tool" for m in msgs[idx[0] + 1:])
        fa = final_answer(r)
        kept += fa is not None and normalize_answer(fa) == normalize_answer(first)
        changed += fa is not None and normalize_answer(fa) != normalize_answer(first)
    fin = Counter(r["finish_reason"] for r in V.values())
    print(f"      challenged {n_ch} questions: searched again {searched}, kept the answer {kept}, changed it {changed}, no final answer {n_ch - kept - changed};"
          f" first answer right {first_ok}, final answer right {final_ok}; finish {dict(fin)}")
