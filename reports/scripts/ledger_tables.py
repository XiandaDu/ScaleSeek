"""Markdown tables for the reports from failure_accounting.json: per run, every dataset's questions split
into one bucket each (percent of all questions), the Wiki total in counts, and the failure classes
sorted by size."""
import json
from pathlib import Path

L = json.loads(Path(__file__).with_name("failure_accounting.json").read_text())
DS = [("nq", "NQ"), ("triviaqa", "TriviaQA"), ("popqa", "PopQA"), ("hotpotqa", "HotpotQA"),
      ("2wikimultihopqa", "2Wiki"), ("musique", "MuSiQue"), ("bamboogle", "Bamboogle")]
CLASSES = [("max_turns", "轮数耗尽"), ("context_exhausted", "上下文写满"), ("no_answer", "无答案"),
           ("abstain", "拒答"), ("form", "答案形式"), ("gold_seen", "见过 gold 答错"),
           ("gold_unseen", "没见过 gold 答错"), ("unmatchable", "无法判断")]
RUNS = ["grepseek", "search_r1_14b__qwen3_emb_4b", "search_r1_14b__e5", "search_o1__qwen3_emb_4b",
        "rag__qwen3_emb_4b", "agentir__agentir_4b"]


def table(run):
    cols = [(k, name) for k, name in CLASSES if any(L[run][d].get(k) for d, _ in DS + [("browsecomp_plus", "")])]
    head = "| 数据集 | N | 答对 | 作答 | 作答准确率 | " + " | ".join(n for _, n in cols) + " | 最大失败类 |"
    lines = [head, "|" + "---|" * (len(cols) + 6)]
    tot = {}
    for d, name in DS + [("browsecomp_plus", "BCP")]:
        c = L[run][d]
        n = sum(c.values())
        unanswered = sum(c.get(k, 0) for k in ("max_turns", "context_exhausted", "no_answer"))
        answered = n - unanswered
        top = max(cols, key=lambda kc: c.get(kc[0], 0))
        cells = [f"{100 * c.get(k, 0) / n:.1f}" for k, _ in cols]
        lines.append(f"| {name} | {n} | {100 * c.get('correct', 0) / n:.1f} | {answered} | "
                     f"{100 * c.get('correct', 0) / answered:.1f} | " + " | ".join(cells) + f" | {top[1]} |")
        if d != "browsecomp_plus":
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
    n = sum(tot.values())
    unanswered = sum(tot.get(k, 0) for k in ("max_turns", "context_exhausted", "no_answer"))
    lines.insert(2, f"| **Wiki 合计（题数）** | {n} | {tot.get('correct', 0)} | {n - unanswered} | "
                    f"{100 * tot.get('correct', 0) / (n - unanswered):.1f} | "
                    + " | ".join(str(tot.get(k, 0)) for k, _ in cols) + " | |")
    fails = sorted(((tot.get(k, 0), name) for k, name in cols if tot.get(k, 0)), reverse=True)
    nf = sum(v for v, _ in fails)
    ranking = "；".join(f"{name} {v}（失败的 {100 * v / nf:.0f}%，全部题的 {100 * v / n:.1f}%）" for v, name in fails)
    return "\n".join(lines), f"Wiki 失败共 {nf} 题，从多到少：{ranking}"


for run in RUNS:
    t, r = table(run)
    print(f"### {run}\n\n{t}\n\n{r}\n")
