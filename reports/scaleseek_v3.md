# ScaleSeek v3 (Prompt)：不训练的搜索 agent

2026-11-20。**未训练**的 Qwen3.5-9B（没有 SFT，没有 RL），靠协议设计在 7 个 QA 数据集上超过 GrepSeek 的论文数值。过程记录见 `research_log.md` 第 2、3 轮。

**要点**
1. 7 个 QA 测试集宏平均 EM 两遍是 49.29 和 48.91，论文的 GrepSeek 是 45.96；按题数加权是 53.33 和 52.93，论文 49.48；宏平均 F1 是 58.69 和 58.56，论文 54.83。
2. 对我们复现的 GrepSeek 做配对检验（3,125 题）：48.4 对 45.2，p = 2.9e-5；第二遍 48.3 对 45.2，p = 9.6e-5。
3. 技术路线与论文不同：论文靠两阶段训练（27B 导师生成 1 万条 SFT 轨迹，再做 GRPO），这里完全不训练。
4. 它用了 e5 索引和精确短语索引，不像 GrepSeek 那样不建索引。BCP 还在跑。

---

## 1. 方法

模型：`Qwen/Qwen3.5-9B` 原始权重。预算与 GrepSeek 相同：每题最多 6 轮、16K 上下文；解码与所有方法相同（temperature 0.6、top_p 1.0、presence_penalty 2.0）；每轮生成上限 2,048 tokens。

| 部件 | 内容 | 代码 |
|---|---|---|
| 工具 | `find`：精确短语检索（大小写和重音折叠的位置索引，标题命中优先）；`search`：e5 语义检索；`answer`：给出最终答案。每次返回 5 段；每轮最多两个并行的 find / search | `scaleseek/agent.py`、`scaleseek/retrieval/exact.py` |
| 草稿答案 | find / search 的每次调用都带 `current_answer`（当前最好的答案或 “unknown”） | `scaleseek/agent.py` |
| 预算提示（budget） | 每轮工具返回的末尾写明还剩几轮检索；用完时写“现在调用 answer” | `agent.budget_note` |
| 最后一轮强制作答（commit） | 最后一轮先让模型自由推理到 `</think>`，然后由 harness 写出 `answer` 调用的开头，模型只填答案（最多 64 tokens）。算一轮 | `agent.COMMIT`、`methods/scaleseek.py` 的 `commit` |

配置：`configs/methods/scaleseek_v3_prompt.yaml`（`variants: [budget, commit]`）。

---

## 2. 每个部件怎么来的

都在**训练题**上验证（NQ train 500 题 + HotpotQA train medium/hard 500 题，每题 2 条轨迹；模型不训练，所以不存在泄漏），测试集各跑一次。

| 协议 | 训练题 NQ / HotpotQA | 测试集（子集 micro EM） |
|---|---|---:|
| v1：工具 + 草稿答案 | 39.0 / 55.2 | 38.0 |
| v2：+ budget | 43.8 / 63.6（p < 0.001） | 44.6 |
| **v3：+ commit** | **47.0 / 67.4**（+3.5，赢 143 输 73，p < 0.0001） | **48.4** |

同一批训练题上的参照：从基座做 RL 训练的 v1 模型是 44.0 / 60.2，Qwen3.5-27B 是 42.9 / 60.4。

试过但没采用的：每轮返回更多段落（不显著，工具 tokens 多 1.7 倍）；find 空结果时放宽（不显著）；最后一轮要求“给出最佳猜测，不要答 unknown”（57.2 → 55.5，变差）。

---

## 3. 测试集结果

7 个 Wiki 测试子集共 3,125 题；没给出答案算 0 分。v3 跑了两遍（第二遍只换服务端随机种子，`scaleseek_v3_prompt_r2.yaml`）。

**严格 EM**

| | NQ | TQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | 7 个宏平均 | 按题数加权 | 子集 micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GrepSeek 论文（Table 6） | 40.47 | 69.25 | 43.63 | 49.49 | 46.27 | 21.43 | 51.20 | 45.96 | 49.48 | — |
| GrepSeek 我们的复现 | 39.0 | 70.4 | 44.4 | 47.6 | 46.8 | 21.6 | 51.2 | 45.86 | 49.70 | 45.2 |
| **v3 第一遍** | 41.4 | 73.2 | 46.2 | 49.6 | 55.0 | 22.8 | 56.8 | **49.29** | **53.33** | **48.4** |
| **v3 第二遍** | 41.4 | 72.4 | 47.4 | 47.4 | 53.4 | 26.0 | 54.4 | **48.91** | **52.93** | **48.3** |
| GrepSeek+ | 42.2 | 69.4 | 49.2 | 51.2 | 57.4 | 22.6 | 50.4 | 48.91 | 54.17 | 48.7 |
| Search-R1-14B + Qwen3-Emb-4B | 48.6 | 69.0 | 48.6 | 47.0 | 48.4 | 25.0 | 52.0 | 48.37 | — | 47.9 |
| Search-R1-14B + e5 | 51.2 | 69.2 | 45.0 | 46.6 | 44.0 | 23.0 | 52.0 | 47.29 | — | 46.7 |

**token F1**

| | NQ | TQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | 7 个宏平均 | 按题数加权 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GrepSeek 论文（Table 1） | 52.23 | 76.73 | 48.61 | 62.31 | 51.78 | 30.06 | 62.12 | 54.83 | 56.91 |
| **v3 第一遍** | 54.3 | 81.0 | 54.6 | 61.1 | 60.2 | 32.5 | 67.2 | **58.69** | **61.63** |
| **v3 第二遍** | 54.1 | 80.4 | 55.4 | 59.3 | 58.2 | 34.9 | 67.7 | **58.56** | **61.08** |

- 两遍的聚合指标只差 0.1–0.4；逐题比较 156 对 160，没有系统差别。单个数据集两遍最多差 3.2（MuSiQue）。
- EM：第一遍 7 个数据集全部高于论文，第二遍 6 个。两遍平均后只有 HotpotQA 低于论文（48.5 对 49.49），但高于我们复现的 GrepSeek（47.6）。
- F1：两遍都是 6 个数据集高于论文，HotpotQA 低 1.2–3.0。
- 与 GrepSeek+ 和 Search-R1-14B + Qwen3-Emb-4B 相当。

**行为**（第一遍）
- 所有题都给出了答案（v2 没作答的比例：NQ 8%，HotpotQA 18%，2Wiki 38%，MuSiQue 39%）。
- 45.2% 的题走到最后一轮被强制作答，这些题 EM 是 26.0%；其余的题按数据集是 45%–82%。
- 10.2% 的题被强制后答了 “unknown”（2Wiki 21.8%，MuSiQue 16.8%，PopQA 11.2%）。
- 每题 find 0.9–1.9 次、search 1.6–3.3 次（GrepSeek 每题 2.3–3.8 次 grep）；每题生成 520–1,030 tokens；结束时上下文 3.1K–5.2K tokens。

---

## 4. 需要如实说明的

- **用了索引**：e5 的 HNSW 索引（69 GB）和一个精确短语索引（4 GB）。GrepSeek 的卖点是不建索引；论文里的 Search-R1、RAG、Search-o1 基线用的也是 e5。
- **BCP 还没有结果**：论文主表（Table 1）是含 BCP 的 8 个数据集宏平均（F1 52.26，EM 43.47）。v3 在 BCP 上高于 7.6 就能超过主表 F1。评测已提交。
- **比较对象**：论文数值是在完整数据集上、不加 presence_penalty 得到的；我们是每个数据集 500 题子集（Bamboogle 125）、所有方法加 presence_penalty 2.0。同样设置下我们复现的 GrepSeek 与论文逐数据集一致（宏平均 EM 45.86 对 45.96）。
- 预算提示本身不是新东西（BATS、BCAS、AnySearch）。这里的结果是：一个未训练的 9B 模型加上这套协议，就达到了 RL 训练的搜索 agent 的水平。

---

## 5. 复现

```bash
bash submit.sh eval --method scaleseek_v3_prompt --dataset hotpotqa --retriever e5      # 7 个数据集各一次
bash submit.sh eval --method scaleseek_v3_prompt_r2 --dataset hotpotqa --retriever e5   # 第二遍
# 训练题上的验证
bash submit.sh teacher --name pilot_commit --nq 2000 --hotpotqa 4000 --samples 2 --take 500 --tp 1 \
     --model Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --variants budget,commit
```

| 内容 | 位置 |
|---|---|
| 协议和工具 | `scaleseek/agent.py`、`scaleseek/methods/scaleseek.py`（commit `8e5f41e`） |
| 配置 | `configs/methods/scaleseek_v3_prompt.yaml`、`scaleseek_v3_prompt_r2.yaml` |
| 测试结果 | `$ROOT/runs/<dataset>/scaleseek_v3_prompt__e5/`、`scaleseek_v3_prompt_r2__e5/` |
| 训练题验证 | `$ROOT/train_data/teacher/pilot_budget/`、`pilot_commit/`、`pilot_guess/` |
| 对比脚本 | `reports/scripts/v3_stability.py`、`gplus_test_cmp.py` |
