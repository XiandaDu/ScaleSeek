# ScaleSeek v3：不训练的搜索 agent，以及加上 SFT 和 RL

2026-11-20。**未训练**的 Qwen3.5-9B（没有 SFT，没有 RL），靠协议设计在 7 个 QA 数据集上超过 GrepSeek 的论文数值。过程记录见 `research_log.md` 第 2、3 轮。

**要点**
1. 7 个 QA 测试集宏平均 EM 两遍是 49.29 和 48.91，论文的 GrepSeek 是 45.96；按题数加权是 53.33 和 52.93，论文 49.48；宏平均 F1 是 58.69 和 58.56，论文 54.83。
2. 对我们复现的 GrepSeek 做配对检验（3,125 题）：48.4 对 45.2，p = 2.9e-5；第二遍 48.3 对 45.2，p = 9.6e-5。
3. 技术路线与论文不同：论文靠两阶段训练（27B 导师生成 1 万条 SFT 轨迹，再做 GRPO），这里完全不训练。
4. BrowseComp-Plus 上 judge 准确率 41.6（GrepSeek 复现 21.6，论文 34.2，AgentIR 31.0）。含 BCP 的 8 个数据集宏平均：F1 口径 56.51 对论文 52.26，EM 口径 48.16 对 43.47。
5. 它用了 e5 索引和精确短语索引，不像 GrepSeek 那样不建索引。

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

### 3.1 BrowseComp-Plus

500 题，Qwen3-32B 判分；16 轮、131K 上下文；检索器是 BCP 语料上的 e5 和精确短语索引，文档显示截到 512 tokens（与其他方法相同）。方法不改，原样运行。

| | judge 准确率 | 给出答案 |
|---|---:|---:|
| **v3** | **41.6** | 498 |
| AgentIR | 31.0 | 175 |
| GrepSeek+ | 25.8 | 377 |
| GrepSeek 复现 | 21.6 | 306 |
| Search-o1 + e5 | 11.6 | 107 |
| GrepSeek 论文 | 34.22 / 26.02 | |

- 对 AgentIR：赢 93 输 40，p = 5e-6；对 GrepSeek：p = 1e-12。
- 判对的 208 题里 185 题与标准答案字符串完全相同（归一化后）；只按完全匹配算是 37.2%。
- 最后一轮之前作答的 144 题里 126 题对；被强制作答的 356 题里 82 题对；95 题答了 “unknown”，全部判错。
- 第二遍（只换随机种子）：42.2（211 题）；两遍逐题比较 50 对 47。第二遍对 AgentIR：p = 5e-6；对按论文方式切段的 GrepSeek（29.6）：赢 125 输 62，p = 5e-6。
- 含 BCP 的 8 个数据集宏平均（都取两遍平均）：F1 口径 56.54（论文 52.26），EM 口径 48.20（论文 43.47）。

**行为**（第一遍）
- 所有题都给出了答案（v2 没作答的比例：NQ 8%，HotpotQA 18%，2Wiki 38%，MuSiQue 39%）。
- 45.2% 的题走到最后一轮被强制作答，这些题 EM 是 26.0%；其余的题按数据集是 45%–82%。
- 10.2% 的题被强制后答了 “unknown”（2Wiki 21.8%，MuSiQue 16.8%，PopQA 11.2%）。
- 每题 find 0.9–1.9 次、search 1.6–3.3 次（GrepSeek 每题 2.3–3.8 次 grep）；每题生成 520–1,030 tokens；结束时上下文 3.1K–5.2K tokens。

---

## 4. 需要如实说明的

- **用了索引**：e5 的 HNSW 索引（69 GB）和一个精确短语索引（4 GB）。GrepSeek 的卖点是不建索引；论文里的 Search-R1、RAG、Search-o1 基线用的也是 e5。
- **BCP 的设置与论文不完全相同**：判分模型（Qwen3-32B 对 GPT-4.1）、题目数（500 对 830）不同；v3 检索的是整篇网页（显示时截到 512 tokens），论文的 GrepSeek 检索的是 700 词的段落（按这种切法我们复现的 GrepSeek 是 29.6）；BCP 上系统提示仍写的是 Wikipedia 语料和“最多 5 轮检索”，实际轮数由每轮的预算提示给出。
- **比较对象**：论文数值是在完整数据集上、不加 presence_penalty 得到的；我们是每个数据集 500 题子集（Bamboogle 125）、所有方法加 presence_penalty 2.0。同样设置下我们复现的 GrepSeek 与论文逐数据集一致（宏平均 EM 45.86 对 45.96）。
- 预算提示本身不是新东西（BATS、BCAS、AnySearch）。这里的结果是：一个未训练的 9B 模型加上这套协议，就达到了 RL 训练的搜索 agent 的水平。

---

## 5. 复现

```bash
bash submit.sh eval --method scaleseek_v3_prompt --dataset hotpotqa --retriever e5      # 7 个数据集各一次
bash submit.sh eval --method scaleseek_v3_prompt_r2 --dataset hotpotqa --retriever e5   # 第二遍
# 训练题上的验证（协议的每个部件）
bash submit.sh rollouts --name pilot_commit --take 500 --variants budget,commit
```

| 内容 | 位置 |
|---|---|
| 协议和工具 | `scaleseek/agent.py`、`scaleseek/methods/scaleseek.py` |
| 配置 | `configs/methods/scaleseek_v3_prompt.yaml`、`scaleseek_v3_prompt_r2.yaml`；消融 `scaleseek_prompt.yaml`（v1）、`scaleseek_v2_prompt.yaml` |
| 测试结果 | `$ROOT/runs/<dataset>/scaleseek_v3_prompt__e5/`、`scaleseek_v3_prompt_r2__e5/` |
| 训练题验证 | `$ROOT/train_data/rollouts/pilot_budget/`、`pilot_commit/` |
| 对比脚本 | `reports/scripts/v3_stability.py`、`gplus_test_cmp.py`；不用 GPU 的循环测试 `commit_test.py` |

---

## 6. 加上 SFT 和 RL（2026-11-24）

问题：v3 不训练已经是 48.4，加上训练还能不能涨。目前的结论：**SFT 之后做与评测协议一致的 RL，在验证集上显著好于未训练的 v3**；SFT 单独没有用，从基座直接 RL 也没有用。测试集结果待填。过程见 `research_log.md` 第 5 轮 §5.3。

**做法**（都在 A100-40GB 上）

| 步骤 | 内容 | 用时 | 代码 |
|---|---|---|---|
| 1. 自己的轨迹 | 未训练的 9B 按 v3 协议在 6,000 道训练题上各跑 2 条轨迹（12,000 条，EM 58.2） | 约 4 小时，1 张卡 | `train/rollouts.py` |
| 2. SFT | 只留答对的，每题留轮数最少的一条，共 3,779 条；1 个 epoch，学习率 5e-6 | 45 分钟，4 张卡 | `train/build_sft.py`、`train/config/sft.yaml` |
| 3. RL | 从 SFT 的 checkpoint 做 GRPO：每步 64 题 × 8 条轨迹，奖励是 EM，学习率 5e-6；rollout 用完整的 v3 协议（含最后一轮强制作答）；取第 90 步 | 24 小时，4 张卡 | `train/scaleseek_rl/agent_loop.py`、`train/config/grpo.yaml` |

```bash
bash submit.sh rollouts --name self_v3                                   # 步骤 1
python train/build_sft.py $ROOT/train_data/rollouts/self_v3 $ROOT/train_data/sft_v3_self
bash submit.sh sft --name sft_v3_self --data $ROOT/train_data/sft_v3_self   # 步骤 2
bash submit.sh rl --name grpo_v3_sft --model $ROOT/train_runs/sft_v3_self/global_step_118/huggingface   # 步骤 3
bash train/export_hf.sh grpo_v3_sft 90                                   # 在作业里跑；输出写进 configs/methods/scaleseek_v3_rl.yaml
bash submit.sh rollouts --name val_grpo_v3_sft_s90 --dev --samples 1 --model local/scaleseek-grpo_v3_sft-step90 --revision <REV>   # 验证集
```

**验证集**（500 道 NQ dev + 500 道 HotpotQA dev，所有行都是同样的 1,000 题；脚本 `reports/scripts/val_table.py`）

| 模型 | EM | NQ | HotpotQA | F1 | 平均轮数 | 对未训练 v3：赢 / 输，p |
|---|---:|---:|---:|---:|---:|---|
| GrepSeek（官方 checkpoint；第二遍 46.1） | 46.5 | 40.8 | 52.2 | 58.6 | — | 79 / 93，p = 0.32 |
| GrepSeek+（另两遍 49.4、49.2） | 50.1 | 44.4 | 55.8 | 63.2 | — | 86 / 64，p = 0.086 |
| v3 未训练 | 47.9 | 43.0 | 52.8 | 61.3 | 4.09 | — |
| v3 + SFT | 49.3 | 44.2 | 54.4 | 62.8 | 4.06 | 48 / 34，p = 0.15 |
| v3 + SFT → RL，第 40 步 | 49.1 | 44.6 | 53.6 | 61.6 | 4.81 | 59 / 47，p = 0.29 |
| v3 + SFT → RL，第 60 步 | 51.1 | 47.2 | 55.0 | 63.3 | 4.67 | 71 / 39，p = 0.003 |
| v3 + SFT → RL，第 80 步 | 50.3 | 46.4 | 54.2 | 63.2 | 4.76 | 63 / 39，p = 0.022 |
| v3 + SFT → RL，第 90 步 | 待填 | | | | | 前 553 题：44 / 25 |

- 对 GrepSeek 逐题比：SFT → RL 第 60 步赢 110 输 64（p = 0.0006），第 80 步赢 102 输 64（p = 0.004）。
- 模型学到的是多搜：平均轮数 4.09 → 4.7，`find` 从每题 1.2 次涨到 1.7–1.9 次。
- 上测试集的 checkpoint 按事先定的规则选：用最后一个（第 90 步）；只有它在验证集上比最好的低 1 分以上才换，并且两个都报。

**两点是必须的**
- **RL 的 rollout 要和评测用同一个协议。** 训练时不强制作答，用完轮数的轨迹奖励是 0，模型学成“早点答”：验证集 47.9 → 42.6。
- **要先做 SFT。** 从基座直接做协议一致的 RL，前 50 步训练 EM 不动，之后崩掉（作答率掉到 63%–71%）。SFT 单独在测试集上是 48.4，与未训练相同（逐题赢 166 输 164）。

**测试集**：待填（配置 `configs/methods/scaleseek_v3_rl.yaml`）。

---

## 7. 训练数据和泄漏检查

| 用途 | 题目来源（FlashRAG 数据集，版本固定在 `configs/base.yaml`） | 数量 |
|---|---|---|
| SFT | NQ `train` 随机 2,000 题 + HotpotQA `train` 随机 4,000 题（只取 medium / hard），种子 20261105；每题 2 条轨迹，留答对的 | 6,000 题 → 3,779 条 |
| RL | NQ `train` + HotpotQA `train` 的全部题（各留出 100 题不用），固定种子打乱，每步 64 题；跑了 92 步 | 5,887 题 |
| 选协议 | 上面 SFT 那 6,000 题里每个来源的前 500 题（§2） | 1,000 题 |
| 选 checkpoint（验证集） | NQ `dev` 随机 500 题 + HotpotQA `dev` 随机 500 题，HotpotQA 先去掉测试子集里的题（`scaleseek/grepseek_pilot.py` 的 `questions`） | 1,000 题 |
| 测试 | 每个数据集固定种子抽 500 题：NQ / TriviaQA / PopQA / Bamboogle（125 题）的 `test`，HotpotQA / 2Wiki / MuSiQue 的 `dev`；BrowseComp-Plus | 3,125 + 500 题 |

这和 GrepSeek、Search-R1 的做法相同：都在 NQ 和 HotpotQA 的 train 上训练。训练只用了这两个数据集，另外 5 个 QA 数据集和 BCP 没有任何题进入训练或选型。

**保证不相交的办法**
1. 训练题只读 `train.jsonl`，测试题只读各数据集的 `test.jsonl` / `dev.jsonl`，是不同的文件。
2. HotpotQA 的测试子集和验证集都来自 `dev.jsonl`，验证集抽样前按题目 id 去掉了测试子集。
3. 协议在训练题上选，checkpoint 在验证集上按事先定的规则选，测试集每个配置只跑一次。
4. 按题目文本实际核对（小写、去标点和冠词后完全相同才算重合；脚本 `reports/scripts/leak_check.py`）：

| 评测集 | 题数 | 与 NQ + HotpotQA 全部训练题重合 | 与 SFT 跑过的 6,000 题重合 | 与 SFT 训练用的 3,779 题重合 | 与 RL 见过的 5,887 题重合 |
|---|---:|---:|---:|---:|---:|
| NQ 测试子集 | 500 | 7 | 1 | 0 | 0 |
| TriviaQA 测试子集 | 500 | 1 | 0 | 0 | 0 |
| PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、BCP 测试子集 | 各自全部 | 0 | 0 | 0 | 0 |
| 验证集 | 1,000 | 7 | 0 | 0 | 1 |

- **模型训练时见过的题（SFT 的 3,779 题和 RL 的 5,887 题）与 8 个测试子集的重合是 0。**
- NQ 的 train 文件里本来就有 7 道题和 NQ 测试子集的题文字相同，这是数据集自带的，在整个 NQ train 上训练的方法（GrepSeek、Search-R1）都会遇到。其中 1 道被抽进了 SFT 的 6,000 题，但它的轨迹没有进入 SFT 训练数据，RL 也没抽到它。
- 验证集有 1 道题和 RL 见过的一道训练题文字相同（1,000 题里的 1 题），只影响选 checkpoint。
- 验证集和测试子集的重合是 0。
- 这个检查查不出来的：换了说法的同一个问题（NQ 已知有这种情况，对所有在 NQ 上训练的方法相同）；底座模型预训练时是否见过这些公开数据集（与 GrepSeek 用的是同一个底座）。
