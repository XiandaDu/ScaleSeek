# 其他方法失败案例分析（ScaleSeek v2 正式运行）

2026-10-20。对象是 5 个强 run，GrepSeek 作为参照列在旁边：
- Search-R1-14B（e5 / Qwen3-Embedding-4B 检索）
- Search-o1（Qwen3.5-9B + Qwen3-Embedding-4B）
- RAG（Qwen3.5-9B + Qwen3-Embedding-4B）
- AgentIR（Qwen3.5-9B + AgentIR-4B）

**做法**
- 全量统计：每道题归入一个结局类别，口径与 `grepseek_failures.md` §0 相同。另外做了两个专项计数：Search-R1 的"中途 EOS"、Search-o1 的检索标签拼错。还跑了两个 presence_penalty 对照 pilot。
- 人工通读分层抽样的案例：AgentIR 72 条，Search-o1 93 条，RAG 70 条，Search-R1-14B e5 70 条、qwen3 24 条。抽样是分层的，所以"x / 抽样数"只说明某种机制存在，不能当成总体比例。
- BCP 只写 query ID 和机制。

**要点**
1. 方法分成两种极端。
   - Search-R1 和 RAG 几乎每题都作答，失败几乎全是"答了但答错"。Search-R1 主要是第一跳没检索到；RAG 主要是拒答。
   - AgentIR 和 Search-o1 有 41% 和 28% 的 Wiki 题没给出答案（轮数用完），但给出的答案准确率不低。
   - GrepSeek 居中。
2. 数据集的难点在各方法之间很一致。
   - NQ 上所有方法的"答案形式"和"见过 gold 答错"都是最高的。
   - 2Wiki 和 MuSiQue 上，会多步搜索的方法都输在轮数不够。
   - MuSiQue 和 BCP 上，单次检索的方法大多根本没检索到 gold。
3. 有三个评测或协议因素会影响解读：
   - EM 对冗长答案很不利：AgentIR 的 cover-EM 比 EM 高 15 个点；
   - RAG 的大量拒答来自 reader prompt；
   - 统一的 presence_penalty 2.0 放大了 Search-o1 和 Search-R1-14B 的格式失败，但不改变 EM 结论。

---

## 0. 口径（与 GrepSeek 报告相同）

- **分母是全部题目，没有题被排除。** 没给出答案的题按 0 分计。EM = 答对 / 全部题；作答准确率 = 答对 / 给出答案的题。
- **这与 GrepSeek 论文的口径一致。** 我们的 GrepSeek F1 按全部题计算，与论文吻合，见下表（`scripts/denominator_rag.py`）。如果只算给出答案的题，2Wiki 和 MuSiQue 会比论文高出 16–20 个点。

  | GrepSeek F1 | NQ | TQ | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 按全部题（我们的口径） | 51.9 | 78.7 | 50.4 | 59.3 | 51.7 | 30.0 | 60.8 |
  | 只算给出答案的题 | 56.1 | 83.0 | 61.6 | 68.3 | 71.3 | 46.2 | 72.4 |
  | 论文 | 52.2 | 76.7 | 48.6 | 62.3 | 51.8 | 30.1 | 62.1 |
- **每道题只归入一类**，按以下顺序判定，先命中者优先：
  1. 答对；
  2. 轮数耗尽；
  3. 上下文写满（包括 RAG/Direct 在 BCP 上撞到 10K 输出上限）；
  4. 无答案（输出格式错误或中途结束，解析不出答案）；
  5. 拒答（"Not mentioned" 一类）；
  6. 答案形式（仅 Wiki：语义对或接近，但 EM 判错）；
  7. 见过 gold 答错（某篇返回文档里出现过 gold）；
  8. 没见过 gold 答错；
  9. 无法判断（gold 太短或是 yes/no）。

  其中 2–4 合称"没给出答案"，5–9 合称"答了但答错"。

---

## 1. 总账

**Wiki（7 个数据集合计 3,125 题）**

| | GrepSeek（参照） | Search-R1-14B qwen3 | Search-R1-14B e5 | Search-o1 | RAG | AgentIR |
|---|---:|---:|---:|---:|---:|---:|
| 给出答案 | 2,573（82%） | 3,071（98%） | 3,070（98%） | 2,266（73%） | 3,113（100%） | 1,839（59%） |
| ↳ 答对（= EM） | 1,413（45.2） | **1,498（47.9）** | 1,460（46.7） | 1,239（39.6） | 1,077（34.5） | 869（27.8） |
| ↳ 答错 | 1,160（37%） | 1,573（50%） | 1,610（52%） | 1,027（33%） | 2,036（65%） | 970（31%） |
| 没给出答案（算 0 分） | 552（18%） | 54（2%） | 55（2%） | 859（27%） | 12（0.4%） | 1,286（41%） |
| 作答准确率 | 54.9 | 48.8 | 47.6 | 54.7 | 34.6 | 47.3 |

**BCP（500 题，judge 判分）**

| | GrepSeek | Search-R1-14B qwen3 | Search-R1-14B e5 | Search-o1 | RAG | AgentIR |
|---|---:|---:|---:|---:|---:|---:|
| 给出答案 | 306 | 380 | 372 | 109 | 400 | 175 |
| ↳ 答对（= judge 准确率） | 108（21.6） | 45（9.0） | 24（4.8） | 63（12.6） | 72（14.4） | **155（31.0）** |
| 没给出答案 | 194 | 120 | 128 | 391 | 100 | 325 |
| 作答准确率 | 35.3 | 11.8 | 6.5 | 57.8 | 18.0 | **88.6** |

AgentIR 在 BCP 上 65% 的题没给出答案，但给出的答案 89% 正确，所以排第一。Search-R1 正好相反：几乎都作答，作答准确率却只有 6.5–11.8%。

---

## 2. 失败分类（从多到少）

**Wiki 各类失败的题数**（括号内为占该方法失败数的比例）

| 类别 | GrepSeek | Search-R1-14B qwen3 | Search-R1-14B e5 | Search-o1 | RAG | AgentIR |
|---|---:|---:|---:|---:|---:|---:|
| 没见过 gold 答错 | 607（35%） | **943（58%）** | **1,040（62%）** | 419（22%） | 732（36%） | 269（12%） |
| 轮数耗尽 | 347（20%） | 0 | 0 | **797（42%）** | 0 | **1,285（57%）** |
| 答案形式 | 272（16%） | 243（15%） | 225（14%） | 283（15%） | 302（15%） | 513（23%） |
| 见过 gold 答错 | 262（15%） | 357（22%） | 312（19%） | 307（16%） | 219（11%） | 184（8%） |
| 拒答 | 1 | 0 | 0 | 9 | **773（38%）** | 1 |
| 上下文写满 | 205（12%） | 0 | 0 | 2 | 12 | 0 |
| 无答案 | 0 | 54（3%） | 55（3%） | 60（3%） | 0 | 1 |
| 无法判断 | 18 | 30 | 33 | 9 | 10 | 3 |
| **失败合计** | **1,712** | **1,627** | **1,665** | **1,886** | **2,048** | **2,256** |

每个方法的失败，从多到少：

| 方法 | 失败排序 |
|---|---|
| Search-R1-14B | 没见过 gold（约 60%）> 见过 gold 答错（约 20%）> 答案形式（约 15%）> 中途 EOS 无答案（3%） |
| Search-o1 | 轮数耗尽（42%）> 没见过 gold（22%）> 见过 gold 答错（16%）> 答案形式（15%）> 标签拼错导致无答案（3%） |
| RAG | 拒答（38%）> 没见过 gold（36%）> 答案形式（15%）> 见过 gold 答错（11%） |
| AgentIR | 轮数耗尽（57%）> 答案形式（23%）> 没见过 gold（12%）> 见过 gold 答错（8%） |

**BCP 各类失败的题数**

| 类别 | GrepSeek | Search-R1-14B qwen3 | Search-R1-14B e5 | Search-o1 | RAG | AgentIR |
|---|---:|---:|---:|---:|---:|---:|
| 没见过 gold 答错 | 175 | 309 | 320 | 26 | 157 | 14 |
| 轮数耗尽 | 0 | 0 | 0 | 269 | 0 | 325 |
| 上下文写满 / 10K 上限 | 192 | 0 | 0 | 74 | 100 | 0 |
| 无答案 | 2 | 120 | 128 | 48 | 0 | 0 |
| 拒答 | 0 | 0 | 0 | 5 | 158 | 1 |
| 见过 gold 答错 | 19 | 20 | 22 | 14 | 12 | 5 |
| 无法判断 | 4 | 6 | 6 | 1 | 1 | 0 |
| **失败合计** | **392** | **455** | **476** | **437** | **428** | **345** |

---

## 3. 各数据集的特点

数值是占该数据集全部题的百分比，每行合计 100%。BCP 不做"答案形式"分类。

### 3.1 Search-R1-14B

qwen3 / e5 两个检索器分开列出，格式为"qwen3 / e5"。

| 数据集 | 答对 | 没见过 gold | 见过 gold 答错 | 答案形式 | 无答案 | 其他 | 作答准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | 48.6 / 51.2 | 23.8 / 23.6 | **17.6 / 16.4** | 8.2 / 7.6 | 0.6 / 0.2 | 1.2 / 1.0 | 48.9 / 51.3 |
| TriviaQA | 69.0 / 69.2 | 16.0 / 18.8 | 5.2 / 4.2 | 9.6 / 7.6 | 0.2 / 0.2 | 0 / 0 | 69.1 / 69.3 |
| PopQA | 48.6 / 45.0 | 33.4 / 38.0 | 12.4 / 10.4 | 5.4 / 6.6 | 0.2 / 0 | 0 / 0 | 48.7 / 45.0 |
| HotpotQA | 47.0 / 46.6 | 29.2 / 33.2 | 11.4 / 7.8 | 10.0 / 9.8 | 1.6 / 1.8 | 0.8 / 0.8 | 47.8 / 47.5 |
| 2Wiki | 48.4 / 44.0 | 25.2 / 30.6 | 11.6 / 10.6 | 7.0 / 5.6 | **4.8 / 5.8** | 3.0 / 3.4 | 50.8 / 46.7 |
| MuSiQue | 25.0 / 23.0 | **52.6 / 54.6** | 12.0 / 12.2 | 6.6 / 6.2 | 3.2 / 3.0 | 0.6 / 1.0 | 25.8 / 23.7 |
| Bamboogle | 52.0 / 52.0 | 33.6 / 36.8 | 4.8 / 3.2 | 7.2 / 6.4 | 0.8 / 0 | 1.6 / 1.6 | 52.4 / 52.0 |
| BCP | 9.0 / 4.8 | 61.8 / 64.0 | 4.0 / 4.4 | — | **24.0 / 25.6** | 1.2 / 1.2 | 11.8 / 6.5 |

- 每个数据集的最大失败类都是"没见过 gold"。在 MuSiQue 上超过一半：只检索 1–2 次，第一跳没拿到就编造桥接（§4.1）。
- 中途 EOS 集中在 2Wiki（约 5%）和 BCP（约 25%）。
- e5 和 qwen3 两个检索器的形态几乎一样，差别只在 PopQA、HotpotQA、2Wiki 上 e5 的"没见过 gold"多 4–5 个点。

### 3.2 Search-o1（qwen3）

| 数据集 | 答对 | 轮数耗尽 | 上下文写满 | 无答案 | 没见过 gold | 见过 gold 答错 | 答案形式 | 其他 | 作答准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | 34.6 | 12.0 | 0 | 1.8 | 15.4 | **20.0** | **16.0** | 0.2 | 40.1 |
| TriviaQA | 65.0 | 11.6 | 0.2 | 1.6 | 8.0 | 4.8 | 8.4 | 0.4 | 75.1 |
| PopQA | 42.8 | 11.8 | 0 | 1.2 | **24.4** | 9.6 | 9.6 | 0.6 | 49.2 |
| HotpotQA | 38.6 | 28.0 | 0 | 1.6 | 11.0 | 12.0 | 8.0 | 0.8 | 54.8 |
| 2Wiki | 37.2 | 45.2 | 0.2 | 1.6 | 4.8 | 5.4 | 4.8 | 0.8 | **70.2** |
| MuSiQue | 16.2 | **47.4** | 0 | **3.4** | 16.4 | 8.2 | 7.8 | 0.6 | 32.9 |
| Bamboogle | 53.6 | 13.6 | 0 | 3.2 | 15.2 | 5.6 | 8.0 | 0.8 | 64.4 |
| BCP | 12.6 | **53.8** | **14.8** | **9.6** | 5.2 | 2.8 | — | 1.2 | 57.8 |

- 多跳数据集上，轮数耗尽是压倒性的失败：HotpotQA 28%，2Wiki 45%，MuSiQue 47%，BCP 54%。
- 但 2Wiki 上一旦作答，准确率是 70%，所有方法里最高。
- 单跳数据集的失败以答错为主：NQ 见过 gold 仍答错 20%，PopQA 没见过 gold 24%。

### 3.3 RAG（qwen3）

| 数据集 | 答对 | 拒答 | 没见过 gold | 见过 gold 答错 | 答案形式 | 上下文写满 / 上限 | 其他 | 作答准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | 38.8 | 7.6 | 20.6 | **15.8** | **16.8** | 0.2 | 0.2 | 38.9 |
| TriviaQA | 66.0 | 4.0 | 15.0 | 4.2 | 10.6 | 0.2 | 0 | 66.1 |
| PopQA | 41.8 | 9.8 | 25.6 | 8.2 | 14.0 | 0.6 | 0 | 42.1 |
| HotpotQA | 33.2 | 23.2 | 26.8 | 7.0 | 8.8 | 0.6 | 0.4 | 33.4 |
| 2Wiki | 18.8 | **60.4** | 14.6 | 2.8 | 2.8 | 0 | 0.6 | 18.8 |
| MuSiQue | 9.8 | 42.2 | **36.4** | 4.6 | 5.4 | 0.8 | 0.8 | 9.9 |
| Bamboogle | 28.0 | 29.6 | 29.6 | 4.8 | 8.0 | 0 | 0 | 28.0 |
| BCP | 14.4 | 31.6 | 31.4 | 2.4 | — | **20.0** | 0.2 | 18.0 |

- 单跳数据集上 RAG 正常，失败以没检索到为主。
- 多跳数据集上拒答占主导：2Wiki 60%，MuSiQue 42%，Bamboogle 30%。这些题只检索一次，只取回了一边的实体，reader 就拒绝作答（§4.3）。
- 这里的"作答准确率"把拒答也算作作答。

### 3.4 AgentIR

| 数据集 | 答对 | 轮数耗尽 | 答案形式 | 没见过 gold | 见过 gold 答错 | 其他 | 作答准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | 15.4 | 31.4 | **31.0** | 10.8 | **11.0** | 0.4 | **22.4** |
| TriviaQA | 52.4 | 20.2 | 17.6 | 8.0 | 1.8 | 0 | 65.7 |
| PopQA | 28.8 | 32.2 | 19.2 | **15.8** | 4.0 | 0 | 42.5 |
| HotpotQA | 31.0 | 37.0 | 15.4 | 6.8 | 9.4 | 0.4 | 49.2 |
| 2Wiki | 24.6 | 60.0 | 10.6 | 1.8 | 3.0 | 0 | 61.5 |
| MuSiQue | 10.8 | **67.8** | 6.8 | 8.6 | 5.8 | 0.2 | 33.8 |
| Bamboogle | 43.2 | 33.6 | 8.0 | 8.0 | 7.2 | 0 | 65.1 |
| BCP | 31.0 | 65.0 | — | 2.8 | 1.0 | 0.2 | **88.6** |

- 每个数据集的最大失败类都是轮数耗尽，从 TriviaQA 的 20% 到 MuSiQue 的 68%。
- 在 NQ 上，答案形式（31%）几乎和轮数耗尽（31%）一样多：NQ 的答案平均 14.8 个词，EM 几乎不可能命中。

### 3.5 横向看：哪个数据集更容易出现哪种失败

| 失败类 | 最严重的数据集（各方法一致） | 原因 |
|---|---|---|
| 没见过 gold | **MuSiQue**（SR1 53–55%，RAG 36%，GrepSeek 26%）；**BCP**（SR1 62–64%） | 2–4 跳组合，第一跳没取到后续全错；BCP 是改写过的多约束描述 |
| 轮数耗尽 | **MuSiQue、2Wiki**（AgentIR 68 / 60%，Search-o1 47 / 45%，GrepSeek 21 / 21%） | 比较题和组合题至少要 4 次查找，6 次调用不够 |
| 答案形式 | **NQ**（AgentIR 31%，RAG 17%，Search-o1 16%，GrepSeek 14%） | NQ 的 gold 写法杂：别名、列表、日期粒度 |
| 见过 gold 答错 | **NQ**（Search-o1 20%，SR1 16–18%，RAG 16%，GrepSeek 13%） | 同上，再加上 NQ 的标注噪声（§5.4） |
| 拒答 | **2Wiki、MuSiQue**（RAG 60 / 42%） | 单次检索只覆盖比较题的一边 |
| 无答案 | **BCP、2Wiki**（SR1 BCP 25%，2Wiki 5%；Search-o1 BCP 10%） | Search-R1-14B 中途 EOS；Search-o1 标签拼错 |
| 上下文写满 | **BCP**（GrepSeek 38%，RAG 20%，Search-o1 15%） | 长网页加长思考 |

结论：**NQ 的失败有相当一部分是评测口径问题**，所有方法在 NQ 上的形式类和"见过仍错"类都最高；**2Wiki 和 MuSiQue 的失败主要是预算问题**；**BCP 和 MuSiQue 的失败主要是检索问题**。

---

## 4. 各方法的失败机理

### 4.1 Search-R1-14B

**行为画像**
- 单跳题平均检索 1.0 次，多跳题约 2 次，从来用不完预算。首个 query 通常照抄问题（单跳抽样 22/22）。
- 拿到 `<information>` 后，104 次生成里有 96 次以 "Based on the search results, …" 开头，紧接着直接断言一个结论。抽样里从没有一次承认"文档没有提到"，也从没有对同一跳换个写法重查。

**没见过 gold → 编造桥接**（第一大类）：第一跳没取到目标实体时，用表面相似的文档顶替，然后继续第二跳。

| 案例 | 怎么编的 |
|---|---|
| 2Wiki `dev_8936` | 从 "Far from Men" 条目编出 'Farfallon' 的导演 |
| 2Wiki `dev_10374` | 把 "The Prince and the Pauper (1977)" 的导演当成 The Pauper Millionaire 的导演 |
| MuSiQue `dev_1111` | 从 "Mark White" 条目编出 "Mark Day was born in Union City" |
| Bamboogle `test_66` | top-3 是 Yangtze、Orinoco、Amazon，却断言 Congo 是第三长河 |
| HotpotQA `dev_2043`、MuSiQue `dev_1503` | Mediastan → "Michael Moore"；NBC → Bob Schieffer |
| BCP 209、963、102、1217 | 答出的实体或数值不在返回的摘录里 |

同一个实体，E5 和 Qwen3-Emb 的结果可以完全不同。MuSiQue `dev_1116`：E5 返回无关条目，Qwen3-Emb 取回了 3 段 Hod Lisenbee。

**见过 gold 答错**（第二大类）
- 选错实体、被干扰项带偏：
  - Bamboogle `test_65` 把 top-1 的 Delhi 当成人口最多的首都；
  - NQ `test_1993` 用第 6 届委员会主席回答第 7 届；
  - 2Wiki `dev_10976` 混淆了 1934 和 1953 两个版本；
  - Bamboogle `test_57` 被"CN Tower 最高"这一过时说法带偏（GrepSeek 这题答对了）。
- 推理和答案不一致：
  - 2Wiki `dev_7467` 推理写的是 "Cheryl Dunye was born later"，答的却是另一部片；
  - NQ `test_2967` think 里写 "channel 15"，答的却是 "KNXV-TV"。
- 抽样的 16 个"见过 gold 答错"里，10 个其实是评测伪影（EM 假阴性或标注噪声）。

**无答案：模型自己在句子中间发 EOS**

| | e5 | qwen3 | bm25 | Search-R1-7B |
|---|---:|---:|---:|---:|
| Wiki 无答案 | 55 | 54 | 49 | 每个 run ≤ 2 |
| BCP 无答案 | 128 | 120 | 117 | 每个 run ≤ 2 |

- 每一例都紧跟一个 `<information>` 块。下一轮生成写了 14–62 个 token 就自己发出 EOS：不是 stop string，也不是长度截断，prompt 只有 655–3,444 token。
- 8/10 正好断在要写出实体、数字或列表项的位置。例如 2Wiki `dev_3813` 断在 `<search> When were directors Tony Ayres and`，MuSiQue `dev_588` 断在 "Based on the search results, Daniel M."。
- 只有 14B 这样，与检索器无关。presence_penalty 的影响见 §5.2：放大了约 1.6 倍，但不是主因。

**BCP**
- 平均 1.27 次检索，16 轮预算基本没用。
- 官方 `_passages2string` 把文档首行当标题，而 BCP 页面以 YAML `---` 开头，所以每篇显示成 `Doc i(Title: ---)`。这与官方代码一致，不影响计分，但输入形式偏离了它的训练分布。

### 4.2 Search-o1

**轮数耗尽（第一大类）：停不下来**
- 抽样的 16 个轮数耗尽，最后一个动作全部是检索。
- **协议因素**：检索上限 = 轮数 − 1。第 6 轮还在发 query 只会收到 LIMIT_MESSAGE，之后没有下一轮。官方配置是 15 轮、检索上限 5，撞上限后还有余量作答。我们这样设置是为了所有方法预算一致（见 `docs/deviations.md`），代价是 Search-o1 失去了这个隐式的强制作答。
- **检索停滞（12/16）**：同义改写对 dense 检索几乎是同一个查询，反复拿到同一组 top-3。
  - NQ `test_2328` 5 次查询返回同样的 3 段；
  - 2Wiki `dev_1121` 第 2–5 次查询的 top-3 完全相同。
- **证据已有却不收敛（4/16）**：
  - HotpotQA `dev_1374` 文档写着 Boston Celtics，模型因为参数知识里此人后来被交易，继续查 "2025 active roster"；
  - MuSiQue `dev_192` 的 RiD 已给出答案，模型又查了 3 次。
- **时间错位**：模型默认"现在"是 2024/25 年，语料是 2018 年左右的 Wikipedia（4 例改变了结局）。
- 已经见过 gold 却耗尽预算：Wiki 255 题，BCP 61 题。

**无答案：检索标签拼错**

| 检索器 | Wiki 有拼错标签的题 | 其中无答案 | Wiki 无答案总数 | BCP 有拼错标签的题 |
|---|---:|---:|---:|---:|
| bm25 | 108 | 45 | 71 | 41 |
| e5 | 92 | 38 | 58 | 35 |
| qwen3 | 105 | 45 | 60 | 38 |

- 典型拼法有 `|end_search_query>`、`<|end_search_query>`：结束标签里与开始标签共享的片段被换掉了。
- 后果是 stop string 不触发、模型随即结束，记为无答案；或者模型自己虚构一个"检索结果"接着往下写（TQ `test_8445`）。
- 抽样的 16/16 个无答案都是格式失败。presence_penalty 使拼错约多 2.8 倍（§5.2）。

**上下文写满与失控生成**
- 2Wiki `dev_8359`：0 次检索，在思考里"假装检索"陷入 "Wait, no." 循环，一直写到 15,976 token。
- BCP 723 首轮生成 130,558 token 的枚举；BCP 927 末轮写了 124,166 token 的候选循环。
- TQ `test_1937`：RiD 写满上限，按官方的 "最后一个 `**Final Information**` 之后" 规则抽取，取到的是思考残片，被插回主链。

**Reason-in-Documents（RiD）**
- "No helpful information" 大多是如实的。
- 问题主要有三类：
  - 只按当前子查询过滤，子查询一错证据就被藏起来（MuSiQue `dev_281`）；
  - RiD 对了、主模型却改写了（Bamboogle `test_96` 的 "Claude, Sr." 被写成 "Claude"）；
  - 偶发幻觉（MuSiQue `dev_465`）。
- 成功案例：2Wiki `dev_8736`，RiD 一次检索就融合了两跳。GrepSeek 这题 4 次调用后上下文写满。

**官方答案抽取的副作用**：
- `\boxed{9:00 \text{ AM}}` 被抽成 ' AM'；
- 模型复述模板时预测被抽成 'YOUR_ANSWER'（HotpotQA `dev_3`）。

例数很少。

### 4.3 RAG

**拒答（第一大类）**
- reader prompt（`prompts/rag.py`，项目自定义）是 "Answer the given question using the retrieved documents below, which may or may not be relevant…"。Qwen3.5-9B 把它理解成"不许用参数知识"，top-3 里没有答案时就输出 "Not mentioned"。
- 同一个模型在 Direct 下几乎不拒答（2Wiki 4/500，MuSiQue 7，HotpotQA 1，Bamboogle 0）。
- 推理里的原话：
  - HotpotQA `dev_1691`："'Not mentioned' is the text-grounded answer"；
  - Bamboogle `test_8`："Prioritize constraint. Output: Not mentioned"。

  这两题 Direct 都答对了。
- 结果是 RAG 在 2Wiki（18.8 对 Direct 29.8）和 Bamboogle（28.0 对 48.0）上反而低于不检索：

  | 数据集 | Direct 对、RAG 错 | RAG 对、Direct 错 |
  |---|---:|---:|
  | 2Wiki | 94 | 39 |
  | Bamboogle | 35 | 10 |
  | PopQA | 28 | 132 |
  | NQ | 33 | 109 |

- 对比较的影响：RAG 是衡量"检索有没有用"的参照线，拒答让它在多跳题上被低估。prompt 已冻结并对三个检索器一致，可以不动，但论文里应说明。

**与旧版（v1）RAG 相比为什么大幅下降**

旧版 2Wiki 是 36–39、Bamboogle 是 43–56，新版是 16–19 和 21–30。对比旧版的 `prompts/rag.py` 和 `configs/baselines.yaml`（`main` 分支）：

| | 旧版（v1） | 新版（v2） |
|---|---|---|
| reader prompt | "Use the passages to answer the question. **If the passages do not contain the answer, use your parametric knowledge.**" | "Answer the given question using the retrieved documents below, which may or may not be relevant." 没有允许用自己的知识 |
| 解码 | temperature 0，思考上限 4,096 token | temperature 0.6，presence_penalty 2.0，不限思考 |
| top-k、模型 | top-3，Qwen3.5-9B | 相同 |
| 题目 | 完整 split | 每个数据集 500 题子集（Bamboogle 125） |

反事实检查：把 RAG 每个拒答换成 Direct 对同一题的答案，再算 EM。

| 检索器 | 2Wiki：拒答题数 / RAG EM / 换成 Direct 后 / 旧版 | Bamboogle：拒答题数 / RAG EM / 换成 Direct 后 / 旧版 |
|---|---|---|
| BM25 | 326 / 15.8 / **34.8** / 36.2 | 51 / 20.8 / **33.6** / 43.2 |
| E5 | 310 / 17.8 / **34.4** / 36.4 | 38 / 30.4 / **41.6** / 56.0 |
| Qwen3 | 302 / 18.8 / **35.0** / 38.8 | 37 / 28.0 / **44.0** / 49.6 |

- **主因是 prompt**：
  - 只要把拒答换成 Direct 的答案，2Wiki 就回到和旧版相差 2–4 个点以内，Bamboogle 也收回了大部分差距。
  - 旧版 prompt 明确允许"文档里没有就用自己的知识"，新版没有这句话。Qwen3.5-9B 在思考模式下把新版 prompt 读成"只能依据文档"。
- **不是解码，也不是子集**：Direct 用的是同一组新旧解码，新旧分数只差 ±2.5（如 2Wiki 27.6 → 29.8）。
- **剩余的差距**：Bamboogle 只有 125 题，95% 置信区间约 ±8.7；另外反事实估计偏保守，因为旧版 RAG 可以把文档里的部分线索和自己的知识结合起来，而这里只是直接换成 Direct 的答案。
- **BM25 拒答更多**（NQ 137 题，Qwen3 只有 38 题）：BM25 取回的文档更不相关，触发拒答也更多。
- **对比较的影响**：RAG 是衡量"检索有没有用"的参照线。当前的 prompt 让它在多跳题上被明显低估。要不要换回允许用自身知识的措辞并重跑 RAG，需要你决定（单次调用，成本很低）。

**没见过 gold（第二大类）**
- 多跳题的 top-3 常被第一跳实体占满：MuSiQue `dev_1778` 的 3 段全是 Pacem in terris。
- 60 个 Wiki 抽样里，43 个的 top-3 至少有 2 段来自同一条目。

**见过 gold 答错**
- 干扰文档压过了正确的先验：Bamboogle `test_70` 答 "Chester W. Nimitz"，Direct 答对了 Gerald R. Ford。
- 答错槽位：HotpotQA `dev_5752`。
- 两头押注：PopQA `test_7260`。

**失控**：Wiki 有 12 次单次 reader 调用写满 16K。BCP 有 100 次撞到 10K 上限，形态是候选逐字母枚举和自检循环。

### 4.4 AgentIR

**轮数耗尽（第一大类）：答案在手仍在核验**
- 抽样的 16 例都是到最后一轮还在搜。其中 9 例本可以在预算内提交正确答案：
  - Bamboogle `test_20` 第 5 次检索已找到答案，reasoning 写着 "Excellent! I found it!"，第 6 次仍去 verify；
  - BCP 295 最后 4 次用几乎相同的 query 核验一个次要约束。
- 已经见过 gold 却耗尽预算：Wiki 529 题，BCP 119 题。
- 14/16 有近乎重复的 query（只差引号、大小写或一个猜测的年份），后期每次 top-3 至少有 2 篇是已经见过的文档。
- 丢弃并行调用（每次回复只执行第一个工具调用，这是公平性设置）只在 2Wiki 上轻度有害，每题最多损失约 1 轮。

**答案形式（第二大类）**
- 官方 QUERY_TEMPLATE 是为 judge 设计的（"Explanation / Exact Answer / Confidence"），没有短答约束。
- Wiki 答案平均 6.7 个词，NQ 平均 14.8 个词。常见写法：
  - yes/no 写成整句；
  - 并列多个实体，如 "UNESCO and the International Labour Organization (ILO)"；
  - 括号别名，如 "Lockheed (SR-71 Blackbird)"。
- 抽样的 20 个"答案包含 gold"型错答里，19 个语义上是对的。

**reasoning-aware 检索的副作用**
- **前缀拖拽**（抽样至少 8 例）：embedding 输入里大部分是 reasoning，query 只占 2–15%。所以 query 已经换成新实体，返回的仍是上一步的文档。
  - 例：2Wiki `dev_11843` 第 2 次 query 返回的 3 篇全是上一步的 Wings for the Eagle 和 Lloyd Bacon；
  - 第 3 次 query 只改了大小写、换了一段 reasoning，返回的 3 篇就完全不同。
- **幻觉被放大**：2Wiki `dev_6253`。在任何检索之前就臆造出一位导演，之后每轮检索都被拉回此人。
- 把 dense 检索当关键词引擎用：16 个轮数耗尽案例里 11 个用了引号 query。
- 罕见专名查不到：PopQA `test_3789`、2Wiki `dev_11843`，这两题 GrepSeek 都答对了。

**BCP 为什么第一**
- 首轮把 3–4 个约束改写成一句自然语言，dense 检索能匹配改写过的描述。BCP 87 第 1 次检索就把 gold 文档排在第 2。
- 以中间实体为跳板构造定向 query（BCP 1097）。
- 一旦提交，89% 判对。

**协议异常**（约 7 / 3,625，非系统性）
- MuSiQue `dev_1453`：末轮工具调用 XML 畸形，被当成最终回答。
- NQ `test_1219` 等 5 例在思考里写了字面的 `<tool_call>`，导致 reasoning 被截断。

---

## 5. 跨方法的评测与协议问题

### 5.1 EM 对 cover-EM（Wiki）

| run | EM | cover-EM | 差 | 错答中"并列多个候选" | 错答中"答案包含 gold" |
|---|---:|---:|---:|---:|---:|
| GrepSeek | 45.2 | 49.5 | +4.3 | 70 | 65 |
| Search-R1-14B qwen3 | **47.9** | **51.4** | +3.5 | 27 | 81 |
| Search-R1-14B e5 | 46.7 | 49.6 | +2.9 | 22 | 68 |
| Search-o1 qwen3 | 39.6 | 45.0 | +5.4 | 106 | 61 |
| RAG qwen3 | 34.5 | 40.4 | +5.9 | 104 | 81 |
| AgentIR | 27.8 | 42.8 | **+15.0** | 283 | 187 |

- 两种口径下排名基本一致，只有 AgentIR 从垫底升到与 RAG 相当。
- 建议主表用 EM，附录给 cover-EM 或 LLM judge 的敏感性分析。
- 抽样中还有几类 EM 处理不了的口径问题：
  - "American" 对 "United States"；
  - "fourth" 对 "4"；
  - "12 episodes" 带单位；
  - 标准化把 "/" 直接删掉，"Musician/Singer" 变成 "musiciansinger"。

### 5.2 共用解码（presence_penalty 2.0）的副作用：对照 pilot

用正式运行的同一份 harness，只把 presence_penalty 从 2.0 改成 0（`scripts/decoding_pilot2.py`，比较脚本 `scripts/compare_pp2.py`）。

| run / 数据集 | 指标 | pp 2.0（正式） | pp 0（pilot） | EM（2.0 → 0） | McNemar |
|---|---|---:|---:|---:|---|
| Search-R1-14B e5 / 2Wiki | 无答案；每千轮中途 EOS | 29；19.6 | 17；11.5 | 44.0 → 44.0 | 36 / 36，p = 1.0 |
| Search-R1-14B e5 / BCP | 无答案；每千轮中途 EOS | 128；112.7 | 80；69.7 | 未跑 judge | — |
| Search-o1 bm25 / NQ | 有拼错标签的题；无答案 | 25；12 | 9；4 | 27.0 → 28.0 | 26 / 31，p = 0.60 |

- **penalty 确实放大了格式失败**：标签拼错约 2.8 倍，中途 EOS 约 1.6 倍。原因是这些标签不是 special token，被切成子词后，结束标签与开始标签共享的子词受到了惩罚。
- **但它不是主因，对 EM 的影响在噪声内**：
  - 去掉 penalty 后，这两种失败依然存在；
  - 救回来的题多数本来也答不对；
  - penalty 同时也在抑制失控循环：Search-o1 NQ 的上下文写满在 pp 0 下从 1 次升到 9 次。
- **只有 BCP 可能有实际影响**：Search-R1-14B 去掉 penalty 后多作答 48 题。按它在 BCP 上的作答准确率粗估，judge 准确率约 +0.6，排名不变。
- **建议**：维持统一解码，在论文里写成 caveat。

### 5.3 预算耗尽时不强制作答
- 所有方法共用 6 次（BCP 16 次）assistant 调用，都没有末轮强制作答，对所有方法一致。
- 受影响最大的是爱核验的 agent：AgentIR 41%、Search-o1 27%、GrepSeek 18%。Search-R1 从不用完预算。
- 分析时应区分"答不出"和"没来得及答"。

### 5.4 标注噪声（所有方法都错，或者预测更对）

| 数据集 | 案例 |
|---|---|
| NQ | `test_727`、`test_718`、`test_1553`、`test_1128`、`test_2003`、`test_2537`（钟面上的 4 常写作 IIII，gold 只有 IV）、`test_1173` |
| PopQA | `test_9057`、`test_8000`（问 Delhi 是哪里的首都，gold 只有历史政权）、`test_11035`（四色旗，gold 只收 red） |
| HotpotQA | `dev_1528`、`dev_633`（gold 是整句）、`dev_2549`、`dev_6989`（gold 是出生全名）、`dev_926` |
| MuSiQue | `dev_71`、`dev_1241`（时效性） |
| TriviaQA | `test_4131`（gold 已过时） |
| Bamboogle | `test_62`（gold 晚于语料） |

### 5.5 BCP judge 与分析启发式的局限
- **judge 偏严**（只写机制）：
  - AgentIR 的 BCP 192，预测与 gold 只差一个称谓前缀，judge 的 reasoning 写了 "the name is the same"，仍然判错；
  - Search-o1 的 BCP 71、72 也类似；
  - 反过来，BCP 1135 的 F1 为 0 却被判对。

  建议人工抽检约 50 条 judge 判错的记录。
- **"见过 gold"启发式**：
  - 比较题必然误报，因为 gold 就是题中的实体；
  - 短 gold 会误报；
  - 非 ASCII 的 gold 会漏报，Wiki 有 277 题。

  所以"见过 / 没见过"两类只适合看量级。

---

## 6. 对 GrepSeek 的启示

1. **别人赢 GrepSeek 的地方，主要靠语义匹配**：
   - 改写过的线索：TQ `test_3310`、`test_4681`，MuSiQue `dev_153`；
   - 拼写变体：NQ `test_1216` 的 Neel / Nil Darpan，PopQA `test_13618` 的 Oporto / Porto；
   - 一次就把两跳放进同一个 top-3：RAG、Search-o1 的 RiD；
   - BCP 上的多约束改写：AgentIR。

   对应 GrepSeek 报告的 §4.1.4、§4.3 和 §4.2。最直接的办法是 grep 落空时允许一次语义回退，或者在工具里提供对大小写、重音、词干都不敏感的匹配。
2. **别人输给 GrepSeek 的地方，是精确性**：
   - Search-o1 把 Chey Chettha III 检索成 IV/V，把 Cryptanthus 检索成 Cryptantha；
   - AgentIR 查不到罕见专名；
   - Search-R1 编造桥接。

   混合检索必须保住这块优势。
3. **别学别人的坏习惯**：AgentIR 的过度核验、Search-o1 的同义改写空转、RAG 的"文档没写就拒答"、Search-R1 的编造桥接。GrepSeek 已经有前两种的变体：见过 gold 仍继续搜；AND 过滤滤空后只换个词再试。
4. **答案形式**：
   - Search-R1 的答案最短、最接近 EM 的 gold 写法，并列答案最少；
   - GrepSeek 用 F1 奖励，并列答案是 Search-R1 的 2–3 倍。
   - 改成 EM 奖励，或加上"只给一个实体"的约束，就能收回这部分分数。

---

## 附录：数字从哪来

脚本都在 `reports/scripts/`，在计算节点上运行。

| 内容 | 脚本 |
|---|---|
| §1–3 的总账、失败分类、各数据集表 | `failure_accounting.py`、`ledger_tables.py` |
| 每题检索次数、见过 gold 后耗尽预算 | `failure_analysis.py`（`summary.json`） |
| EM 对 cover-EM、答案形式 | `answer_form.py` |
| Search-R1 中途 EOS | `sr1_no_answer.py` |
| Search-o1 标签拼错 | `tag_errors.py` |
| 解码 pilot | `decoding_pilot2.py`、`compare_pp2.py`；数据在 `/scratch/a32du/scaleseek_v2/dev/pilots/` |

抽样案例：`/scratch/a32du/scaleseek_v2/analysis/failures/cases/others/<run>/<ds>/<category>/<id>.txt`。
