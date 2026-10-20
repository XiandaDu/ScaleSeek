# GrepSeek 失败案例分析（ScaleSeek v2 正式运行）

2026-10-20。对象：官方 checkpoint `alireza7/GrepSeek-Qwen3.5-9B-GRPO`，运行在官方 `inference/` harness 上。7 个 Wiki 数据集共 3,125 题，BrowseComp-Plus（下称 BCP）500 题。

**做法**
- 对全部 3,625 条轨迹做自动特征抽取。每次 grep 调用记录：命令、返回行数、是否为空/截断/超时、gold 是否出现在输出里，以及每轮思考的 token 数。
- 人工通读 206 条分层抽样的完整轨迹（每个数据集 26 条，覆盖各失败类别和"GrepSeek 独对"）。
- 另外逐条看了 78 道"GrepSeek 错、强基线对"的抽样题在 GrepSeek 一侧的命令序列（§5）。

**BCP 写法**：BCP 有 canary 保护，本文只写 query ID 和机制，不复述题面、答案或文档内容。

数字的出处见附录 A，案例索引见附录 B。

---

## 0. 结论速览

1. **一半的失败不在"搜不到"，而在预算和收尾。** Wiki 共 1,712 个失败：

   | 失败类型 | 数量 | 说明 |
   |---|---:|---|
   | 预算耗尽 | 552 | max_turns 347 + 上下文写满 205；其中 139 个**已经在工具输出里见过 gold** |
   | 见过 gold 仍答错 | 443 | |
   | 从未见过 gold | 698 | |
   | gold 无法匹配检查 | 19 | |

2. **工具返回自己把上下文吃掉了。**
   - 每次调用的 JSON 里，`stdout` 之外还有一份逐行拷贝的 `information_lines`。平均 977 token 的 stdout 因此变成 2,157 token 的 payload（2.2 倍）。
   - Wiki 窗口是 16K：4 次被截断的调用（每次 2,048 token stdout）就装不下了。
   - 205 条 Wiki 和 192 条 BCP 的上下文写满记录，若在写满那一刻去掉这份重复，**全部还装得下**（静态估计）。
3. **BCP 上一次 grep 基本只看到一篇文档的开头。**
   - BCP 语料一行一篇网页，中位 10.5 KB、均值 33 KB。每次调用截断在 6,000 token。
   - 79% 的非空调用只显示**一篇文档的前一部分**，16% 显示一篇半。
   - 显示顺序是语料文件顺序，不是相关度。
4. **grep 的检索语义是：精确子串 + 文件顺序 + AND 过滤。**
   - 9,449 次 Wiki 调用全部是 `rg -F A corpus.jsonl | rg -i -F b | … | head -n K` 这一种写法。
   - 从未用过 `-w`、`-o`、`wc`；锚定标题只有 1 次。
   - 20% 的调用返回空。把空返回在语料上重放：52% 是第一段本来有命中、被 AND 过滤全滤掉了（如 "death toll"、"nickname"、"director" 对不上原文的 "directed by"）；27% 是第一段的字符串在语料里根本不存在。
   - 36% 的调用被 `head` 截满：看到的是文件最前面的 K 行，而不是最相关的 K 行。
5. **大小写和 Unicode 让实体锚点失效。**
   - 第一段（实体锚点）99.8% 是区分大小写的 `-F`；`-i` 只出现在后面的过滤段。
   - 语料把非 ASCII 字符存成 `\uXXXX` 转义，所以带重音的名字（Valentín、Luis García Berlanga、Éric Rohmer）原样 grep 永远为空。
   - 模型随后会断定"语料里没有"，转而凭记忆作答。
   - 重放实验：含非 ASCII 字符的命令有 275 次，其中 **273 次返回空**。只差大小写或只差转义的空返回共 387 次，占全部空返回的 20%（2Wiki 是 38%）。这些都可以零成本修掉（§3.4）。
6. **推理失控。**
   - 612 题出现过单轮 ≥2,000 token 的思考，这些题只有 11% 答对，其余题是 48%。
   - 265 个"无动作"轮次（既没有工具调用也没有答案）中，264 个是思考一直写到窗口末尾。
7. **答案形式（EM 口径）。**
   - Wiki 的 1,160 个错答里，有 272 个（23%）属于形式问题。
   - 典型是并列多个候选，如 "Ross Parker and Hughie Charles"，共 70 个；Search-R1-14B 同类只有 22–27 个。其次是只答年份（"1789" 对 "April 30, 1789"）、过度具体或不够具体。
   - 训练奖励是 F1，这类答案能拿部分分，所以 RL 没有压力把答案收敛到单一实体。

**改进优先级**（详见 §6）：

1. 去掉 payload 里的重复字段，并在预算用尽时强制作答。
2. BCP 按段落切行，或改用 `rg -o` 取片段窗口。
3. 第一段默认大小写不敏感；语料存成 UTF-8。
4. 提供标题锚定、候选标题枚举、命中排序。
5. 给每轮思考设上限。
6. 奖励改成 EM，或惩罚并列答案。
7. 扩展训练分布：多跳、长文档、16 轮。
8. 与 dense 检索混合：§2.3 的 oracle 显示两者互补性很大。

---

## 1. 影响解读的设置

**评测时**
- **Harness 与解码**：官方 `inference/` 未改动。解码是 Table 5 的 temperature 0.6 / top_p 1.0，加所有方法共用的 presence_penalty 2.0。每轮不设 token 上限（`max_tokens_per_turn: 0`）。
- **预算**：Wiki 最多 6 次 assistant 调用、16K 窗口；BCP 最多 16 次、131K 窗口。工具输出按 tokenizer 截断：Wiki 2,048，BCP 6,000。截断后 stdout 末尾追加一行 `[... output truncated at N tokens]`。
- **工具 payload**：`{command, stdout, stderr, exit_code, timed_out, truncated, information_lines}`，其中 `information_lines` 是 stdout 的非空行列表，与 stdout 内容重复。SFT 数据生成和 RL 工具用的是同一格式，所以这是训练分布的一部分，不是我们的 harness 引入的。
- **System prompt**（`inference/agent.py`，与 cold-start SFT 数据逐字节一致）：
  - 说语料是 "one Wikipedia passage per line"、"21 million lines"，建议 `| head -n 3`，最多到 `| head -n 8`。
  - 给了三个示范：`rg -F "distinctive phrase" corpus.jsonl | head -n 3`、`rg -F "phrase1" corpus.jsonl | rg -i -F "phrase2" | head -n 3`、`… | wc -l`，并提到 `-w`。
  - 答案要求是 "concise — typically a short noun phrase, name, or date"，没有要求只给一个实体。
  - **BCP 用的也是这份 prompt**（`build_system_prompt()` 不带参数）：模型被告知自己在 grep 2,100 万行 Wikipedia 段落，实际一行是一整篇网页。

**训练时**
- GRPO，5 个 rollout，奖励为 F1 加格式门槛。
- 16K 上下文（1,024 prompt + 15,360 response），最多 5 次工具调用加 1 次作答。
- 只用 NQ + HotpotQA，只有 Wikipedia。
- 所以 BCP（16 轮、长网页）和 2Wiki/MuSiQue 式的组合问题都在训练分布之外。

**复现与计分**
- 我们的 F1 与论文一致：

  | 数据集 | NQ | TQ | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 我们 | 51.9 | 78.7 | 50.4 | 59.3 | 51.7 | 30.0 | 60.8 |
  | 论文 | 52.2 | 76.7 | 48.6 | 62.3 | 51.8 | 30.1 | 62.1 |

- 只有 `finish_reason == answer` 的记录计分。Wiki 用 EM，BCP 用 Qwen3-32B judge，judge 只看最终答案。
- 上下文写满在 vLLM 0.25 上表现为 `api_error`，计分时归为 `context_exhausted`。

## 2. 结果总览

### 2.1 主表

Wiki 列为 EM，BCP 列为 judge accuracy。下表只列强基线，全表见 `runs/<ds>/REPORT.md`。

| run | Bamboogle | PopQA | TriviaQA | NQ | HotpotQA | 2Wiki | MuSiQue | BCP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GrepSeek** | 51.2 | 44.4 | **70.4** | 39.0 | **47.6** | 46.8 | 21.6 | 21.6 |
| Search-R1-14B e5 | 52.0 | 45.0 | 69.2 | **51.2** | 46.6 | 44.0 | 23.0 | 4.8 |
| Search-R1-14B qwen3 | 52.0 | **48.6** | 69.0 | 48.6 | 47.0 | **48.4** | **25.0** | 9.0 |
| Search-o1 qwen3 | **53.6** | 42.8 | 65.0 | 34.6 | 38.6 | 37.2 | 16.2 | 12.6 |
| RAG qwen3 | 28.0 | 41.8 | 66.0 | 38.8 | 33.2 | 18.8 | 9.8 | 14.4 |
| AgentIR | 43.2 | 28.8 | 52.4 | 15.4 | 31.0 | 24.6 | 10.8 | **31.0** |

GrepSeek 在 TriviaQA 和 HotpotQA 上第一，在 BCP 上第二。NQ 落后 Search-R1-14B 12 个点，是差距最大的一项。

### 2.2 GrepSeek 失败类别

"见过 gold" = 某次工具输出里出现过 gold 字符串。

| 数据集 | 对 | 错，见过 gold | 错，没见过 gold | max_turns | 上下文写满 | 每题调用 | 空返回 | 截断 | 超时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bamboogle (125) | 64 | 6 | 34 | 13 | 7 | 3.38 | 50 | 72 | 15 |
| PopQA (500) | 222 | 64 | 123 | 30 | 61 | 2.67 | 262 | 219 | 15 |
| TriviaQA (500) | 352 | 43 | 79 | 20 | 6 | 2.25 | 243 | 110 | 9 |
| NQ (500) | 195 | 124 | 140 | 23 | 14 | 2.65 | 249 | 156 | 16 |
| HotpotQA (500) | 238 | 83 | 110 | 47 | 19 | 2.93 | 255 | 139 | 14 |
| 2Wiki (500) | 234 | 54 | 66 | 107 | 30 | 3.75 | 502 | 234 | 15 |
| MuSiQue (500) | 108 | 69 | 146 | 107 | 68 | 3.79 | 355 | 263 | 15 |
| **Wiki 合计 (3,125)** | **1,413** | **443** | **698** | **347** | **205** | 3.02 | 1,916 | 1,193 | 99 |
| BCP (500) | 108 | 19 | 175 | — | 192 | 6.77 | 300 | 3,009 | 0 |

另外：Wiki 有 19 个错答的 gold 太短或是 yes/no，无法检查是否见过；BCP 有 4 个这样的错答，另有 2 个 parse_error。

各数据集的瓶颈不同：

| 数据集 | 主要失败 |
|---|---|
| NQ | 见过 gold 仍答错（124，多为答案形式和标注口径，见 §3.7） |
| PopQA | 上下文写满（61）和歧义标题（§3.5） |
| 2Wiki / MuSiQue | 轮数耗尽（各 107） |
| BCP | 上下文写满（192） |

### 2.3 和强基线的互补性

强基线集合：Search-R1-14B e5/qwen3、Search-o1 qwen3、RAG qwen3、AgentIR。

| 数据集 | GrepSeek 错，≥1 强基线对 | GrepSeek 对，强基线全错 | 15 个 run 任一答对 |
|---|---:|---:|---:|
| Bamboogle | 29 | 3 | 98 / 125 |
| PopQA | 99 | 21 | 346 / 500 |
| TriviaQA | 48 | 9 | 414 / 500 |
| NQ | 126 | 14 | 347 / 500 |
| HotpotQA | 86 | 12 | 350 / 500 |
| 2Wiki | 123 | 27 | 403 / 500 |
| MuSiQue | 96 | 23 | 233 / 500 |
| BCP | 142 | 46 | 287 / 500 |

- 在 Wiki 上，"GrepSeek 错、强基线对"共 607 题，按 GrepSeek 的结局分：

  | GrepSeek 的结局 | 题数 | 占比 |
  |---|---:|---:|
  | 见过 gold 仍答错 | 192 | 32% |
  | 没见过 gold | 200 | 33% |
  | 预算耗尽（max_turns 118 + 上下文写满 86） | 204 | 34% |
  | gold 无法匹配检查 | 11 | 2% |

  也就是说，三分之二不是检索问题。
- 两组一对一的对比（GrepSeek 错而对方对 / GrepSeek 对而对方错）：
  - NQ 上对 Search-R1-14B e5：97 / 36。
  - BCP 上对 AgentIR：110 / 63。

---

## 3. 失败机理

按对改进的价值排序。每条给出量化数字、机理和代表案例。

### 3.1 预算：上下文和轮数（Wiki 552 题，BCP 192 题）

**上下文写满**

| | 写满时平均调用次数 | 思考 token | payload token |
|---|---:|---:|---:|
| Wiki（205 题） | 4.09 | 3,770 | 12,866 |
| BCP（192 题） | 8.76 | 32,125 | 100,800 |

- 重复的 `information_lines` 平均每条占掉 Wiki 6,307 token、BCP 50,137 token，约为写满时窗口的 38%。
- 把它减掉后，这 397 条在写满那一刻都还在窗口内。这是静态估计：去掉重复后模型会继续往下走，最终能否答对需要重跑才知道。
- Wiki 的典型路径是：三四次调用都被截在 2,048 token，每次实际占约 4.5K；再加一次长思考，16K 就满了。

| 案例 | 经过 |
|---|---|
| 2Wiki `dev_8736` | 问 Stanislav Neuhaus 父亲的卒日。4 次调用每次都截断在 2,061 token，没等到读出 Heinrich Neuhaus 的卒日窗口就满了。Search-o1 一次搜索就答对 |
| Bamboogle `test_46`、`test_91`，NQ `test_216`、`test_2558`，HotpotQA `dev_2317`、`dev_3288`，2Wiki `dev_5183`，MuSiQue `dev_153`、`dev_648` | 写满之前**都已经在输出里见过 gold**：前 4 题是在最后一次调用里，其余在更早的调用里。写满前最后一轮的思考从约 120 到 6,600 token 不等（`dev_648` 6,586，`dev_2317` 4,462，`test_216` 3,827），多数时候是 payload 本身把窗口吃满 |

**轮数耗尽**（Wiki 347 题）
- 2Wiki 和 MuSiQue 各 107 题。
- "比较两部电影导演谁先出生"这类题至少要 4 次查找（2 部电影 → 2 位导演 → 2 个生日），6 轮预算里只要有一两次空返回就不够了。

**见过 gold 却耗尽预算**
- Wiki 139 题，BCP 27 题。见到 gold 之后平均又做了 2.6 次调用。
- 例：PopQA `test_13618` 第 5 次调用已拿到 Afonso 的母亲，第 6 次又去查 John II of Portugal。HotpotQA `dev_5214` 第 2 次调用就有 "singer"，之后又搜了 4 次。

**为什么这是头号问题**
- 预算失败的记录没有任何答案，按 0 分计。
- 去重、给思考设上限、末轮强制作答，这三件事都不需要改变检索能力。

### 3.2 BCP：一次调用 ≈ 一篇文档的开头

| 指标 | 数值 |
|---|---|
| 调用次数 | 3,385（每题 6.77） |
| 非空调用 | 3,085，其中 3,009 被截断在 6,000 token |
| 非空调用显示的行数 | 2 行 2,436 次（79%），即一篇文档的开头加截断提示；3 行 506 次（16%），即一篇完整文档加第二篇的开头；≥4 行 83 次 |
| 文档长度（`offsets.npy`） | p25 4.3 KB，中位 10.5 KB，p75 26.5 KB，p90 63 KB；27% 超过 24 KB（约 6K token） |
| 超时 | 0 |
| 单词模式 | 49% 的调用用单个词做模式 |

对比：其他方法每次检索拿到 3 篇按相关度排序的文档，各截前 512 token，共约 1.5K token。GrepSeek 每次花约 12K token 的上下文（截断 stdout 加重复字段），看到的却是**按文件顺序**第一个命中的文档开头。

后果：
- `head -n 8` 在 BCP 上完全不起作用，第 2 到第 8 个命中永远看不到。
- 多约束题只能靠"把某个罕见字符串直接 grep 中"碰运气。BCP 题面一般是改写过的，这种字符串很少。
- 16 轮 × 12K 远超窗口，192 题写满。写满时平均做了 8.76 次调用、思考 32K token。

代表案例（只写机制）：

| query ID | 经过 |
|---|---|
| 783 | 11 次调用全是 2–4 个常见词的 AND 过滤，每次都只显示 1 篇文档的开头，最后上下文写满 |
| 1097 | 11 次调用，其中多次是同一组过滤词换个顺序，最后上下文写满 |
| 1138 | 第 9 次调用的输出里已出现 gold，但模型继续坚持第 2 轮就提出的错误候选，最终答错。这是锚定 |
| 602 | 第 5、6 次调用（同一命令执行了两次）的输出都含 gold，最终仍答了另一个候选 |
| 1198、1135 | 单轮思考分别约 80K 和 79K token，把 131K 窗口直接写满 |
| 1223 | 单条轨迹的思考累计约 33.7 万字符 |
| 160 | 思考陷入重复词序列 |
| 1148 | 连续 5 轮几乎相同的命令 |
| 896 | 同类重复 |

### 3.3 检索语义（Wiki "没见过 gold" 698 题的主因）

模型只用一种句式：`rg -F "<实体>" corpus.jsonl | rg -i -F "<词1>" | rg -i -F "<词2>" | head -n K`。这正是 system prompt 里第二个示范，只是把过滤段串得更长。

prompt 里还提到了 `wc -l`（先计数再收窄）和 `-w`，允许的工具也包括 `sort`、`uniq`、`cut`、`awk`，但 9,449 次调用里一次都没用过。SFT 和 RL 把行为收敛到了单一模板。

这种写法没有排序，也没有任何"相关度"概念。具体有四种失败方式。

**3.3.1 文件顺序 + `head`**
- 36% 的调用被 `head` 截满。对常见实体，前 K 行多半是"顺带提到该实体"的其他条目。
- Bamboogle `test_9`：问达尔文的最后一本书，答错为 *The Expression of the Emotions*。`"Charles Darwin" | "book" | "final"` 和 `"Charles Darwin" | "works"` 都被截满，讲 *The Formation of Vegetable Mould* 是最后著作的段落不在前 20 行内。
- Bamboogle `test_53`（Starbucks）同理；`test_114`（2012 奥运排球）结果全是无关条目，最后凭记忆答 "United States"。
- NQ `test_1283`：`"Battle of Britain Flight"` 只命中 1 行（正式名称是 *Battle of Britain Memorial Flight*）。之后漂移到 1969 年的同名电影，最后凭记忆答 RAF Biggin Hill。

**3.3.2 AND 过滤的词汇对不上**
- 1,914 次非超时的空返回中，1,000 次（52%）第一段本来有命中，是后面的过滤词把结果滤空了（重放实验，见 §3.4）。另有 526 次（27%）第一段的字符串在语料里根本不存在：要么是改写过的措辞，要么是编造的名称。
- 过滤词取自题面或模型自己的措辞，而 Wikipedia 原文用的是另一种说法：

  | 案例 | 过滤词 → 结果 |
  |---|---|
  | Bamboogle `test_43` | Pinatubo + "death toll" / "deaths" 都为空。最后答了原文措辞 "over 800"，gold 是 847，模型自己其实知道 847 |
  | Bamboogle `test_16` | Salk + "religion" 为空，换 "Jewish" 后命中，答 "Judaism"（gold "Jewish"，又是形式问题） |
  | HotpotQA `dev_1246` | "nickname" 为空，之后一轮思考写了 47,822 字符，上下文写满 |
  | HotpotQA `dev_4193` | "debut" → 被带到 Geneva Motor Show |
  | HotpotQA `dev_4124` | "introduced"、"Representative" 都为空，最后凭记忆编了一个国会选区 |
  | HotpotQA `dev_2744` | "overthrown"、"deposed" 为空 |
  | 2Wiki `dev_8598` | "director" 对不上原文的 "directed by" |
  | Bamboogle `test_106` | "founded" |
  | NQ `test_2309` | "drawing" |

**3.3.3 过滤词里写进了自己的假设**
- 过滤条件常常就是模型当前的猜测，于是只能看到支持猜测的段落：

  | 案例 | 过滤条件 | 后果 |
  |---|---|---|
  | PopQA `test_11838` | `"Smalley" | "cricket"` | 找到一个打板球的 Smalley，gold 是足球 |
  | PopQA `test_5539` | `"The Men" | "1993"` | 年份是猜的，实际是 1950 年那部 |
  | MuSiQue `dev_2247` | 把"水下的意大利城市"假设为 Baiae | 实为 Venice |
  | 2Wiki `dev_9152` | 假设 Princess Margaret of Prussia 的父亲是 Wilhelm II（其实是她哥哥） | 从未验证 |

**3.3.4 题面是改写过的线索**
- TriviaQA 的谜语式题目，原文里根本没有题面那些词。dense 检索一次就能命中，grep 不行：

  | 案例 | 内容 |
  |---|---|
  | `test_3310` | Renault 的菱形标志 |
  | `test_4681` | 沙滩足球的场地尺寸 |
  | `test_3806` | AIDA |
  | `test_3780` | O.J. Simpson |
  | `test_10161` | Game On：题面写 "Mathew"，原文是 "Matthew"，第 6 次调用才命中 |
  | `test_5613` | GEICO：`"gecko" | "mascot"` 先命中 Crystal Dynamics 的吉祥物 Gex，于是答错；gecko + caveman 不在同一段，AND 为空 |

### 3.4 大小写与 Unicode 转义

- 9,449 个 Wiki 第一段里只有 21 个带 `-i`（0.2%）；`-i` 只出现在过滤段。这与 prompt 示范一致：示范的第一段是 `rg -F`，第二段才是 `rg -i -F`。
- 语料 `wiki_corpus.jsonl` 是 `json.dumps` 默认的 ASCII 输出，`í` 存为 `í`，所以 `rg -F "Valentín"` 永远为空。
- 250 个第一段（2.6%）含非 ASCII 字符。

**空返回重放实验**（`scripts/empty_grep_experiment.py`，计算节点上对本地语料副本运行）

做法：把 1,914 次非超时的 Wiki 空返回命令的第一段在语料上依次重跑三遍：
1. 原样；
2. 大小写不敏感；
3. 把非 ASCII 字符写成 `\uXXXX` 转义（NFC 和 NFD 两种形式，同时大小写不敏感）。

| 空返回的原因 | 次数 | 占空返回 |
|---|---:|---:|
| 第一段有命中，被后面的 AND 过滤滤空 | 1,000 | 52% |
| 第一段的字符串在语料里根本不存在（三种写法都没有） | 526 | 27% |
| **只差大小写**：加 `-i` 就能命中 | 162 | 8% |
| **只差 Unicode 转义**：写成 `\uXXXX` 就能命中 | 225 | 12% |
| 命令无法解析 | 1 | — |

| 数据集 | 空返回 | 过滤滤空 | 不存在 | 大小写 | 转义 | 大小写 + 转义占比 |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 502 | 160 | 150 | 96 | 96 | **38%** |
| MuSiQue | 355 | 247 | 57 | 10 | 41 | 14% |
| HotpotQA | 255 | 157 | 56 | 4 | 38 | 16% |
| PopQA | 260 | 89 | 127 | 18 | 26 | 17% |
| NQ | 249 | 159 | 61 | 23 | 6 | 12% |
| TriviaQA | 243 | 159 | 57 | 9 | 17 | 11% |
| Bamboogle | 50 | 29 | 18 | 2 | 1 | 6% |

结论：
- **带非 ASCII 字符的命令在这个语料上基本不可能成功**：275 次中 273 次返回空。纯 ASCII 命令的空返回率是 18%（1,643 / 9,174）。
- 大小写和转义合计 387 次，占空返回的 20%、全部调用的 4%。这些都是零成本可修的：第一段默认 `-i`，语料存成 UTF-8 或在工具层自动转义。
- 2Wiki 最严重。它的题面实体多是首字母大写的片名，例如 "Fighting With Buffalo Bill"（原文是小写 with），还有带重音的欧洲人名。
- "不存在"的 526 次说明，即使大小写和编码都修好，仍有约四分之一的空返回来自模型自己的措辞或编造的名称。这部分要靠改写、语义回退或候选枚举来解决（§6 第 5、7、11 条）。

案例（第一段为空之后，模型通常认定"语料没有这个条目"，转而凭记忆作答或放弃）：

| 类型 | 案例 |
|---|---|
| 大小写 | 2Wiki `dev_8196`：`"Fighting With Buffalo Bill"` 为空，第 3 次换成小写 with 才命中，已来不及，max_turns |
| 大小写 | 2Wiki `dev_7291`：`"M3 (Band)"` 为空 |
| 大小写 | 2Wiki `dev_5906`（Cherchez L'Idole）、`dev_12040`、`dev_2332`、`dev_10518`、`dev_8598` |
| 重音 | PopQA `test_6841`：`"Valentín"` 连续 3 次为空，去掉重音后又命中一堆别的 Valentin，max_turns |
| 重音 | 2Wiki `dev_8106`：`"Luis García Berlanga"` 为空 |
| 重音 | 2Wiki `dev_5599`（Éric Rohmer） |
| 重音 | MuSiQue `dev_1860`（Hana Mandlíková） |
| 拼写变体 | NQ `test_1216`：题面写 "Neel Darpan"，语料主条目是 "Nil Darpan"，最后编了个译者 |

### 3.5 歧义短标题（PopQA）

PopQA 的题目很多是 "Who was the director of X?"，X 是常见词或多义标题。grep 拿到的是文件里前 K 个含 X 的段落，模型从不枚举"标题为 X (…) 的所有条目"。

| 案例 | 经过 |
|---|---|
| `test_4684` | 搜 "Crash (film)"，条目实为 "Crash (2004 film)"；之后思考 27,567 字符 |
| `test_9637` | 正确条目已在输出里，却选了 *Big Trouble in Little China* |
| `test_6670` | Basement 选错了片 |
| `test_8435` | Hell 猜成 2014 年的片 |
| `test_12561` | New Europe 是期刊还是 Palin 的书 |
| `test_10299` | Frozen |
| 同类 | `test_1811`（Paris）、`test_5114`（I Do）、`test_5164`（Around the World）、`test_9399`（Pilot）、`test_12887`（Relic）、`test_3310`（The Goldfinch） |

Wikipedia 用 `Title (year film)` 区分同名条目。按标题锚定加枚举（如 `rg -o '"contents": "\\"Crash \([^)]*\)'`）一次就能列出全部候选，但模型 9,449 次调用里只锚定过 1 次标题。

### 3.6 推理控制：思考失控、找到不收、锚定、重复

**思考失控**
- 612 / 3,625 题出现单轮 ≥2,000 token 的思考，这些题只有 66 题（11%）答对；其余 3,013 题答对 1,455 题（48%）。
- 265 个"无动作"轮次（harness 回填 `format error: no <tool_call> or <answer> found`）：Wiki 162，BCP 103。其中 264 个上一轮的 finish_reason 是 length，185 个思考 ≥2,000 token。也就是说，无动作基本都是一轮思考写满了剩余窗口。
- 例：Bamboogle `test_34`（25,764 字符）、`test_11`（22,067 字符），PopQA `test_4684`（27,567 字符），HotpotQA `dev_1246`（47,822 字符），BCP 1223、1198、1135。

**找到不收**
- 见过 gold 却耗尽预算的有 166 题（Wiki 139 + BCP 27）。
- 例：Bamboogle `test_106`、`test_78`、`test_91`，PopQA `test_13618`，HotpotQA `dev_5214`。

**锚定**
- BCP 1138、602：gold 已出现在输出里，仍坚持之前的候选。
- 2Wiki `dev_8598`：拿到 gold 段落后又思考了 8.6K token，最后选了另一部片。

**重复**：BCP 1148 连续 5 轮几乎相同的命令；BCP 896 同类。

**有据可依的答案可信得多**
- 答案字符串出现在某次工具输出里的有 2,131 题，答对 1,307（61%）。
- 没出现的有 748 题，只答对 214（29%）。
- 后者基本是凭记忆补全，见 §3.8。

### 3.7 答案形式（只影响 EM，Wiki 272 / 1,160 个错答）

| 形式 | GrepSeek | SR1-14B qwen3 | SR1-14B e5 | 例（GrepSeek） |
|---|---:|---:|---:|---|
| 并列多个候选（"X and Y"、"X (or Y)"） | **70** | 27 | 22 | PopQA `test_10738` "Ross Parker and Hughie Charles"、`test_10299`、`test_4684`、`test_6799`；NQ `test_1975` "Algeria and Angola"、`test_2281`、`test_2438` "United Kingdom (or British)" |
| 只答年份（gold 是完整日期） | 23 | 12 | 17 | Bamboogle `test_5` "1789"、`test_6` "1797"、`test_23`、`test_53`、`test_95` "1904" |
| 过度具体（答案包含 gold） | 65 | 81 | 68 | TriviaQA `test_5984` "Athina Onassis"、PopQA `test_13026` "staff at General Mills"、`test_7367` |
| 不够具体（gold 包含答案） | 114 | 123 | 118 | Bamboogle `test_13` "Lockheed"、PopQA `test_1268` "Horror"、MuSiQue `dev_1170` |
| 其他（实质错误） | 888 | 1,330 | 1,385 | |

- cover-EM（gold 出现在答案里就算对）下，GrepSeek 从 45.2 升到 49.5，SR1-14B qwen3 从 47.9 升到 51.4。GrepSeek 的形式损失略大，主要来自"并列多个候选"。
- 原因很直接：RL 奖励是 F1。"Ross Parker and Hughie Charles" 对 gold "Ross Parker" 的 F1 约 0.67，"1789" 对 "April 30, 1789" 的 F1 是 0.5。部分分让策略没有动力只给一个答案、给完整日期。
- 还有一部分 EM 失败其实答对了：NQ `test_2766` "Richard Keith" 是 Keith Thibodeaux 的艺名；NQ `test_3280` "16 June 1976"；TriviaQA `test_8675` "fear itself"；HotpotQA `dev_671` "United States" 对 gold "American"。

### 3.8 凭记忆回退、幻觉、答错槽位

**凭记忆回退与幻觉**

| 案例 | 经过 |
|---|---|
| NQ `test_1216` | 见 §3.4 |
| PopQA `test_9368` | 输出写的是 Godard，却答 "Jean-Pierre Gorin" |
| HotpotQA `dev_4124` | 编了一个国会选区 |
| 2Wiki `dev_9152` | 把 Wilhelm II 当成父亲，答了 1941 年的日期 |
| Bamboogle `test_114` | 答 "United States" |

**答错槽位（找到了对的条目，回答了题目问的另一个属性）**

| 案例 | 答了什么 |
|---|---|
| TriviaQA `test_5831` | 问花名，答 "Mexico" |
| Bamboogle `test_49` | 问奥巴马父亲的母亲，答奥巴马的母亲 Ann Dunham |
| MuSiQue `dev_930` | 问负责人是谁，答 "Governor" |
| MuSiQue `dev_404` | 答 Jefferson Davis 卸任总统的 1865，题目问的是他在美墨战争中作战结束的 1848 |
| HotpotQA `dev_5144` | 问集团的所有人，答了集团名 Synergy Group |
| HotpotQA `dev_3262` | 桥接演员选错，且按日期判断错 |

### 3.9 不该算在 GrepSeek 头上的

**标注噪声**：这些题所有方法都错，或 GrepSeek 的答案其实更对。
- NQ `test_727`、`test_718`、`test_1553`、`test_1128`、`test_2003`；NQ `test_606`（McCormick 收割机常见说法是 1831，gold 是 1834/1837）。
- PopQA `test_9057`（1987 年那部的编剧确是 Steven de Souza）。
- HotpotQA `dev_1528`；HotpotQA `dev_2549`（Catcha Lake 在 Nova Scotia，"Nova Scotia and British Columbia" 比 gold 更准确）。
- MuSiQue `dev_1241`（时效性题目）。

**基础设施**
- Wiki 有 99 次超时（约 1%），BCP 为 0。超时都出现在"第一段是常见实体、过滤很严"的管道上：rg 要扫完 14 GB 的语料才凑得够 `head`。
- 超时与负载有关：同一条 Theranos 命令第 1 轮超时，第 5 轮成功。
- 我们按官方默认 `engine_mode none` 运行，16 个 worker 用 12 个 CPU。
- 每次超时白白消耗一轮预算，例：Bamboogle `test_5`、`test_9`。分片引擎或更多 CPU 可以消除。

---

## 4. GrepSeek 赢在哪里（24 道"GrepSeek 对、强基线全错"的抽样）

- **罕见精确字符串**，dense 检索在这类问题上反而容易被近义内容淹没：
  - HotpotQA `dev_6618`："taser match" 直接命中 Bam Bam Bigelow 对 Scott Hall 的比赛。
  - HotpotQA `dev_5407`：Les Scadding → EuroMillions 首期开奖日。
  - HotpotQA `dev_423`：通过 Arizona State 找到 Phil Bennett 任 defensive coordinator。
  - PopQA `test_12811`：书名 *The Chaos Code* 精确命中作者条目。
  - 2Wiki `dev_3229`：*Teenage Wolfpack* → Georg Tressler。
- **9B RL 模型自身的知识加上部分证据**：
  - Bamboogle `test_1`（Voyager 2 → Titan IIIE）、`test_57`（Tokyo Skytree 2012）、`test_79`（Yamate Tunnel 1992）。
  - NQ `test_2353`、`test_1144`，TriviaQA `test_71`、`test_1494`。
  - 这些题的工具输出里并没有直接给出答案。
- **2Wiki 的比较题**：精确标题能同时锁定两部电影，如 `dev_2461`、`dev_3229`。

所以第 6 节的方向应是：**保留精确匹配的优势，补上"排序、改写、预算"三块短板**。不应该把 grep 换成 dense 检索。

## 5. 78 道"GrepSeek 错、强基线对"抽样题的逐题归因

每个数据集抽约 10 题，看 GrepSeek 一侧的命令和输出。

| 主因 | Wiki 题数（共 68） | 案例 |
|---|---:|---|
| 答案形式 / EM 口径（多数其实答对了） | 19 | Bamboogle `test_5`、`test_6`、`test_95`；PopQA `test_1159`、`test_10299`、`test_10738`；TQ `test_5671`、`test_5984`、`test_8675`、`test_10057`；NQ `test_1218`、`test_1975`、`test_2281`、`test_2766`、`test_3280`；HotpotQA `dev_671`；2Wiki `dev_3140`、`dev_3353`；MuSiQue `dev_1170` |
| 见过 gold 后上下文写满 | 11 | Bamboogle `test_46`、`test_91`；NQ `test_216`、`test_235`、`test_2558`；HotpotQA `dev_2317`、`dev_3288`；2Wiki `dev_5183`、`dev_8736`；MuSiQue `dev_153`、`dev_648` |
| 检索语义：AND 词汇、改写线索、无排序 | 12 | Bamboogle `test_9`、`test_43`；TQ `test_1881`、`test_3310`、`test_4681`、`test_5613`、`test_10161`；HotpotQA `dev_1246`、`dev_2744`、`dev_4124`；2Wiki `dev_8598`；MuSiQue `dev_1881` |
| 推理：答错槽位、少一跳、错误桥接 | 9 | Bamboogle `test_49`；TQ `test_5831`；HotpotQA `dev_4729`、`dev_5144`；2Wiki `dev_9152`；MuSiQue `dev_404`、`dev_465`、`dev_930`、`dev_1102` |
| 大小写、Unicode、标题写法 | 6 | PopQA `test_6841`；2Wiki `dev_7291`、`dev_8106`、`dev_8196`、`dev_9580`；NQ `test_1216` |
| 歧义标题、过滤里写进假设 | 6 | PopQA `test_5539`、`test_6670`、`test_8435`、`test_11838`、`test_12561`；MuSiQue `dev_2247` |
| 找到不收（max_turns） | 2 | PopQA `test_13618`；HotpotQA `dev_5214` |
| 标注噪声、时效性 | 3 | NQ `test_606`；HotpotQA `dev_2549`；MuSiQue `dev_1241` |

BCP 的 10 题（783、64、602、1198、1263、1097、1135、1138、87、380）背后都是"一次一篇文档开头、按文件顺序"这一条。在此之上：
- 4 题上下文写满（783、1097、1198、1135），其中 2 题是单轮思考约 80K token。
- 2 题 gold 已出现却锚定在旧候选上（602、1138）。
- 其余 4 题一直没见到 gold，最后给了错误候选。

和全量统计一致：**这 68 题里约 45% 是形式和预算问题，修起来不需要更强的检索。**

---

## 6. 改进建议（按"预期收益 / 成本"排序）

收益估计都是上界或粗估，必须以重跑为准。改动 GrepSeek 只针对我们自己的新方法；正式 baseline 的数字保持不变。

| # | 改动 | 针对的失败 | 粗估收益 | 成本 |
|---|---|---|---|---|
| 1 | **去掉 `information_lines` 重复**：payload 只留 stdout，或只留 info 行 | 上下文写满 Wiki 205、BCP 192；payload 降到约 1/2.2 | 写满的 397 题都能继续。其中 77 题（Wiki 50、BCP 27）已见过 gold，按"有据答案"61% 的准确率算，约 Wiki +30、BCP +16（BCP 约 +3 分）；全部 397 题按平均水平算是上界 | 改格式需在 SFT/RL 里同步，否则有轻微分布偏移；可以先零样本试 |
| 2 | **预算末尾强制作答**：最后一轮或剩余窗口 < X 时注入"现在必须作答" | Wiki 552、BCP 192 个无答案记录 | 见过 gold 的 166 题最直接受益；其余题按凭记忆作答约 29% 的准确率算 | 只改推理逻辑；RL 时加入同样的规则 |
| 3 | **BCP 按段落切行**：每行 ≤512 token 一段，行首带 `docid` 和标题；或训练/提示使用 `rg -o '.{0,300}PAT.{0,300}'` 片段窗口，并用 `--max-columns` 截断长行 | 79% 的 BCP 调用只看到一篇文档开头 | 每次调用可见文档从约 1 篇增到 8 段，每次上下文成本也大幅下降 | 语料预处理加 prompt；最好用长文档数据重训 |
| 4 | **第一段默认 `-i`**：把 prompt 和 SFT 示范的第一段改成 `rg -i -F`。**语料存 UTF-8**（`ensure_ascii=False`），或在工具里把非 ASCII 模式自动转成转义形式 | 大小写、重音导致的空返回（§3.4） | 387 次空返回（占空返回的 20%，2Wiki 38%）可直接变成有结果；含非 ASCII 的命令目前 99% 为空 | 语料或工具层零训练成本；改示范需要 SFT |
| 5 | **标题锚定和候选枚举**：提供或教会 `rg -o '"contents": "\\"<Title>( \([^)]*\))?\\"'` 这类写法，以及 `… | sort | uniq -c | sort -rn | head` | PopQA 歧义标题；文件顺序 + `head` 的噪声 | PopQA 是 GrepSeek 和最佳基线差距最大的数据集之一（44.4 vs 48.6）。上下文写满 61 题，多数先在歧义标题里打转 | 需要 SFT 示范 |
| 6 | **命中排序**：工具把 grep 命中按"标题 = 锚点实体 > 正文首句提及 > 其他"重排，或接一个 BM25 重排 | 36% 调用被 `head` 截满；文件顺序 | 直接减少"没见过 gold"的 698 题 | 工具层；只需轻量重训 |
| 7 | **AND 过滤改成 OR 或词干**：`-e directed -e director`；教模型用词干（`direct`、`nicknam`）而不是题面原词 | AND 滤空 1,000 次；字符串不存在 526 次 | 每次滤空都白耗一轮；2Wiki、MuSiQue 的轮数耗尽会明显下降 | SFT 示范和 prompt |
| 8 | **每轮思考上限**（如 1,024–2,048 token），超限截断后强制给出动作；检测重复命令或思考 | ≥2K 思考的 612 题只有 11% 答对；265 个无动作轮次 | 能把一部分写满和轮数耗尽转成作答 | 推理期即可做；RL 里同步 |
| 9 | **奖励改成 EM，或给多候选、"(or …)" 答案罚分**；prompt 的答案要求从 "concise — typically a short noun phrase" 改成"只给一个实体，问 when 时给完整日期" | 形式错答 272（Wiki 23%） | cover-EM 与 EM 相差 4.3 分，其中可回收的部分 | 重训 |
| 10 | **训练分布扩展**：16 轮 / 长上下文 rollout；加入 2Wiki/MuSiQue 式比较和组合题、首字母大写和带重音的实体、改写型 trivia、BCP 式长网页 | 2Wiki/MuSiQue 的轮数耗尽；BCP 的分布外行为 | 结构性收益 | 重训 |
| 11 | **混合检索**：给 agent 再加一个 dense `search` 工具，或在 grep 为空时自动回退 | Wiki 607 题、BCP 142 题是"GrepSeek 错、强基线对" | oracle 上界很高（如 NQ 任一 run 对 347 vs GrepSeek 195）；但要保住 §4 的精确匹配优势 | 改方法本身 |
| 12 | 基础设施：分片引擎（`engine_mode`）或更多 CPU | 99 次超时 | 约 1% 的调用 | 零训练成本 |

**建议的落地顺序**

1. 先做零训练的 1、2、4、8、12，在 Wiki 和 BCP 上各重跑一遍 GrepSeek（作为新方法的消融，不替换 baseline），量化收益。
2. 再做需要 SFT 示范的 3、5、6、7，以及需要重训的 9、10。
3. 混合检索（11）放在最后，作为单独的方法变体评估。

---

## 附录 A：数字从哪来

所有脚本在 `reports/scripts/`（数据路径写死为 `/scratch/a32du/scaleseek_v2`），在计算节点上运行；输出在 `/scratch/a32du/scaleseek_v2/analysis/failures/`。

| 脚本 | 输出 | 内容 |
|---|---|---|
| `failure_analysis.py` | `summary.json`、`grepseek_features.jsonl`（3,625 行）、`others_features.jsonl`、`cases/`（584 个抽样案例文件） | 逐调用特征、失败类别、互补性 |
| `failure_stats2.py` | `stats2.json` | 命令习惯、有据/无据答案、思考失控、无动作轮次、重复 payload 的静态估计、见过 gold 后耗尽预算 |
| `answer_form.py` | `answer_form.json` | 错答形式分桶、EM 对 cover-EM |
| `bcp_visibility.py` | `bcp_visibility.json` | BCP 文档长度分布、每次调用可见行数 |
| `grepseek_side.py` | — | 抽样"GrepSeek 错、强基线对"题的 GrepSeek 命令序列 |
| `empty_grep_experiment.py` | `empty_grep_experiment.json` | 空返回在本地语料上的重放分类 |

**定义**
- "见过 gold"：任一工具输出（规范化后）包含任一 gold 字符串。
  - 少于 3 个字符的 gold 和 yes/no 不参与，所以短数字类 gold 可能有误报。
  - 工具输出里非 ASCII 字符是 `\uXXXX` 转义，所以非 ASCII 的 gold 会漏检。Wiki 有 277 题含非 ASCII gold，"没见过 gold"的 698 题里有 55 题属于这类。因此 698 是上界。
- "有据答案"：最终答案字符串出现在某次工具输出里。

## 附录 B：案例索引

正文出现过的案例：

| 数据集 | 案例 |
|---|---|
| Bamboogle | `test_1` `test_5` `test_6` `test_9` `test_11` `test_13` `test_15` `test_16` `test_23` `test_34` `test_43` `test_46` `test_49` `test_53` `test_57` `test_67` `test_78` `test_79` `test_91` `test_92` `test_95` `test_106` `test_114` |
| NQ | `test_216` `test_235` `test_446` `test_606` `test_718` `test_725` `test_727` `test_984` `test_1128` `test_1144` `test_1216` `test_1218` `test_1257` `test_1283` `test_1484` `test_1553` `test_1975` `test_2003` `test_2281` `test_2309` `test_2353` `test_2438` `test_2558` `test_2766` `test_3280` |
| PopQA | `test_1159` `test_1268` `test_1811` `test_3310` `test_4684` `test_5114` `test_5164` `test_5539` `test_6670` `test_6799` `test_6841` `test_7367` `test_8435` `test_9057` `test_9368` `test_9399` `test_9637` `test_10299` `test_10738` `test_11118` `test_11838` `test_12561` `test_12811` `test_12887` `test_13026` `test_13425` `test_13618` |
| TriviaQA | `test_71` `test_1494` `test_1881` `test_2888` `test_3310` `test_3780` `test_3806` `test_4595` `test_4681` `test_5613` `test_5671` `test_5831` `test_5984` `test_8675` `test_10057` `test_10161` |
| HotpotQA | `dev_423` `dev_671` `dev_743` `dev_1241` `dev_1246` `dev_1528` `dev_2317` `dev_2430` `dev_2549` `dev_2744` `dev_3262` `dev_3288` `dev_4124` `dev_4193` `dev_4729` `dev_5144` `dev_5214` `dev_5407` `dev_5915` `dev_6618` |
| 2Wiki | `dev_2332` `dev_2461` `dev_3140` `dev_3229` `dev_3353` `dev_4970` `dev_5183` `dev_5599` `dev_5906` `dev_7291` `dev_8106` `dev_8196` `dev_8598` `dev_8736` `dev_9152` `dev_9580` `dev_10518` `dev_12040` |
| MuSiQue | `dev_87` `dev_153` `dev_404` `dev_465` `dev_648` `dev_930` `dev_1102` `dev_1170` `dev_1241` `dev_1860` `dev_1881` `dev_2247` `dev_2391` `dev_2404` |
| BCP（只给 ID） | 64 87 160 380 602 783 896 1097 1135 1138 1148 1198 1223 1263 |

**怎么查看完整轨迹**
- 抽样案例：`/scratch/a32du/scaleseek_v2/analysis/failures/cases/grepseek/<ds>/<category>/<id>.txt`。
- 任意题：`runs/<ds>/grepseek/trajectories.jsonl`，按 `id` 检索。
