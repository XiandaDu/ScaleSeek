# 其他方法失败案例分析（ScaleSeek v2 正式运行）

2026-10-20。对象是 5 个强 run：
- Search-R1-14B（检索器 e5 / Qwen3-Embedding-4B）
- Search-o1（Qwen3.5-9B + Qwen3-Embedding-4B）
- AgentIR（Qwen3.5-9B + AgentIR-4B）
- RAG（Qwen3.5-9B + Qwen3-Embedding-4B）

Direct 和 Search-R1-7B 只在相关处提及。

**做法**
- 全量统计：`summary.json` 的失败类别、`answer_form.json` 的 EM 对 cover-EM，以及两个专项计数：Search-R1 的"中途 EOS"、Search-o1 的检索标签拼错。
- 人工通读分层抽样的案例：AgentIR 72 条、Search-o1 93 条、RAG 70 条、Search-R1-14B e5 70 条、Search-R1-14B qwen3 24 条。每个数据集每类约 2 条，并包括"本方法对、GrepSeek 错"的案例。

**口径**
- 抽样是分层的，所以"x / 抽样数"只说明机制存在，不能当总体比例。
- BCP 只写 query ID 和机制。
- GrepSeek 的分析见另一份报告 `grepseek_failures.md`。

---

## 0. 结论速览

| 方法 | 主要失败（Wiki，3,125 题） | 主要失败（BCP，500 题） | 它比 GrepSeek 强在哪 |
|---|---|---|---|
| Search-R1-14B | 检索只做 1–2 次，第一跳没取到就**编造桥接**；**中途自发 EOS**（no_answer 54–55）；从不 max_turns | 平均只检索 1.3 次；**24–26% 中途 EOS**；judge 准确率 4.8 / 9.0 | 单实体短答案；dense 检索一次命中改写过的线索 |
| Search-o1 | **max_turns 797（25%）**：同义改写拿到同一组 top-3，不收敛；**检索标签拼错**导致 no_answer | max_turns 269 + 上下文写满 74 + no_answer 48 | Reason-in-Documents 一次融合两跳；语义匹配拼写变体 |
| AgentIR | **max_turns 1,285（41%）**：答案已在手仍反复核验；答案冗长使 EM 27.8 而 cover-EM 42.8 | max_turns 325（65%），但**一旦作答 89% 被判对**，所以 BCP 第一（31.0） | 多约束改写的语义匹配；以中间实体为跳板 |
| RAG | **大量拒答**："Not mentioned" 在 2Wiki 302/500、MuSiQue 210/500；多跳题检索不到第二跳 | 拒答 158/400；10K 上限截断 100 | 一次检索就把桥接条目和答案条目放进同一个 top-3 |

**跨方法的五点**
1. **预算耗尽且不强制作答**是 agent 类方法的最大损失。AgentIR 41%、Search-o1 25% 的 Wiki 题没有答案，其中金标已出现过的分别有 529 和 255 题。GrepSeek 的同类问题见它的报告。
2. **答案形式**：cover-EM（金标出现在答案中即算对）比 EM 高的幅度为 AgentIR +15.0，Search-o1 +5.4，RAG +5.9，GrepSeek +4.3，Search-R1-14B +2.9~3.5。不同方法受 EM 的惩罚程度差别很大，见 §6.1。
3. **共用解码 presence_penalty 2.0 放大了"标签协议"的格式失败，但不影响 EM 结论**。对照 pilot 中去掉 penalty 后：
   - Search-o1 拼错标签的题从 25 降到 9，EM 27.0 → 28.0（p = 0.60）；
   - Search-R1-14B 中途 EOS 的题在 2Wiki 从 29 降到 17、在 BCP 从 128 降到 80，2Wiki EM 不变（44.0）。

   建议维持统一解码，写成 caveat，见 §6.2。
4. **RAG 拒答来自项目自定义的 reader prompt**，不是代码 bug，但它让 RAG 在 2Wiki 和 Bamboogle 上低于 Direct。论文里需要说明，见 §5。
5. **Qwen3.5-9B 的失控思考**：Direct 在 PopQA 上 16% 写满 16K 窗口，在 BCP 上 84% 撞到 10K 上限。所有用 Qwen3.5-9B 的方法都有这一底色。

---

## 1. 失败分布总表（Wiki 7 个数据集合计 / BCP）

| run | 对 | max_turns | 上下文写满 / 上限截断 | no_answer | 错，检索到过 gold | 错，没检索到 | gold 无法检查 | 每题检索 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GrepSeek（参照） | 1,413 | 347 | 205 | — | 443 | 698 | 19 | 3.0 |
| Search-R1-14B e5 | 1,460 | 0 | 0 | 55 | 458 | 1,119 | 33 | 1.0–2.1 |
| Search-R1-14B qwen3 | **1,498** | 0 | 0 | 54 | 521 | 1,019 | 33 | 1.0–2.0 |
| Search-o1 qwen3 | 1,239 | 797 | 2 | 60 | 534 | 483 | 10 | 1.9–3.9 |
| AgentIR | 869 | 1,285 | 0 | 1 | 624 | 314 | 32 | 2.6–5.1 |
| RAG qwen3 | 1,077 | — | 12 | — | 511 | 1,463 | 62 | 1 |
| **BCP** | | | | | | | | |
| GrepSeek | 108 | — | 192 | 2（parse） | 19 | 175 | 4 | 6.8 |
| Search-R1-14B e5 | 24 | 0 | 0 | 128 | 22 | 320 | 6 | 1.3 |
| Search-R1-14B qwen3 | 45 | 0 | 0 | 120 | 20 | 309 | 6 | 1.3 |
| Search-o1 qwen3 | 63 | 269 | 74 | 48 | 14 | 31 | 1 | 10.6 |
| AgentIR | **155** | 325 | 0 | 0 | 5 | 15 | 0 | 13.6 |
| RAG qwen3 | 72 | — | 100（10K 上限） | — | 18 | 301 | 9 | 1 |

注：
- "检索到过 gold"是启发式判断，意思是某篇返回文档包含金标字符串。它对比较题（金标本来就是题中实体）和短金标会误报，对非 ASCII 金标会漏报，见 §6.5。
- BCP 的"对 / 错"由 judge 判定。

---

## 2. Search-R1-14B

### 2.1 行为画像
- 单跳题平均检索 1.0 次，多跳题约 2 次。6 轮（BCP 16 轮）的预算从来不是约束：所有数据集都**没有**一个 max_turns 或上下文写满。
- 首个 query 通常照抄问题：单跳 NQ/TQ/PopQA 抽样 22/22；BCP 8/10，抄整题或把全部约束塞进一句。
- 拿到 `<information>` 后，104 次生成中有 96 次以 "Based on the search results, …" 开头，紧接着直接断言一个结论。抽样中从没有一次承认"文档没有提到"，也从没有对同一跳换个写法重查。

### 2.2 失败分类（e5 抽样 54 个失败 + qwen3 抽样）

**编造桥接或事实（e5 抽样 12/54；算上 no_answer 和侥幸答对的，70 个样本中 18 个含编造）。** 第一跳没有取到目标实体时，模型用表面相似的文档顶替，然后继续第二跳。

| 案例 | 怎么编的 |
|---|---|
| 2Wiki `dev_8936` | 从 "Far from Men" 条目编出 'Farfallon' 的导演 |
| 2Wiki `dev_10374` | 把 "The Prince and the Pauper (1977)" 的导演当成 The Pauper Millionaire 的导演 |
| MuSiQue `dev_1111` | 从 "Mark White" 条目编出 "Mark Day was born in Union City" |
| Bamboogle `test_66` | top-3 是 Yangtze、Orinoco、Amazon，却断言 Congo 是第三长河 |
| HotpotQA `dev_2043` | Mediastan → "Michael Moore" |
| MuSiQue `dev_1503` | NBC / Bob Schieffer |
| BCP 209、963、102、1217 | 输出的实体或数值不在返回的 512 token 摘录里，或取自一个几乎空白的占位页标题 |

同一实体，E5 和 Qwen3-Emb 的表现可以截然不同。MuSiQue `dev_1116` 查 "Where did Hod Lisenbee die?"：E5 返回 Lou Henry Hoover 等无关条目，RAG 用 Qwen3-Emb 取回了 3 段 Hod Lisenbee。

**选错实体、被干扰项带偏（10/54）**

| 案例 | 经过 |
|---|---|
| Bamboogle `test_65` | 把 top-1 的 Delhi 当成人口最多的首都 |
| NQ `test_1993` | 用第 6 届薪酬委员会主席回答第 7 届 |
| 2Wiki `dev_10976` | A Night in Venice 混淆了 1934 和 1953 两个版本 |
| Bamboogle `test_57` | 被检索到的"CN Tower 是最高塔"这一过时说法带偏；gold 是 2012 年的 Tokyo Skytree，这题 GrepSeek 答对了 |
| BCP 966 | gold 页排第 1，模型却答了先验记忆里一个著名的同类实体 |
| BCP 199 | gold 页排第 3，被忽略 |

**推理和答案形式（8/54）**
- 2Wiki `dev_7467`：推理写的是 "Cheryl Dunye was born later"，答案却是 "Takeoff"。
- NQ `test_2967`：think 里写 "channel 15"，答案却是 "KNXV-TV"。
- 4/70 例在宣布"下一步要查国籍"之后，紧接着就写了 `<answer>`（2Wiki `dev_10374`、`dev_9624`，HotpotQA `dev_2842`，BCP 209）。看起来是训练中养成的"检索两次就答"的习惯。

**评测伪影（14/54）**：7 个 EM 假阴性加 7 个标注噪声。16 个"错，检索到过 gold"里有 10 个属于伪影，真正读错的只有 4 个。

### 2.3 no_answer：模型自己在句子中间发 EOS

| | e5 | qwen3 | bm25 | Search-R1-7B（三个检索器） |
|---|---:|---:|---:|---:|
| Wiki no_answer | 55 | 54 | 49 | 0–2 |
| BCP no_answer | 128 | 120 | 117 | 0–2 |
| 每千轮 dead-EOS（2Wiki / BCP） | 19.6 / 112.7 | 15.9 / 106.5 | 11.9 / 102.3 | ≤ 4 |

- **机制**：每一例都紧跟一个 `<information>` 块。下一轮生成在 14–62 个 token 后，模型自己发出 EOS（stop_reason=null）。
  - 不是 stop string，不是长度截断，也与长上下文无关：prompt 只有 655–3,444 token。
  - 7/10 断在 `<think>` 里，3/10 断在 `<search>` query 中间。8/10 正好断在要写出实体、数字或列表项的位置。
  - 例：2Wiki `dev_3813` 断在 `<search> When were directors Tony Ayres and`；MuSiQue `dev_588` 断在 "Based on the search results, Daniel M."；BCP 830、769 也是写到候选名的一半就停。
- **只有 14B 有这个问题，与检索器无关。** 7B 在同一 harness、同一解码下 no_answer 每个 run 不超过 2。
- **1 个 token 的空转是良性的**，Wiki 共约 10 次。这些是上一轮已经写完 `<answer>…</answer>`、后面又跟了垃圾 `</search>`；harness 按官方 `get_query` 重跑旧 query，下一轮模型立即 EOS，答案仍然保留。
- **另一个固定习惯（与 penalty 无关）**：几乎每个答案轮在 `</answer>` 后都带尾巴，如 `\n</answer>\n\n</`、`</s`。7B 和 14B 都有，带 penalty 和不带 penalty 都有。答案抽取不受影响。
- **与 presence_penalty 的关系**：pilot 显示 penalty 让这一现象约多 1.6 倍，但去掉后仍有 2Wiki 17、BCP 80 个 no_answer，且 2Wiki EM 不变，见 §6.2。机理上，dead-EOS 常发生在下一个 token 本轮从未出现过的位置，而 presence penalty 不会惩罚这种 token，所以主因是 14B checkpoint 自身的习惯。

### 2.4 BCP
- 平均 1.27 次检索、2.27 轮，16 轮预算基本没用上。抽样 8/10 只检索 1 次。
- 答对的 2 例（1263、380）都是第一条长 query 直接命中答案页。
- **格式问题（与官方代码一致）**：官方 `_passages2string` 把文档首行当标题。BCP 页面以 YAML `---` 开头，所以 Search-R1 看到的每篇都是 `Doc i(Title: ---) …`，真实标题在正文里。RAG 用的是 `split_title(bcp_front_matter)`，能拿到真标题。这不影响计分，但 Search-R1 在 BCP 上的输入形式偏离了它的训练分布。

---

## 3. Search-o1（Qwen3.5-9B + Qwen3-Embedding-4B）

### 3.1 max_turns（Wiki 797，BCP 269）：停不下来
- 抽样的 16 个 max_turns，**最后一个动作全部是检索**，没有一个是在准备作答。
- **协议因素**：轮数 = profile turns（Wiki 6），检索上限 = turns − 1。第 6 轮如果还在发 query，只会收到 LIMIT_MESSAGE，之后没有下一轮，所以没有"被迫作答"的机会。
  - 官方配置是 max_turn 15、检索上限 5/10，模型撞到上限后还有余量作答。
  - 我们这样设计是为了所有方法预算一致（已记录在 `docs/deviations.md`），代价是 Search-o1 失去了这个隐式的强制作答。
- **检索停滞（12/16）**：同义改写对 dense 检索几乎是同一个查询，反复拿到同一组 top-3。
  - NQ `test_2328`：5 次查询返回同样 3 段 "Taarak Mehta Ka Ooltah Chashmah"。
  - 2Wiki `dev_1121`：第 2–5 次查询（Spielberg birthplace / born where city / …）的 top-3 完全相同。
  - 这 16 例共 100 次检索，RiD 报 "No helpful information" 58 次。对照组：成功样本 8/43，"错，检索到过 gold"的样本 0/42。
- **证据已有却不收敛（4/16）**：
  - HotpotQA `dev_1374`：文档写着 Boston Celtics，模型因为参数知识里此人后来被交易，继续查 "2025 active roster"。
  - MuSiQue `dev_192`：RiD 已给出 Bill Pullman，模型又查了 3 次。
  - NQ `test_1104`、PopQA `test_2687` 类似。
- **把臆测写进查询（5/16）**：例如 2Wiki `dev_5213` 编了两位"父亲"去查。
- **时间错位**：模型默认"现在"是 2024/25 年，而语料是 2018 年前后的 Wikipedia。改变结局的有 4 例（HotpotQA `dev_1374`，MuSiQue `dev_281`，NQ `test_2395`，Bamboogle `test_62`）。
- **budget_fail_gold_seen**（启发式）：Wiki 255（2Wiki 78，MuSiQue 64，HotpotQA 48），BCP 61。

### 3.2 no_answer：检索标签拼错（全量计数，`scripts/tag_errors.py`）

| 检索器 | Wiki 中有拼错标签的题 | 其中 no_answer | Wiki no_answer 总数 | BCP 中有拼错标签的题 |
|---|---:|---:|---:|---:|
| bm25 | 108 | 45 | 71 | 41 |
| e5 | 92 | 38 | 58 | 35 |
| qwen3 | 105 | **45** | **60** | 38 |

- 典型拼法：`|end_search_query>`、`|end_search_query|>`、`<|end_search_query>`。也就是结束标签里与刚生成的开始标签**共享的片段**被换掉了：`<|` 变成 `|`，`|>` 变成 `>`。
- Wiki 上约一半、BCP 上约 2/3 发生在第一轮，即 0 次检索。
- 后果：
  - stop string 没触发，模型随后输出 EOS，harness 找不到 `\boxed{}`，记为 no_answer。
  - 或者模型自己写一个"检索结果"块继续往下。TQ `test_8445` 就是虚构了摘要后答错。
- 抽样的 16/16 个 no_answer 都是格式失败：结束标签拼错 10，开始标签错 4，`</boxed{` 1，只用文字作答 1。
- **机理**：这些标签对 Qwen3.5 不是 special token，会被切成子词，而 presence_penalty 2.0 正好惩罚本次请求里已经生成过的子词。pilot 证实了这一点：NQ bm25 去掉 penalty 后，有拼错标签的题从 25 降到 9，no_answer 从 12 降到 4，但 EM 只从 27.0 变成 28.0，不显著（§6.2）。

### 3.3 上下文写满（BCP 74）与失控生成
- 2Wiki `dev_8359`：0 次检索。在思考里"假装检索"并陷入循环："Okay, I found that *Gang Leader* (1967) is directed by **Hiroshi Teshigahara**? Wait, no." 一直写到 15,976 token。
- BCP 723：首轮生成 130,558 token 的模板化枚举。BCP 927：末轮写了 124,166 token 的 "Wait, could it be X? No." 循环。
- TQ `test_1937`：RiD 生成 15,363 token 撞上限，结尾是一串零宽空格。
  - 官方的 "最后一个 `**Final Information**` 之后" 抽取规则，这时取到的是一段思考残片，被插回主链，主链随即写满窗口。
  - 这和此前 "No helpful information" 统计失效是同一个根源：RiD 输出里包含了 thinking。

### 3.4 RiD（Reason-in-Documents）质量
- **"No helpful information" 大多是如实的**，反映的是检索没有取到。
- **按错的子查询过滤**：RiD 只看当前子查询，子查询一错，正确证据就被藏起来。
  - MuSiQue `dev_281`：子查询是 "NHL 2024-25"，RiD 因为年份不符把 2018–19 赛季的条目判为无信息。
  - PopQA `test_2687` 同理。
- **RiD 对了，主模型却丢掉或改写了**：
  - Bamboogle `test_96`：RiD 给出 "Claude, Sr."，主模型 boxed "Claude"。
  - 2Wiki `dev_6618`：RiD 给出 "Carlsruhe (now Pokój)"，主模型 boxed "Carlsruhe"。
- **幻觉**：
  - MuSiQue `dev_465`：RiD 写出文档里没有的 "founded by … Allen Klein"。
  - Bamboogle `test_69`：摘要逐步漂移，最后答 Spain。
- **促成成功**：
  - 2Wiki `dev_8736`：一次检索同时取回儿子和父亲两个条目，RiD 融合了两跳。GrepSeek 这题 4 次调用后上下文写满。
  - BCP 1198：RiD 用算术匹配了一条数字线索。
- **成本**：3 段约 100 词的文档，RiD 常写几百到 7K token，思考部分大多在复述自己的指令。

### 3.5 官方答案抽取（`extract_answer(mode='qa')`）的副作用
- 贪婪的 `\boxed\{(.*)\}` 先匹配，再取 `\text{}`，最后 `strip("()")`：
  - `\boxed{9:00 \text{ AM}}` 被抽成 ' AM'（MuSiQue `dev_2364`）。
  - `\boxed{日本 (Nihon/Nippon)}` 被抽成 '日本 (Nihon/Nippon'。
- 模型复述任务模板时，预测会被抽成 'YOUR_ANSWER'（HotpotQA `dev_3`）。
- 例数很少，但都会直接变成错答。

---

## 4. AgentIR（Qwen3.5-9B + 官方 OSS client + AgentIR-4B）

### 4.1 全量画像

| 数据集 | EM | max_turns | 作答精度（对 / 作答） | 答案均长（词） | 丢弃并行调用的题 |
|---|---:|---:|---:|---:|---:|
| Bamboogle | 43.2 | 42/125 | 65.1% | 3.1 | 0 |
| PopQA | 28.8 | 161 | 42.5% | 4.8 | 0 |
| TriviaQA | 52.4 | 101 | 65.7% | 4.4 | 4 |
| NQ | 15.4 | 157 | **22.4%** | **14.8** | 0 |
| HotpotQA | 31.0 | 185 | 49.2% | 5.0 | 47 |
| 2Wiki | 24.6 | 300 | 61.5% | 4.5 | **161** |
| MuSiQue | 10.8 | 339 | 33.8% | 6.8 | 14 |
| BCP | 31.0（judge） | 325 | **88.6%**（155/175） | 2.8 | 19 |

### 4.2 max_turns：答案在手仍在核验
- 抽样的 16 例，检索次数都等于轮数上限，没有轮次被解析失败吃掉。问题纯粹是到最后一轮还在搜。
- **已拿到答案仍核验到超时（5/16）**：
  - Bamboogle `test_20`：第 5 次检索找到 "Quezon City"，reasoning 写 "Excellent! I found it!"，第 6 次仍去 verify。
  - NQ `test_984`、TQ `test_10085`、TQ `test_5613`。
  - BCP 295：第 4 次检索已出现正确候选，第 12 次拿到关键证据，最后 4 次用几乎相同的 query 核验一个次要约束。这题 GrepSeek 答对了。
- **有明显候选不肯提交（2/16），金标已在结果里却没注意（2/16）**。
- 合计 9/16 本可在预算内提交正确答案。全量 budget_fail_gold_seen：Wiki 529 / 1,285，BCP 119 / 325。
- **近重复 query 加结果饱和**：14/16 至少有 3 条 query 只差引号、大小写或补了一个猜测的年份；后期每次 top-3 中至少 2 篇是已经见过的文档。
- **丢弃并行调用**（每次回复只执行第一个工具调用，这是公平性设置）：
  - 2Wiki 有丢弃的题 max_turns 73%，没有丢弃的 54%。比较题天然想同时查两个实体。
  - 但 HotpotQA 方向相反。总体是局部轻度有害，每题最多损失约 1 轮。

### 4.3 reasoning-aware 检索的两个副作用
- **前缀拖拽（抽样至少 8 例）**：query 已经换成新实体，返回的仍是上一步实体的文档。原因是 embedding 输入里 reasoning 占了大头，query 只占约 2–15%。
  - 2Wiki `dev_11843`：第 2 次 query "Son Of D'Artagnan film director" 返回的 3 篇全是 Wings for the Eagle 和 Lloyd Bacon，这些都是上一步的实体。
  - 第 3 次 query 只改了大小写，换了一段 reasoning 后，返回的 3 篇完全不同。
- **幻觉被检索放大**：2Wiki `dev_6253`。首轮 7,185 token 的规划在任何检索之前就臆造出一位导演；第一次检索 top-1 就是此人传记，之后每轮都被拉回，最终答错城市。
- **把 dense 检索当关键词引擎用**：16 个 max_turns 中 11 个用了引号 query；BCP 598 有 9/13 次是引号短语。这些算子对 dense 检索几乎不起作用。
- **罕见专名查不到**：HotpotQA `dev_2343`（蛾类学名）、PopQA `test_3789`、2Wiki `dev_11843`。后两题 GrepSeek 都答对了。

### 4.4 答案形式：EM 27.8，cover-EM 42.8
- 官方 BCP QUERY_TEMPLATE 是为 judge 设计的（"Explanation / Exact Answer / Confidence"），没有短答约束。
- Wiki 的 970 个错答中，513 个（53%）与金标互相包含。抽样的 20 个包含型错答里 19 个语义正确。常见形式：
  - yes/no 写成整句；
  - 整句解释；
  - 并列或多标签，如 "UNESCO and the International Labour Organization (ILO)"；
  - 括号别名，如 "Lockheed (SR-71 Blackbird)"；
  - 多余限定，如 "May 27, 1827" 对 "1827"。
- 形式问题约占 Wiki 错答的 55–70%。用 cover-EM 比较时，AgentIR 与 GrepSeek 的差距从 17.4 缩小到 6.7 个点（42.8 vs 49.5）。

### 4.5 协议异常（约 7 / 3,625，非系统性）
- MuSiQue `dev_1453`：末轮输出了畸形的 XML 工具调用（缺 `<function=search>`），parser 不识别，被当成最终回答，记为 no_answer。
- 5 例在思考里写了字面的 `<tool_call>`，reasoning 在此截断，剩下的思考被当成最终回答。NQ `test_1219` 就是这样：模型说 "I will simulate the search result"，0 次检索，引用了虚构的 docid。
- 全量另有 29 个 Wiki 作答是 0 次检索，其中 9 个正确。
- 2Wiki `dev_79`：1 次客户端 PydanticUserError，计作 1 轮。

### 4.6 BCP：为什么它是第一
- 首轮把 3–4 个约束改写成一句自然语言，dense 检索能在没有字面锚点的情况下匹配改写过的描述。例：BCP 87 第 1 次检索就把 gold 文档排在第 2。
- 以中间实体为跳板构造定向 query。例：BCP 1097 第 7 次检索拿到 gold。
- 大多数题能在预算内提交，且提交得很准（89%）。
- 反面：
  - 295 是过度核验，导致预算耗尽。
  - 1208 是锚定首个候选：10 次 query 有 5 次带着它的名字。
  - 598 是引号短语反复召回同一个干扰文档，最后仍答了它（"I must provide an answer"）。

---

## 5. RAG（Qwen3-Embedding-4B top-3 + Qwen3.5-9B 单次阅读）

### 5.1 拒答是 RAG 的头号失败
reader prompt（`prompts/rag.py`，项目自定义）是 "Answer the given question using the retrieved documents below, which may or may not be relevant…"。Qwen3.5-9B 把它读成"不许用参数知识"，top-3 不含答案时就输出 "Not mentioned" 或 "Information not available"。

| | NQ | TQ | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | BCP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG 拒答 | 38 | 20 | 49 | 116 | **302/500** | **210/500** | 37/125 | 158/400 作答 |
| Direct EM | 23.6 | 56.6 | 21.0 | 26.6 | **29.8** | 8.2 | **48.0** | 0.2 |
| RAG EM | 38.8 | 66.0 | 41.8 | 33.2 | **18.8** | 9.8 | **28.0** | 14.4 |

- 同一个 Qwen3.5-9B 在 Direct 下几乎不拒答：2Wiki 4/500，MuSiQue 7，HotpotQA 1，Bamboogle 0。
- 推理里的原话：
  - HotpotQA `dev_1691`："Can I infer 'No'? No… 'Not mentioned' is the text-grounded answer"。Direct 答对。
  - Bamboogle `test_8`："Conflict. Resolution: Prioritize constraint. Output: Not mentioned"。Direct 答对。
- 比较题结构上最容易拒答：抽样的 8/8 个比较题，top-3 只覆盖一边的实体。
- **RAG 对 Direct 的逐题对比**：

  | 数据集 | Direct 对、RAG 错 | RAG 对、Direct 错 |
  |---|---:|---:|
  | 2Wiki | 94 | 39 |
  | Bamboogle | 35 | 10 |
  | PopQA | 28 | 132 |
  | NQ | 33 | 109 |

  检索在单跳上大幅帮忙，在多跳上因为拒答而伤分。
- **对比较的影响**：RAG 是"检索有没有用"的参照线。拒答使它在多跳题上被低估。这个 prompt 已冻结并对三个检索器一致，可以保持不动，但论文里应说明。如果要做敏感性分析，可以试"文档无用时可以用自己的知识"这类措辞。

### 5.2 其他失败
- **多跳题检索不到第二跳**（"错，没检索到"在 2Wiki 319、MuSiQue 387）：top-3 常被第一跳实体占满。
  - MuSiQue `dev_1778`：3 段全是 Pacem in terris。`dev_2038`：3 段全是 Kharg Island。
  - 60 个 Wiki 样本里 43 个的 top-3 至少有 2 段来自同一条目。
- **文档里有答案仍读错（10/54）**：干扰项或过时文档压过正确先验。
  - Bamboogle `test_70`：Nimitz 文档说它是最大航母，于是答 "Chester W. Nimitz"。Direct 答对 Gerald R. Ford。
  - 答错槽位：HotpotQA `dev_5752` 答了出发地机场。
  - 两头押注：PopQA `test_7260` 答 "Avi Nesher and Frank Borzage"。
- **从错误实体的文档里硬凑答案（6/54）**：TQ `test_10291` 从 ruddy duck 的学名 *jamaicensis* 推出 "Jamaica"。
- **失控**：Wiki 12 次单次 reader 调用写满 16K，completion 15.8K token，而 prompt 只有约 1K。BCP 有 100 次撞到 10K 上限。形态有三种：候选逐字母枚举、每轮换一个填充词的自检循环、在 "Football" 和 "football" 之间反复犹豫。

### 5.3 RAG 比 GrepSeek 强在哪
一次检索就把桥接条目和答案条目都放进了 top-3：

| 案例 | RAG 怎么对的 | GrepSeek 怎么错的 |
|---|---|---|
| HotpotQA `dev_5144` | 桥接页和答案页同在 top-3 | 停在中间实体 |
| Bamboogle `test_46` | Augustine Washington 条目里就写了他父亲，两跳在一页内完成 | 上下文写满 |
| HotpotQA `dev_2317` | Mos Def 条目提到 Talib Kweli | 上下文写满 |

在 RAG 赢的 14 个 Wiki 抽样里，GrepSeek 有 6 个根本没作答。

---

## 6. 跨方法的评测与协议问题

### 6.1 EM 对 cover-EM（Wiki 3,125 题，`answer_form.json`）

| run | EM | cover-EM | 差 | 错答中"并列多个候选" | 错答中"答案包含金标" |
|---|---:|---:|---:|---:|---:|
| GrepSeek | 45.2 | 49.5 | +4.3 | 70 | 65 |
| Search-R1-14B qwen3 | **47.9** | **51.4** | +3.5 | 27 | 81 |
| Search-R1-14B e5 | 46.7 | 49.6 | +2.9 | 22 | 68 |
| Search-o1 qwen3 | 39.6 | 45.0 | +5.4 | 106 | 61 |
| RAG qwen3 | 34.5 | 40.4 | +5.9 | 104 | 81 |
| AgentIR | 27.8 | 42.8 | **+15.0** | 283 | 187 |

- **排名在两个口径下基本一致**，只有 AgentIR 从垫底升到与 RAG 相当。
- 建议论文主表用 EM，附录给 cover-EM 或 LLM judge 的敏感性分析。
- 抽样中还发现几类不包含型的口径问题，只能靠 judge 或别名处理：
  - 国籍形容词对国名（"American" vs "United States"）；
  - 数字的英文写法（"fourth" vs "4"）；
  - 单位后缀（"12 episodes"）；
  - 标准化把 "/" 直接删掉，"Musician/Singer" 变成 "musiciansinger"。

### 6.2 共用解码（presence_penalty 2.0）的副作用

对照 pilot（`scripts/decoding_pilot2.py`）：用正式运行的同一份 903a200 harness，只把 presence_penalty 从 2.0 改成 0，其余设置完全不变。比较脚本是 `scripts/compare_pp2.py`。

| run / 数据集 | 指标 | pp 2.0（正式） | pp 0（pilot） | EM（2.0 → 0） | McNemar |
|---|---|---:|---:|---:|---|
| Search-R1-14B e5 / 2Wiki | no_answer；每千轮 dead-EOS | 29；19.6 | 17；11.5 | 44.0 → 44.0 | 36 / 36，p = 1.0 |
| Search-R1-14B e5 / BCP | no_answer；每千轮 dead-EOS | 128；112.7 | 80；69.7 | （未跑 judge） | — |
| Search-o1 bm25 / NQ | 有拼错标签的题；no_answer | 25；12 | 9；4 | 27.0 → 28.0 | 26 / 31，p = 0.60 |

结论：
- **penalty 确实放大了这两种格式失败**：Search-o1 的拼错标签约为无 penalty 时的 2.8 倍，Search-R1-14B 的中途 EOS 约 1.6 倍。机理与 §3.2 的假设一致：结束标签里与开始标签共享的子词被惩罚了。
- **但它们不是主因，对 EM 的影响也在噪声内。**
  - 不加 penalty，Search-R1-14B 仍有 17 / 80 个 no_answer，Search-o1 仍有 9 题拼错标签。这是模型自身的习惯。
  - penalty 2.0 救回来的那些题多数本来也答不对，所以 EM 不变（2Wiki）或只差 1 个点且不显著（NQ）。
  - 反过来，penalty 也在抑制失控循环：Search-o1 NQ 的上下文写满在 pp 0 下从 1 次升到 9 次。
- **BCP 是唯一可能有实际影响的地方**：Search-R1-14B 去掉 penalty 后多作答 48 题。按它在 BCP 上的作答精度（24 / 372 ≈ 6.5%）粗估，judge 准确率约 +0.6 个点，排名不变。精确数字需要对 pilot 跑一次 judge（Qwen3-32B，TP=4，约半小时）。
- **建议**：维持统一解码（presence_penalty 2.0）不变，在论文里把这一点写成 caveat 并附上这张表。

### 6.3 预算耗尽不强制作答
- 所有方法共用"6 次 assistant 调用"（BCP 16 次），都没有末轮强制作答，这对所有方法是一致的。
- 受影响最大的是爱核验的 agent：AgentIR 41%、Search-o1 25%、GrepSeek 18%。Search-R1 不受影响，它从不用完预算。
- 这是方法本身的行为，不建议改动正式设置。但在分析里应区分"答不出"和"没来得及答"。

### 6.4 标注噪声（所有方法都错，或预测更对）

| 数据集 | 案例 |
|---|---|
| NQ | `test_727`、`test_718`、`test_1553`、`test_1128`、`test_2003`、`test_2537`（钟面上的 4 常写作 IIII，gold 只给了 IV）、`test_1173` |
| PopQA | `test_9057`、`test_8000`（问 Delhi 是哪里的首都，gold 只有历史政权）、`test_11035`（四色国旗，gold 只收 red） |
| HotpotQA | `dev_1528`、`dev_633`（gold 是整句）、`dev_2549`、`dev_6989`（gold 是出生全名）、`dev_926` |
| MuSiQue | `dev_71`、`dev_1241`（时效性） |
| TriviaQA | `test_4131`（gold 已过时） |
| Bamboogle | `test_62`（gold 晚于语料） |

### 6.5 BCP judge 和分析启发式的局限
- **judge 偏严**（只写机制）：
  - AgentIR 的 BCP 192：预测与 gold 只差一个称谓前缀和连字符，judge 的 reasoning 写了 "the name is the same"，仍判错。
  - Search-o1 的 BCP 71、72：预测是 gold 去掉开头一个修饰词，判错。
  - 反过来 BCP 1135：F1=0（Unicode 标点差异）却判对，说明 BCP 不能用字符串指标。
  - 建议人工抽检约 50 条 judge 判错。
- **"检索到过 gold"启发式**：
  - 对比较题必然误报，因为金标就是题中实体。
  - 对短金标会误报，如 "Iran"、"201" 匹配到 "2017"。
  - 对**非 ASCII 金标会漏报**：语料和工具输出里是 `\uXXXX` 转义。Wiki 共有 277 题含非 ASCII 金标，GrepSeek 的"没见过 gold"里有 55 题属于这一类。
  - 所以第 1 节的"检索到 / 没检索到"只适合看量级。

---

## 7. 对 GrepSeek 的启示

1. **别人赢 GrepSeek 的地方，主要靠语义匹配**：
   - 改写过的线索：TQ `test_3310`、`test_4681`，MuSiQue `dev_153`。
   - 拼写和音译变体：NQ `test_1216` 的 Neel / Nil Darpan，PopQA `test_13618` 的 Oporto / Porto。
   - 一次把两跳放进同一个 top-3：RAG 和 Search-o1 的 RiD。
   - BCP 上的多约束改写：AgentIR。

   这些正好对应 GrepSeek 报告里的 §3.3.4、§3.4 和 BCP 一节。最直接的办法是在 grep 落空时**允许一次语义回退**，或在工具里提供大小写、重音、词干都不敏感的匹配。
2. **别人输给 GrepSeek 的地方，是精确性**：
   - 数字、年份、序号、形近词：Search-o1 把 Chey Chettha III 检索成 IV/V，把 Cryptanthus 检索成 Cryptantha。
   - 罕见专名和精确标题：AgentIR 的 R4，Search-R1 的编造桥接。

   混合检索必须保住这块优势。
3. **别学别人的坏习惯**：
   - AgentIR 的过度核验；
   - Search-o1 的同义改写空转；
   - RAG 的"文档没写就拒答"；
   - Search-R1 的编造桥接。

   GrepSeek 自己也有前两种的变体：见过 gold 仍继续搜，以及 AND 过滤滤空后只换个词再试。
4. **答案形式**：
   - Search-R1 的答案最短、最像 EM 金标（均长约 2 个词），并列答案最少。
   - GrepSeek 的 F1 奖励让它比 Search-R1 多出 2–3 倍的并列答案。
   - 用 EM 奖励或给"只给一个实体"加约束，就能把这部分分数拿回来。

---

## 附录：数字从哪来

| 内容 | 出处 |
|---|---|
| 失败类别、每题检索次数、budget_fail_gold_seen | `analysis/failures/summary.json` |
| EM 对 cover-EM、错答形式 | `analysis/failures/answer_form.json`（`scripts/answer_form.py`） |
| Search-R1 dead-EOS | `analysis/failures/sr1_no_answer.json`（`scripts/sr1_no_answer.py`） |
| Search-o1 拼错标签 | `analysis/failures/tag_errors.json`（`scripts/tag_errors.py`） |
| 解码 pilot | `/scratch/a32du/scaleseek_v2/dev/pilots/sr1_14b_e5_*_pp0/`、`/scratch/a32du/scaleseek_v2/dev/pilots/so1_bm25_nq_pp0/`（`scripts/decoding_pilot2.py`，与正式运行同一份 903a200 harness） |
| 抽样案例 | `analysis/failures/cases/others/<run>/<ds>/<category>/<id>.txt` |
