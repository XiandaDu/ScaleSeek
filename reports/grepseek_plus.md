# GrepSeek+：方法、验证与结果

2026-11-18。对象：GrepSeek 官方 checkpoint `alireza7/GrepSeek-Qwen3.5-9B-GRPO` 和官方 `inference/` harness，外加我们在 harness 一侧的五个改动。零训练。

**要点**
1. 7 个 Wiki 测试集上严格 EM 从 45.2 升到 **48.7**（+3.5，配对检验 p = 1.5e-7），F1 从 54.0 升到 58.4。与最强基线 Search-R1-14B + Qwen3-Emb-4B（47.9）持平（p = 0.34）。BrowseComp-Plus 上从 21.6 升到 25.8（p = 0.020），仍低于 AgentIR 的 31.0。
2. 提升来自两条路径：没给出答案的题从 17.7% 降到 7.6%；工具输出里出现过正确答案的题从 58.0% 升到 65.1%。
3. 单个改动在验证集上都不显著，组合才显著。消融里只有“标题优先”单独显著。
4. 配置只用验证集选，测试集只跑了一次。验证集上的提升（+3.5）和测试集一致。

---

## 1. 方法

**不变的东西**：同一个 checkpoint；每题最多 6 轮；16K 上下文；每次工具输出上限 2,048 tokens；解码与所有方法相同（temperature 0.6、top_p 1.0、presence_penalty 2.0）。仍然只用 grep，不建索引，不用 dense 检索。

**五个改动**（`configs/methods/grepseek_plus.yaml` 的 `harness.variants`）

| 名称 | 做什么 | 针对的失败（`grepseek_failures.md`） |
|---|---|---|
| `budget` | 最后一轮之前、或上下文快满时，在工具返回里提示“现在作答”；工具输出按剩余空间截短，始终给作答留出 1,536 tokens | 没给出答案 17.7%（轮数耗尽 11.1%，上下文写满 6.6%） |
| `dedup` | 工具返回去掉 `information_lines`（stdout 的逐行重复） | 上下文写满；重复字段占写满时窗口的约 38% |
| `fold` | 固定字符串模式里的非 ASCII 字符改写成语料里的 `\uXXXX` 形式；结果为空时加 `-i` 重试 | 大小写和 Unicode 转义造成的空返回（387 次调用） |
| `relax` | 结果为空而第一段有命中时，去掉最后一个过滤段重试，再不行去掉全部过滤段，并在返回里说明 | AND 过滤词把结果滤空（1,000 次调用） |
| `title` | 命中按标题分层排序：实体自己的条目（`X`、`X (…)`）最前，标题含 X 的其次，其余按文件顺序 | 按文件顺序 + `head`，36% 的调用被截满 |

改动做了什么都写在工具返回的 `notice` 字段里；`command` 字段保持模型原样。

**实现**（`scaleseek/grepseek_plus.py`）
- 官方循环 `inference/agent.py` 一行没改。只拦截三个调用点：
  - `client.chat.completions.create`：统计当前题的轮数和上下文用量；
  - `inference.agent.run_tool`：执行命令并加上面的兜底和排序；
  - `inference.run.run_agent_on_example`：每题重置状态。
- 由 `scaleseek/grepseek_entry.py` 在 `settings["variants"]` 非空时安装；GrepSeek 基线不经过这段代码。
- 改动全关时，工具返回与官方逐字节相同（7 条命令核对）。
- 标题分层只对 `rg|grep -F "X" corpus.jsonl [| 过滤段] [| head -n K]` 这种写法生效（模型 9,449 次调用几乎全是这一种）。结果被 `head` 截满时，再跑最多两次标题锚定的扫描；没截满时在本地重排。
- 语料格式的两个细节：标题只有含空格时才带引号；正则里不能有 `|`（官方校验按 `|` 切分命令）。

---

## 2. 怎么选的

**验证集**：500 道 NQ dev + 500 道 HotpotQA dev（不在测试子集里）。只用这两个数据集，与 GrepSeek 的训练域一致；这些题不在它的训练数据里。代码 `scaleseek/grepseek_pilot.py`，输出 `$ROOT/pilots/grepseek/<name>/`。每个配置跑一遍，temperature 0.6。

**噪声**：同一配置跑两遍，官方 46.5 / 46.1，组合版 50.1 / 49.4。1 分以内的差别不能当结论。

**单个改动**（严格 EM，1,000 题；官方 46.5）

| 配置 | NQ | HotpotQA | 合计 | 相对官方 |
|---|---:|---:|---:|---|
| 官方 | 40.8 | 52.2 | 46.5 | — |
| budget | 41.2 | 51.2 | 46.2 | −0.3 |
| budget + countdown | 40.8 | 51.8 | 46.3 | −0.2 |
| dedup | 41.8 | 50.4 | 46.1 | −0.4 |
| 每轮思考上限 2,048 | 40.0 | 50.0 | 45.0 | −1.5 |
| fold | 40.6 | 51.8 | 46.2 | −0.3 |
| relax | 40.8 | 52.0 | 46.4 | −0.1 |
| title | 44.0 | 51.6 | 47.8 | +1.3（p = 0.30） |
| soft（按满足过滤词数排序） | 39.4 | 49.4 | 44.4 | −2.1（p = 0.07） |
| **budget + dedup + fold + relax + title** | **44.4** | **55.8** | **50.1** | **+3.6（p = 0.002）** |
| budget + dedup + fold + soft + title | 44.2 | 55.4 | 49.8 | +3.3（p = 0.006） |
| fold + soft + title | 43.6 | 52.6 | 48.1 | +1.6（p = 0.20） |

**组合 vs 官方**：两边各跑两遍，四种配对全部显著（+2.9 到 +4.0）。合并 2 × 1,000 条：46.3 → 49.8（+3.5，赢 162 输 93，p < 0.001）；NQ 41.0 → 43.1，HotpotQA 51.6 → 56.4。

**逐个去掉一个改动**（完整组合两次平均 49.8）

| 去掉 | EM | 相对完整组合 | 显著 |
|---|---:|---:|---|
| title | 47.0 | −2.8 | 是（p = 0.009 / 0.037） |
| dedup | 48.3 | −1.5 | 否 |
| relax | 48.7 | −1.0 | 否 |
| fold | 48.8 | −1.0 | 否 |
| budget | 49.2 | −0.5 | 否 |

只开 budget + dedup 是 47.4，只开 title 是 47.8。没有哪个改动拖后腿，最终配置保留全部五个（与试跑前定的组合相同）。

**为什么组合才有效**：只开 budget 时，多答出来的题大多答错，因为证据没变好（作答的题里 EM 从 50.5 降到 48.3）。title 让证据变好，dedup 腾出上下文多查几次，budget 把这些转成答案。

**机制**（验证集）：工具输出里出现正确答案的题，官方 62.3%，只开 title 68.2%，组合 70.9%，去掉 title 的组合 65.5%。见到答案时答对率约 63%，没见到约 17%，各配置下几乎不变。多见到 8.6 个点折合约 +4 分 EM，与实测吻合。

**没采用的**
- `countdown`（每轮报剩余次数）：没有收益。
- 每轮思考上限 2,048：变差。
- `soft`：单独开时每次返回更多行，上下文写满的题从 41 道翻到 82 道；与其他改动组合后和 `relax` 没有差别。

---

## 3. 测试集结果

7 个 Wiki 测试子集，共 3,125 题。全部题在分母里，没给出答案算 0 分。

**严格 EM**

| | NQ | TQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GrepSeek | 39.0 | 70.4 | 44.4 | 47.6 | 46.8 | 21.6 | 51.2 | 45.2 |
| **GrepSeek+** | 42.2 | 69.4 | 49.2 | 51.2 | 57.4 | 22.6 | 50.4 | **48.7** |
| 差值 | +3.2 | −1.0 | +4.8 | +3.6 | +10.6 | +1.0 | −0.8 | +3.5 |
| p（配对 McNemar） | 0.10 | 0.47 | 0.009 | 0.025 | <0.0001 | 0.61 | 1.0 | 1.5e-7 |
| Search-R1-14B + Qwen3-Emb-4B | 48.6 | 69.0 | 48.6 | 47.0 | 48.4 | 25.0 | 52.0 | 47.9 |
| Search-R1-14B + e5 | 51.2 | 69.2 | 45.0 | 46.6 | 44.0 | 23.0 | 52.0 | 46.7 |

**token F1**

| | NQ | TQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GrepSeek | 51.9 | 78.7 | 50.4 | 59.3 | 51.7 | 30.0 | 60.8 | 54.0 |
| **GrepSeek+** | 54.9 | 78.7 | 56.8 | 63.0 | 62.6 | 32.3 | 66.5 | **58.4** |
| Search-R1-14B + Qwen3-Emb-4B | 57.9 | 76.7 | 54.3 | 58.7 | 54.7 | 34.4 | 65.5 | 56.5 |

- 合并 3,125 题：赢 272 题，输 162 题。
- 对 Search-R1-14B + Qwen3-Emb-4B：48.7 对 47.9，赢 329 输 304，p = 0.34。只能说持平。

**结局构成**（占全部题）

| | 答对 | 没给出答案 | 见过 gold 答错 | 没见过 gold 答错 | 工具输出里出现过 gold |
|---|---:|---:|---:|---:|---:|
| GrepSeek | 45.2% | 17.7% | 14.4% | 22.7% | 58.0% |
| GrepSeek+ | 48.7% | 7.6% | 19.0% | 24.7% | 65.1% |

| 数据集 | 没给出答案 | 出现过 gold | 作答的题里的 EM |
|---|---|---|---|
| NQ | 7.4% → 1.6% | 63.0% → 69.6% | 42.1 → 42.9 |
| TriviaQA | 5.2% → 4.0% | 73.8% → 78.0% | 74.3 → 72.3 |
| PopQA | 18.2% → 5.0% | 57.4% → 68.0% | 54.3 → 51.8 |
| HotpotQA | 13.2% → 6.8% | 61.0% → 67.4% | 54.8 → 54.9 |
| 2Wiki | 27.4% → 10.2% | 57.4% → 66.2% | 64.5 → 63.9 |
| MuSiQue | 35.0% → 18.4% | 38.4% → 43.2% | 33.2 → 27.7 |
| Bamboogle | 16.0% → 4.8% | 45.6% → 56.8% | 61.0 → 52.9 |

- GrepSeek 原来没答的 552 题，GrepSeek+ 答了 380 题，其中 123 题答对。
- 预算提示：567 题被提示作答，354 题在下一轮作答（62%），其中 105 题答对。
- 兜底触发次数：大小写重试 276 次，放宽过滤 976 次。

### 3.1 BrowseComp-Plus

500 题，Qwen3-32B 判分（与其他方法相同）。预算 16 轮、131K 上下文、每次工具输出上限 6,000 tokens。

| | judge 准确率 | 给出答案 | 上下文写满 | 轮数耗尽 |
|---|---:|---:|---:|---:|
| GrepSeek | 21.6（108 题） | 306 | 192 | 0 |
| **GrepSeek+** | **25.8**（129 题） | 377 | 91 | 30 |
| AgentIR | 31.0（155 题） | — | — | — |

- 对 GrepSeek：+4.2，赢 48 输 27，p = 0.020。
- 对 AgentIR：低 5.2，赢 81 输 107，p = 0.068，差距不显著但方向是落后。
- 这里的语料一行是一整篇网页，开头是 `---\ntitle: …`，标题分层不生效；起作用的只有 budget、dedup 和两个兜底。配置是在 Wiki 式的验证题上选的，没有针对 BCP 调过。

**按论文方式切段的 BCP**（2026-11-21；700 词一段、步长 630，共 862,948 段，论文是 864,837 段；配置 `grepseek_passages.yaml`、`grepseek_plus_passages.yaml`）

| | 整篇网页一行 | 切段 |
|---|---:|---:|
| GrepSeek | 21.6 | 29.6 |
| GrepSeek+ | 25.8 | 29.0 |

- 切段让 GrepSeek 涨了 8.0 分（p = 0.0002），接近论文的 26.0–34.2。我们之前的 BCP 基线偏低是语料格式造成的。
- 在切段设置下 GrepSeek+ 没有额外收益（赢 42 输 45）。它在整篇网页设置下的提升来自修补“一次只看到一篇文档开头”，切段已经解决了这个问题。
- 所以 GrepSeek+ 的结论应限定在 7 个 QA 数据集上；BCP 上它与 GrepSeek 持平。

**用量和耗时**（Wiki 7 个数据集）

| | GrepSeek | GrepSeek+ |
|---|---:|---:|
| 每题调用次数 | 3.08 | 2.96 |
| 返回空的调用 | 19.9% | 5.7% |
| 每题工具输出 tokens | 6,526 | 3,658 |
| 每题上下文 tokens | 7,192 | 5,499 |
| 每次工具调用耗时 | 2.3 s | 6.3 s |
| 每题模型生成耗时 | 40.6 s | 35.7 s |

工具变慢是因为标题分层和兜底需要额外扫描语料（一次全量扫描空载约 1.4 秒，带 `-i` 约 12 秒）。

---

## 4. 没解决的问题

- **见过 gold 仍答错**从 14.4% 升到 19.0%：更多题检索到了答案，读错和答案形式的问题跟着变多。需要训练才能修。
- **MuSiQue 基本没动**：作答比例从 65% 升到 82%，但作答的题里 EM 从 33.2 降到 27.7；只有 43% 的题检索到 gold。多跳检索仍是瓶颈。
- **思考失控变多**：出现超过 8K 字符单轮的题从 346 道增加到 473 道，原因没查。
- **TriviaQA、Bamboogle 没有提升**（−1.0 / −0.8，不显著）。
- **BrowseComp-Plus 有提升但还不够**：见 §3.1。上下文写满的题仍有 91 道，与 AgentIR 的差距没有补上。

---

## 5. 需要如实说明的

- 改动的设计参考了在**测试子集**上做的失败分析（`grepseek_failures.md`）。配置的选择只用了验证集；测试集只跑了一次。
- 模型是 GrepSeek 官方 checkpoint，没有任何训练。这是 harness 一侧的改进，不是新训练的模型。
- 每个配置只跑一遍（temperature 0.6）。验证集上量过噪声；测试集没有重复。
- 预算提示本身不是新东西（BATS、BCAS、AnySearch）；harness 与 RL 结合见 Harness-1；带计数的工具反馈见 IndexAct（基于索引）。

---

## 6. 复现

```bash
# 验证集试跑（一个配置一个作业，约 45 分钟，1 张卡）
bash submit.sh grepseek-pilot --name base
bash submit.sh grepseek-pilot --name all --variants budget,dedup,fold,relax,title

# 测试集
bash submit.sh eval --method grepseek_plus --dataset hotpotqa     # 7 个 Wiki 数据集各一次
bash submit.sh eval --method grepseek_plus --dataset browsecomp_plus --time 36:00:00
bash submit.sh judge --dataset browsecomp_plus --runs grepseek_plus
bash submit.sh score --dataset hotpotqa
```

| 内容 | 位置 |
|---|---|
| 方法 | `scaleseek/grepseek_plus.py`（commit `7e0de7e`、`186eb92`） |
| 正式配置 | `configs/methods/grepseek_plus.yaml`（commit `460ea57`） |
| 验证集试跑 | `scaleseek/grepseek_pilot.py`；结果在 `$ROOT/pilots/grepseek/` |
| 测试结果 | `$ROOT/runs/<dataset>/grepseek_plus/` |
| 验证集对比、消融、机制 | `reports/scripts/gplus_pilot_cmp.py`、`gplus_pilot_loo.py`、`gplus_pilot_mech.py` |
| 测试集对比、机制 | `reports/scripts/gplus_test_cmp.py`、`gplus_test_mech.py` |
| 工具层和循环的无模型测试 | `reports/scripts/gplus_tool_test.py`、`gplus_loop_test.py` |

脚本里的数据路径写死为 `/scratch/a32du/scaleseek_v2`，都在计算节点上运行。
