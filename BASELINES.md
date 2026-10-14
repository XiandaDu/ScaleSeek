# Baseline 评测：从零重跑

本文件只覆盖 baseline，环境、数据、索引、各方法的实现和全部实验都要从头完成，不复用任何已有的结果、索引或缓存。

## 0. 总要求

- **参数写进 yaml 文件**：模型、revision、采样参数、轮数、top-k、BM25 参数、prompt hash 都放在这里。
  代码只从这个文件读取参数，不允许在脚本里藏默认值。
- **prompt**：自己写的 prompt 作为 Python 常量放在 `prompts/`；从官方仓库照搬的 prompt 必须与固定 commit 的原文做自动 byte diff。
- **全部固定版本**：所有 HF 模型和数据记录 `repo_id + revision`，所有官方仓库记录 commit（见 §6），不能写 `main`。
- **固定 seed 抽样**：每个数据集抽 `min(数据集大小, 500)` 题，具体规则见 §2。所有方法共用同一份子集，
  不允许另用 `-n`、`--offset` 或 first-N。smoke 测试的数字不得进结果表。
- **解码设置对齐 GrepSeek 论文**：所有方法统一使用 GrepSeek 论文 Table 5 的解码设置（见 §4 开头），不再沿用各方法原仓库的默认值。
- **每条输出都要能溯源**：数据集 manifest、方法 config id、模型 revision、索引 manifest、prompt hash、harness commit、随机种子和完整轨迹。

## 1. 要跑什么

| 方法 | 运行数 | 生成模型 | 检索 |
|---|---:|---|---|
| Direct | 1 | Qwen3.5-9B | 无 |
| RAG | 3 | Qwen3.5-9B | bm25 / e5 / qwen3_emb_4b，top-3 |
| Search-R1 | 6 | 官方 7B 和 14B 两个 checkpoint | 每个模型 × 3 个检索器，top-3 |
| Search-O1 | 3 | Qwen3.5-9B | bm25 / e5 / qwen3_emb_4b，top-3 |
| GrepSeek | 1 | 官方 GrepSeek-Qwen3.5-9B-GRPO | 无检索器，用 shell/grep 访问物化后的语料 |
| AgentIR | 1 | Qwen3.5-9B | AgentIR-4B，top-3 |

每个数据集共 15 个 run。BrowseComp-Plus 另外需要一个独立的 judge（Qwen3.5-9B）。

## 2. 数据集

| 数据集 | 来源 | split | 全量（预期） | 评测题数 |
|---|---|---|---:|---:|
| PopQA | FlashRAG | test | 14,267 | 500 |
| TriviaQA | FlashRAG | test | 11,313 | 500 |
| NQ | FlashRAG | test | 3,610 | 500 |
| HotpotQA | FlashRAG | dev | 7,405 | 500 |
| 2WikiMultiHopQA | FlashRAG | dev | 12,576 | 500 |
| MuSiQue | FlashRAG | dev | 2,417 | 500 |
| Bamboogle | FlashRAG | test | 125 | 125（全量） |
| BrowseComp-Plus | `Tevatron/browsecomp-plus` | — | 830 | 500 |

- FlashRAG 数据来自 `RUC-NLPIR/FlashRAG_datasets`。
- 前七个数据集共用 Wiki-18 语料，用 GrepSeek 论文同一份：`PeterJinGo/wiki-18-corpus`（`wiki-18.jsonl.gz`，约 2,100 万段，约 14GB）。
  BrowseComp-Plus 用它自己的完整语料，两者的索引不能混用。
- 每次正式运行前做 preflight：核对来源 revision、原始行数、规范化后行数、唯一 ID 数、重复 ID、空问题、空答案和文件 SHA256。
  与预期不符就停下来查原因，不能自己截断凑数。
- **抽样规则**：
  - 先对全量数据做 preflight，再抽样。题数 ≤ 500 的数据集直接全量使用。
  - seed 写进 yaml。抽样结果只能由 seed 和规范化后的数据文件决定，与机器和运行顺序无关。
  - 每个数据集只抽一次，生成子集文件和子集 manifest，内容包括 seed、全量文件的 SHA256、子集 ID 列表和子集 ID 的 SHA256。
  - 所有方法、所有检索器都读同一个子集文件；以后重新抽样必须换新的 manifest，不能覆盖旧的。
- 顺序：**先只跑 PopQA，汇报结果并确认没问题后，再跑其余七个数据集。**

## 3. 检索器

三个后端必须基于**同一份语料、同一套 doc_id、同样的文档文本**建索引，并为每个索引记录文档数、字节数、SHA256、构建参数和构建命令。

| ID | 模型 | 设置 |
|---|---|---|
| `bm25` | Pyserini / Lucene BM25 | `k1=1.2`、`b=0.75`，固定不扫参 |
| `e5` | `intfloat/e5-base-v2` | query 前缀 `query: `，文档前缀 `passage: `，masked mean pooling，L2 归一化，max_length 256 |
| `qwen3_emb_4b` | `Qwen/Qwen3-Embedding-4B` | 官方 query instruction（`Given a web search query, retrieve relevant passages that answer the query`），last-token pooling，L2 归一化 |

稠密检索索引与 GrepSeek 论文相同：FAISS HNSW，`M=32`，`efConstruction=128`，`efSearch=128`。

另外还需要两样东西：
- AgentIR 需要用 AgentIR-4B 对同一语料编码，单独建一个 FAISS 索引。官方 searcher 能加载 HNSW 的话，用同样的参数；不能的话按官方方式建，并记录下来。
- GrepSeek 需要把语料物化成文件，供 grep 使用。

Wiki-18 和 BrowseComp-Plus 各自都要建一套。

## 4. 各方法设置

### 统一解码设置（所有方法，取自 GrepSeek 论文 Table 5）

| 项 | FlashRAG 七个数据集 | BrowseComp-Plus |
|---|---:|---:|
| temperature | 0.6 | 0.6 |
| top_p | 1.0 | 1.0 |
| 最多 assistant 轮数 | 6 | 16 |
| 最大上下文（`max-model-len`） | 16,384 | 131,072 |
| 每轮生成上限 | 不设，只受上下文限制 | 同左 |

- 不另设 top_k、repetition penalty 或 thinking budget。
- Direct 和 RAG 只调用一次模型，所以轮数这一项对它们不适用。
- BrowseComp-Plus 的 judge 不在此列，设置见下文。

### Direct
- prompt 由本项目自定义（GrepSeek 论文有 Direct 对照，但没有公开 prompt），文档中标注为 project-defined。
- 单次调用，解码按统一设置。

### RAG
- 一次检索取 top-3，把完整 passage 交给 reader，不按字符截断。三个检索器使用完全相同的 reader prompt 和 passage 格式。
- reader prompt 参照 GrepSeek 的 RAG 协议，由本项目自定义并冻结。解码按统一设置。

### Search-R1（7B、14B）
- 模型：`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3` 和 `...-14b-em-grpo-v0.3`。两个都要跑满 3 个检索器。
- 不重新训练。GrepSeek 论文里的 Search-R1 是作者在 Qwen3.5-9B 上重训的，但没有在 HF 上公开，所以这里只能用官方 checkpoint。
  以后如果作者公开了这些 checkpoint，再改用。
- 照搬官方仓库 `PeterGriffinJin/Search-R1` 的 inference prompt、stop token，以及 `<search>/<information>/<answer>` 续写循环；
  使用 checkpoint 自带的 chat template，不额外加 system message。
- 每次检索取 top-3。解码按统一设置：原版的 `B=4`、`max_new_tokens=1024`、`temperature=0.7` 改为 6 轮、不设每轮上限、0.6。

### Search-O1
- 照搬官方仓库 `RUC-NLPIR/Search-o1` 的 `scripts/prompts.py` 和 `scripts/run_search_o1.py`。
- prompt 分支：
  - PopQA、NQ、TriviaQA 用 `get_singleqa_search_o1_instruction`；
  - HotpotQA、2Wiki、MuSiQue、Bamboogle 用 `get_multiqa_search_o1_instruction`；
  - task prompt 走非 QwQ 的 `get_task_instruction_openqa`；
  - Reason-in-Documents 用 `get_webpage_to_reasonchain_instruction`。
- 解码按统一设置，主推理和 RiD 都一样。答案取最后一个 `\boxed{}`。
- 与原版的差异，都要在文档中注明：
  - 原版 `max_turn=15`、单跳最多搜索 5 次、多跳 10 次。这里改为 6 轮，所以搜索上限统一为"轮数 − 1"（FlashRAG 为 5，BCP 为 15）。
    官方 prompt 模板原样照搬（byte diff 仍要通过），搜索次数是模板的 `MAX_SEARCH_LIMIT` 参数，只把传入的值改成这个上限即可。
    多跳 prompt 里显示的次数因此从 10 变成 5。
  - 原版的 `temperature=0.7`、`top_p=0.8`、`top_k=20`、repetition penalty 1.05、`max_new_tokens=8192` 都改为统一设置。
  - 原版用 Bing 搜索、取 top-10；这里换成本地语料、取 top-3。

### GrepSeek
- 直接用官方仓库 `alirezasalemi7/grepseek` 的 `inference/` 作为 harness，模型用官方 checkpoint `alireza7/GrepSeek-Qwen3.5-9B-GRPO`，不能换成未微调的 Qwen3.5-9B。
- 设置与论文 Table 5 完全相同：

  | 项 | FlashRAG 七个数据集 | BrowseComp-Plus |
  |---|---:|---:|
  | 最多 assistant 轮数 | 6 | 16 |
  | 工具输出上限 `tool_max_tokens`（用该 checkpoint 的 tokenizer 计数） | 2,048 | 6,000 |
  | 最大上下文 | 16,384 | 131,072 |
  | temperature / top_p | 0.6 / 1.0 | 0.6 / 1.0 |

- 生成端不设每轮上限，否则 thinking 会被截断。部署时开启 `--reasoning-parser qwen3`。
- Wiki 语料用 `PeterJinGo/wiki-18-corpus`，与论文相同。
- 不要给 GrepSeek 加 BM25 或 E5 检索器。

### AgentIR
- 检索器用 `Tevatron/AgentIR-4B`，生成模型用 Qwen3.5-9B，agent loop 用官方仓库 `texttron/AgentIR` 的原版实现。
- 查询必须同时包含 agent 的 reasoning trace 和它生成的 search query，并加官方 instruction 前缀；向量做 normalize。
- 每次检索取 top-3，与其他 search agent 保持一致（官方原版为 top-5，文档中要注明这一替换）。
- 解码按统一设置：原版最多 100 次迭代改为 6 轮（BCP 为 16 轮）。

### BrowseComp-Plus judge
- 使用独立的 Qwen3.5-9B 服务，`temperature=0`，`top_p=1`，`max_new_tokens=4096`。
- judge prompt 从官方仓库逐字照搬。judge 只能看到问题、gold 答案和 agent 的最终回答，不能复用 agent 的对话上下文，
  也不能把 agent 的自评当作分数。所有方法的 BrowseComp-Plus 结果都用同一个 judge 判分，并保存判卷原文。

## 5. 执行顺序

1. 搭建环境，下载 §6 中的全部模型、数据和官方仓库，并固定版本。
2. 下载数据集并通过 preflight，按固定 seed 抽出各数据集的子集并生成子集 manifest；准备 Wiki-18 和 BrowseComp-Plus 语料，生成语料 manifest。
3. 建索引：bm25、e5、qwen3_emb_4b、AgentIR，以及 GrepSeek 用的物化语料。验证三个检索器的 doc_id 完全对齐，
   并且同一 query 多次检索结果一致。
4. 实现或接入六个方法和 judge，把全部参数写进 `configs/baselines.yaml`，官方 prompt 做 byte diff。
5. smoke 测试：每个方法用真实模型跑几道题，确认"检索 → 工具 → 回答"闭环正常、解析正确（数字不入表）。
6. 正式跑 PopQA 的 15 个 run。允许中断后续跑，但最终输出的 ID 集合必须与子集 manifest 完全一致（算分前自动校验）；
   临时的服务错误要重试，无法恢复的失败要保留并计入失败统计。
7. 汇报 PopQA 结果，确认后再跑其余七个数据集。

## 6. 版本固定

| 对象 | ID | revision / commit |
|---|---|---|
| 生成模型和 judge | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Search-R1 7B | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3` | `395b18f1fecee52f1b51fb22f898c220f0a08ec3` |
| Search-R1 14B | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3` | `d65f11c88d3c129a01466f0c154aaad7d9b09225` |
| GrepSeek | `alireza7/GrepSeek-Qwen3.5-9B-GRPO` | `a79563970cfdd2ced3cc5fde481737d0ebea6fa4` |
| AgentIR | `Tevatron/AgentIR-4B` | `e31abb637caa227c4b7d04176a24ecff1bcb10f4` |
| E5 | `intfloat/e5-base-v2` | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| Qwen3 Embedding | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| FlashRAG 数据 | `RUC-NLPIR/FlashRAG_datasets` | `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c` |
| Wiki-18 语料 | `PeterJinGo/wiki-18-corpus` | `69c1c00ffe7c5554c68d8548355cb22e46aabc51` |
| Search-R1 仓库 | `PeterGriffinJin/Search-R1` | `598e61bd1d36895726d28a8d06b3a15bed19f5d3` |
| Search-o1 仓库 | `RUC-NLPIR/Search-o1` | `c76a700fb2a948039ada577c03b3be958aa57282` |
| GrepSeek 仓库 | `alirezasalemi7/grepseek` | `1f6ea58372defe774213e22c7650b7fd1b842ab8` |
| AgentIR 仓库 | `texttron/AgentIR` | `9626bdc7ab90649608d483e66febe13c6eebab2c` |

BrowseComp-Plus 的 query、语料和 judge prompt 仓库在下载时固定 revision，并补进这张表。

## 7. 汇报内容

- **Wiki QA**：EM、token F1、answered rate、finish reason 分布、parse/tool/API 错误数、平均轮数、平均检索或工具调用次数、总 wall time。
- **BrowseComp-Plus**：judge accuracy、judge 失败数、agent 失败数、平均轮数和工具调用数、总 wall time。
- 每个结果标明 n 和 95% 置信区间。n=500 时 EM 的置信区间约为 ±4.4 个点。
- 显著性检验与 GrepSeek 论文相同：EM 在同一批题目上做配对的 McNemar 检验，与最强 baseline 比较，p<0.05 才算显著。
  不显著的差距不得表述为方法之间的差异。
- 三个检索器的并列表，以及 Search-R1 7B / 14B 的并列表。
- 每个方法抽查几条成功、检索失败、解析失败的轨迹（只作说明，不代替子集上的整体指标）。
- 与论文数字不能直接比较的原因：
  - 只评测了 500 题子集；
  - 所有方法统一用 GrepSeek 的解码设置，而不是各自原版的设置；
  - Search-R1 用的是官方 Qwen2.5 checkpoint，而 GrepSeek 论文里的 Search-R1 是在 Qwen3.5-9B 上重训的；
  - top-k 与原版不同；
  - BCP 的 judge 不同（我们用 Qwen3.5-9B，GrepSeek 论文用 GPT-4.1）。

## 8. 不允许

- 为不同检索器更换 reader prompt、top-k、语料或 passage 预算，却声称只比较了检索器。
- 因为某组 BM25 参数结果更好，就偏离 `1.2/0.75`。
- 把本项目自己写的 prompt 说成官方原文。
- 把 500 题子集上的结果说成完整 split 的结果。
- 不同方法或不同检索器使用不同的子集，或者看到结果后重新抽样。

