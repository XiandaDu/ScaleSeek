# ScaleSeek baseline 修正与全量评测任务

## 总目标

先把所有 baseline 修正成可追溯、可解释、可复现的实现，再进行全量实验。任何实现都必须能回答以下问题：

1. 论文、官方仓库、官方 prompt/harness 和模型权重分别来自哪里？
2. 哪些设置来自原方法，哪些是为了统一比较而进行的替换？
3. 是否使用检索器；若使用，检索器、top-k、索引、query/document 模板和 BM25 参数是什么？
4. generator、judge、采样参数、轮数、工具返回预算和停止条件是什么？
5. 结果能否从固定配置、固定模型 revision、固定数据 manifest 和完整轨迹重新生成？

本任务分为三个阶段。阶段二结束后必须向用户汇报并停止；只有用户明确确认结果可接受后，才能开始阶段三。

---

## 全局实验约束

### 模型

- 除 Search-R1 和 AgentIR 的方法固有模型外，所有 prompt-based generator/agent 统一使用 [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)。
- GrepSeek 使用官方 [`alireza7/GrepSeek-Qwen3.5-9B-GRPO`](https://huggingface.co/alireza7/GrepSeek-Qwen3.5-9B-GRPO)，不能替换成未微调的 Qwen3.5-9B；该 checkpoint 的 backbone 本身是 Qwen3.5-9B，因此满足统一模型要求。
- Search-R1 只保留并分别测试以下两个官方 checkpoint：
  - [`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3`](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3)
  - [`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3`](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3)
- AgentIR 固定使用 [`Tevatron/AgentIR-4B`](https://huggingface.co/Tevatron/AgentIR-4B) 作为 reasoning-aware retriever；其回答 generator 仍使用 Qwen3.5-9B。
- BrowseComp-Plus 的 judge 统一使用 Qwen3.5-9B。agent 和 judge 必须使用独立、明确记录的 prompt；不得把 agent 自评当作正式判分。
- 所有 Hugging Face 模型必须在配置 manifest 中固定 `repo_id + revision/commit hash + dtype`，不能只记录可漂移的 `main`。
- 本项目自写或实质修改的 prompt 必须作为 `prompts/*.py` 中的 Python 常量保存；逐字照抄的第三方 prompt 放在对应 `eval/*.py` 中并自动与固定上游 commit 做 byte diff。训练、评测和 provenance 必须引用同一个常量并对实际传给模型的字符串计算 hash。只由官方外部 harness 消费的 prompt 保留在固定上游 checkout，不在本项目重复复制。

### 统一检索器矩阵

只要替换检索器不会改变方法本身的核心语义，就必须分别测试以下三种后端：

| ID | 模型/实现 | 必须固定的设置 |
|---|---|---|
| `bm25` | Lucene/Pyserini BM25 | `k1=1.2`、`b=0.75`；所有固定 BM25 baseline 禁止自行扫参后挑最好结果 |
| `e5` | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2)，约 110M | 官方 `query:` / `passage:` 前缀；mean pooling、attention-mask、L2 normalize；固定 revision |
| `qwen3_emb_4b` | [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 使用官方 query instruction、官方 pooling/归一化方式；固定 revision |

三个索引必须基于同一份 corpus manifest、同一套 `doc_id` 和同一文档文本，不允许某个检索器使用不同语料快照。每个数据集/语料都要记录：文档数、字节数、SHA256、索引参数和构建命令。

BM25 的 `1.2/0.75` 是本项目统一采用的常见经典设置，不再声称它是 Robertson 唯一规定的参数。ScaleSeek 本身若以“agent 主动选择 `k1/b`”作为方法能力，则保留该能力；它的默认值、缺省值和所有固定 BM25 对照仍必须是 `1.2/0.75`，并在结果中单独标明这是方法特例。

除非方法的官方实现本来就用数百或数千篇候选文档构造外部工作区，否则所有普通 top-k 检索统一为 `top_k=3`。因此 Search-R1、Search-O1、RAG 和 IRCoT 都固定 top-3；不能因为某个检索器在 top-5 上更好而改变。允许保留大 top-k 的方法只有其核心协议明确要求高召回工作区时，例如 DR-DCI 的动态 300–600 篇 pull 和 RISE 的每子查询 `K=1000`。大 top-k 的文档不会直接全部塞进上下文，而是物化到外部 workspace。

### 可替换性边界

| 方法 | 是否跑三检索器矩阵 | 原因 |
|---|---:|---|
| RAG | 是 | 单次检索接口可直接替换 |
| Search-R1 | 是 | GrepSeek 已按 BM25/E5/Qwen embedding 比较该类 baseline；策略和 top-k 可保持不变 |
| Search-O1 | 是 | 以本地 corpus retriever 替换 Bing 后，三种后端可共享同一接口 |
| IRCoT | 是，但仅在官方实现修正通过后保留 | 每一步的 retriever 可替换，推理/交替流程不变 |
| DR-DCI | 否 | 保留官方 task-specific pull retriever：Wiki-18 用 E5，BrowseComp-Plus 用官方 Qwen3-Embedding-8B |
| RISE | 否 | 方法原生使用 BM25 为外部 interaction space 建边界，固定每子查询 K=1000；替换成 dense retriever 不属于本任务 |
| Direct | 否 | 无检索 |
| GrepSeek | 否 | 方法使用 shell/grep 访问物化语料，不是 top-k retriever |
| DCI | 否 | 原方法使用 bash/read/grep 访问 corpus，不是 top-k retriever |
| AgentIR | 否 | AgentIR-4B 本身就是该方法指定的 reasoning-aware retriever；替换后不再是 AgentIR |
| ScaleSeek 主设置 | 否 | 当前方法的工具语义包含 BM25 的 `k1/b` 控制；dense 版本只能作为另行命名的 ablation，不能混入主结果 |

### 数据与评测

- 正式实验禁止 `-n/--n`、`--offset`、随机抽样、固定 1500 子集或只跑 first-N。
- 阶段一允许使用极小 smoke fixture 验证代码，但 smoke 数字不得进入结果表。
- `popqa` 必须唯一表示 FlashRAG PopQA 完整 test split（预期 14,267 条）。删除/迁移 `popqa_full`、`popqa_full14267`、`popqa_longtail` 等会改变数据含义的别名。
- 每次正式运行前执行 dataset preflight：来源、split、原始行数、规范化后行数、唯一 ID 数、重复 ID 数、空问题数、空答案数和文件 SHA256。任一项不符合 manifest 时立即终止。
- Wiki QA 使用统一的 Wiki-18 corpus；BrowseComp-Plus 使用其完整专用 corpus，不能混用 Wiki 索引。
- Wiki QA 至少报告 EM、token F1、answered rate、finish-reason 分布、平均轮数/工具调用数、检索 recall（有 qrels 时）和总 wall time。
- BrowseComp-Plus 报告完整 judge accuracy、judge failure 数、agent failure 数、平均轮数/工具调用数和总 wall time，同时保存原始 judge verdict/reason。
- 所有输出都必须包含 `dataset_manifest_id`、`method_config_id`、generator revision、retriever revision/index manifest、prompt hash、harness commit、随机种子和完整轨迹位置。

---

# 第一阶段：修正并验证全部 baseline

## 阶段目标

这一阶段只解决“代码是否正确、配置是否有出处、不同方法是否公平可比”。不得把已有旧结果当成验收依据。先建立来源与测试，再替换错误实现；全部通过后，删除错误或过时的 baseline 代码、启动脚本和 checkpoint。

## 1. 建立统一基础设施

- [x] 新建 machine-readable 方法配置目录，例如 `configs/baselines/*.yaml`。禁止继续依赖散落在 shell 脚本里的隐式默认值。
- [x] 新建 `BASELINE_SOURCES.md`，逐项固定论文 URL、官方仓库 commit、prompt 文件及行/函数、harness 文件、HF revision 和本地修改说明。
- [x] 将当前 `BM25Retriever`/`E5Retriever` 重构为统一 retriever protocol；参数命名从 `bm25_top_k` 等改成与后端无关的 `retrieval_top_k`，避免 dense 后端继续伪装成 BM25。
- [x] 添加 `qwen3_emb_4b` 索引构建和查询实现，并为 E5/Qwen 分别加入官方前缀、pooling 和归一化单元测试。
- [ ] **服务器验收**：对三个完整索引做同 query 的稳定性测试；本地已完成确定性合并、去重和 manifest 契约测试。
- [x] 正式模式加入 `--full-eval` 或等价保护：检测到 `--n`、`--offset`、抽样文件或不匹配的数据 manifest 时拒绝启动。
- [x] 修正 dataset registry，使 `popqa` 直接加载完整 test split；删除“popqa 实际指向 longtail”之类的 override。
- [x] 统一结果 schema、错误重试、resume 去重和配置指纹。resume 后仍必须得到完整且唯一的 ID 集合。
- [x] 为 prompt 生成 snapshot tests；官方 prompt 必须与固定 commit 的源文本做自动 diff，而不是靠注释声称“verbatim”。
- [ ] **服务器验收**：完成所有真实模型端到端 smoke；本地 parser、fake-retriever loop 和官方 launcher dry-run 已通过。

## 2. 各方法修正清单与最终参数

### 2.1 Direct

- **generator**：Qwen3.5-9B。
- **retriever/BM25**：无。
- **prompt/harness 来源**：GrepSeek 论文包含 Direct 对照，但没有公开其表格 baseline 的逐字 prompt。保留一个最小、冻结的本项目 prompt，并在 `BASELINE_SOURCES.md` 中明确标记为“project-defined”，不得声称来自官方。
- **参数**：`temperature=0`、`top_p=1`；短答案格式与统一 parser；输出预算必须足以容纳 thinking 和最终答案，不能用过小的固定单轮上限截断答案。
- [x] 用 Qwen3.5 tokenizer/chat template 替换旧 Qwen3-4B 假设。

### 2.2 RAG（三检索器）

- **generator**：Qwen3.5-9B。
- **retriever**：`bm25`、`e5`、`qwen3_emb_4b` 三行；共享 `top_k=3`，以复现 GrepSeek 的统一 retriever 比较口径。
- **BM25**：`k1=1.2`、`b=0.75`。
- **prompt/harness 来源**：参考 GrepSeek 的 RAG 实验协议；其逐字 reader prompt未公开，因此 reader prompt必须标为 project-defined、冻结并记录 hash。三个 retriever 必须使用完全相同的 reader prompt和 passage 渲染。
- **参数**：`temperature=0`、`top_p=1`；传入完整的 Wiki-18 passage，不再使用无出处的每篇前 1500 字符截断；若因 context 必须设预算，应按 token 从低排名 passage 整段丢弃并记录 `shown/total`。
- [x] 将现有 `bm25_rag` 改为后端无关的 `rag`；旧 CLI 仅可保留短期兼容警告，阶段末删除。

### 2.3 Search-R1（2 个模型 × 3 个检索器）

- **generator**：指定的 7B v0.3 和 14B v0.3 checkpoint，二者都必须跑全矩阵。
- **retriever**：BM25/E5/Qwen3-Embedding-4B；`top_k=3`。
- **BM25**：`k1=1.2`、`b=0.75`。
- **prompt/harness 来源**：[`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1) 固定 commit 的官方 inference prompt、stop token 和 `<search>/<information>/<answer>` 原文续写循环；使用 checkpoint 自带 tokenizer/chat template，不再硬编码额外 system message。
- **关键参数**：action budget `B=4`；`max_new_tokens=1024`；`temperature=0.7`、`top_p=1.0`，与公开 inference 默认一致。若 v0.3 固定 commit 提供不同的专用 evaluation config，以该 config 为准，但必须先更新来源 manifest，不能静默修改。
- **停止/解析**：只在官方结束条件、EOS、预算耗尽或明确错误时结束；保存每次 query、top-3 doc_id、注入文本和最终 parse 状态。
- [x] 删除旧 3B checkpoint 配置、`max_turns=8` 默认、修正版 prompt 文本和额外 `You are a helpful assistant` system 注入。

### 2.4 Search-O1（三检索器）

- **generator**：Qwen3.5-9B；主推理和 Reason-in-Documents 均用同一固定 revision。
- **retriever**：BM25/E5/Qwen3-Embedding-4B；本地 corpus 比较统一 `top_k=3`。同时在文档中说明原版 Bing Web Search 是 top-10，此处 top-3 是 GrepSeek-style local-retriever protocol，不是原版 Bing 数字的直接复现。
- **BM25**：`k1=1.2`、`b=0.75`。
- **prompt/harness 来源**：[`RUC-NLPIR/Search-o1`](https://github.com/RUC-NLPIR/Search-o1) 固定 commit 的 `scripts/prompts.py` 与 `scripts/run_search_o1.py`。
- **prompt 选择**：PopQA/NQ/TriviaQA 使用完整 `get_singleqa_search_o1_instruction`；HotpotQA/2Wiki/MuSiQue/Bamboogle 使用完整 `get_multiqa_search_o1_instruction`；Qwen3.5-9B 走非 QwQ 的 `get_task_instruction_openqa` 分支；Reason-in-Documents 使用完整官方 `get_webpage_to_reasonchain_instruction`。
- **关键参数**：单跳数据 `max_search_limit=5`；多跳数据 `max_search_limit=10`；`max_turn=15`；`temperature=0.7`、`top_p=0.8`、`top_k_sampling=20`。保持连续 completion、搜索 stop token、RiD 注入和最后一个 `\boxed{}` 提取机制。
- **文档输入**：Wiki-18 已是 passage corpus，向 RiD 提供完整 top-3 passages；不再声称本地的“前 1500 字符”复现了原版围绕 Bing snippet 的前后各 3000 字符。
- [x] 补回当前本地 prompt 缺失的 multi-hop 示例/Remember 段，修正错误的 task 分支并加入官方文本 diff test。

### 2.5 GrepSeek

- **generator**：官方 GrepSeek-Qwen3.5-9B-GRPO checkpoint。
- **retriever/BM25**：无；工具在物化 corpus 上执行 shell/grep。不要给 GrepSeek 添加 E5/BM25 并继续沿用同一方法名。
- **prompt/harness 来源**：[`alirezasalemi7/grepseek`](https://github.com/alirezasalemi7/grepseek) 固定 commit 的 `inference/agent.py`、官方 system prompt、tool call parser 和 shell 输出结构。
- **关键参数**：`temperature=0.6`、`top_p=1.0`、最多 6 assistant turns；`tool_max_tokens=2048`，用该 checkpoint tokenizer 计数；生成端不设固定 `max_tokens_per_turn=2048`，保留模型总 context/sequence 上限；连续解析错误策略必须与固定官方 harness 一致。
- [x] 用自动 trace 对比证明本地 wrapper 与官方 harness 在相同问题、相同模型输出下产生相同工具调用和回包；否则直接调用官方 harness。

### 2.6 DCI

- **generator**：Qwen3.5-9B。
- **retriever/BM25**：无；使用 bash/read/grep 在完整、物化的 corpus 上工作。
- **prompt/harness 来源**：[`DCI-Agent/DCI-Agent-Lite`](https://github.com/DCI-Agent/DCI-Agent-Lite) 固定 commit；system prompt 使用官方 `prompts/system_prompt.txt`，任务 prompt 使用 `scripts/bcplus_eval/run_bcplus_eval.py` 的官方构造函数。
- **关键参数**：官方 Pi/DCI-Agent-Lite harness；最多 300 turns；L3 context management；单命令 timeout 30 秒；工具输出和 compaction 使用固定官方配置，不沿用当前自写 8 轮/8000 字符实现。
- [x] 用官方 Lite 完整替换 `eval/dci_agent.py`、`prompts/dci_prompt.txt` 和自写 shell 行为；若需要 adapter，只允许做数据格式和结果格式转换，不得重写 agent loop。

### 2.7 DR-DCI

- **agent generator**：Qwen3.5-9B。
- **BrowseComp-Plus judge**：独立 Qwen3.5-9B。
- **retriever**：不做三检索器替换。Wiki-18 数据集使用官方 E5 pull 路线；BrowseComp-Plus 使用官方 Qwen3-Embedding-8B index。具体模型 revision、索引和 query instruction 固定到 manifest。
- **BM25**：不使用。
- **prompt/harness 来源**：[`EigenTom/DR-DCI`](https://github.com/EigenTom/DR-DCI) 固定 commit 的官方 Pi harness 和任务脚本；Wiki 配置参考官方 Wiki-18 E5 脚本，BCP 配置参考官方 Qwen3-Embedding-8B/root-flat 脚本。
- **关键参数**：300 turns、L3、high reasoning 配置、`pull(query, topK)` 动态 300–600、去重后 root-flat 物化、rank-aware top-20 导航预览、正文由 bash/read 查看、单命令 timeout 30 秒。
- **judge 参数**：使用官方 BrowseComp-Plus judge prompt/答案格式，`temperature=0`、`top_p=1`；judge 不读取 gold 以外的额外轨迹信息，也不复用 agent 对话上下文。
- **实现边界**：不得为了统一矩阵替换 `pull` 的召回/排序后端；topK 动态策略、去重、物化、预览、Pi loop、prompt 和 context management 必须保持官方 task-specific 设置。
- [x] judge 与 agent 已分离为独立服务/配置和独立结果步骤；官方轨迹、workspace/pull 产物与 judge 原始输出均由固定 launcher/adapter 保留。

### 2.8 RISE

- **论文/代码来源**：[`Towards Retrieving Interaction Spaces for Agentic Search`](https://arxiv.org/abs/2606.06880) 与官方 [`texttron/RISE`](https://github.com/texttron/RISE) 固定 commit。优先直接使用官方 `scripts/run_rise.py`、`src/rise/` 工具和 prompt，只增加 Qwen3.5-9B provider/data adapter。
- **generator**：Qwen3.5-9B。
- **retriever**：只使用 BM25，不参与三检索器矩阵。
- **BM25**：本项目统一使用 `k1=1.2`、`b=0.75`。必须同时在差异表中记录论文/官方代码原值是 `k1=1.5`、`b=0.75`；这是用户指定的统一参数替换，不能声称参数逐字复现。
- **大 top-k 例外**：每个自然语言子查询固定检索 `K=1000` 篇，多个子查询结果取并集并单调加入 per-query workspace；只向模型返回每个子查询 top-10 预览，完整 top-1000 文件供后续 bash/read 使用。不得改成 top-3。
- **interaction space**：retrieved documents 以保持 corpus-relative path 的方式物化/硬链接到有界工作区；agent 只能在该工作区内执行 `bash`、`rg/grep` 和 line-range `read`。
- **RISE 完整版**：使用官方离线结构化协议，在原始正文前加入经过验证的 line-numbered TOC/section anchors；正文不得被摘要、删除或改写。BrowseComp-Plus 优先使用官方预构建 [`Tevatron/browsecomp-plus-md-toc-gpt5.4-nano`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-md-toc-gpt5.4-nano)。
- **Wiki 数据要求**：不能给 Wiki-18 的约 100-word DPR passage 逐段生造 TOC 后称为 RISE。阶段一必须准备与 Wiki-18 时间快照一致的 article/file-level Wikipedia corpus；若没有现成的同快照文章资产，则按 title/article id 将同一 Wiki-18 passage corpus 确定性地还原成文章文件，并记录排序规则、passage↔article 映射和内容校验。随后使用官方结构化 pipeline 生成 TOC corpus。该资产是阶段二启动前的必需项，不能用 RISE-BM25 代替。
- **离线结构化模型**：BrowseComp-Plus 使用官方预构建 TOC 资产；自建 Wiki article TOC 时使用固定 revision 的 Qwen3.5-9B，并保留 section proposal、anchor 验证率和失败文档清单。该替换必须与论文使用 gpt-5.4-nano low reasoning 的原设置并列记录。
- **关键参数**：最多 100 model calls；每 query 一小时 wall-clock cap；不在预算耗尽时强制生成答案；bash stdout 4000 字符；每个 subprocess 60 秒；line-range read 默认最多 2000 行；保存 search union、最终 workspace、每次 bash/read 和停止原因。
- **judge**：BrowseComp-Plus 仍使用本项目统一的独立 Qwen3.5-9B judge和官方 Appendix F prompt。
- [ ] **服务器验收**：用完整 Wiki article TOC 资产验证 RISE full method；代码已只允许 structured-docs 主行，boundary-only 不会冒充正式 RISE。

### 2.9 AgentIR

- **retriever**：固定 AgentIR-4B，不参与三检索器矩阵。
- **generator**：Qwen3.5-9B。
- **BM25**：不使用。
- **prompt/harness 来源**：[`texttron/AgentIR`](https://github.com/texttron/AgentIR) 与官方 HF model card。query 必须包含 agent 的 reasoning trace 和它生成的 search query，并使用官方 instruction prefix。
- **关键参数**：pooling、normalization、query/document 编码和检索循环全部按官方实现固定；top-k/最大交互步数取官方 evaluation config并写入 manifest。
- [x] 删除或改名当前“只编码原始 question，然后交给通用 RAG reader”的 `agentir_rag`。该实现最多可以作为 `rag_agentir_embedding` 消融，不能作为 AgentIR baseline。

### 2.10 IRCoT

- **generator**：Qwen3.5-9B。
- **retriever**：BM25/E5/Qwen3-Embedding-4B；每步统一 `top_k=3`。
- **BM25**：`k1=1.2`、`b=0.75`。
- **prompt/harness 来源**：[`StonyBrookNLP/ircot`](https://github.com/StonyBrookNLP/ircot) 固定 commit，使用官方 dataset-specific demonstrations、paragraph formatting、retrieval-generation interleaving和答案提取。
- **参数**：temperature 和最大步数采用固定官方数据集配置，逐数据集写入 manifest，不允许沿用当前零样本 prompt 的隐藏默认值。
- [x] 当前 `eval/ircot_agent.py` 是 zero-shot 机制近似，不是官方 baseline。阶段一必须完成官方化；如果 PopQA 无法定义有依据的官方 adapter，则删除该 baseline，而不是留下一个标为 IRCoT 的自实现。

### 2.11 ScaleSeek（本项目方法，不属于外部 baseline）

- **generator**：Qwen3.5-9B。
- **主检索器**：BM25；固定 BM25 对照和默认/回退参数为 `1.2/0.75`。若 agent 仍可主动选择 `k1/b`，必须记录每次选择并把该能力明确列为 ScaleSeek 方法的一部分。
- **工具**：`bm25_retrieve(query, top_k<=50, k1, b, mode)`、`grep_workspace`、`read_doc`；有界 workspace；最多 8 assistant turns；每次工具响应预算 2048 token。
- **prompt/harness 来源**：本项目版本化 prompt；记录 prompt hash 和 RL 环境协议。`bm25_retrieve`/`grep_workspace` 对 passage 列表按排名整段丢弃，`read_doc` 的 token 截断行为要单独记录，不能笼统称为全部“整段丢弃”。
- **dense ablation**：若未来实现 dense retriever 工具，必须命名为 `scaleseek_dense_*` ablation，并移除/重定义无意义的 `k1/b` 参数；不计入本阶段 baseline 修正门禁。

## 3. 第一阶段验收与清理

以下条件全部满足，第一阶段才算完成：

- [x] 每个保留方法都有固定来源、配置文件、prompt hash、harness commit、模型 revision 和一条可运行命令。
- [ ] **服务器验收**：三个完整索引共享同 corpus/doc_id manifest；代码已强制 BM25 `1.2/0.75` 并完成 dense pooling/合并数值测试。
- [ ] **服务器验收**：每个 baseline 的真实模型端到端 smoke；本地 parser、loop、retrieval、resume 与 dry-run 已通过。
- [ ] **服务器验收**：Qwen3.5-9B、BCP judge、Search-R1 7B/14B 的真实 serving 闭环；本地 prompt/chat-template/tool-parser 契约已锁定。
- [x] `popqa` 完整数据加载通过 14,267 条、14,267 个唯一 ID 的 preflight；不存在抽样 override。
- [x] README、运行脚本、配置文档和 CLI help 与最终实现一致，不再出现 Qwen3-4B generator、Search-R1 3B 或 `popqa_full=1500` 等过时说明。
- [x] 生成 `cleanup_manifest.json`，列出每个待删除路径、删除原因和替代实现。
- [x] 已删除工作区内所有错误/过时 baseline 代码和启动脚本：旧 Search-R1 3B/Qwen3-4B 配置、自写 DCI/GrepSeek loop、残缺 Search-O1、伪 AgentIR/IRCoT、`-n 1500` 脚本及 PopQA 别名；服务器 checkpoint 按 manifest 在验收后清理。
- [x] 不删除有效的历史结果与轨迹；将其标记为 `legacy_invalid_for_main_table`，避免被新脚本 resume 或合并。若结果本身损坏或来自错误数据，记录后删除。

第一阶段交付物：正确代码、测试、配置、来源文档、数据/索引 manifest、清理 manifest，以及一份“保留/删除/改名”总结。第一阶段不需要产出可用于论文的 benchmark 数字。

---

# 第二阶段：完整 PopQA 单数据实验

## 启动条件

只有第一阶段全部验收并完成清理后才能开始。

## 数据准备

- [ ] 从 [`RUC-NLPIR/FlashRAG_datasets`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) 下载完整 PopQA test split。
- [ ] 保存原始文件和规范化文件的 SHA256、来源 revision、下载时间及转换脚本 commit。
- [ ] preflight 必须显示 14,267 条有效样本和 14,267 个唯一 ID；若上游 revision 的真实计数发生变化，停止并向用户说明，不能自行抽成 14,267。
- [ ] 为 Wiki-18 完整 corpus 建好 BM25/E5/Qwen3-Embedding-4B/AgentIR 所需索引，并验证 doc_id 对齐。
- [ ] 正式命令中禁止出现 `-n`、`--n`、`--offset`、sampling、subset 或只跑失败样本后错误汇总。

## 必跑矩阵

| 方法 | PopQA 完整运行数 | 配置 |
|---|---:|---|
| Direct | 1 | Qwen3.5-9B |
| RAG | 3 | BM25 / E5 / Qwen3-Embedding-4B |
| Search-R1 | 6 | 7B、14B各自 × 三检索器 |
| Search-O1 | 3 | Qwen3.5-9B × 三检索器 |
| GrepSeek | 1 | 官方 GRPO checkpoint + grep workspace |
| DCI | 1 | Qwen3.5-9B + 官方 DCI-Agent-Lite |
| DR-DCI | 1 | Qwen3.5-9B + Wiki-18 官方 E5 pull backend |
| RISE | 1 | Qwen3.5-9B + BM25 K=1000 + 阶段一验收过的完整 Wiki article TOC interaction space |
| AgentIR | 1 | AgentIR-4B retriever + Qwen3.5-9B agent/generator |
| IRCoT | 3 或 0 | 仅当阶段一官方化通过时跑三检索器；否则从项目删除并说明 |
| ScaleSeek | 1 | Qwen3.5-9B + 主 BM25 方法设置 |

每个配置必须覆盖完整 PopQA。允许因机器故障 resume，但最终输出必须通过集合校验：输出唯一 ID 集合与 dataset manifest 完全相等；临时 API/服务错误必须重试，无法恢复的 method failure要保留并计入失败分布。

## 第二阶段汇报

完成后向用户提交：

1. 完整配置矩阵和每个 run 的 config ID。
2. EM/F1、answered rate、parse/tool/API error、平均轮数、平均检索次数、平均最终 workspace 大小和 wall time。
3. 三检索器的并列表，以及 Search-R1 7B/14B 的并列表。
4. 数据、索引、模型、prompt 和 harness revision。
5. 至少对每个方法抽查成功、检索失败、解析失败各若干条轨迹，但不得用这些抽查样本代替全量指标。
6. 任何与论文数字不可直接比较的原因，尤其是 generator/judge 替换和本地 Wiki-18 corpus 差异。

**强制门禁：汇报第二阶段结果后立即停止。未收到用户明确的“结果没问题，可以开始第三阶段”确认，不得下载第三阶段新增数据、构建其索引或启动实验。**

---

# 第三阶段：扩展到全部目标数据库

## 启动条件

仅在用户审核第二阶段 PopQA 结果并明确批准后开始。

## 完整数据集

下载并使用以下数据集的完整官方 evaluation split，不做任何 sampling：

1. NQ
2. TriviaQA
3. PopQA
4. HotpotQA
5. 2WikiMultiHopQA
6. MuSiQue
7. Bamboogle
8. BrowseComp-Plus

为每个数据集生成独立 dataset manifest。普通 retrieval baseline 在前七个 Wiki QA 数据集上统一使用同一 Wiki-18 passage corpus/index manifest；DR-DCI 使用官方 E5 pull 索引。RISE 因方法要求 document-level 可导航对象，使用与 Wiki-18 同时间快照的 article/file-level manifest，并单独记录 passage↔article 映射。BrowseComp-Plus 使用完整专用 corpus和方法各自的官方索引。不得复用不相容的 Wiki doc_id、qrels 或 passage cache。

BrowseComp-Plus preflight 预期为完整 830 条 query；若固定的上游 revision 数量不同，必须停止并报告差异，不能静默截取到 830。

## 实验执行

- [ ] 在每个数据集上运行阶段二已批准的完整方法矩阵，配置不允许按结果临时调参。
- [ ] Search-O1 根据单跳/多跳数据选择阶段一固定的官方 prompt 分支和 search limit。
- [ ] Search-R1 继续同时跑 7B/14B × 三检索器，保持 `B=4` 和 top-3。
- [ ] RAG、Search-O1、IRCoT 等普通检索方法统一 top-3；只有官方协议原本构造数百/数千篇外部工作区的方法保留大 top-k。
- [ ] 所有固定 BM25 baseline 继续使用 `k1=1.2`、`b=0.75`。
- [ ] DR-DCI 不做 retriever sweep：Wiki 数据固定官方 E5，BrowseComp-Plus 固定官方 Qwen3-Embedding-8B。
- [ ] RISE 保留官方每子查询 BM25 K=1000、top-10 preview、100-call budget 和 TOC interaction-space 协议；在 BrowseComp-Plus 使用官方 structured corpus，在 Wiki QA 使用阶段一验收过的 article-level structured corpus。
- [ ] BrowseComp-Plus 的所有正式结果由独立 Qwen3.5-9B judge统一重判；保存判卷原文并验证无漏判/重复判卷。
- [ ] 任何单次 run 只有在输出 ID 集合与数据 manifest 完全一致后才标记为 complete。

## 最终交付

- [ ] 八数据集主表与 retriever/model ablation 表。
- [ ] 每个方法和数据集的完整配置、来源和差异说明。
- [ ] EM/F1 或 BCP judge accuracy，以及检索质量、运行成本、失败率和轨迹统计。
- [ ] 可从空缓存开始重建数据、索引并恢复所有运行的 runbook。
- [ ] 明确区分：官方协议复现、统一模型替换、统一检索器对照、本项目方法和额外 ablation。

---

## 不允许的做法

- 不允许把 1500 条随机集、long-tail 子集或 first-N 称为完整 PopQA。
- 不允许为不同检索器更换 reader prompt、top-k、corpus 或 passage 预算后仍声称只比较 retriever。
- 不允许把未公开的 baseline prompt写成“官方逐字 prompt”。
- 不允许把自写 DCI/IRCoT/AgentIR 机制近似继续标成官方 baseline。
- 不允许用 Qwen3-4B 的旧结果与新的 Qwen3.5-9B 主表混合。
- 不允许因为某一 BM25 参数结果更好而偏离统一 `1.2/0.75`。
- 不允许阶段二结束后自动进入阶段三。
