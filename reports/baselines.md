# ScaleSeek Baseline 实现与复刻对照（详细版）

每个 baseline：对标论文与官方资产（精确到 HF snapshot 哈希）、我们的完整参数、
与论文 setup 的逐项差异、论文数字 vs 我们的数字、忠实度分级与升级计划。

**忠实度分级**：A=与官方实现头对头验证过；B=官方资产/提示词逐字对齐；
C=官方机制复刻但有明确记录的替换（如换模型/检索器）；D=自实现（无官方 prompt 可抄）。

**统一口径**：popqa_full = FlashRAG popqa/test（14,267）**随机抽的固定 1500 样本**（seed 不可考），EM/F1
（SQuAD 规范化，与 GrepSeek 官方 scoring.py 逐行等价）。检索语料 wiki-18
（21,015,324 DPR 段落）。prompt 型 baseline 底座 = [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)
（snapshot `1cfa9a72...`，vLLM，thinking 模式）除非另注明。

**通用不可避免差异（适用所有 prompt 型 baseline）**：原论文多用前沿模型
（GPT-5.4-nano / QwQ-32B / 9B 训练模型），我们统一用本地 Qwen3-4B——这保证
**方法间**（vs scaleseek）严格可比，但**绝对值**天然低于论文。

## 资源链接索引
| baseline | 论文 | 代码/资产 |
|---|---|---|
| grepseek | [arxiv 2605.29307](https://arxiv.org/abs/2605.29307) | ckpt: [alireza7/GrepSeek-Qwen3.5-9B-GRPO](https://huggingface.co/alireza7/GrepSeek-Qwen3.5-9B-GRPO)；官方推理代码：本地 `/data/rech/mofengra/grepseek/inference/` |
| search_r1 | [arxiv 2503.09516](https://arxiv.org/abs/2503.09516) | repo: [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)；ckpt: [PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo) |
| search_o1 | [arxiv 2501.05366](https://arxiv.org/abs/2501.05366) | repo: [sunnynexus/Search-o1](https://github.com/sunnynexus/Search-o1)（prompt 逐字取自 `scripts/prompts.py`）|
| dci | [arxiv 2605.05242](https://arxiv.org/abs/2605.05242) | repo: [DCI-Agent/DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite)；本地可用副本在 `dr_dci_official/`（`uv run dci-agent-lite`）|
| dr_dci | [arxiv 2606.14885](https://arxiv.org/abs/2606.14885) | repo: [EigenTom/DR-DCI](https://github.com/EigenTom/DR-DCI)（本地 `/data/rech/mofengra/dr_dci_official/`）；检索索引: [Tevatron/browsecomp-plus-indexes](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes) |
| bm25 参数出处 | [Pi-Serini, arxiv 2605.10848](https://arxiv.org/abs/2605.10848) | 索引: Pyserini/Lucene 自建（`scripts/build_bm25_index.py`）|
| agentir_rag | [arxiv 2606.06880](https://arxiv.org/abs/2606.06880) | 模型: [Tevatron/AgentIR-4B](https://huggingface.co/Tevatron/AgentIR-4B)；索引自建（`scripts/build_agentir_index.py`）|
| 底座模型 | — | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)（snapshot `1cfa9a72...`）|
| E5（升级用）| — | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)（本地 `checkpoints/e5-base-v2`）|
| 评测数据 | — | [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)；BCP: [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) + [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) |
| 相关方法（未实现）| [s3, arxiv 2505.14146](https://arxiv.org/abs/2505.14146) | 训练侧参考，非 baseline |

---

## 1. grepseek —— 忠实度 **A（已头对头验证）**
| | |
|---|---|
| 对标论文 | GrepSeek (arxiv 2605.29307) |
| 官方模型 | [`alireza7/GrepSeek-Qwen3.5-9B-GRPO`](https://huggingface.co/alireza7/GrepSeek-Qwen3.5-9B-GRPO)（snapshot `a7956397...`，18GB bf16，[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 基座，GRPO 训练，教师 27B）|
| 服务方式 | 独立 vLLM :8002。octal30(A5000)需 TP=2 + `--disable-custom-all-reduce` + `NCCL_P2P_DISABLE=1`；**必须** `--reasoning-parser qwen3` + `--max-model-len 32768` |
| Prompt | system 与官方 `inference/agent.py` **逐字节一致**（difflib 已对 inference/SFT/RL 三处验证）；user turn `"Query: {q}"` |
| 采样/循环 | temperature **0.6**（官方）、top_p 1.0、**6 轮**（5 tool+1 answer）、**生成端无上限**（设 2048 会腰斩 think→12% parse_error）、连续 2 次解析错终止 |
| 工具 | shell 白名单执行器；stdout 用**模型自带 tokenizer 截 2048 token**（官方 `--tool_max_tokens 2048`）；工具回包 JSON 含 `information_lines`（SFT 同构）|
| 论文数字 | PopQA **F1 0.4861**（完整 14,267 集）；7 集 micro-avg EM .4948 / F1 .5691 |
| 我们数字 | **官方 harness 原码**跑本地语料（前 1000）：F1 **0.4154**；我们实现同题同服务器：F1 **0.4187**（**差 0.003，复刻闭环**）。popqa_full 随机 1500：EM .3440 / F1 .3870 |
| 结论 | 实现无差异。paper 的 .4861 与官方原码的 .4154 之间 ~7 F1 的差距属 checkpoint/语料快照/口径，**不可用公开资产复现**（证据链见 metric_support.md）|

## 2. search_r1 —— 忠实度 **B**（检索器待升级 → B+）
| | |
|---|---|
| 对标论文 | Search-R1 (arxiv 2503.09516, Jin et al.) |
| 官方模型 | [`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo)（snapshot `d4d2f71e...`，[Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) GRPO）|
| 服务方式 | 独立 vLLM :8001，**max-model-len 必须 32768**（8192 → 31% 上下文溢出）|
| Prompt/循环 | user prompt 逐字官方 infer.py；completions 原文续写：stop `</search>` → 检索注入 `<information>` → 续写至 `<answer>`；Qwen2.5 ChatML 硬编码 |
| 我们参数 | temperature 0.0（贪心）、top_p 1.0、top-3、≤8 轮 |
| 论文 setup | **E5 检索器 top-3**、wiki-18、action 预算 **B=4**、（训练 rollout temp 1.0；评测未明说，推断贪心）|
| **差异** | ①检索器 BM25 vs 论文 E5（**主差异**）；②轮上限 8 vs 4（宽松侧，理论无害）|
| 论文数字 | PopQA **EM 0.413**（3B GRPO + E5 top3），7 集平均 0.365 |
| GrepSeek 论文复刻版（**不可直接对标**）| Table 1 "Search-R1+BM25" PopQA F1 **.4194**——那是 **Qwen3.5-9B 底座、与 GrepSeek 同设置重训**的版本，非公开 3B ckpt（见文末附录 A）|
| 我们数字 | popqa_full：EM **0.308** / F1 0.345（BM25 top3）|
| 升级计划 | **建 E5 索引**（本地已有 `checkpoints/e5-base-v2`，110M 模型，21M 编码约 6h 单卡）→ E5 top-3 重跑对齐论文；顺带 `--max-turns 4` |

## 3. search_o1 —— 忠实度 **B-/C**（机制与提示词官方，后端与底座替换）
| | |
|---|---|
| 对标论文 | Search-o1 (Li et al. 2025, sunnynexus/Search-o1)；在六篇对标论文中作为"本地语料检索 agent"跑法（Bing→本地检索器）|
| 官方原版 setup | **QwQ-32B-Preview + Bing Web API + Reason-in-Documents** |
| Prompt | `get_multiqa_search_o1_instruction` / `get_task_instruction_openqa` / **Reason-in-Documents 蒸馏 prompt** 三段全部逐字官方 |
| 机制 | **单条连续 completions 流**（我们曾用 chat 多轮注入——不忠实，已重写废弃）：stop `<|end_search_query|>` → 检索 → RiD 蒸馏调用 → `<|begin_search_result|>` 内联续写 → 取最后 `\boxed{}`（平衡括号+LaTeX 清洗）。搜索超限注入官方拒绝文案 |
| 我们参数 | temp 0.0、top_p 1.0、BM25 top-5、搜索上限 5、≤10 段续写、每页截 1500 字符 |
| **差异** | ①底座 QwQ-32B→Qwen3-4B；②Bing→BM25 wiki-18（论文本地化改法用 E5）；③官方逐页 fetch 网页，我们直接用检索段落 |
| 论文数字 | 原论文不测 PopQA（其 QA 集为 NQ/HotpotQA 等 + Bing），无直接对标数字 |
| GrepSeek 论文复刻版（**不可直接对标**）| Table 1 "Search-O1+BM25" PopQA F1 **.4003**——底座是 **Qwen3.5-9B**（我们 4B）；见文末附录 A |
| 我们数字 | popqa_full：EM **0.2547** / F1 0.2959 |
| 升级计划 | E5 索引建成后换 dense 后端（与 search_r1 同一次升级）|

## 4. dci —— 忠实度 **D（当前自实现）→ 计划升级到官方 Lite（A/B）**
| | |
|---|---|
| 对标论文 | Beyond Semantic Similarity (arxiv 2605.05242)，官方开源 = **DCI-Agent-Lite**（Pi harness）|
| 官方原版 setup | GPT-5.4-nano（high reasoning）+ bash/read 工具 + **L3 运行时上下文管理**（单工具结果截 20K 字符 + 240K 累计压缩 + 保留最近 12 轮）+ **300 轮预算** |
| 我们当前实现 | 自写 prompt（`prompts/dci_prompt.txt`，官方未发布单条 prompt）+ 自写 shell 执行器 + ≤8 轮 + stdout 8000 字符（实测 20K 对 4B 有害：EM -5，弱模型被 grep 噪声淹没）|
| **差异** | prompt/循环/轮数/截断全部非官方——定位是"**未训练 4B 裸 grep 下界**"而非论文复刻 |
| 论文数字 | 不测 PopQA。Wiki QA（LLM-judge 准确率%，GPT-5.4-nano）：NQ 72 / TriviaQA 84 / HotpotQA 72 / 2Wiki 68 / MuSiQue 40 / Bamboogle 72 |
| 我们数字（自实现）| popqa_full：EM **0.2260** / F1 0.2665（并发铁律 ≤2，否则 44% grep 磁盘超时）|
| **官方 dci-agent-lite（2026-07-14，同 4B 底座，popqa n=50）** | **EM .3800 / F1 .4834**（我们 EM/F1 口径）；LLM-judge acc .36。官方 harness（read,bash grep 14GB 语料 + L3 上下文 + 300 轮 + thinking），provider=vllm/model=agent，judge=agent |
| **关键结论** | 官方 lite **大幅超自实现**（EM +15/F1 +22 点，即便计入 n=50 噪声 SE≈7 下界仍超）→ **我们自实现 dci 严重低估 DCI 方法**；主表 dci 行应标注"未训练 4B 裸 grep 下界，非 DCI 方法上限"。扩大 n 可待拍板 |

## 5. dr_dci —— 忠实度 **A-（官方 harness 原码，仅换模型端点，待跑）**
| | |
|---|---|
| 对标论文 | DR-DCI (arxiv 2606.14885)，官方 repo EigenTom/DR-DCI（Pi TS coding-agent）|
| 决策 | **不重写**——跑官方原码，pi `models.json` 定义 `vllm` provider（baseUrl→:8000，api=openai-completions）；vLLM 需 `--enable-auto-tool-choice --tool-call-parser hermes` |
| 官方 setup 保留 | 300 轮 / L3 / `pull(query, topK)` 300-600 / rank_aware 预览 / root_flat 物化 / 30s bash 超时 / 官方 qwen3-emb-8b 检索索引（Tevatron/browsecomp-plus-indexes 4 分片已下载）|
| **差异** | 仅模型：GPT-5.4-nano→Qwen3-4B；judge 暂用 agent（冒烟）|
| 论文数字 | BCP **71.2%**（+reset 73.25%）；wiki-18 file-per-doc（LLM-judge，50/集）：NQ 62 / TriviaQA 82 / HotpotQA 68 / 2Wiki 58 / MuSiQue 44 / Bamboogle 64，avg 63.0 |
| 我们状态 | ✅ **全量 830 完赛（2026-07-13）：accuracy 43.25%（359/830），830/830 判卷、0 失败**。论文 71.2%（GPT-5.4-nano）→ 差距 = 纯底座（4B vs GPT-5.4-nano），scaffold 官方原码。judge=4B（verdict 可用、reason 有幻觉倾向）——trajectories 全存，可后续换更强 judge 离线重判。排障三项见 RESUME_STATE §3 |

## 6. bm25_rag —— 忠实度 **C**（通用 baseline，参数出处明确）
- **实现**：问题原文 → Pyserini/Lucene（wiki-18 全 21M 索引，`--storeRaw`）top-5，
  k1/b 每查询可调 → 每段 1500 字符 → `prompts/bm25_rag.txt` reader（自写，无官方版）单次调用，temp 0。
- **五组 (k1,b) 出处**（Pi-Serini/2605.10848）：0.9/0.4 Anserini 默认、25/1 其长文档主设置、
  16/1 其网格最优、1.2/0.75 经典 Robertson、1.5/0.75 旧默认。
- **popqa_full 扫参实测（2026-07-12，n=1500，全零 api_error；1.2/0.75 即主表行复用）**：

  | k1/b | EM | F1 | 备注 |
  |---|---|---|---|
  | 0.9/0.4 | .3100 | .3582 | Anserini 默认 |
  | 1.2/0.75 | .3067 | .3550 | 经典 Robertson（主表行）|
  | **1.5/0.75** | **.3120** | **.3590** | 名义最优，但见下 |
  | 16/1.0 | .2613 | .3038 | Pi-Serini 网格最优（长文档）|
  | 25/1.0 | .2507 | .2908 | Pi-Serini 长文档主设置 |

  三组短文档配置差距 ≤0.5 EM 点（n=1500 时 SE≈1.2 点）→ **统计上平手**，主表
  继续用 1.2/0.75；两组 Pi-Serini 长文档配置掉 5-6 EM 点——其论文自述"BM25 默认
  为短 passage 调优、长文档需大 k1/b"在短 passage 语料上反向获证。
  （旧 longtail 扫参结论 1.2/0.75 最优 .3388 属首代 longtail 集，文件已按用户
  2026-07-11 批准删除，不再引用。）
- **论文参照**：GrepSeek Table 1 RAG(BM25)（9B reader）PopQA F1 **.3239**；
  我们（4B reader，1.2/.75）popqa_full F1 **.3550**——小 reader 但更优 BM25 参数，可比且合理。

## 7. direct —— 忠实度 C（无检索对照）
单次调用 + `<answer>` 抽取，temp 0，max_tokens 2048（thinking 含）。
论文参照：GrepSeek Table 1 Direct（9B）PopQA F1 **.2364**；我们（4B）F1 **.2129**。

## 8. agentir_rag —— 忠实度 **C**（官方检索模型，索引自建，reader 同 bm25_rag）
| | |
|---|---|
| 对标 | AgentIR / Towards Retrieving Interaction Spaces (arxiv 2606.06880)；作为 dense 检索 agent 底座 |
| 官方模型 | [`Tevatron/AgentIR-4B`](https://huggingface.co/Tevatron/AgentIR-4B)（snapshot `e31abb63...`，2560 维，last-token pooling，官方 query 指令前缀逐字使用）|
| 索引 | 21M 段落 fp16 编码（transformers v5 须 `dtype=`；`torch_dtype` 被静默忽略→fp32 慢 4×）→ **6×sq8_flat 分片**（SQ8 精确暴力搜索；HNSW 入图 ~100/s 单线程不可行已弃）；每 1M checkpoint + `--resume` |
| 评测 | `precompute_agentir_retrieval.py` 逐片批检索（单片 9G 轮流入内存，归一化内积跨片合并=全局精确 top-5）→ `--agentir-cache` reader（与 bm25_rag 同 reader、同 temp 0）|
| 论文参照 | 无直接 PopQA 数字；最近参照 = GrepSeek Table 1 RAG(Qwen3-4B-emb)（9B reader）F1 **.5046** |
| **结果（2026-07-12）** | popqa_full **EM .4453 / F1 .5172**（n=1500，0 api_error）——全表第一，比第二名 grepseek（.3440/.3870）高 10+ 点 |
| 交叉验证 | 与 GrepSeek Table 1 检索器梯度完全吻合：RAG 随检索器 BM25→E5-110M→Qwen3-4B-emb 为 .3239→.4468→.5046；我们 4B reader + AgentIR-4B（agentic 训练过的 4B 稠密检索器）落在 .5172 同档 → PopQA 上检索器质量主导，数字可信 |

## 9. scaleseek（我们的方法，非复刻）
三工具 prompt agent：`bm25_retrieve(query, top_k≤50, k1, b, mode)` / `grep_workspace` /
`read_doc`；工作区有界；ChatML ≤8 轮；工具响应 2048 token 预算（整段丢弃式，与未来
RL 训练一致）；`bm25_calls`+`workspace_doc_ids` 全落盘。popqa_full EM **.3120** / F1 .3648
——唯一超过全部 RAG 系的 prompt 方法。

---

## 第二轮结果汇总（2026-07-13/14）

> 本轮完成：E5 检索升级、BCP 全量 830、DR-DCI 全量+wiki 六集、hotpotqa 第二数据集、
> 三组消融、search-o1-7B 尝试。口径：popqa_full=随机 1500（固定样本文件）（**用户 2026-07-13
> 正式定为标准集**，抽样误差 ±2.3 F1）；EM/F1 = SQuAD 规范化。

### 9a. popqa_full 主表（1500，EM / F1，按 F1 降序）
| # | agent | EM | F1 | 备注 |
|---|---|---|---|---|
| 1 | **rag_e5 (top5)** | **.4513** | **.5238** | E5 稠密检索，新榜首 |
| 2 | agentir_rag | .4453 | .5172 | AgentIR-4B 稠密 |
| 3 | rag_e5 (top3) | .4447 | .5155 | 超 GrepSeek 表 RAG+E5(9B) .4468 |
| 4 | **search_r1_e5** | **.4387** | **.4740** | **反超原论文 EM .413**（见 9b）|
| 5 | scaleseek | .3120 | .3648 | 我方方法 |
| 6 | grepseek | .3440 | .3870 | 9B 官方 ckpt |
| 7 | bm25_rag (1.2/.75) | .3067 | .3550 | |
| 8 | search_r1 (BM25) | .3080 | .3454 | ckpt 训练分布错配 |
| 9 | search_o1_e5 | .3160 | .3695 | 修复畸形标记后（见 9i）|
| 10 | search_o1 (4B, BM25) | .2813 | .3258 | 修复后；仍 <naive（39% 零检索）|
| 11 | dci | .2260 | .2665 | 自实现（严重低估，见 §4）|
| 12 | direct | .1740 | .2129 | 无检索 |

### 9b. E5 升级 = "检索器轴"三处闭环
Search-R1 公开 ckpt 是 **E5 分布上 RL 训**的；之前配 BM25 = 检索质量 + 策略双重错配。
- popqa RAG：BM25 .3550 → **E5 .5238**（+17 F1）
- **search_r1 同 ckpt**：BM25 EM .3080 → **E5 EM .4387**（+13 点，且**超原论文 .413**）
- hotpotqa（见 9c）：连 **Gold R@W** 都是 E5 赢 → 优势来自**检索器本身**，非 reader 运气

### 9c. hotpotqa 多跳第二数据集（1500，首个 wiki 域 Gold R@W）
| agent | EM | F1 | Gold R@W |
|---|---|---|---|
| **rag_e5** | **.3473** | **.4495** | **.4997** |
| bm25_rag (1.2/.75) | .3307 | .4326 | .4743 |
| scaleseek | .3173 | .4352 | .4050 |
| IRCoT+E5 | .3140 | .4122 | .4700 |
| IRCoT+BM25 | .2980 | .4000 | .4567 |
| search_o1+BM25 (修复版) | .2480 | .3361 | .2277 |
| search_o1+E5 (修复版) | .2373 | .3264 | .2577 |
| direct | .1633 | .2464 | — |
| *search_r1_e5（训练域内，跑批中）* | | | |

多跳上 IRCoT 略低于朴素 RAG（4B 中间 query 弱，交替检索未占优），但 E5 轴仍成立
（IRCoT: E5 > BM25 全指标）。**search_o1 修复版**（旧 buggy .2333/.3093/.1917）：54%
零检索拖累，E5 找回更多 gold（GR .2577>.2277）但答案 F1 未涨 → 瓶颈是"模型不搜+多跳弱"
非检索器（与 popqa §9i 一致）。

E5 三项全赢（含检索召回）。scaleseek 的 Gold R@W 反低于单发 BM25 → 多跳检索不足
（与 9e① 消融印证）。search_o1 召回最低 → RiD 压缩每轮丢弃大部分检索文档。

### 9d. BrowseComp-Plus 全量 830（唯一全指标集）
| agent | EM | F1 | Gold R@W | 备注 |
|---|---|---|---|---|
| bm25_rag (25/1.0) | .0928 | .1365 | .0781 | 长文档参数领先（Pi-Serini 反向验证）|
| grepseek (9B) | .0434 | .0602 | .0198 | **api_err 21%**：grep 长文档撑爆上下文（方法边界）|
| bm25_rag (1.2/.75) | .0229 | .0490 | .0301 | 短文档参数在长文档域失效 |
| scaleseek | .0193 | .0469 | .0394 | 仅 0.9 检索/题（域外）|
| direct | .0024 | .0322 | 0 | |
| **DR-DCI (官方 harness)** | — | — | — | **acc 43.25%（359/830，LLM judge）** vs 论文 nano 71.2% |

Pi-Serini 论断双向验证：25/1.0 在 wiki 短段落输 5-6 EM 点、在 BCP 长文档赢 10 倍+。

### 9e. DR-DCI wiki 六集（各 50，官方协议，4B agent + 自建 E5 检索 + LLM judge）
| | NQ | TriviaQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | **AVG** |
|---|---|---|---|---|---|---|---|
| 我们(4B) | 40 | 50 | 44 | 44 | **48** | 58 | **47.3** |
| 论文(nano) | 62 | 82 | 68 | 58 | 44 | 64 | 63.0 |

差距=底座（4B vs gpt-5.4-nano），**MuSiQue 反超论文**——越考检索的多跳集底座差距越小。
自建 E5 索引（21M，行序天然对齐 wiki_corpus.jsonl）直接喂官方
`searchr1_wiki18_dci_server`，零适配零下载。

### 9f. 消融（详见 `reports/ablations.md`）
① **scaleseek prompt 无数字 vs 给参考值**（3 域×500）：分数统计平手（维持"给参考值"），
   但复读现象被"无数字"变体治好（自选参数从配方复读→有机取值）；真瓶颈是多跳检索不足。
② **生成 max_tokens**：wiki 集 2048 够用（4096 无增益），BCP 需 4096（2048 parse_err 54%）。
③ **grepseek tool_max_tokens**：1024 .3380/.3810(apiE.045)｜2048 .3440/.3870(官方)｜
   4096 **.3573/.4034**(apiE.014)。单调递增（非"甜点"），但 4096 vs 2048 差 ~1.3 EM 在
   噪声内；2048 作忠实默认（匹配 SFT 训练分布）。
④ **search-o1-7B（DeepSeek-R1-Distill-Qwen-7B）负结果**：99% 零检索、一轮 \boxed 直答
   (EM .0833)——纯推理模型不遵守 Search-o1 检索协议。**Search-o1 无官方 7B**（论文用
   QwQ-32B）。**用户拍板：保留 4B search_o1(.2547) 为主表行**，7B 结果留档不入表。

### 9g. IRCoT 新 baseline（2026-07-14 新增，`eval/ircot_agent.py`）
Trivedi et al. ACL 2023（arxiv 2212.10509）。机制：交替"生成一句 CoT → 该句当检索
query → 累积文档 → 继续"，直到"answer is"。零样本指令版（官方用数据集专属 few-shot，
GrepSeek 表的 IRCoT 也是重实现——机制一致，few-shot 是唯一记录差异）。popqa_full 1500：
| 检索后端 | EM | F1 | GrepSeek 表 IRCoT 参照 |
|---|---|---|---|
| BM25 top3 | .2807 | .3200 | .2756（略高）|
| **E5 top3** | **.4347** | **.5122** | .3970（高 +11 F1；甚至超其 IRCoT+Qwen3-4B-emb .4548）|

E5 轴再验证：+BM25 .3200 → +E5 .5122（**+19 F1**），.5122 逼近 rag_e5(.5238)、超
search_r1_e5(.4740)——好检索器下 IRCoT 很能打。no_answer 0，平均 1.4 轮交替检索。

### 9i. search_o1 为何 ≈/< naive BM25（2026-07-15 诊断+修复）
纯 prompt 零训练（Qwen3-4B + 官方 sunnynexus prompt 逐字）。诊断 50% 题**零检索**：
- **76% 无标记**：4B 在长尾 popqa 过度自信、根本不发 `<|begin_search_query|>`，凭记忆
  瞎答（`Easy Living` 作曲→Richard Marx）。**Search-o1 把"要不要搜"交给模型，弱底座
  滥用"我知道"**——纯 prompt 方法对底座强度的固有脆弱（官方用 QwQ-32B/GrepSeek 用 9B
  才肯乖乖搜）。→ 真·能力问题，非 bug。
- **24% 畸形标记**：模型发了 `|<end_search_query>|`（管道符错位），vLLM 没在 stop token
  停下，模型自己脑补检索结果块。→ 我们 bug，**已修**（`_LENIENT_QUERY_RE` 宽松识别 +
  截断脑补尾巴让真检索器跑）。

修复前后（v1→v2）：BM25 .2547/.2959→**.2813/.3258**（零检索 50→39%）；
E5 .2827/.3326→**.3160/.3695**（51→40%）。修复救回 ~3-4 F1；剩余差距是 39-40% 零检索
（模型行为救不了）。E5 v2 .3695 已超 naive BM25 .3550，但远低于 naive rag_e5 .5238——
一半 E5 检索力被零检索浪费。**结论**：search_o1 落后主因是弱底座不肯检索，非实现缺陷。

**三数据集零检索率（修复版）——越难越不搜的病理**：
| 数据集 | 难度 | 零检索率 | search_o1 F1 |
|---|---|---|---|
| popqa 单跳 | 中 | 39% | .3258(BM25)/.3695(E5) |
| hotpotqa 多跳 | 高 | 54% | .3361/.3264 |
| BCP 长文档谜题 | 极高 | **79%** | .0241（低于 direct！）|
越难的题越需要检索，4B 反而越不肯搜——弱模型的"自信"与真实知识不相关，在最该搜时
跳过检索。这是 Search-o1（纯 prompt、检索决策交给模型）对弱底座脆弱性最强的量化证据。
（对标 GrepSeek 表 Search-O1+BM25 .4003 亦为纯 prompt 重实现，差别纯在底座 4B vs 9B。）

### 9h. 对齐条款（用户 2026-07-13）
各 baseline 设置尽量对齐六篇 related work；**结果冲突时以 grepseek 的最优设置为准**。

---

## 忠实性差距总表与升级队列
| baseline | 等级 | 主要残余差异 | 升级动作 |
|---|---|---|---|
| grepseek | **A** | 无（paper 绝对值差距已证非实现问题）| ✅ 完（含 tool_token 消融）|
| dr_dci | **A-** | 模型替换（必然）| ✅ 全量 830(43.25%) + wiki 六集(AVG 47.3) |
| search_r1 | **B+** | ~~BM25 vs E5~~ 已修 | ✅ E5 版 EM .4387 **超原论文 .413** |
| search_o1 | B-/C | 底座+后端 | ✅ E5 版 + 7B 尝试（无官方 7B，保留 4B）|
| bm25_rag/direct | C | reader prompt 自写（无官方）| 无必要（含 5 组扫参 + BCP 长文档验证）|
| agentir_rag | C | 索引自建（官方只发模型）| ✅ 已出数（EM .4453/F1 .5172）|
| **dci** | **D→计划 A/B** | 全套自实现 | 待拍板：改跑官方 dci-agent-lite（资产在盘上）|

**待拍板池（#15，不阻塞）**：① DR-DCI 830 trajectories 换强 judge 离线重判
（4B judge verdict 可用但 reason 有幻觉）；② 六 wiki 集是否给我方 8-agent 全铺；
③ 官方 dci-agent-lite 替换自实现 dci；④ IRCoT 是否实现。

---

## 附录 A：GrepSeek 论文 Table 1 PopQA 列（F1，2026-07-11 自 arxiv/html 抄录）
所有 baseline 均为 **Qwen3.5-9B 底座**、检索 top-3、"trained (when applicable)
and evaluated under the same settings"（即 Search-R1 等训练型方法按各检索器重训）。
与我们表的差异：底座（9B vs 我们 4B/3B ckpt）+ 训练检索器匹配。**只用于方法间
相对趋势参照，绝对值不可对标**；且其 GrepSeek 行 .4861 已被我们证实公开资产
只能跑出 .4154（§1），读全表时应保留同等怀疑度。

| 方法 | +BM25 | +E5-110M | +Qwen3-4B-emb |
|---|---|---|---|
| Direct（无检索）| .2364 | — | — |
| RAG | .3239 | .4468 | .5046 |
| IRCoT | .2756 | .3970 | .4548 |
| Rejection Sampling | .3827 | .4298 | .4630 |
| Search-O1 | .4003 | .4322 | .4731 |
| Search-R1 | .4194 | .4747 | **.5101** |
| GrepSeek（无检索器）| .4861↓（vs 最优 baseline 显著降）| | |
