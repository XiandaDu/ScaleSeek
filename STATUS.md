# 交接状态（2026-10-14 晚）

## 10-14 的决定已全部落实，重新冒烟 0 失败

| 决定 | 实现 |
|---|---|
| AgentIR 用官方 OSS client（`oss_client.py`），官方 BCP `QUERY_TEMPLATE` | `scaleseek/agentir_entry.py` 直接调用官方 `main()`，只覆盖 sampling、轮数上限、k=3 |
| EM/F1 取 `Exact Answer:` 那一行 | `scaleseek/methods/agentir.py::exact_answer()` |
| BCP 文档按 Qwen3-0.6B 截到 512 token（RAG / Search-R1 / Search-o1 / AgentIR），top-3 不变 | 检索服务的 `display_function`；AgentIR 用官方 `snippet_max_tokens=512`，配置检查强制两边一致 |
| BCP 上 Direct / RAG 每次回答最多 10,000 token | `profiles.bcp.single_call_max_tokens`，finish reason 记为 `max_tokens_cap` |
| GrepSeek 在 BCP 上原样跑官方 harness | 不改 |
| judge 改用 Qwen3-32B（BrowseComp-Plus 官方开源评测的默认 judge），原始输出全部保存 | `configs/methods/bcp_judge.yaml`，TP=4，占一整个节点；`judge__bcp_judge.jsonl` 里存 judge prompt、judge 原文和 agent 回答；如果文件里已有别的 judge 模型的判决，会拒绝运行，不会混在一起 |
| 代码 commit 并 push 到 `new` 分支 | run 的 provenance 记录 git commit；formal run 不接受未 commit 的改动，也不会在另一个 commit 下续跑；`submit.sh` 拒绝未 commit 的改动，每个作业在该 commit 的 git worktree（`$ROOT/harness/<commit>`）里运行 |

PopQA 15 个 run 和 BCP 15 个 run，再加上 judge 和报告，全部跑通（`smoke/{popqa,browsecomp_plus}/REPORT.md`）。冒烟用的是 mini 语料、每个 run 4 题，数字不能当结果。

## BCP judge 换成 Qwen3-32B（10-14 决定）

Qwen3.5-9B 在冒烟中对 AgentIR 的 3 个回答（3 个都答对了）都没写 `correct:` 这一行，官方 parser 判为失败，AgentIR 在 BCP 上因此得 0%。现已改用 Qwen3-32B，并在冒烟的 15 个 BCP run 上重新判分：

- 32B 每个回答都给出了 `correct:`，15 个 run 的 judge 失败数都是 0；
- AgentIR 的 3 题都判为正确（BCP 冒烟 0% → 75%）；
- 其余 14 个 run 的 57 题，32B 和 9B 的判决逐题完全一致；
- 加载很快：61 GiB 权重拷到本地盘 105 s，vLLM（TP=4）启动 175 s。

9B 的判决和旧报告移到了 `smoke/_superseded_20261014/judge_qwen3.5-9b/`。完整冒烟（`run.py smoke`）现在需要 4 张 GPU 的 allocation，因为 judge 是 TP=4。

## 索引

| 语料 | bm25 | e5 | qwen3_emb_4b | agentir_4b | retriever check |
|---|---|---|---|---|---|
| BCP（100,195 篇） | ✅ | ✅ | ✅ | ✅ | ✅ |
| Wiki-18（21,015,324 段） | ✅ | ✅ | ✅（HNSW 221 GB，构建 4.5 小时） | ✅ | 排队中（5564796） |

- qwen3 有 7 个 shard 曾经 OOM：原来按「字符数/3」估算 token 数会低估。现在按真实 tokenizer 长度分 batch，7 个 shard 已重跑成功。作业号见 `$ROOT/dev/index_jobs.txt`。
- `submit.sh` 新增 `--tasks 13,15,...`（配合 `--shards N`），可以只重跑指定的 shard。

## 10-14 晚：代码精简

- 删掉了多余的防御性代码：GrepSeek / AgentIR 的 transient 重跑（冒烟里从未触发）、编码 OOM 时减半 batch 的兜底、旧格式兼容、重复的存在性检查等。Search-o1 的 `max_search_limit` 不再写进 yaml 再校验，直接由 turns − 1 算出。
- 修了一个 bug：`slurm/job.sh` 的 `cd "$SLURM_SUBMIT_DIR"` 让作业回到了 `~/ScaleSeek`，「冻结副本」其实从未生效，作业跑的一直是当时的工作区代码。另外 `.gitignore` 里的 `env/` 会让 lock 文件进不了 git。两处都已修好。

## 10-14 晚：公平性修正（commit 2fbdad8）

1. AgentIR 每次模型调用最多执行一次搜索：官方 OSS client 会执行一次回复里的所有工具调用，现在第一个之后的调用都丢掉（次数记在 sidecar 里）。AgentIR 的 README 也写了，要和 BCP 官方 baseline 公平比较，就关掉并行搜索。
2. 统一的答案规则：只有以答案结束（`finish_reason == "answer"`）的记录才计 EM/F1、才送 judge（`metrics.final_answer`）。BCP judge 对所有方法都只看最终答案这个短串，AgentIR 只送 `Exact Answer:` 的值，不再送整段回答。
3. Search-o1 只从最后一个 `</think>` 之后取 `\boxed{}`，和 Direct/RAG 一致。
4. GrepSeek 每次工具调用的上限保持 BASELINES 的 2,048 / 6,000 token（你的决定），写进了 deviations.md，报告里作为说明列出。

## 正式 run（10-15 开始）

- 顺序（你定的）：Direct × 8 个数据集 → Search-R1-7B / 14B + BM25 × 8 → Search-R1 + e5 / qwen3_emb_4b × 8 → RAG → Search-o1 → GrepSeek → AgentIR（后四个由我排序，都跑 8 个数据集）→ BCP judge → 全部 `score`。每个方法跑完我检查一遍，再提交下一个。
- 所有正式 run 都从 commit `8aa3ae9` 的 worktree（`$ROOT/harness/8aa3ae938c51`）提交：它和冒烟用的 `2fbdad8` 相比只多改了 STATUS.md，代码完全相同；之后 STATUS.md 再怎么改，也不会让 run 落到不同的 commit 上。作业日志在该 worktree 的 `logs/` 下，作业号记在 `$ROOT/dev/formal_jobs.txt`。
- 前提都已满足：全新冒烟（5615537）0 失败；Wiki-18 retriever check 通过（4 个索引都是 21,015,324 篇，重复查询结果一致，命中 id 都在语料内）。
- 冒烟里看到的现象：AgentIR 在 PopQA 上 4 题都跑满 6 轮。它每轮都在正常检索，只是反复换说法核实，没有给出答案。没有并行调用被丢弃（0 次），所以和 10-14 的改动无关；轮数预算和其他方法一样，不改，报告里会写。

| 方法 | 状态 |
|---|---|
| Direct × 8（旧解码） | 作废，移到 `$ROOT/runs_superseded/20261015_no_presence_penalty/`（死循环，见下） |

**10-15 发现：Qwen3.5-9B 的 thinking 死循环。** Direct 里用完 16K 窗口、没给出答案的比例：PopQA 269/500、TriviaQA 113/500、NQ 159/500、HotpotQA 165、2Wiki 210、MuSiQue 162、Bamboogle 26/125；BCP 有 462/500 撞到 1 万 token 上限。前三个数据集里，95% 以上的截断记录最后 4K 字符是同一句话的重复（PopQA 253/269、TriviaQA 111/113、NQ 155/159）。Qwen3.5-9B 没有 `generation_config.json`，所以不是服务端默认参数的问题，而是 BASELINES 的解码设置（temperature 0.6、top_p 1.0、不发其他参数）在 thinking 模式下容易死循环；Qwen3.5 模型卡对 thinking 模式的建议是 `top_p=0.95, top_k=20, presence_penalty=1.5`。

**10-15 决定：所有方法统一加 `presence_penalty=1.5`**（temperature 0.6 / top_p 1.0 不变；judge 不加）。vLLM 不能在服务端设 presence_penalty 的默认值（`--override-generation-config` 只认 temperature/top_p/top_k/min_p/repetition_penalty），所以由每个客户端随请求发送同一个值：我们自己的方法走 `LLMClient.complete`，GrepSeek 走新的 `scaleseek/grepseek_entry.py`（包装官方 harness 的 OpenAI 客户端，官方代码不改），AgentIR 走 `agentir_entry.py`（Responses API 的 `extra_body`）。正式 run 改从新 commit 的 worktree 提交；先在 PopQA 上跑 Direct，看死循环是否消失，再跑其余数据集。

## 提醒

- 10/27–11/03 集群维护停机。
- /scratch 有 60 天清理：最终的轨迹和报告要归档到 /project 或 /nearline。
