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
| Wiki-18（21,015,324 段） | ✅ | ✅ | 64 个 shard 都已编码（OOM 的 7 个重跑完成，每个约 47 分钟）；HNSW 构建中（5564785） | ✅ | 排在 HNSW 之后（5564796） |

- OOM 的原因：原来按「字符数/3」估算 token 数，遇到数字和表格多的段落会低估，batch 因此过大。现在先用真实 tokenizer 算出长度，再按 padding 后的 token 数分 batch；万一仍然 OOM，就把 batch 减半重试（向量结果不变）。
- 重跑以后的顺序：qwen3 wiki18 的 7 个 shard → `build-hnsw qwen3_emb_4b wiki18 --shards 64`（400G 节点）→ `check-retrievers --corpus wiki18`。作业号见 `$ROOT/dev/index_jobs.txt`。
- `submit.sh` 新增 `--tasks 13,15,...`（配合 `--shards N`），可以只重跑指定的 shard。

## 10-14 晚：代码精简

- 删掉了多余的防御性代码：GrepSeek / AgentIR 的 transient 重跑（冒烟里从未触发）、编码 OOM 时减半 batch 的兜底、旧格式兼容、重复的存在性检查等。Search-o1 的 `max_search_limit` 不再写进 yaml 再校验，直接由 turns − 1 算出。
- 修了一个 bug：`slurm/job.sh` 的 `cd "$SLURM_SUBMIT_DIR"` 让作业回到了 `~/ScaleSeek`，「冻结副本」其实从未生效，作业跑的一直是当时的工作区代码。另外 `.gitignore` 里的 `env/` 会让 lock 文件进不了 git。两处都已修好。

## 下一步（按 BASELINES 的顺序）

1. Wiki-18 retriever check 通过后：`bash submit.sh eval-all --dataset popqa`（15 个 run）→ `score` → 汇报。
2. 汇报之后再跑其余 7 个数据集。BCP 还要跑 `judge`（Qwen3-32B，整节点 4 GPU）和 `score`。

## 提醒

- 10/27–11/03 集群维护停机。
- /scratch 有 60 天清理：最终的轨迹和报告要归档到 /project 或 /nearline。
