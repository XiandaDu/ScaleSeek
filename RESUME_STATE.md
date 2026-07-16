# RESUME_STATE — 恢复运行指南（快照时间：2026-07-12）

> **2026-07-12 现场**：会话直接跑在 octal35 上。shard4(GPU0)/shard5(GPU1) 已用
> driver_gpu.sh 复跑；**接力脚本 `logs/pipeline_after_shards.sh` 已挂**（日志
> `logs/pipeline.log`），会自动完成：shard4 完工→起 4B vLLM@:8000→BM25 四组补扫
> （§2b）→shard5 完工→6/6 校验→agentir 检索预计算→agentir_rag 评测（§2）。
> 即 §2 + §2b 无需手动执行，只需盯 `tail logs/pipeline.log`。
> 完成标志：`PIPELINE_ALL_DONE`。octal30 当前无 slurm job，登不上。

> 下次会话读这个文件即可恢复全部运行。配套：`reports/WEEKLY_2026-07-10.md`
> （周报）、`reports/metric_support.md`（指标定义+发现）、`RUNBOOK.md`（完整命令手册）。

## 0. 铁律（血泪教训，违反必翻车）
- **slurm job 重启会 cgroup 连坐杀掉所有 setsid 进程**（2026-07-12/13 多次实录：
  vLLM/检索/E5 构建/eval 全灭）。**run_eval 现已支持 `--resume`**（跳过 output
  里已完成的 ID，api_error/exception 行重跑，结束合并）——所有长跑 eval 都带上它。
  恢复：直接重发对应链脚本（幂等）。当前两条主链：
  `logs/chain_gpu1_grepseek.sh`（9B → BCP grepseek --resume → tool_tok 消融）、
  `logs/chain_gpu0_search.sh`（3B → search_r1_e5 → 7B → search_o1_7b×2）。
  两条独立、各占一卡、可直接 `setsid nohup bash ... &` 重发。
- DR-DCI 官方 harness 用 `DCI_RESUME_RUN=1`（wrapper 旁路，按题跳过已完成）。
- **⚠️ `--resume` 陷阱**：若 output 文件是**旧配置**的遗留（如 hotpotqa_scaleseek 那个
  7405 行、缺 bm25_calls/workspace_doc_ids 字段的 pre-trace 版），resume 会静默
  跳过全部、按旧数据出分。跑新配置前先确认 output 不存在或确属当前配置。旧文件
  移开存为 `.STALE_xxx`。另：output 行数 > -n 时 resume 保留全部旧行并按其 n 出分。
- **⚠️ 最高规则：任何 heavy 命令前先 `hostname`！** 跳板机 `arcade*` 上严禁跑
  编码/评测/vLLM/大 IO（2026-07-10 事故：恢复驱动发到 arcade03 崩溃循环 11h）。
  GPU 节点：`octal30`=4×A5000/24G/大内存，`octal35`=2×A6000/48G/可用内存~32G，
  **两节点 GPU 不同，启动参数不可混用**。`driver_gpu.sh` 已内置 hostname 守卫。
- python 一律用绝对路径：`PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python`
  （裸 `python` = base env，无 vllm/faiss）。
- 长任务一律 `setsid nohup ... > /data/rech/mofengra/ScaleSeek/logs/xxx.log 2>&1 &`
  （日志放 NFS 持久目录，不放 /tmp scratchpad——会话重启会清）。
- 杀进程：按 GPU-UUID 或括号模式（`[b]uild_agentir`），且**杀与启永远分开两条命令**
  （同一命令里启动行的明文字符串会被自己的 pgrep 匹配→自杀）。
- 裸 grep agent（dci）并发 ≤2；grepseek ≈16；indexed agent（bm25/search_o1）24-48。
- 当前节点 octal35：2×A6000(48G)，可用内存 ~32G。NFS 偶发卡死→进程 D-state 等待即可自愈。

## 1. 当前现场（2026-07-10 快照）
- **AgentIR 6 片索引 @ `/data/rech/mofengra/data/agentir_index_v2/`**：
  shard0-3 ✅ 完工（各 3,502,554 条 + index.faiss）；
  shard4 @2.12M、shard5 @2.00M（**已用 --resume 续跑**，各剩 ~1.4M ≈ 8-9h）。
- 编码速率物理极限 ~46-60 条/s/卡（fp16）。检查进度：
  `for i in 0 1 2 3 4 5; do wc -l /data/rech/mofengra/data/agentir_index_v2/shard$i/doc_ids.txt; done`
- 驱动脚本：`logs/driver_gpu.sh <gpu> "<shard:skip> ..."`（内置 3 次重试 + --resume）。
  若 builder 死了重发：
  ```bash
  L=/data/rech/mofengra/ScaleSeek/logs
  setsid nohup bash $L/driver_gpu.sh 0 "4:14010216" >/dev/null 2>&1 &
  setsid nohup bash $L/driver_gpu.sh 1 "5:17512770" >/dev/null 2>&1 &
  ```
- vLLM 服务器当前未起（等索引完工后按 §2 起）。

## 2. 索引完工后的收尾序列（按序执行）
```bash
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index DATASETS=/data/rech/mofengra/datasets

# (a) 逐片批检索（校验 6/6 都是 3502554 后）
CUDA_VISIBLE_DEVICES=0 $PY scripts/precompute_agentir_retrieval.py \
  --dataset popqa_full -n 1500 --index-root /data/rech/mofengra/data/agentir_index_v2 \
  --top-k 5 --device cuda --out results/popqa_full_agentir_retrieval.jsonl
# (b) 4B 服务器（注意 hermes 工具解析 flags——DR-DCI 需要）
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 > logs/vllm8000.log 2>&1 &
# (c) agentir_rag 评测（补 popqa_full 表第 8 行）
$PY -m eval.run_eval --dataset popqa_full --agent agentir_rag --concurrency 32 \
  --agentir-cache results/popqa_full_agentir_retrieval.jsonl \
  --output results/popqa_full_agentir.jsonl
$PY scripts/compute_metrics.py --results results/popqa_full_agentir.jsonl \
  --out results/popqa_full_agentir.metrics.json
```

## 2b. ✅ 已完成（2026-07-12 11:36）BM25 扫参 popqa_full 重扫
结果：0.9/0.4 = .3100/.3582；1.2/0.75 = .3067/.3550（主表行复用）；
1.5/0.75 = .3120/.3590；16/1.0 = .2613/.3038；25/1.0 = .2507/.2908。
短文档三组统计平手（SE≈1.2 EM 点），主表继续用 1.2/0.75；长文档两组掉 5-6 点。
10 个旧 longtail 扫参文件（jsonl+metrics）已按批准删除。baselines.md §6 已更新。
原计划（留档）：
背景：原扫参跑在 popqa_longtail 上（当时 `--dataset popqa` 的本地默认文件），
用户决定**只要 full 版本**。1.2/0.75 那组不用重跑——`popqa_full_bm25.jsonl`
就是它（主表行），直接复用，保证主表行和扫参表数字一致。只需补 4 组：
```bash
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek   # 前提：4B vLLM @ :8000 已起（命令见 §2 的 (b) 步）
for kb in "0.9 0.4" "1.5 0.75" "16 1.0" "25 1.0"; do set -- $kb
  $PY -m eval.run_eval --dataset popqa_full --agent bm25_rag --concurrency 32 \
    --bm25-k1 $1 --bm25-b $2 --output results/popqa_full_bm25_k1-$1_b-$2.jsonl
  $PY scripts/compute_metrics.py --results results/popqa_full_bm25_k1-$1_b-$2.jsonl \
    --out results/popqa_full_bm25_k1-$1_b-$2.metrics.json
done
```
跑完并核对后：删除 5 个 `results/popqa_bm25_k1-*` 旧扫参文件（用户已批准），
并更新 `reports/baselines.md` 的扫参表为 full 数字。
其余 longtail 旧结果（A/B 组：`popqa_direct_4b/popqa_dci/popqa_scaleseek_4b/
popqa_search_r1/popqa_search_o1/popqa_grepseek_faithful/popqa_bm25_rag_4b`）
用户拍板**先不动**，未经再次确认不得删除。

## 3. ✅ DR-DCI 官方冒烟通过（2026-07-12 15:5x，qid=769 correct=True acc=1.0）
链路：官方 harness→vLLM(agent)→pull 物化 343 篇→DCI→答对 Queen Arwa University
→judge(agent) 判卷成功。**全量 830 待用户拍板**（连同 judge 用什么模型——4B judge
的 reason 文本有幻觉痕迹（"Berlin capital"），verdict 对但解释不可靠，真跑建议换）。
**必需 env（缺一必翻车，教训实录）**：
- `DCI_VIEW_CACHE_ROOT=/data/rech/mofengra/dr_dci_official/.view_cache`
  —— 必须与 corpus 同文件系统！默认 /tmp 会让多行文档 hardlink 全 EXDEV
  （吞成 missing，399/400 物化失败的假象）。
- `DCI_JUDGE_BASE_URL=http://127.0.0.1:8000/v1/responses`
  —— judge 原码写死 api.openai.com；vLLM 原生支持 /v1/responses（已实测含
  reasoning.effort）。补丁：pi_rpc_runner.py:346 与 run_bcplus_eval.py:3735
  改为 env 可覆盖（默认值不变）。
- `DCI_JUDGE_MAX_OUTPUT_TOKENS=2048` —— 原 180 会被 Qwen3 thinking 吃光正文
  （judge 重试 8 次全空）。补丁：pi_rpc_runner.py:338 env 可覆盖。
- venv 补装：CUDA torch（`uv pip install --reinstall --index-url
  https://pypi.org/simple torch`）+ peft（tevatron encoder 需要）。
**已知固有行为（非 bug）**：root_flat 压平文件名撞名（history.txt ×26 等）
→ 每 400 hit 约 54 个记 missing；官方模式官方语料同样如此，忠实。
```bash
cd /data/rech/mofengra/dr_dci_official
# retriever 现在可用 GPU1（索引完工后空闲）：先给它的 venv 换 CUDA torch
uv pip install torch  # 替换 cpu 版
set -a; source .env; set +a
CUDA_VISIBLE_DEVICES=1 setsid nohup .venv/bin/python tools/dense_retriever/faiss_searcher.py \
  --index-path 'indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl' \
  --model-name Qwen/Qwen3-Embedding-8B --port 8002 --max-top-k 5000 \
  > /data/rech/mofengra/ScaleSeek/logs/drdci_retriever.log 2>&1 &
# :8002/retrieve 通了之后：
BCP_LIMIT=1 DCI_RUN_NAME=smoke_vllm DCI_OVERWRITE_RUN=1 \
  bash scripts/bcplus_eval/run_smoke_vllm.sh   # 已改好 provider=vllm/model=agent
```
已打通项：models.json（vllm provider）、agent-dir 拷贝逻辑、依赖（faiss/tevatron/torch）、
数据（data/bcplus_qa.jsonl + corpus/bc_plus_docs 都在）。判卷 judge=agent（仅冒烟用）。

## 3c. E5 升级结果（2026-07-13）+ search_o1-7B 发现
- **rag_e5 top5 EM .4513/F1 .5238**（popqa_full 新榜首，超 agentir .4453/.5172）
- **rag_e5 top3 EM .4447/F1 .5155**（超 GrepSeek 表 RAG+E5 9B 的 .4468）
- **search_r1_e5（E5 top3+4轮）EM .4387/F1 .4740 —— 超原论文 PopQA EM .413！**
  同 ckpt BM25 版仅 .3080 → E5 版 .4387，涨 13 点。坐实"检索器轴"论证：
  Search-R1 公开 ckpt 是 E5 分布上 RL 训的，配 BM25 = 双重惩罚；换 E5 后忠实复现。
- search_o1_e5(4B) EM .2827/F1 .3326
- **⚠️ search_o1-7B(#10) 用 DeepSeek-R1-Distill-Qwen-7B 失败**：99% 题零检索、
  平均 1.01 轮——R1-Distill 是纯推理模型，不发 Search-o1 检索标记，一轮 think 完
  直接 \boxed{} 拍脑袋（EM .0833，比 direct 还差）。**Search-o1 官方无 7B ckpt**
  （其论文用 QwQ-32B）。**#10 已拍板（2026-07-13）：保留现有 4B search_o1
  (.2547/.2959) 为主表 search_o1 行；不上 7B。** 承认 Search-o1 无官方 7B。
  证据文件 `results/popqa_full_search_o1_7b_bm25.jsonl`（EM .0833，99% 零检索）
  留档但不入主表。

## 3d. DR-DCI wiki 六集 ✅（2026-07-14 07:32，各50题，4B agent+自建E5检索+4B judge）
NQ 40 / TriviaQA 50 / HotpotQA 44 / 2Wiki 44 / MuSiQue 48 / Bamboogle 58，AVG **47.3**
（论文 gpt-5.4-nano AVG 63.0）。差距=底座；**MuSiQue 48 反超论文 44**——越考检索的
多跳集底座差距越小。0 失败。自建 E5 索引（21M）直接喂官方 searchr1_wiki18_dci_server
（行序天然对齐 wiki_corpus.jsonl，零适配），产物 `dr_dci_official/outputs/qa/*_wiki18_e5_vllm4b/`。
补装 fastapi/uvicorn 到 dr_dci .venv。

## 3e. 🔄 hotpotqa 多跳第二数据集（2026-07-14 起，GPU0，chain_gpu0_hotpot.sh）
direct/bm25_rag(1.2/.75)/rag_e5/scaleseek/search_o1 @hotpotqa n=1500，出 EM/F1 +
Gold R@W（title 库 /data/.../corpus_title_index.db 已在）。验证 E5>>BM25 是否推广到多跳。

## 4. popqa_full 对比表 ✅ 8/8 行齐（2026-07-12 14:53）
**agentir .4453/.5172** ≫ grepseek .3440/.3870 > scaleseek .3120/.3648 >
search_r1 .3080/.3454 ≈ bm25(1.2/.75) .3067/.3550 > search_o1 .2547/.2959 >
dci .2260/.2665 > direct .1740/.2129
（agentir 与 GrepSeek Table 1 检索器梯度交叉验证吻合，见 baselines.md §8）
现场：6/6 索引完工；vLLM 4B @ :8000 在 GPU0 常驻；GPU1 空闲。

## 5. 三线并发现场（2026-07-12 傍晚，用户拍板"DR-DCI/E5/BCP 都跑"后）
**线 1 ✅ DR-DCI 全量 830 完赛（2026-07-13 早）：accuracy 43.25%（359/830），
830/830 判卷、0 失败**。论文 71.2%（GPT-5.4-nano）→ 差距=纯底座。产物
`dr_dci_official/outputs/bcplus_eval/full830_vllm/`。judge=4B，可换强 judge
离线重判（重跑 run_bcplus_eval 会按题跳过 agent、只补判卷）。中断续跑用
`DCI_RESUME_RUN=1`（斜杠 wrapper 已支持）。
**线 2 🔄 E5**：索引构建 GPU1（`logs/e5_build.log`，~9h，崩溃带 --resume 重发）；
**接力已挂** `logs/e5_relay.sh`（日志 `logs/e5_relay.log`）→ 完工自动跑 4 个
popqa_full 评测：rag_e5(top5)、rag_e5_top3（GrepSeek 参照 .4468）、
search_r1_e5（论文口径 top3+4 轮，对标 EM .413）、search_o1_e5。
新代码：`eval/e5_retriever.py`（BM25Retriever 同签名即插即用，offsets.npy
懒建）+ `run_eval --retrieval-backend {bm25,e5}`（$E5_INDEX_DIR）。
**线 3 🔄 BCP 我方 agent**：`logs/bcp_smoke_chain.sh`（日志同名 .log）50 题 ×
{direct, bm25(1.2/.75), bm25(25/1.0), scaleseek, grepseek(--corpus-path BCP)}，
BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index；metrics 带
--bcp-qrels/--bcp-doclen（Gold/Qrel R@W + Coverage/Localization 全开）。
datasets.py 已注册 `browsecomp_plus`（$DATASETS/browsecomp_plus/queries.jsonl）。
冒烟过目后 → 全量 830 铺开。
**GPU 布局**：GPU0=vLLM:8000（DR-DCI+BCP 共用）；GPU1=E5 构建 + DR-DCI 检索:8002。

## 6. 后续队列
1. 全量数据集铺开（RUNBOOK；hotpotqa/2wiki 出全量 Gold R@W）
2. 官方 dci-agent-lite 跑法
3. 待拍板：IRCoT 是否实现、popqa_full 是否扩到全 14,267、DR-DCI 重判 judge 选型
