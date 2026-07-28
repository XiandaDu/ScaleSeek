# ScaleSeek — 敌意审稿报告（ICLR reviewer + reproducibility auditor）

审稿时间：2026-07-27　审稿环境：arcade02（跳板机，只做静态审计 + 纯 CPU 微验证，未跑任何 GPU/重任务）
审稿对象：`main` @ `08bd315`，外加 `logs/`、`results/phase2/`、`reports/` 全部落盘产物。

**证据分级**：
- **[已验证]** = 我在本会话中用代码/日志/落盘文件直接证实。
- **[高风险]** = 代码路径推断明确，但需一次真实运行才能最终确认。
- **[待查]** = 需要更多信息或作者澄清。

**先说公道话**（下面全是刀子，这段是唯一的糖）：`results/phase2/*.jsonl` 每行带
`method_config_id / prompt_sha256 / dataset_manifest / retriever_manifest /
harness_source_sha256 / upstream_harness_commit`，`--full-eval` 有 ID 全集校验和抽样禁令，
`BASELINE_SOURCES.md` 逐条标注忠实度等级并明确拒绝把自实现叫官方——这套 provenance 纪律
**高于我审过的大多数 ICLR 投稿**。`reports/baselines.md` §10f 用 McNemar 配对检验主动
撤回自己 n=500 的 "top-k=3 更好" 结论、§10j 主动揭发自家 judge 有 11–13% 机械矛盾、
§9f④ 主动标注旧 7B 结果被 bug 污染不可引用——这三处自我证伪是真本事。

**但是**：以上全部是**流程质量**，不是**结果**。这篇稿子现在的状态是——
**核心方法一个数字都没有，而 Stage-2 的训练信号在冻结配置下恒等于 0。**

---

# 零、审稿人最先看到的事实

| 项 | 状态 |
|---|---|
| Phase-2 冻结协议下 **ScaleSeek 本体**的结果 | **不存在**（`results/phase2/` 无 scaleseek 文件）|
| Phase-2 冻结协议下 **direct / rag** 对照 | **不存在**（同上）|
| 已完赛的行 | 只有 `search_r1` 7B/14B × {bm25,e5} 四格 + `search_o1_bm25` 一格 |
| 正在跑 | `search_o1_e5` 12,700/14,267（截至 07-27 23:38，未完）|
| `reports/baselines.md` 里那张 14 方法 × 6 数据集大表 | **全部来自已被废弃的 Qwen3-4B / 随机 1500 抽样体制**，与当前冻结配置（Qwen3.5-9B / 全量 14,267）**没有一个数字可以共用** |

也就是说：**目前 repo 里唯一"看起来像论文主表"的东西，和目前唯一"真跑出来"的东西，
属于两个互不兼容的实验体制，而提出的方法在两个体制里都没有可用数字。**

---

# 一、FATAL（这几条任意一条单独就足以 reject）

## F1. 冻结的 production mode 让 RL 奖励对**完全正确的答案**恒等于 0

**被挑战的主张**：`README.md` 表格 "2c. RL (GRPO) — implemented"；
`08bd315` 冻结 `enable_thinking: false` 为 scaleseek production mode。

**证据（已验证，本会话实测）**：
```
nothink rollout -> format gate: (False, 'unbalanced <think>')
  reward score = 0.0   em = 1.0   format_pass = 0.0
thinking rollout -> score = 0.998  format_pass = 1.0
```
机制在 `train/reward.py:213-216`：
```python
fmt_input = solution_str
if not fmt_input.lstrip().startswith("<think>"):
    fmt_input = "<think>\n" + fmt_input      # 补开标签
format_pass, _ = _check_format(fmt_input)     # 但没有 </think> -> 计数不平衡
```
`_check_format` 要求 `<think>` 与 `</think>` **成对**。`enable_thinking=false` 时
生成段里两个标签都没有，补上开标签后变成 1 open / 0 close → `unbalanced <think>`
→ `score = 0.0 if not format_pass`（`reward.py:240`）。

**为什么是 fatal 而不是 major**：GRPO 的 advantage 是组内 reward 的 z-score。
一组 5 个 rollout 全是 0.0 → 标准差 0 → advantage 0 → **梯度恒为 0**。
训练不会崩、不会报错、loss 曲线好看，**它只是什么都不学**。
这是最坏的一类 bug：静默的 no-op。而且 `trainer.total_training_steps: 200`、
`save_freq: 40` 会老老实实存 5 个和初始化权重无差别的 checkpoint。

**严重度**：**FATAL**。Stage-2 的 RL 贡献在冻结配置下不可能存在。

**最小修复**：
1. `_check_format` 改成 thinking-agnostic：只要求 `<answer>` 存在且 `<tool_call>` 配对，
   `<think>` 改为"若出现则必须配对"。
2. 加一条回归测试（现在没有）：
   `assert scaleseek_reward(..., "<answer>Paris</answer>", golds=["Paris"])["score"] > 0`。
   注意现有 `tests/test_reward.py` 的 fixture 是
   `sol = f"I searched the corpus.</think>\n<answer>...</answer>"`——**它特意塞了一个裸
   `</think>` 来凑平衡**，等于把"必然有 thinking"这个假设焊死进了测试夹具，
   于是这条 fatal bug 被 5 个 PASS 的测试完美掩盖。

---

## F2. 枪毙掉 direct/rag/scaleseek 的那道 gate，本身是个 parser bug；而 production freeze 建立在它的输出之上

**被挑战的主张**：`configs/baselines.yaml:211-216` + `eval/run_eval.py:254-258` +
`sbatch/p2_local.sbatch:64-67` 三处写着：
> "greedy thinking never closes within any usable budget (probe 07-23), and vLLM's
> thinking_token_budget forced closure corrupts `<tool_call>` blocks (vllm#44676),
> leaving `enable_thinking=false` as the only working mode."

**证据（已验证）**：

(a) 我用当前代码和 07-24 之前的代码分别解析日志里出现的那类字符串：
```
CURRENT parser                       -> 'Politician'
with instruction-quote in think, CURRENT -> 'Politician'
OLD-style (search full text)         -> ''          <-- 就是它把一切判成 0
```
`19198ee` 的 commit message 自己写得很清楚：模型在 thinking 里逐字复读格式说明中的
空 `<answer></answer>`，旧 parser 用 `re.search` 命中**第一个**匹配 → 永远返回空串。

(b) 时间线（`git log --date=iso` 已验证）：

| 时间 | 事件 |
|---|---|
| 07-23 23:51 | `p2_rag_bm25` GATE-FAIL |
| 07-24 00:25 | `p2_direct` GATE-FAIL |
| 07-24 01:05 | `p2_rag_e5` GATE-FAIL |
| 07-24 01:39 | `p2_scaleseek` GATE-FAIL |
| **07-24 15:09** | **`19198ee` 修好 parser** |
| 07-26 14:55 | `08bd315` 依据上面四份 gate 表冻结 `enable_thinking=false` |

**四次 GATE-FAIL 全部发生在修复之前。修复之后 gate 一次都没有重跑。**

(c) 我把四份日志里每个失败 probe 的 `content_head` 拉出来数了一遍——
**模型其实已经吐出了合法闭合的非空 `<answer>`，只是 parser 看不见**：

| 日志 | 模式 | gate 报告 | 失败 probe 中实际含合法 `<answer>` | `finish=length` |
|---|---|---|---|---|
| p2_scaleseek | **budget_think** | 0/16 | **14/16** | **0** |
| p2_direct | **budget_think** | 0/16 | **14/16** | **0** |
| p2_rag_bm25 | **budget_think** | 0/16 | **12/16** | **0** |
| p2_rag_e5 | **budget_think** | 0/16 | **15/16** | **0** |
| （四份）| greedy_think | 0/16 | 3–5/16 | 9–12 |

`budget_think`（= 现在 `baselines.yaml` 里 direct/rag 的 production 配置）
**在四份日志里 `finish=length` 全是 0、completion_tokens 中位数稳定在 4119
(=thinking_budget 4096 + 答案)、每一发都干净收尾**。它工作得**完美**。
gate 给它打了 0/16。按 threshold=0.8 复算：14/16=0.875、14/16、12/16=0.75、15/16=0.9375
→ **四道门里三道本该 GATE-PASS**。

(d) `greedy_think` 那部分结论**是真的**（9–12/16 `finish=length` 在 28000 token 上限撞墙）。
但 `baselines.yaml:56` 写的 "**provably never closes** ... 0/16 answered" 是被 bug 放大的
表述——实际有 3–5/16 正常闭合，是 ~60–75% 不闭合，不是 100%。

**严重度**：**FATAL**（对 Stage-1 全部 prompt-based 行）。
四个 job、数十 GPU 小时被一个已修复的正则 bug 烧掉；更糟的是，**一个改变方法本体的
配置决策（给提出的 agent 关掉 reasoning）是基于这份被污染的证据做出的**，而且到今天
（07-27）还冻结在 `configs/baselines.yaml` 里。

**最小修复**：在 `19198ee` 之后的代码上，对 direct/rag/scaleseek 各重跑一次
`probe_thinking_closure.py`（16 题 × 4 模式，几分钟的事）。在拿到新表之前，
`enable_thinking=false` 这个冻结**必须撤销**，三处引用它的注释一并撤销。

---

## F3. gate 从来没有测过它所把守的那个 agent

**被挑战的主张**：`sbatch/p2_local.sbatch` 按 agent 分派 `GATE_MODE`
（`direct|rag) budget_think;; scaleseek) greedy_nothink`），读起来像是每个 agent
用自己的配置过闸。

**证据（已验证）**：`scripts/probe_thinking_closure.py:32-35`
```python
def probe_one(client, model, question, mode, max_tokens, thinking_budget):
    messages = [
        {"role": "system", "content": prompts.load("direct")},   # <-- 写死
        {"role": "user",   "content": f"Question: {question}"},
    ]
```
**system prompt 永远是 `direct`**，`main()` 里根本没有 agent 参数。于是：
- `rag` 的 gate 没有注入任何检索段落 → 测的就是 direct；
- `scaleseek` 的 gate 没有 tool schema、不会产生 `<tool_call>`、workspace 全程为空
  → 测的还是 direct。

**直接后果**：F2 里那句 "thinking_token_budget **corrupts `<tool_call>`** (vllm#44676)"
——**repo 里唯一被引为证据的那份 probe，其 prompt 根本不可能产生 `<tool_call>`**。
我全仓 grep 了 `44676`：只在 `baselines.yaml` / `p2_local.sbatch` / `run_eval.py`
三处**注释**里出现，没有任何 probe 日志、测试、复现脚本、issue 链接。
（上游 issue 可能真实存在，但**本仓库没有任何证据支持这个 agent 在这个配置下的这个行为**。）

**严重度**：**FATAL**（方法论）。一道不测被测对象的 gate，等于没有 gate；
用它的输出去论证一个它物理上观测不到的现象，是循环论证。

**最小修复**：`probe_thinking_closure.py` 加 `--agent {direct,rag,scaleseek}`，
按 agent 加载对应 prompt；scaleseek 模式下必须断言至少产生一个可解析的 `<tool_call>`。
在此之前，`vllm#44676` 那句话从三个文件里删掉，或替换成一条真实的复现记录。

---

## F4. 提出的方法在当前实验体制下，没有任何数字

**被挑战的主张**：`README.md` "1. Evaluation — implemented"；
`reports/baselines.md` §10a/§10b "scaleseek_e5 多跳均 EM .3335 排第一，
压过 9B 的 grepseek"、"我方方法在 2wiki / musique / bamboogle 三个集上直接夺冠"。

**证据（已验证）**：
- `ls results/phase2/` → 只有 `search_r1_{7b,14b}_{bm25,e5}` + `search_o1_bm25`
  （+ 未完成的 `search_o1_e5`）。**没有 scaleseek、没有 direct、没有 rag。**
- `reports/baselines.md` 开篇口径："popqa_full = ... **随机抽的固定 1500 样本
  （seed 不可考）**"、"prompt 型 baseline 底座 = **Qwen/Qwen3-4B**"、
  Search-R1 用 **3B** ckpt。
- 当前冻结配置：`configs/baselines.yaml` generator = **Qwen3.5-9B**，
  Search-R1 = **7B/14B**，`popqa_test_expected_count: 14267`。
- 而 `TASK.md:65` 明文：**"正式实验禁止 `-n/--n`、`--offset`、随机抽样、
  固定 1500 子集或只跑 first-N。"**

**所以**：那张 14 方法 × 6 数据集的大表，用的底座、检索器 ckpt、样本量、抽样协议
**四项全部**与当前冻结协议冲突，且其抽样方式被项目自己的规则书明令禁止，
**且 seed 不可考**（`metric_support.md` §0 原话）→ **任何人（包括作者自己）
都无法重现那 1500 是哪 1500**。

**严重度**：**FATAL**。论文当前**没有主结果**。
`reports/ablations.md` 顶上有 `legacy_invalid_for_main_table` 横幅——
**但 `baselines.md` 和 `metric_support.md` 没有**，而它们才是最像主表的两份文件。

**最小修复**：
1. 立刻在 `reports/baselines.md` 和 `reports/metric_support.md` 顶部加同样的
   `legacy_invalid_for_main_table` 横幅，并注明"底座 4B / 抽样 1500 / seed 不可考，
   与 Phase-2 冻结协议不兼容"。
2. 先解 F1/F2/F3，再按冻结协议跑 scaleseek + direct + rag 全量 14,267。
   **在那之前，任何对外材料里不得出现 "多跳第一 / 三集夺冠"。**

---

# 二、MAJOR

## M1. 延迟指标被并发污染，而并发度根本没落盘——TARGET.md 的核心命题无法评估

**被挑战的主张**：`TARGET.md:144-149` "The target is to achieve: **higher accuracy
under the same latency budget; lower latency under the same accuracy target**"。
这是整个 project 的中心命题。

**证据（已验证）**：
- `eval/agent.py:480-487` — `t_llm = perf_counter()` 包住一次**阻塞 HTTP 调用**，
  而 `eval/run_eval.py:361` 用 `ThreadPoolExecutor(max_workers=conc)` 并发。
  测到的是 **service time + 排队等待**，随并发度线性劣化。
- 各 agent 并发度**不同**（`sbatch/p2_local.sbatch`）：
  `search_r1_7b → CONC=8`、`search_r1_14b → CONC=6`、其余（含 scaleseek/direct/rag/search_o1）
  → `CONC=16`。
- `grep -n "concurrency" eval/run_eval.py scripts/compute_metrics.py` → **只在 print 里出现，
  从不写进 jsonl 行，也不写进 `.metrics.json`**。落盘的 26 个字段里没有并发度。

落盘数字的荒谬程度：
| run | `sec_per_query` | `llm_time_s_mean` | 并发 |
|---|---|---|---|
| `search_o1_bm25` | **208.95** | 208.87 | 16 |
| `search_r1_14b_bm25` | **2.68** | 2.66 | 6 |
| `search_r1_14b_e5` | 14.02 | 2.12 | 6（`tool_time` 11.90 > `llm_time`）|

208.95 s/query × 14,267 = **828 GPU-小时的"LLM 时间"**，而该 job 墙钟约 53 小时——
差的 15.6× 就是并发系数。**这不是延迟，这是吞吐的倒数。**
拿 208.95 去和 2.68 比"效率"，比的是两次不同的批处理压力。

**严重度**：**MAJOR**（对核心命题是 fatal）。
efficiency 轴上现在一个可用数字都没有，而它是 TARGET.md 的两条成功判据之一。

**最小修复**：
1. 把 `concurrency` 写进每行 jsonl 和 `.metrics.json`（一行改动，立刻做）。
2. 效率结论只能来自 **`--concurrency 1`** 的专门子集跑（如 n=200，各方法同 n 同机同卡），
   或改报 **throughput (queries/GPU-hour)** + **token 成本 + tool-call 次数**这类
   并发不变量。现有 `n_tool_calls_mean` / `n_bm25_calls_mean` 是干净的，可以直接用。
3. 主表的 latency 列在 (1)(2) 完成前**必须删掉**，不能只加脚注。

## M2. RL 的验证集就是评测集

**证据（已验证）**：`scripts/prepare_rl_data.py:43-48`
```python
# (dataset_name, eval_split, has_train_split)
("nq",              "test",  True),
("hotpotqa",        "dev",   True),
("2wikimultihopqa", "dev",   True),
("musique",         "dev",   True),
```
docstring：`Val: NQ (test) + HotpotQA (dev) + 2Wiki (dev) + MuSiQue (dev)`。
这四个 split 正是 `reports/baselines.md` §10a 主表评测所用的 split。
而 `train/config/grpo_trainer.yaml`：`val_before_train: true`、`test_freq: 40`、
`save_freq: 40`、`total_training_steps: 200` → **每 40 步在测试集上验证并存 checkpoint**。

**严重度**：**MAJOR**。哪怕不显式挑 best checkpoint，在测试集上画验证曲线本身就是泄漏；
若有任何 checkpoint 选择行为，则是彻底的 test-set fitting。

**最小修复**：从各数据集 **train split** 切出 held-out val（NQ/HotpotQA/2Wiki/MuSiQue
都有 train）；评测 split 在整个训练期间**物理隔离**（不同目录 + 一条 assert）。

## M3. 训练数据优势恰好落在宣称获胜的那三个集上

**被挑战的主张**：§10b "我方方法在 **2wiki / musique / bamboogle** 三个集上直接夺冠"，
且强调 "注意 scaleseek 用的是 4B 底座，grepseek 是 9B 且经 GRPO 训练"。

**证据（已验证）**：ScaleSeek RL 训练混合 = NQ + HotpotQA + **2Wiki + MuSiQue** (train)；
Search-R1 公开 ckpt = `SearchR1-**nq_hotpotqa_train**-...`（`BASELINE_SOURCES.md:46-47`），
**没有** 2Wiki / MuSiQue。

**严重度**：**MAJOR**。一旦 RL 跑起来，"2wiki/musique 夺冠"就是
**多两个 in-domain 训练集**的直接推论，不是方法优势。
（当前 §10a 的 scaleseek 行还是 prompt agent、未训练，所以这条现在还是**前瞻性**缺陷——
但它会在第一次 RL 结果出来的瞬间变成 fatal。）

**最小修复**：二选一——(a) 训练混合削到 NQ+HotpotQA 与 Search-R1 严格对齐，
2Wiki/MuSiQue/Bamboogle 全部作为 **held-out 泛化集**；或 (b) 保留现混合，
但主表明确标注 in-domain / out-of-domain，且**跨方法宣称只允许在 OOD 集上做**。

## M4. SFT 报告里唯一的定量表是循环论证

**被挑战的主张**：`reports/param_policy_findings.md` §3
| policy | recall (gold in workspace) |
|---|---|
| omit | 0.31 |
| heuristic | 0.75 |
| **search-then-teach** | **1.00** |

结论："**search wins**"。

**证据（已验证）**：`train/sft/coldstart.py:401-421`
```python
def _search_params(retriever, query, targets, bm25_idx):
    ...
    for k1 in _K1_GRID:
        for b in _B_GRID:
            rank = _gold_rank(retriever.retrieve(query, top_k=top_probe, k1=k1, b=b), targets)
            ...  # 取使 gold rank 最小的 (k1,b)
    top_k = next((t for t in _TOPK_LADDER if t >= rank), top_probe)   # 恰好装下 gold 的 top_k
```
`_gold_rank`（:391）判定标准是"检索到的 passage 是否**包含 gold 答案串**"，
调用处（:569-570）`targets = [hop.expected] + list(hop.forms)` 即**金标准答案**。

也就是说：search 策略**用 gold 做网格搜索，直到 gold 进 workspace，然后把
"gold 是否在 workspace"当成评价指标**。recall=1.00 不是实验结果，
**它是这个搜索的终止条件**。只要网格里存在任一组参数能让 gold 上榜，recall 必然=1.00。

**严重度**：**MAJOR**。作为 oracle teacher 去蒸馏是完全正当的方法；
**但把 oracle 自己的目标函数当作 policy comparison 的指标报出来是无效比较**。
这是该报告唯一的定量表。

**最小修复**：这张表改报**学生**的 held-out recall（而不是 teacher 的构造成功率），
或至少把该列改名为 `oracle grid coverage` 并明确"按构造 ≤1.00，不可与其他行比较"。
报告已经诚实地在 §4 说了学生没学到 k1/b——把那个才当主结果。

## M5. SFT 在教模型做它在推理时做不到的事

**证据（已验证）**：

(a) **模板化的伪推理**（`coldstart.py:425-427`），被拼进 SFT 的 `<think>` 里：
```python
why = (f"A default search ranks the most on-topic passage around position "
       f"{default_rank or 'off the list'}; with k1={...} and b={...} it moves to "
       f"position {rank}, so I'll use those and set top_k={top_k} ...")
```
学生在推理时**永远不可能知道 gold 的 rank**。这是典型的
imitation-of-unattainable-behavior，且是**固定模板**——学生唯一能学的就是背下这个句式。
`param_policy_findings.md` §4 观测到的正是这个：heuristic 学生 → "a single constant
`1.5/0.75/5`"；search 学生 → top_k 有变化但 `k1/b` 恒定。**报告把这归因于
"数据里没有 k1/b 信号"，但同样成立的解释是"学生背下了模板但填不出数字"**——
两个假设在现有证据下无法区分。

(b) **默认放行未验证的 hop**：`ColdStartConfig.strict: bool = False`（:57，
注释 "skip examples whose hops fail to verify"）。默认 `False` ⇒
**检索没验证到 expected 的 hop 照样进数据集**。

(c) **答案无条件钉死**：`coldstart.py:588-590`
```python
final_reason = _final_reasoning(teacher, question, history, cfg)
answer_turn = _fmt_answer_turn(final_reason, golds[0])    # 恒等于 gold
```
(b)+(c) 合起来 = **workspace 里可能压根没有答案，轨迹却断言 gold**。
这是在系统性地教幻觉。唯一防线 `QUALITY_JUDGE` 是**同一个 teacher 家族自评**
（`_quality_pass`, :281-289），而 `reports/baselines.md` §10j 恰好已经证明
本地小 judge 有 **11–13% 机械矛盾**（自己抽出的答案与 gold 逐字相同却判错）。

(d) **`grep_workspace` 允许含答案**（`README`/`coldstart.py:17` 的 answer-leak 规则：
"a `bm25_retrieve` query never contains expected[i]; **grep may**"）。
于是学生被教着 grep 一个它推理时不知道的字符串。

**严重度**：**MAJOR**。

**最小修复**：
1. `strict` 默认改 `True`；未验证 hop 一律丢弃并统计丢弃率。
2. `<answer>` 钉 gold **之前**加一道**程序化**（非 LLM）检查：
   final workspace 至少一个 passage 含 gold 串，否则整条轨迹标 `unsupported` 丢弃。
3. `why` 模板里所有 oracle 量（`default_rank`、`rank`）删掉，只保留可从工具返回值
   观测到的表述。
4. grep pattern 也纳入 answer-leak 禁令，或明确论证为何 grep 泄漏无害
   （现在 `README` 只是陈述规则，没有论证）。

## M6. train / eval 的答案抽取是两个不同函数，语义不同

**被挑战的主张**：`README.md:8-10` "Every stage — evaluation, SFT generation, RL —
shares the same system prompt, the same three tools, and **the same output format**,
so a trajectory is **interchangeable** across them."

**证据（已验证，本会话实测）**：
```
输入: '<think>rehearsing</think>\n<answer>RIGHT</answer>\n<think>second thought <answer>LATE</answer></think>'
  eval  parse_assistant   -> 'RIGHT'    # think 外的第一个块
  RL    _extract_prediction -> 'LATE'    # 全文最后一个块
```
`eval/agent.py:346-363` 先剥 `<think>` 再取**第一个**；
`train/reward.py:56-65` 在**全文**上 `findall` 取**最后一个**，**完全不剥 think**。

**严重度**：**MAJOR**。RL 优化的目标函数与最终汇报的指标不是同一个。
`19198ee` 修了 eval 一侧，**忘了 reward 一侧**——同一个 bug 的另一半还活着。

**最小修复**：`train/reward.py` 直接 `from eval.agent import parse_assistant, clean_answer`，
删掉 `_extract_prediction`。加一条 property test 断言两侧在随机轨迹上恒等。

## M7. 随机解码、单次运行、无 seed、无置信区间

**证据（已验证）**：
- `grep -n "seed\|random" eval/run_eval.py eval/datasets.py` → **零命中**。
- `configs/baselines.yaml`：search_r1 `temperature: 0.7`、search_o1 `0.7`、
  grepseek `0.6`（`run_eval.py` 里 search_r1 硬编码 `temperature=0.7`，
  search_o1 `temperature=0.7, top_p=0.8, sampling_top_k=20`）。
- 五份 `.metrics.json` 全部是 **n=14267 的单次抽样**，无 seed、无重复、无 CI。

于是 `search_r1_14b_e5 EM .4959` vs `search_r1_7b_e5 EM .4517` 这类比较
**没有任何不确定性刻画**。n=14267 时 EM 的 SE≈0.42 点，所以 4.4 点差是稳的；
但 `search_o1_bm25 F1 .4299` vs GrepSeek 论文 `Search-O1+BM25 .4003` 这种
跨来源比较、以及未来 scaleseek vs grepseek 的小差距，**必须有 CI 才能说话**。

**严重度**：**MAJOR**（可低成本修复）。

**最小修复**：(1) `--seed` 落盘进 jsonl；(2) 对随机解码方法至少 3 seeds，
或对 EM/F1 报 bootstrap 95% CI（n=14267 上 1000 次 bootstrap 是秒级 CPU 操作）；
(3) 方法间比较一律用**同题配对检验**（§10f/§10g 已经会做 McNemar 了，推广到主表即可）。

## M8. "一条 pipeline" 实际横跨三个不同模型

**证据（已验证）**：
| 阶段 | 模型 | 出处 |
|---|---|---|
| Stage-1 eval generator | **Qwen3.5-9B** | `baselines.yaml:8` |
| Stage-2c RL 默认 base | **Qwen3-8B** | `grpo_trainer.yaml:112` `${oc.env:SCALESEEK_MODEL_PATH,Qwen/Qwen3-8B}` |
| Stage-2a SFT student | **Qwen3-1.7B**（teacher Qwen3-4B/8B）| `train/sft/README.md:54,56` |

`README.md` 的 pipeline 图声称三阶段共享 prompt/tools/format，"a trajectory is
interchangeable across them"。模型不同 → chat template 不同 → thinking 行为不同
（F1 就是被这个咬的）→ **轨迹不可互换**。

**严重度**：**MAJOR**（可比性）。训好的 ScaleSeek 若是 8B，去和 9B 底座的 prompt
baseline 同表比较，是 apples-to-oranges；`reports/baselines.md` §10b 已经在用
"注意 scaleseek 是 4B、grepseek 是 9B" 这种话把底座差异当**加分项**说了。

**最小修复**：把 `SCALESEEK_MODEL_PATH` 的 fallback 从 `Qwen/Qwen3-8B` 改成
`baselines.yaml` 里的冻结 generator，并加一条 contract test 断言两者一致；
或在主表显式增加 "base model / params" 列，所有跨方法宣称都必须同底座。

## M9. 报告与代码/规则书自相矛盾（至少 5 处）

全部**已验证**：

| # | 文档说 | 代码/规则书说 |
|---|---|---|
| 1 | `baselines.md`：主表用**随机 1500**（seed 不可考）| `TASK.md:65`：**正式实验禁止**随机抽样和固定 1500 子集 |
| 2 | `metric_support.md` §1：PopQA 本地文件 = `popqa/popqa_longtail.jsonl` | `eval/datasets.py:27,38`：`("popqa","test")`, `EXPECTED_FULL_COUNTS=14267`；`README:104` "no popqa_full or long-tail aliases" |
| 3 | `ablations.md` §1 结论："**维持 B 为标准 prompt**"（给参考数字那版）| `ccf9d8f` 把 **A（无数字）**设为默认；`param_policy_findings.md` 反过来论证 A |
| 4 | `baselines.md` §2/§3：search_r1/search_o1 "我们参数 **temperature 0.0（贪心）**" | `baselines.yaml:96,109` + `run_eval.py`：**0.7** |
| 5 | `PHASE1_STATUS.md`："**20 passed** offline tests" | 实际 `grep -c '^def test' tests/*.py` = **36** |
| 6 | `train/reward.py:18` docstring："Workspace stats are logged but **not yet included** in the training signal" | `grpo_trainer.yaml:277` `enable_workspace_penalty: **true**`；`44f641d` 就是实现它的 commit |

**严重度**：**MAJOR**（累积）。审稿人交叉核对时每命中一条，对全文可信度的折扣是乘性的。
第 1、2 条尤其致命：它们让人怀疑主表究竟跑在哪份数据上。

**最小修复**：一次性文档过账。`baselines.md` / `metric_support.md` 加 legacy 横幅
（见 F4）；#2 #4 #5 #6 直接改字；#3 补一段"决策已于 `ccf9d8f` 反转，理由见
`param_policy_findings.md`"。

---

# 三、MINOR（但审稿人会逐条问）

| # | 问题 | 证据 | 修复 |
|---|---|---|---|
| m1 | RL prompt 预算可能装不下 system prompt | `grpo_trainer.yaml:94` `max_prompt_length: 1024`（注释 "system prompt + question"）；`prompts/scaleseek_prompt.py` = **4,630 字符 ≈ 1,100–1,160 token**，还没加 question。**[高风险，需用真 tokenizer 确认]** | 用 Qwen tokenizer 实测；不够就提到 2048，并加 assert |
| m2 | scaleseek 的 eval 超参硬编码，绕过"冻结配置" | `run_eval.py:261` `max_turns=8, max_tokens=8192` 写死；同一个函数里 direct/rag 却老老实实读 `mcfg`。`baselines.yaml` 的 `max_tool_calls: 7` / `max_new_tokens: 8192` 对 scaleseek **无效** | 全部改从 `method_cfg` 读；加 contract test |
| m3 | 训练/评测轮数预算不一致 | eval `max_turns=8`；RL `max_assistant_turns: 6`, `max_user_turns: 5` | 统一，或在论文里显式说明并做敏感性分析 |
| m4 | 关键 bug 零测试覆盖 | 没有任何测试覆盖 (a) `parse_assistant` 处理真实 thinking 输出（烧掉 4 个 job 的那个）、(b) nothink 下的 reward 路径（F1）、(c) eval/RL 抽取一致性（M6） | 三条回归测试，各 5 行 |
| m5 | 结果产物不入版本控制 | `.gitignore:1` `results`；`git ls-files results/` 为空 | 至少把 `.metrics.json`（<1KB）纳入版本控制，jsonl 走 LFS 或存 checksum |
| m6 | `search_o1` 2.6% 无答案未在报告中说明 | `finish_reason` 分布：13891 answer / **211 no_answer** / **165 max_turns**；`.metrics.json` 的 `parse_error_rate: 0.0` 掩盖了这 376 例 | 主表加 `answer_rate` 列（数据已在，只是没报） |
| m7 | `reports/` 里多处"待拍板"项从未收敛 | `baselines.md` 文末待拍板池 ①（换强 judge 重判，"仍待拍板"、"仓库里没有离线重判脚本"）；§10j 已把它定性为 "**必须做**" | 要么做，要么把所有 LLM-judge 口径数字从论文里删掉 |

---

# 四、Novelty 与 significance（最伤的一刀）

**被挑战的主张**：`TARGET.md:43` "The main contribution is a **trained retrieval and
workspace-control policy**, rather than a fixed BM25-to-DCI pipeline."，
其可学习动作空间列为：是否检索 / query / `top_k` / **`k1`、`b`** / merge-vs-replace / 何时停。

**问题**：最强的相关工作已经占了绝大部分设计空间。按 `BASELINE_SOURCES.md` 自己的记录：
- **RISE**（2606.06880）：**BM25 boundary** + 每子查询 K=1000 + top-10 预览 +
  **monotonic file workspace** + bash/read 在 workspace 内检索。
  → "先 BM25 圈定 workspace，再在其中 grep/read" 这个**核心架构 RISE 已经有了**。
- **DR-DCI**（2606.14885）：标题就是 "Scaling DCI via **Dynamic Workspace Expansion**"，
  `pull(query, topK)` 300–600、≤10 pull queries、rank-aware 预览。
  → "**动态**调整 workspace 大小"也已经有了。
- **GrepSeek**（2605.29307）：GRPO 训练一个 DCI 搜索 agent。
  → "**训练**这件事"也已经有了。

于是 ScaleSeek 的**增量**只剩：把 BM25 的**参数**（`k1/b/top_k/mode`）交给学到的策略控制。

**而这个增量，被项目自己的报告否掉了大半**。
`reports/param_policy_findings.md` 的结论逐字如下：
> "**Do not rely on SFT cold-start to teach `k1/b` adaptation.** Scope the adaptive
> action space to **`top_k` / `merge`-vs-`replace` / query reformulation**"
> "`k1` **cannot be isolated in BM25**"（唯一词高 idf → 任何 k1 下都排第一）
> "only **`b`** is physically isolable ... and only within a **narrow, hand-tuned window**"

**即：论文的核心新意（自适应 BM25 参数控制）中，`k1` 被证明在 BM25 里物理上不可辨识，
`b` 只在人工构造的窄窗口内有效。** 剩下 `top_k` + merge/replace ——
这两个相对 RISE / DR-DCI 的 workspace 管理是很薄的 delta。

而 §10f 又实测：`top_k ∈ {3,5,10}` 在 popqa/2wiki/musique 上做 n=1500 配对 McNemar
**全部不显著**（p=0.74 / 0.093 反向 / 0.50），k1/b 在 wiki 域"各档差 ≤0.4 EM"。
**连固定参数的敏感性都测不出来，那学一个自适应策略去调它，上限在哪？**

**严重度**：**MAJOR / 接近 FATAL**。这是我作为 reviewer 会写在 meta-review 第一句的话：
*"The paper's own negative results eliminate most of its claimed contribution;
what remains is a thin delta over RISE + DR-DCI."*

**最小修复（这条最重要，也最难）**：重新定位贡献。**诚实的、仍然有价值的**定位有两个：
1. **"BM25 参数在 passage 级语料上不可学、在长文档域才有效"** ——
   把 `param_policy_findings.md` + §10d（BCP 上 k1/b 换档差 9 F1，wiki 上差 ≤0.4 EM）
   + §10f 合成一篇**负面/诊断性论文**。这个证据链现在**已经跑完了**，而且很扎实，
   是 repo 里最有说服力的东西。
2. **"workspace 规模 vs 语料规模的 scaling 行为"** —— `TARGET.md:107` 原本的主实验
   ("how performance changes as **corpus size increases**") **一次都没做过**。
   全仓没有任何 corpus-size sweep。这才是 "Scale"Seek 名字所承诺的东西，
   而它是唯一一个 RISE/DR-DCI/GrepSeek 都没系统做过的轴。

---

# 五、五条最强 rejection 理由

1. **Stage-2 的训练信号在冻结配置下恒等于 0**（F1，已实测）。
   `enable_thinking=false` × format gate 要求 `<think>` 配对 ⇒ 正确答案得 0.0 ⇒
   GRPO advantage 恒 0 ⇒ RL 静默空转。而 5 个 PASS 的 reward 测试用一个塞了裸
   `</think>` 的 fixture 把它完美掩盖。

2. **提出的方法没有任何符合当前协议的实验数字**（F4，已验证）。
   `results/phase2/` 里没有 scaleseek/direct/rag。唯一的主表来自 4B 底座 +
   seed 不可考的 1500 随机子集，而该抽样方式被 `TASK.md:65` 明令禁止。

3. **一个改变方法本体的配置决策，建立在已被修复的 parser bug 的输出之上**（F2/F3，已验证）。
   四道 gate 全在 `19198ee` 之前跑；`budget_think` 实际有 12–15/16 合法答案却被记 0/16
   （按 threshold 复算三道本该 PASS）；据此关掉了提出 agent 的 reasoning，至今冻结。
   而据以论证的那道 gate **根本没加载被测 agent 的 prompt**（写死 `direct`）。

4. **项目自己的负面结果消灭了大部分新意**（第四节）。
   `k1` 在 BM25 中不可辨识、`b` 只在人工窄窗内有效、`top_k` 的固定值差异 n=1500
   配对检验全不显著。相对 RISE（BM25 boundary + monotonic workspace）和
   DR-DCI（dynamic workspace expansion）的残余 delta 极薄。

5. **中心命题（同延迟下更高精度）所需的指标不可用**（M1，已验证）。
   latency = 并发下的墙钟排队时间，各 agent 并发度 6/8/16 不同且**不落盘**。
   208.95 s/q vs 2.68 s/q 之比几乎全是并发系数。同时 RL 验证集 = 评测集（M2），
   训练混合在宣称获胜的三个集上有 in-domain 优势（M3）。

---

# 六、投稿前五个最高优先级动作

按"改一行能救多少"排序：

1. **修 F1 并加回归测试**（工时：1 小时）。
   `_check_format` 改 thinking-agnostic + 一条
   `assert score("<answer>Paris</answer>", golds=["Paris"]) > 0`。
   **在这条修好之前跑任何 RL 都是纯烧电。**

2. **在 `19198ee` 之后的代码上重跑四道 gate，并撤销 `enable_thinking=false` 冻结**
   （工时：几分钟 GPU + 半天决策）。同时给 `probe_thinking_closure.py` 加
   `--agent`，让 scaleseek 的 gate 真的走 scaleseek prompt 并断言产生
   可解析 `<tool_call>`。`vllm#44676` 那句话在拿到真实复现前从三个文件里删掉。

3. **按冻结协议跑出 scaleseek + direct + rag 的全量 14,267**（工时：GPU 天级）。
   在这之前论文没有主结果。跑之前先把 `concurrency` 和 `seed` 写进落盘字段（M1/M7）。

4. **切断 RL val / eval 的重叠，并对齐训练混合**（工时：2 小时）。
   val 从 train split 切；2Wiki/MuSiQue/Bamboogle 转为 held-out，
   或主表加 in-domain/OOD 标注（M2/M3）。

5. **文档过账 + 重新定位贡献**（工时：1 天）。
   `baselines.md`/`metric_support.md` 加 legacy 横幅；M9 的 6 处矛盾逐条改字；
   然后按第四节把 story 从"自适应 BM25 参数策略"改成
   **"BM25 参数可学性的负面诊断"** 或 **"workspace × corpus-size scaling"**。

---

# 七、必须弱化或删除的表述

| 现表述 | 出处 | 处理 |
|---|---|---|
| "scaleseek_e5 多跳均 EM .3335 **排第一**，压过 9B 的 grepseek"、"**三个集上直接夺冠**" | `baselines.md` §10b | **删**。4B/1500/seed 不可考体制，且训练混合有 in-domain 优势 |
| "2c. RL (GRPO) — **implemented**" | `README.md` 表 | 改 "scaffolded; reward signal degenerate under frozen config (see F1); **never run**" |
| "greedy thinking **provably never closes**" / "0/16 answered" | `baselines.yaml:56`、三处注释 | 改 "~60–75% fail to close at 28k budget"，并补 budget_think 的真实通过率 |
| "thinking_token_budget **corrupts `<tool_call>`** (vllm#44676)" | `baselines.yaml:215`、`p2_local.sbatch:67`、`run_eval.py:257` | **删**，或补一份真实复现记录（当前零证据，且据以论证的 probe 不产生 tool_call） |
| "**search wins**: recall **1.00**" | `param_policy_findings.md` §3 | 改 "oracle grid coverage（按构造 ≤1.00，不可跨行比较）" |
| "Every stage ... **the same output format**, so a trajectory is **interchangeable**" | `README.md:8-10` | **删** "interchangeable"。三个不同模型（9B/8B/1.7B）+ 两套答案抽取语义 |
| "**popqa_full** = 随机 1500 ... 正式定为 stage-1 主对比标准集" | `metric_support.md` §0 | 标 legacy；与 `TASK.md:65` 直接冲突 |
| "20 passed offline tests" | `PHASE1_STATUS.md` | 改 36；并注明覆盖率缺口（m4） |
| latency / `sec_per_query` 列 | 所有主表 | **删列**，直到 `--concurrency 1` 或换成吞吐/token 成本 |

---

# 八、不必做的实验（省下来的 GPU 拿去做第 6 节第 3 条）

1. **更多 `k1/b` 扫参**。§10f（wiki 各档 ≤0.4 EM）+ §6（三组短文档配置统计平手）
   + `param_policy_findings.md`（k1 在 BM25 中不可辨识）已经三重证死。**够了，收工。**
2. **`max_tokens` 2048 vs 4096 再补跑**。§10g 八个 baseline 全部配对检验完毕，
   净变化 0，且已抓出并更正了唯一那个"显著"（是 bug 修复的功劳）。这块做得比论文要求还细。
3. **`search_o1` 7B/更多底座**。§9f④ 已定性（纯推理模型不遵守检索协议），
   且该数字自己已标注被 bug 污染。留档即可，不值得重跑。
4. **BCP 上的 ScaleSeek 域外扩展**。§10d 已经说清瓶颈是 4B reader 处理 19KB 长文档，
   不是检索器。换 9B 底座前做任何 BCP 扫参都是在测 reader 不是测方法。
5. **DR-DCI wiki 六集扩样**。§10 待拍板池 ⑤ 已查证 `dci-bench` 每集只有 50 题，
   物理上不可扩。别再试了。

---

# 九、诚实的 ICLR 判决

## 分数

| 项 | 评分 |
|---|---|
| **Rating** | **3 / 10 — reject** |
| **Confidence** | **4 / 5** |
| **Soundness** | 1 / 4 |
| **Presentation** | 2 / 4 |
| **Contribution** | 1 / 4 |
| **Reproducibility** | 2 / 4（provenance 基建 4/4；实际可重现的数字 1/4）|

**Confidence 为什么是 4 而不是 5**：F1（reward 恒 0）、F2（parser bug 时间线）、
F3（gate 写死 direct prompt）、M4（oracle 循环）、M6（抽取分歧）、M1（并发污染）
六条我都做了直接验证（实测执行 / 日志逐行统计 / git 时间戳）。
扣 1 分是因为：m1（prompt 长度超 1024）我只用 4 字符/token 估算、没跑真 tokenizer；
`vllm#44676` 上游是否真实存在我无法离线核实（我的结论是"**本仓库没有证据**"，
不是"该 bug 不存在"）；`search_o1_e5` 还在跑，结论可能变。

## Meta-review 我会写的话

> 这个项目的**工程纪律远超其科学产出**。provenance ledger、commit-gated 官方
> harness、frozen prompt hashes、full-eval ID 校验、以及三处主动的自我证伪
> （McNemar 撤回 top-k=3、揭发 judge 的 11–13% 机械矛盾、标注被 bug 污染的旧结果）
> ——这些是我今年审到的最认真的复现基建之一。
>
> 但审稿看的是**主张与证据的匹配**，而这里的缺口是结构性的：
> 提出的方法在当前协议下**一个数字都没有**；唯一的主表跑在一个被项目自己规则书
> 禁止、且 seed 不可考的抽样上；Stage-2 的训练信号在冻结配置下**恒为零**；
> 而据以做出该冻结决策的证据，来自一个已修复的 parser bug 加一道不加载被测
> agent prompt 的 gate。
>
> 更根本的是：项目自己最扎实的那份负面结果（`k1` 在 BM25 中不可辨识、
> `b` 只在人工窄窗有效、`top_k` 差异统计不显著）**恰好否掉了标题所主张的核心新意**。
> 作者已经掌握了一篇好论文的材料——只是那篇论文的题目不是
> "Training Adaptive Retrieval Agents"，而是
> **"BM25 参数在 passage 级检索中不可学：一份负面诊断"**，
> 或者 `TARGET.md` 里写了却从未执行的那个实验——
> **workspace 机制随 corpus size 增长的 scaling 行为**。后者是 "Scale"Seek 这个名字
> 所承诺、而 RISE / DR-DCI / GrepSeek 都没系统做过的唯一一条轴。

## 达到 borderline accept 的最短路径

1. 修 F1 + F2/F3 → 跑出 scaleseek/direct/rag 全量（第 6 节 1–3 条）；
2. 切断 M2 的 val/eval 重叠、对齐 M3 的训练混合；
3. 主表全部改成**同题配对检验 + bootstrap CI**（§10f/§10g 的做法推广到全表）；
4. 补 `TARGET.md:107` 那个从未做过的 **corpus-size scaling 主实验**；
5. 把 contribution 从 "adaptive k1/b" 改成 "workspace-size control under corpus scaling"，
   并把 k1/b 的负面结果作为**独立贡献**正面写出来，而不是藏在 `reports/` 里。

前三条是**修 bug + 重跑**，不需要新想法。第 4–5 条才是这篇论文能不能活的关键。
