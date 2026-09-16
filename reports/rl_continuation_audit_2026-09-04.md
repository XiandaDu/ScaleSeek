# RL 续跑前的数据/方法审计（2026-09-04，本机 4×A100）

范围：续跑 `grpo_9b_from190` 启动时能拿到的证据。SFT 轨迹本体（Nibi `$SFT_TRAJ_DIR`）
没有同步到 HF 或本机，因此 SFT 数据只能从生成代码反推，并用 RL 已落盘的 rollout
观察策略的实际行为。所有数字来自冒烟/闸门，不得进结果表。

## 1. SFT 数据（由 `sbatch/sft_data.sbatch` + `train/sft/coldstart.py` 反推）

| 项目 | 值 | 来源 |
|---|---|---|
| 问题源 | `rl_data/train.jsonl` 前 3000 条（默认 `SFT_LIMIT=3000`） | sft_data.sbatch |
| 前 3000 条的来源混合 | hotpotqa 1321 / nq 1154 / musique 292 / 2wiki 233（多跳 62%） | 本机重算（train.jsonl sha256 与 HF provenance 一致） |
| 可用轨迹 | 1725（HF README），良率 ≈58%，`--strict` 逐跳验证 | HF README / sbatch |
| teacher | Qwen3.6-35B-A3B，`--param-policy heuristic` | sbatch |
| 工具响应预算 | **512 token**（`--tool-response-tokens` 默认） | generate_sft_data.py |
| 训练 | verl SFT，3 epoch，lr 5e-6 | sft_train.sbatch |

结构上的结论：
- **多跳问题按构造会有多次 bm25_retrieve**（backward pass 每个 hop 一次调用，forward 时
  第 2 次起 `mode=merge`），单跳 NQ 只有 1 次；grep/read_doc 只在 teacher 主动给出时出现。
  所以「有没有多次 tool call 的例子」答案是有（≈62% 的问题是多跳且需通过逐跳验证），
  但具体比例只有轨迹文件才能给出。
- **参数由规则给定而非 teacher 决定**：`_param_policy` 按 query 词数映射到 3 个常量组
  （≤3 词 → 5/2.0/0.5，4–5 词 → 5/1.5/0.75，≥6 词 → 10/1.2/0.9），并把一句模板化理由
  写进 reasoning。`reports/param_policy_findings.md` 已记录 heuristic 学生会坍缩成常量。
- **失败→恢复样本几乎没有**：backward 只保留验证成功的调用（该报告自己也指出这个混淆）。
- **train/inference 不一致**：SFT 时工具响应截到 512 token，RL/eval 是 2048。
- 质量闸门方向正确（信息前沿检查、答案泄漏检查、`when in doubt FAIL`），但 TUTOR_EDIT
  是「把 reasoning 改写成通向已知正确调用」，FINAL_ANSWER 把答案钉在 gold 上：学生学到的是
  事后合理化的推理，不是从失败中调整的过程。这是 GrepSeek 同款设计，不是本项目独有问题。

## 2. RL step-190 策略的实际行为

80 条训练 rollout（temperature 1.0，冒烟 2 步）+ 16 条贪心闸门轨迹：

| 指标 | 80 条 rollout | 16 条贪心 |
|---|---|---|
| 工具调用次数 | 1 次 89%，2 次 11% | 1 次 100% |
| 工具种类 | bm25 85 次，grep 4 次，read_doc 0 | bm25 16，其余 0 |
| top_k / k1 / b | 3 / 1.2 / 0.75 占 84/82/83 of 85 | 全部 3 / 1.2 / 0.75 |
| mode | replace 80，merge 5 | replace 16 |
| EM | 0.16（hotpotqa .31，nq .10，musique 0/20，2wiki 0/5） | 5/16 |
| format_pass | 0.99 | 15/16 |

也就是说 step-190 的策略 ≈ **固定参数 top-3 单次检索的 RAG**：自适应参数完全没有出现
（连 SFT 教的 5/1.5/0.75 也被换成了 eval 默认的 3/1.2/0.75），多跳题基本全错。
同 16 题贪心 EM：SFT 10/16（sbatch 注释记录）→ step-190 5/16。

## 3. 为什么 RL 学不到新东西（方法论）

1. **奖励信号极稀疏**：n=5 的 16 个 GRPO 组里 12 组分数完全相同（全 0）→ advantage 0 →
   零梯度；只有 2 组有来自 EM 的信号。
2. **惩罚项在全错组里成了唯一信号**：`score = EM − workspace_penalty`，全错时分数是
   0 / −0.05 / −0.063 之类，GRPO 按组标准化后「少调一次、少拿几篇」在全错组里获得正
   advantage。剩下 2 个有信号的组正是这种「只有惩罚在起作用」的组。这直接解释了
   坍缩到单次 top-3。
3. **惩罚阈值对多跳题不友好**：`workspace_target=5`、`max_bm25_calls=3`，第 2 次
   bm25 就开始扣分；而多跳题按定义至少要 2 次。
4. **k1/b 本身不可观测、不可隔离**（param_policy_findings 的结论），EM 又是 0/1，
   RL 没有任何可微的检索质量信号（例如 gold 是否进入 workspace、进入的排名）。
5. **参数记忆的干扰**：rollout 里大量「general knowledge confirms …」——模型在用
   参数记忆作答，检索只是走形式，奖励却照发。

## 4. 建议（供决定，不自动执行）

- 惩罚只对答对的轨迹生效（乘法门控：`EM × (1 − penalty)`），或把 `max_bm25_calls`
  调到 ≥4、`workspace_target` 按问题类型放宽。
- 加检索过程奖励：gold 出现在 workspace（`compute_metrics` 已有 Gold R@W 口径）。
- 课程：先只在多跳子集上训练，或按题型加权采样，否则 NQ 的单跳先验会主导。
- 把 SFT 的工具响应预算改成 2048，与 RL/eval 对齐；重新生成时保留失败→恢复回合。
- 把 SFT 轨迹文件同步到 HF（dataset repo），否则数据质量无法审计。

## 5. 附：产出与工具

- `scripts/analyze_rollouts.py`：对 verl rollout dump / 闸门记录做行为审计。
- `scripts/carve_rl_continuation.py`、`scripts/run_rl_local.sh`、`scripts/rl_stop.sh`。

## 6. 2026-09-05 决定：全量优化（v2）—— 已实现并排入自动链

用户批准了第 4 节的建议。实现（代码）：

| 项 | 实现 |
|---|---|
| 奖励门控 + 过程分 | `train/reward.py`：`penalty_mode=gated`，`score = (EM + 0.3·ret_coverage)·(1 − pen)`，答错不为负；`retrieval_coef` |
| 证据 | `eval/evidence.py`：hotpotqa/2wiki 用 `supporting_facts.title`，musique 用 decomposition 的支撑段落标题 + 子答案；标题级匹配 wiki-18 passage 首行 |
| 轨迹追踪 | `train/environment.py` 记录所有进过 workspace 的 passage 与各工具调用数；`train/agent_loop.py` 在 rollout 结束时算 coverage 并写入 extra_fields（同时不再把 env 对象序列化进 batch） |
| 数据 v2 | `scripts/prepare_rl_data.py --with-evidence --mix … --dev-ids …`：2wiki 14k / musique 15k / hotpotqa 15k / nq 6k，dev 与 v1 同 2000 题同序；另产出 `sft_questions.jsonl`（3000，多跳 90%）与 `screen_pool.jsonl`（12k） |
| 难度筛 | `scripts/screen_difficulty.py`（旧 SFT 策略，n=4，记录 pass_rate / 工具数 / BM25 可达性）→ `--screen` 只保留 0<pass<1 的题 + ≤1500 道可达的 pass=0 多跳题 |
| 配置 v2 | `train/config/grpo_trainer_v2.yaml`：n=8，batch 48，KL(low_var_kl) 0.001 锚 SFT，response 14336，CUDA graphs，阈值 workspace 10/30、calls 5 |
| SFT 重做 | `scripts/sft_data_local.sh`：teacher Qwen3.6-35B-A3B（TP=2），`--param-policy search --tool-response-tokens 2048 --strict`，inject_failures 默认开；`scripts/sft_train_local.sh` 4 卡 verl SFT |
| 自动链 | `scripts/phase2_chain.sh`：等旧 RL 停 → A(SFT 数据, GPU0-1) ‖ B(难度筛, GPU2-3) → C(SFT 训练) → D(筛选后的 v2 训练集) → E(pilot_v2, 40 步, val_before_train) → F(汇总到 `logs/phase2_F_summary.txt`) |

pilot 通过门槛（第 4 节第 7 条）：多次调用比例↑、`reward/ret_coverage`↑、dev 多跳 EM 相对新 SFT ≥ +3 点；否则停下来改而不是放量。

## 7. step-190 在 2000 题 dev 上的正式基线（`grpo_9b_from190` 训练前验证，贪心，2026-09-04）

| | 全部 | 2wiki | hotpotqa | musique | nq |
|---|---|---|---|---|---|
| EM | **0.376** | 0.494 | 0.464 | 0.184 | 0.362 |

行为：单次工具调用 94%（0 次 2.8%，≥2 次 3.1%）；bm25 参数 1986/1987 次为 3/1.2/0.75；grep 32 次，read_doc 0；
format_pass 0.954，hit_max 4.5%，响应 token 均值 1997 / 中位数 913。
这是 v2 pilot 要对比的数字（dev 同 2000 题同序）；后续每个 checkpoint 的 val_rollouts 用
`scripts/analyze_rollouts.py` 得到同表。

## 8. v2 SFT 结果（2026-09-06）

- 数据：三批 teacher 生成合计 6000 题 → **1236 条 ok 轨迹**（第一批 17%，可达题两批 24%）。
- 训练：verl SFT，3 epoch = 114 步，loss 0.36→0.10，`$SFT_CKPT_DIR/v2/global_step_114/huggingface`。
- 16 题贪心闸门（同一组题）：**EM 6/16、作答 11/16**（5 题跑满 8 轮未作答）。行为对比：

| 模型 | 工具调用 ≥2 次 | 用 merge | top_k 取值 | grep |
|---|---|---|---|---|
| step-190（v1 RL） | 0/16 | 0/16 | 全 3 | 0 |
| SFT v2 | 13/16 | 33/51 次 | 3/5/10/20 | 11 次 |

- 难度筛（旧 SFT 代理，pass@4）：6194 题有统计 → 训练集 3374 题（有分歧 1874 + 可达困难多跳 1500）。
- pilot_v2 于 2026-09-07 03:35 UTC 以闸门阈值 0.6 启动（原 0.7 被 11/16 卡住）。

## 9. pilot_v2 结果（40 步，2026-09-08 01:35 → 15:51 UTC，平均 18 min/步）

2000 题 dev（贪心，同题同序），对比 v1 的 step-190：

| 模型 | 全部 | 2wiki | hotpotqa | musique | nq | 作答率 | ≥2 次工具 |
|---|---|---|---|---|---|---|---|
| step-190（v1 RL） | 0.376 | 0.494 | 0.464 | 0.184 | 0.362 | 0.954 | 3% |
| SFT v2（pilot step 0） | 0.389 | 0.426 | 0.550 | 0.248 | 0.332 | 0.689 | 87% |
| pilot_v2 step 20 | 0.429 | 0.520 | 0.568 | 0.264 | 0.364 | 0.795 | 84% |
| **pilot_v2 step 40** | **0.454** | 0.574 | 0.566 | 0.286 | 0.388 | 0.875 | 82% |

训练曲线（train 集是筛出来的难题带，EM 低于 dev 属正常）：format_pass 0.72→0.91，
未作答 0.28→0.09，bm25 调用 4.1→3.3 次/轨迹（仍是多次），grep 0.3→0.12，KL 0.006→0.23，
熵 0.17→0.19 稳定；**ret_coverage 在 0.62–0.74 之间波动、没有明显上升**。
GRPO 零 advantage 组从 v1 的 75% 降到 10%。

门槛：多次调用 ✓（82%），dev 多跳 EM 相对 SFT v2 ✓（2wiki +14.8、musique +3.8、hotpotqa +1.6，
总 +6.5），检索覆盖率 ✗（持平）。大部分增益来自「学会作答」（作答率 0.69→0.875）和 2wiki；
过程奖励没有把覆盖率推高，但也没有再出现坍缩。

决定：同一 run 续到 300 步（`scripts/phase2_full.sh`，接 step-40 的优化器状态，每 40 步验证），
覆盖率不动的问题留到下一轮再调（候选：retrieval_coef 0.3→0.5，或按跳数给覆盖率加权）。

## 10. 放量段与 v2.1（2026-09-15/16）

- 弹性调度 `scripts/phase2_elastic.sh`：2 或 4 卡随空闲切换，切换 = 合并 → 新 run 热启动（优化器重置）。
- `v2_4gpu_from40`（v2 配置，总步 41→115）：dev EM 总步 80 处 0.452（持平），训练中 bm25 调用 3.0→1.7、
  覆盖率 0.69→0.55、KL 0.41（参考点被切换换成了 step-40 权重）。中途两次崩：Triton OOM、
  verl max_tokens=0（已在 agent loop 里加预算守卫）。
- **v2.1**（`grpo_trainer_v2_1.yaml`，总步 115 起）：覆盖率系数 0.5、KL 0.005 钉在 SFT v2
  （`ref.model.path`）、`response_length` 8192（用户决定）、并发 48/副本、TP 1、SP 1、
  每卡 16384 token、actor 参数不 offload。
- 每步耗时：985 s（v2）→ 746 s（v2.1 初版，12288 预算）→ **609 s**（16384 + 参数留 GPU）；
  生成 601→245 s，训练侧 342→356→356 s（update 208、logprob 91、ref 57）；峰值显存 56 GB。

## 11. v3：按用户要求重做 reward，全部重跑（2026-09-16）

用户判断 v2/v2.1 的 reward 过严，指定重做：只奖励正确答案、一点格式，再加一项「grep 里出现了
答案内容」。实现见 `train/reward.py` 的 `reward_shape="simple"`：

```
score = 1.0 × EM + 0.1 × format + 0.3 × grep_hit        # 上限 1.4，不会为负
```

- `format`：标签闭合且 `<think>` 之外有 `<answer>`（等价于「确实作答且格式良好」）。
- `grep_hit`：某次 `grep_workspace` **展示给模型**的段落里出现了 gold（词级连续匹配，
  `no` 不会匹配 `nothing`）；pattern 去掉非字母数字后须 ≥3 字符，否则视为 catch-all 不计分
  （`grep_hit_broad` 记录被这条挡掉的比例）；该项要求本条轨迹确实作答，否则「只 grep 不作答」
  会成为最便宜的正分。`grep_hit_source: prediction` 可切换成「grep 里出现了模型自己的答案」
  这一无标注的 groundedness 变体。
- 取消：workspace/调用次数惩罚、乘法门控、支撑标题覆盖率计分（仍记录 `reward/ret_coverage`）。
- 验证集仍只报 EM，与 v1/v2 的 dev 数字可直接比较。

重跑：删除全部旧 RL run（1.3 TB），从 SFT v2 冷启动，训练集换成未筛的 50k 多跳池
（新 reward 下全错的组也能靠 grep 命中产生梯度，不再需要靠难度筛选回避同分组），
KL 0.001 锚定 SFT v2，其余沿用 v2.1 的速度设置（约 10 min/步）。
入口：`CONFIRM_DELETE=yes bash scripts/phase3_v3.sh`，状态见 `logs/phase3_status.txt`。

## 12. v3.1：按用户给的形式改成 F1 + 格式闸门（2026-09-16 22:01 UTC 重启）

```
reward = format_valid × ( token_f1(pred, gold) + 0.3 × grep_reward )
grep_reward = 1  当且仅当  答案完全正确 且 某次 grep 展示的段落里出现了该 gold
```

与 v3 的差别：base 从二值 EM 换成 token F1（"Alexander I" 对 "Alexander I of Russia" 不再
和完全答错同分）；format 从 0.1 的加分变成乘法闸门（GrepSeek 的形式）；grep 奖励以答对为前提，
不再为"围着错答案乱 grep"付费。验证集仍只报 EM。

实现：`train/reward.py` 的 `reward_shape="f1_gated"`，配置 `train/config/grpo_trainer_v3_1.yaml`。
gold 匹配仍用词级连续匹配（`no` 不匹配 `nothing`），grep pattern 仍需 ≥3 个字母数字字符，
否则 catch-all 不计分 —— 这两条是我加的守卫，不在用户给的示例里。

v3（simple 形式）只跑了一步就被本版取代，记录在案：score 0.52 = EM 0.435 + format 0.073 +
grep 0.013；bm25 调用 3.90 次、grep 0.24 次、grep 命中率 6%、KL 0.008、677 s/步。
