# RESUME_STATE — 恢复运行指南（快照：2026-07-20）

> 新会话读这一个文件即可恢复上下文。配套文档：
> `reports/baselines.md`（各 baseline 实现细节 + 三轮结果汇总，**最权威**）、
> `reports/ablations.md`（消融）、`reports/metric_support.md`（指标定义）、
> `reports/WEEKLY_2026-07-19.md`（最新周报）、`RUNBOOK.md`（命令手册）。

---

## 0. 铁律（违反必翻车）

- **⚠️ 最高规则：任何 heavy 命令前先 `hostname`。** 跳板机 `arcade*` 上严禁跑
  评测/vLLM/大 IO。GPU 节点的卡各不相同，**启动参数不可互抄**：

  | 节点 | GPU | 关键约束 |
  |---|---|---|
  | octal30 | 4×A5000 24G | 9B 必须 TP=2；4B+E5 同卡时 util ≤0.72 |
  | octal25 | 1×RTX3090 24G | 只有 14 核，别排 faiss/E5 重活 |
  | octal40/41 | 4×L40S 48G | **必须 `export CUDA_HOME`**（见下） |
  | abaque01/02 | 4×2080Ti 11G | Turing 无 bf16，须 `--dtype float16` + TP=2 |

- **无 NVLink 的卡（A5000 / 2080Ti / 3090）跑 vLLM 张量并行会静默死锁**：日志停在
  `vLLM is using nccl==…` 后再无输出，显存几百 MiB 却 100% 利用率。
  必须 `NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce`。
  （9B 单卡放不下：32 层/4 KV 头/head_dim 256 → 128 KB/token，32k 单序列就要 4.00 GB。）

- **octal40/41 没有 `/usr/local/cuda`**，不设 `CUDA_HOME` 则 flashinfer JIT 找不到
  nvcc → vLLM 全灭且**静默**（run_eval 照跑，每行 api_error，照样打印 ALL_DONE）。
  ```bash
  export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
  export PATH=$CUDA_HOME/bin:$PATH
  export VLLM_USE_FLASHINFER_SAMPLER=0
  ```

- **`/tmp` 是 tmpfs（占物理内存），跨作业不清理。** 跑 dci-lite / pi coding-agent 前
  必须 `export TMPDIR=/var/tmp/<name>`（本地 NVMe）+ `trap 'rm -rf $TMPDIR' EXIT`。
  否则 pi-bash 日志能堆到 101GB 内存，表现为莫名的写失败/JSON 截断。

- **sbatch 不写 `--cpus-per-task` 只给 1 核（=2 线程）**，faiss/grep/vLLM 会被饿死。
  一律显式写（octal30/40/41 用 48）。

- **只 `wait` lane 的 PID，不要裸 `wait`** —— 裸 wait 会把永不退出的 vLLM 也等进去，
  作业跑完后照样空占整机（2026-07-16 实录空转 9.5 小时）。

- **每个作业的输出文件名必须与其他作业严格切分**，不要靠"时间上应该错开"避让
  （2026-07-18 两个进程同时写 `musique_grepseek`，数据侥幸没坏但白烧 7 GPU 小时）。

- python 一律绝对路径 `PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python`
  （裸 `python` = base env，无 vllm/faiss）。

- 长任务 `setsid nohup ... > logs/xxx.log 2>&1 &`，日志放 NFS 不放 /tmp。
  杀进程用括号模式（`[v]llm`）且**杀与启分成两条命令**（否则 pgrep 自杀）。

- **`--resume` 陷阱**：若 output 是**旧配置**的遗留，resume 会静默跳过全部、按旧数据
  出分。跑新配置前先确认 output 不存在或确属当前配置。

- **看到 `ALL_DONE` 不等于有数据。** 信任任何数字前先查 api_error 占比。
  脚本里保留两道闸：`waitp` 端口不通就放弃该 lane、`sane()` 拒绝 api_error 过半的结果。
  写监控守候时同理：**过滤条件必须覆盖"静默挂死"**，只匹配成功和显式报错会让循环空转
  （实录空转 13 小时）。

---

## 1. 数据现状（2026-07-20）

**主表已全部铺满**：14 个 agent 变体 × 6 个 wiki 数据集 × n=1500（bamboogle 全集 125）。
另有 popqa_full 标准集 22 格、BCP 830 共 11 格、消融/扫参 33 格。共 152 格，
`results/` 无孤儿文件、无重复版本。

**核心结论**（详见 `reports/baselines.md` §10）：
- 单跳由检索器决定（rag_e5 单跳均 EM .5310 最优，所有 agentic 方法都跑不赢单发稠密检索）
- 多跳由工作区机制决定（**scaleseek_e5 多跳均 EM .3335 第一**，压过 9B 的 grepseek .3193，
  且延迟只有它的 1/8）
- grepseek 全量 14267：EM .3347 / F1 .3793

**口径**：popqa_full = 从 popqa/test(14,267) 随机抽的固定 1500 样本（2026-07-13 定为标准集）；
EM/F1 = SQuAD 规范化；检索语料 wiki-18（21,015,324 段落）；
prompt 型 baseline 底座统一 Qwen3-4B；**全员 max_tokens=2048**（BCP 域用 4096）。

---

## 2. 已知问题（会影响结论，未解决）

1. **4B judge 判定不可信**：judge 自己抽取的答案与 gold 完全相同却判错——
   DR-DCI 六集 13.3%、dci-lite 11.4% 机械矛盾。**所有 LLM-judge 口径数字被系统性
   低估 11-13 点**（含 DR-DCI 的 43.25% 和 AVG 47.3）。

2. **自实现 `dci` 严重低估 DCI 方法**：官方 harness 六集均 EM .5467 vs 我们 .2136，
   **差 33 点**。主表该行只能当"未训练 4B 裸 grep 下界"，不能用来论断 DCI 类方法。

3. **dci-lite 的 `write ENOBUFS` 未解**：`/tmp` 干净时仍出现 29/92 次，有独立于内存的
   成因。降并发能压低但压不到 0，是剩余 31/500 题失败的主因。

4. **DR-DCI 六集每集只有 50 题**——那是 dci-bench 的**全部**，不是抽样。
   ±14 点不确定性是固有属性，**不能靠重跑收窄**。

---

## 3. 待拍板 / 下一步

- **换强 judge 重判**（优先级最高，影响面最大）：`Qwen3.5-27B` 已在本地缓存，
  TP 死锁已修，A5000 上 TP=4 可行；**但仓库里没有离线重判脚本**（官方 harness
  内联判卷），需先写。写好后可一次重判 DR-DCI 六集 + BCP 830 + dci-lite 500 三批。
- **scaleseek merge/replace 消融**：多跳题上 60-70% 的后续检索仍是 `replace`，把已找到的
  gold 冲掉（与 scaleseek 在 hotpotqa 的 Gold R@W .4050 < 单发 BM25 .4743 吻合）。
  ⚠️ 模型 **100% 显式写 `mode`**，改代码默认值无效；只能改 prompt 或服务端强制覆盖。
  建议先跑"强制 merge"测机制天花板（4 格），有信号再调 prompt。
- **报修节点**：abaque01（NVML 与内核模块版本不匹配，重载即可）、octal35（`cuInit=999`）。

---

## 4. 集群使用备忘

- QOS 限制**每人同时跑 2 个作业**（提交上限 4），排队的会自动接上，不用守着。
- 提交：`sbatch logs/sbatch_octal30_r6.sh`（最近一次的模板，含全部守卫，可照抄改）。
- 查看：`squeue -u mofengra`；进度 `grep '^\[o30f\]' logs/sbatch_o30f_<jobid>.log`。
- 附到运行中的作业：`srun --jobid=<id> --overlap bash -c '...'`。

---

## 5. 陈旧产物清理记录（2026-07-20）

删除了约 90MB 会造成误导的旧产物，**不要再从 git 历史里翻出来用**：
- `_quarantine_job7159_api_error/`（服务未启动产生的 100% api_error 假数据）
- **search_o1 畸形标记 bug（commit a260465 @07-16 14:18）修复前的全部结果** ——
  修复后的 `_v2` 已改名为正式名，现在每格只有一个版本
- `*smoke50*`（07-06 管线验证，已被全量运行取代）
- popqa-longtail 时代结果（`popqa_*_4b` 等，该数据集已被 popqa_full 取代）
- `deprecated/`（错误 BM25→rerank 管线产物）
- `hotpotqa_scaleseek.STALE_pre-trace`（n=7405 错误行数）

**这次清理的直接起因**：用 bug 修复前的 `popqa_full_search_o1` 当基线，得出了
"search_o1 用 4096 显著提升 p=.0040"的**错误结论**，差点据此改主表配置。
换正确基线后是净 −5、p=0.74，无效果。
→ **跨版本比较必须核对两侧的代码版本，不能只看文件名。**
