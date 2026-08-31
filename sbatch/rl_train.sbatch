#!/usr/bin/env bash
# 阶段 3/3：GRPO 强化学习训练（verl main_ppo + ScaleSeekAgentLoop）。
#
#   SCALESEEK_MODEL_PATH=<SFT checkpoint>/huggingface sbatch sbatch/rl_train.sbatch
#
# 不设 SCALESEEK_MODEL_PATH 会从 base 模型冷启动（无 SFT 热启动），
# run_rl.sh 会打印一条 note 提醒 —— 那通常不是你想要的。
#
# 覆盖项：
#   SCALESEEK_MODEL_PATH   SFT checkpoint    [未设则从 base 冷启动]
#   RL_GPUS                GPU 数            [4]
#   RL_BATCH               每步题数          [64]（× rollout.n=5 = 320 条轨迹）
#   SCALESEEK_TRAIN_FILES  RL 训练集         [$RL_DATA_DIR/train.jsonl]
#   SCALESEEK_VAL_FILES    RL 验证集         [$RL_DATA_DIR/dev.jsonl]
#   ULYSSES_SP             序列并行度，须整除 NPROC  [2]
#   WANDB_API_KEY/ENTITY   两个都设才开 W&B
#
# 产物：$RL_CKPT_DIR/$RL_RUN_NAME/  （续跑链共享，见下方 RL_RUN_NAME）
#
# 4 卡：这就是 grpo_trainer.yaml 的目标拓扑（fsdp_size: 4，三重 offload 全开，
# 注释写明按 4×A5000-24GB 调过），H100 80GB 是当初目标显存的 3.3 倍，余量充足。
# 先前的 8 卡整节点是我的偏离，代价是排队 22 小时（68 个 8 卡作业竞争、零空闲
# 整节点）；4 卡能从 mix 节点回填，调度容易得多。
# ULYSSES_SP 与 RL_ROLLOUT_TP 都必须整除 GPU 数（4 % 2 == 0）。
#
# 短墙钟续跑链而非一次长跑：12h 请求曾被排到 3.5 天后（Reason=Priority，不是
# 资源形状问题 —— 换 GPU 数救不了）。配合 save_freq + resume_mode=auto，作业链
# 的总吞吐远高于赌一次长跑。段长由 chain_rl.sh 传入（默认 6h，那里有实测的
# 开销/排队权衡表）；本文件的 3h 只是单独提交时的保守默认。
#   bash sbatch/chain_rl.sh 6      # 挂 6 段 6h = 36h 墙钟 / 约 158 步
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_rl
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=1000G
#SBATCH --time=3:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
export NPROC="${RL_GPUS:-4}"
require_full_gpus "$NPROC"
trap cleanup_tmpdir EXIT

export ULYSSES_SP="${ULYSSES_SP:-2}"
if [ $(( NPROC % ULYSSES_SP )) -ne 0 ]; then
  echo "FATAL: ULYSSES_SP=$ULYSSES_SP 必须整除 NPROC=$NPROC"; exit 1
fi

# vLLM rollout 的 tensor-parallel 度。grpo_trainer.yaml 不设它，于是继承 verl
# rollout.yaml 的默认值 2；GPU 数不是它的整数倍时，作业会在 Ray 起 worker 之后
# 才抛 "rollout world_size: N is not divisible by infer_world_size: 2"，白等几分钟。
# 这里显式传参并前置校验，单卡调试时设 RL_ROLLOUT_TP=1。
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-2}"
if [ $(( NPROC % RL_ROLLOUT_TP )) -ne 0 ]; then
  echo "FATAL: RL_ROLLOUT_TP=$RL_ROLLOUT_TP 必须整除 NPROC=$NPROC"; exit 1
fi
# 输出目录按 RUN NAME 而非 JOB ID —— 短墙钟续跑链的每个作业必须写同一个目录，
# verl 的 trainer.resume_mode=auto 才能捡起上一段的 global_step_*。用 JOB ID
# 会让每段各写各的，续跑等于每次从头开始。
export RL_RUN_NAME="${RL_RUN_NAME:-grpo_9b}"
export SCALESEEK_OUTPUT_DIR="${SCALESEEK_OUTPUT_DIR:-$RL_CKPT_DIR/$RL_RUN_NAME}"
export SCALESEEK_TRAIN_FILES="${SCALESEEK_TRAIN_FILES:-$RL_DATA_DIR/train.jsonl}"
export SCALESEEK_VAL_FILES="${SCALESEEK_VAL_FILES:-$RL_DATA_DIR/dev.jsonl}"

# --- 前置检查 ---
[ -d "$GREPSEEK_VERL_DIR/verl/trainer" ] || { echo "FATAL: verl 不存在: $GREPSEEK_VERL_DIR
  先跑: \$PY scripts/bootstrap_official_repos.py --root \$OFFICIAL_ROOT --repo grepseek"; exit 1; }
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: BM25 索引不存在: $BM25_INDEX_DIR/index
  先跑: \$PY scripts/build_bm25_index.py"; exit 1; }
[ -f "$SCALESEEK_TRAIN_FILES" ] || { echo "FATAL: RL 训练集不存在: $SCALESEEK_TRAIN_FILES
  先跑: \$PY scripts/prepare_rl_data.py --out_dir $RL_DATA_DIR"; exit 1; }
# ── SFT checkpoint 交接 ──────────────────────────────────────────────────────
# 作业链提交时 SFT 还没跑，checkpoint 路径那时并不存在，所以 submit_training.sh
# 无法把它作为环境变量传进来。这里在**运行时**解析：取 $SFT_CKPT_DIR 下最新的
# global_step_*/huggingface。
# （此前 submit_training.sh 的注释声称本脚本会做这件事，但代码从未实现 ——
#  后果是整条链会静默地从 base 模型冷启动，SFT 那一级白跑，且因为下面的格式
#  闸门以 SCALESEEK_MODEL_PATH 非空为前提，闸门也会被一起跳过。）
if [ -z "${SCALESEEK_MODEL_PATH:-}" ]; then
  _latest=$(find "$SFT_CKPT_DIR" -type d -name huggingface -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "$_latest" ] && [ -f "$_latest/config.json" ]; then
    export SCALESEEK_MODEL_PATH="$_latest"
    echo "[rl] 自动解析到 SFT checkpoint: $SCALESEEK_MODEL_PATH"
  else
    echo "warn: $SFT_CKPT_DIR 下找不到 SFT checkpoint —— 将从 base 模型冷启动，"
    echo "      且格式闸门会被跳过。确认这是你要的，否则先跑 sft_train。"
  fi
  unset _latest
fi

mkdir -p "$SCALESEEK_OUTPUT_DIR"
echo "[rl] model=${SCALESEEK_MODEL_PATH:-<base>}  out=$SCALESEEK_OUTPUT_DIR"
echo "[rl] gpus=$NPROC  ulysses_sp=$ULYSSES_SP  rollout_tp=$RL_ROLLOUT_TP"

# ── 格式遵守闸门 ─────────────────────────────────────────────────────────────
# train/reward.py 对"没输出 <answer>"和"答错"都给 0 分。若模型格式遵守率低，
# GRPO 一组 rollout 的 reward 全为 0 -> advantage 全 0 -> 梯度全 0，整个作业会在
# 零信号上空转到超时。2026-07-30 的冒烟正是如此：4 条 rollout 里 3 条推理完直接
# 用散文收尾没打标签，pg_loss 和 grad_norm 都是 0.0。
# 这里花几分钟先验一下，好过烧掉 8 卡 × 24h。设 RL_SKIP_FORMAT_GATE=1 跳过。
if [ "${RL_SKIP_FORMAT_GATE:-0}" != "1" ] && [ -n "${SCALESEEK_MODEL_PATH:-}" ]; then
  echo "[rl] 前置检查：格式遵守率（阈值 ${RL_MIN_FORMAT_PASS:-0.7}）"
  if ! "$PY" scripts/run_scaleseek_smoke_eval.py \
        --model "$SCALESEEK_MODEL_PATH" \
        --index-dir "$BM25_INDEX_DIR" \
        --questions "$SCALESEEK_VAL_FILES" \
        --limit "${RL_GATE_N:-16}" \
        --min-format-pass "${RL_MIN_FORMAT_PASS:-0.7}"; then
    echo "FATAL: 格式闸门未通过 —— 先修 SFT 的格式遵守，再提交 RL。"
    echo "       确认要无视时用 RL_SKIP_FORMAT_GATE=1 重投。"
    exit 1
  fi
fi

# dataloader worker 数跟随分配的核数。verl 默认 8，核数少于 8 时 torch 会警告
# "suggested max number of worker in current system is N"，且这些 worker 会 fork
# 出已装载 vLLM/FSDP 的父进程地址空间，在小分配上把内存压爆（冒烟里就吃到
# "DataLoader worker is killed by signal: Killed"）。
RL_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} < 8 ? ${SLURM_CPUS_PER_TASK:-8} : 8 ))

# ── 反挂死措施（2026-08-08，作业 19083387 的教训）────────────────────────────
# 那次 TaskRunner 在 processor 加载后静默挂死 22 小时，8×H100×24h 烧完零训练步。
# 三层防护：
#   1. HF_HUB_OFFLINE：一切所需模型/分词器已在本地缓存；任何意外的 Hub 网络调用
#      从"无限挂起"变成"立即报错或走缓存"。
#   2. 看门狗：每 15 分钟打时间戳 + GPU 利用率 + 主进程 py-spy 栈 —— 再挂死时
#      日志里直接有挂在哪一行的证据。
#   3. 断路器：无训练步 **且** GPU 连续闲置才判挂死（RL_STALL_MIN，默认 100 分钟）；
#      另有 RL_HARD_MAX（210 分钟，利用率 <50%）兜底。只看「有没有 step」会误杀
#      慢但健康的作业 —— 2026-08-31 就发生过。
export HF_HUB_OFFLINE=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

RL_LOG=/home/a32du/ScaleSeek/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
PYSPY="$SCALESEEK_VENV/bin/py-spy"

bash scripts/run_rl.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$RL_ROLLOUT_TP" \
    data.dataloader_num_workers="$RL_WORKERS" \
    data.train_batch_size="${RL_BATCH:-64}" \
    trainer.save_freq="${RL_SAVE_FREQ:-10}" \
    trainer.resume_mode=auto \
    trainer.experiment_name="${RL_RUN_NAME}" \
    "$@" &
MAIN_PID=$!

STALL_MIN="${RL_STALL_MIN:-100}"   # 配合 GPU 闲置判据
HARD_MAX="${RL_HARD_MAX:-210}"     # 兜底（仍要求 GPU 利用率 <50%）
(
  # set -e 必须在这个子 shell 里关掉。2026-08-28 作业 20657741/20657743 各烧满
  # 6h 零训练步：看门狗在 +30min 打完最后一行后就静默死了，断路器再没机会检查
  # "有没有训练步"。
  #
  # 机制是继承自外层的 `set -euo pipefail` 撞上看门狗里两处必然非零的构造
  # （两者都实测复现过）：
  #   * `"$PYSPY" dump ... | head -25` —— head 读够就关管道，py-spy 收 SIGPIPE
  #     退出 141，pipefail 把它变成整条管道的状态，set -e 随即退出子 shell。
  #   * `util=$(nvidia-smi ... | paste)` —— 简单赋值的退出码就是命令替换的退出码。
  # （注：`for p in $(pgrep ...)` 反而是安全的 —— 命令替换的状态不构成 for 的
  #  状态，无匹配时循环迭代零次并成功。）
  # 看门狗全是"允许失败"的探针，正确做法是整段关掉 errexit。
  set +e
  t=0; idle=0
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 900; t=$((t+15))
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
    echo "[watchdog] +${t}min gpu_util=[${util}]"
    if [ -x "$PYSPY" ]; then
      for p in $(pgrep -f "ray::TaskRunner|verl.trainer.main_ppo" 2>/dev/null | head -2); do
        echo "[watchdog] py-spy dump pid=$p:"
        "$PYSPY" dump --pid "$p" 2>&1 | head -25
      done
    fi
    # 「无训练步」单独不足以判定挂死 —— 它也可能只是慢。2026-08-31 作业
    # 20839641 就被误杀：四张卡全程 40-100% 利用率明明在算 rollout，只是
    # 20480 上下文让每序列 KV 大了 43%、并发下降，首步落到 100 分钟之后。
    # 所以要求「无训练步」且「GPU 基本闲置」同时成立，再加一个不看利用率的
    # 硬兜底，防止 GPU 空转式的死锁。
    mean_util=$(echo "$util" | tr ',' '\n' | awk 'NF{s+=$1;n++} END{print (n?int(s/n):0)}')
    if [ "${mean_util:-0}" -lt 10 ]; then idle=$((idle+1)); else idle=0; fi
    if ! grep -aq "step:" "$RL_LOG" 2>/dev/null; then
      if [ "$t" -ge "$STALL_MIN" ] && [ "$idle" -ge 3 ]; then
        echo "[watchdog] FATAL: ${STALL_MIN} 分钟无训练步且 GPU 连续闲置 —— 判定挂死"
        kill -TERM "$MAIN_PID" 2>/dev/null; sleep 60; kill -KILL "$MAIN_PID" 2>/dev/null
        break
      fi
      # 硬兜底同样要看利用率，否则又变成误杀慢作业（首步实测已 >105 分钟）。
      # 这里放宽到 <50%：真死锁不会持续吃满 GPU，而健康的 rollout 会。
      if [ "$t" -ge "$HARD_MAX" ] && [ "${mean_util:-0}" -lt 50 ]; then
        echo "[watchdog] FATAL: ${HARD_MAX} 分钟仍无训练步且 GPU 利用率仅 ${mean_util}% —— 硬兜底终止"
        kill -TERM "$MAIN_PID" 2>/dev/null; sleep 60; kill -KILL "$MAIN_PID" 2>/dev/null
        break
      fi
    fi
  done
) &
WATCHDOG_PID=$!

RC=0; wait "$MAIN_PID" || RC=$?   # set -e 下裸 wait 失败会直接退脚本，绕过收尾
kill "$WATCHDOG_PID" 2>/dev/null || true
[ "$RC" -eq 0 ] || { echo "FATAL: run_rl.sh 退出码 $RC"; exit "$RC"; }

echo "[rl] 完成。checkpoints -> $SCALESEEK_OUTPUT_DIR"
echo "评测结果请写入 \$RESULTS ($RESULTS)，checkpoint 留在 \$CKPT (/scratch)。"
