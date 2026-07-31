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
#   RL_GPUS                GPU 数            [8]
#   SCALESEEK_TRAIN_FILES  RL 训练集         [$RL_DATA_DIR/train.jsonl]
#   SCALESEEK_VAL_FILES    RL 验证集         [$RL_DATA_DIR/dev.jsonl]
#   ULYSSES_SP             序列并行度，须整除 NPROC  [2]
#   WANDB_API_KEY/ENTITY   两个都设才开 W&B
#
# 产物：$RL_CKPT_DIR/<jobid>/
#
# RL 比 SFT 重得多：GRPO 要同时驻留 actor(FSDP) + reference + vLLM rollout 引擎，
# 9B student 三者相加远超 4 张卡，因此默认整节点 8 卡 + --mem=0（整机内存）。
# ULYSSES_SP 必须整除 GPU 数。
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_rl
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
export NPROC="${RL_GPUS:-8}"
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
export SCALESEEK_OUTPUT_DIR="${SCALESEEK_OUTPUT_DIR:-$RL_CKPT_DIR/$SLURM_JOB_ID}"
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

bash scripts/run_rl.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$RL_ROLLOUT_TP" \
    data.num_workers="$RL_WORKERS" \
    trainer.experiment_name="nibi_${SLURM_JOB_ID}" \
    "$@"

echo "[rl] 完成。checkpoints -> $SCALESEEK_OUTPUT_DIR"
echo "评测结果请写入 \$RESULTS ($RESULTS)，checkpoint 留在 \$CKPT (/scratch)。"
