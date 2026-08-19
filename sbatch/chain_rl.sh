#!/usr/bin/env bash
# 用 afterany 把 N 段短墙钟 RL 作业串成续跑链。
#
#   bash sbatch/chain_rl.sh [段数=4]
#
# 为什么 afterany 而不是 afterok：每段都会被 3h 墙钟杀掉（State=TIMEOUT），
# 那是预期行为而非失败；afterok 会因此断链。代价是真崩了也会继续排 ——
# 第一段出结果后确认一下，坏了就 scancel 剩下的。
# 所有段共享 RL_RUN_NAME，写同一个 checkpoint 目录，靠 resume_mode=auto 接力。
set -euo pipefail
cd /home/a32du/ScaleSeek
N="${1:-4}"
# 段长：实测每段固定 ~71min 初始化开销（vLLM 加载 + Ray + FSDP 恢复），每步 ~11min。
#   3h -> 有效 109min / 10 步 / 开销 39%
#   6h -> 有效 289min / 26 步 / 开销 20%   <- 默认
#   12h-> 有效 649min / 59 步 / 开销 10%，但排队惩罚重（>6h 档积压 3214 个作业，
#         3-6h 档只有 278 个；上次 12h 请求被排到 3.5 天后）
WALL="${RL_WALL:-6:00:00}"
RUN="${RL_RUN_NAME:-grpo_9b}"
DEP=""
for i in $(seq 1 "$N"); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $DEP --time="$WALL" \
        --export=ALL,RL_RUN_NAME="$RUN",RL_SKIP_FORMAT_GATE=1 \
        sbatch/rl_train.sbatch)
  echo "[chain] 第 $i/$N 段 -> $JID  wall=$WALL${DEP:+  ($DEP)}"
  DEP="--dependency=afterany:$JID"
done
echo
echo "checkpoint 目录: \$CKPT/rl/$RUN"
echo "断链请 scancel 后续作业；加段直接再跑一次本脚本（会接在队尾之外，注意）"
