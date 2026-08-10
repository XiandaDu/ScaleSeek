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
RUN="${RL_RUN_NAME:-grpo_9b}"
DEP=""
for i in $(seq 1 "$N"); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $DEP \
        --export=ALL,RL_RUN_NAME="$RUN",RL_SKIP_FORMAT_GATE=1 \
        sbatch/rl_train.sbatch)
  echo "[chain] 第 $i/$N 段 -> $JID${DEP:+  ($DEP)}"
  DEP="--dependency=afterany:$JID"
done
echo
echo "checkpoint 目录: \$CKPT/rl/$RUN"
echo "断链请 scancel 后续作业；加段直接再跑一次本脚本（会接在队尾之外，注意）"
