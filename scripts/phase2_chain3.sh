#!/usr/bin/env bash
# Phase-2 链，第三版（2026-09-05 15:30 UTC）。第二版在 A2（teacher 显存未释放）和 C
# （set -e + find 空目录）上断掉。本版：把剩下的 3000 道可达题拆成两半，两对 GPU 各起一个
# teacher 并行生成（A2: GPU 0,1 / A3: GPU 2,3），合并 A+A2+A3 → C SFT → D 筛选 → E pilot → F。
set -uo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"; mkdir -p logs
STATUS=logs/phase2_status.txt
mark() { echo "$(date -u '+%F %T') $*" | tee -a "$STATUS"; }
V2=$DATA/rl_data_v2; SFT=$SFT_TRAJ_DIR
PILOT_STEPS="${PILOT_STEPS:-40}"
mark "chain3 start"
# 共享机器：别人的作业占着卡时不要硬起（2026-09-06 00:00 的恢复就是这么断的：外部进程占 76G，
# teacher 起不来、SFT 直接 OOM）。等到四张卡都 <2G 再走，最多等 GPU_WAIT_H 小时。
wait_gpus() {
  local need=$1 t=0
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l)
    [ "$free" -ge "$need" ] && return 0
    [ "$t" -eq 0 ] && mark "waiting for $need free GPUs (now $free free)"
    sleep 120; t=$((t+2)); [ "$t" -ge $(( ${GPU_WAIT_H:-24} * 60 )) ] && { mark "FATAL: GPUs busy for ${GPU_WAIT_H:-24}h"; exit 1; }
  done
}
wait_gpus 4

if [ ! -s "$V2/sft_questions_b1.jsonl" ]; then
  n=$(wc -l < "$V2/sft_questions_b.jsonl"); h=$(( (n + 1) / 2 ))
  head -n "$h" "$V2/sft_questions_b.jsonl" > "$V2/sft_questions_b1.jsonl"
  tail -n +"$((h + 1))" "$V2/sft_questions_b.jsonl" > "$V2/sft_questions_b2.jsonl"
fi
mark "[A2/A3] $(wc -l < "$V2/sft_questions_b1.jsonl") + $(wc -l < "$V2/sft_questions_b2.jsonl") reachable questions"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 5
SFT_GPUS=0,1 SFT_PORT=8001 SFT_QUESTIONS="$V2/sft_questions_b1.jsonl" SFT_OUT_DIR="$SFT/v2b" \
  bash scripts/sft_data_local.sh > logs/phase2_A2_sft_data.log 2>&1 & P2=$!
SFT_GPUS=2,3 SFT_PORT=8003 SFT_QUESTIONS="$V2/sft_questions_b2.jsonl" SFT_OUT_DIR="$SFT/v2c" \
  bash scripts/sft_data_local.sh > logs/phase2_A3_sft_data.log 2>&1 & P3=$!
wait $P2; mark "[A2] done rc=$?"; wait $P3; mark "[A3] done rc=$?"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 20

MERGED=$SFT/v2all; mkdir -p "$MERGED"
cat "$SFT/v2/trajectories.jsonl" "$SFT/v2b/trajectories.jsonl" "$SFT/v2c/trajectories.jsonl" 2>/dev/null > "$MERGED/trajectories.jsonl"
"$PY" -m train.sft_dataset --in "$MERGED/trajectories.jsonl" --out "$MERGED/sft_train.parquet" > logs/phase2_merge.log 2>&1 \
  || { mark "FATAL: parquet merge failed (logs/phase2_merge.log)"; exit 1; }
mark "[merge] ok trajectories: $(grep -c '"status": "ok"' "$MERGED/trajectories.jsonl") -> $MERGED/sft_train.parquet"

wait_gpus 4
mark "[C] sft_train_local (4 GPU) start"
SCALESEEK_SFT_TRAIN=$MERGED/sft_train.parquet SCALESEEK_SFT_OUTPUT=$SFT_CKPT_DIR/v2 \
  bash scripts/sft_train_local.sh > logs/phase2_C_sft_train.log 2>&1; mark "[C] rc=$?"
HF_CKPT=$(find "$SFT_CKPT_DIR/v2" -type d -name huggingface 2>/dev/null | sort | tail -1 || true)
[ -n "$HF_CKPT" ] && [ -f "$HF_CKPT/config.json" ] || { mark "FATAL: [C] no HF checkpoint (logs/phase2_C_sft_train.log, logs/sft_train_v2.log)"; exit 1; }
mark "[C] done: $HF_CKPT"

if [ -s "$V2/screen_stats.jsonl" ]; then
  "$PY" scripts/prepare_rl_data.py --out_dir "$V2" --with-evidence \
      --mix 2wikimultihopqa=14000 musique=15000 hotpotqa=15000 nq=6000 --dev-ids "$RL_DATA_DIR/dev.jsonl" \
      --screen "$V2/screen_stats.jsonl" --band 0.0,1.0 --keep-hard-multihop 1500 > logs/phase2_D_data.log 2>&1 \
    || { mark "FATAL: [D] failed"; exit 1; }
  mark "[D] done: $(wc -l < "$V2/train.jsonl") train rows"
else mark "[D] no screening; full pool"; fi

wait_gpus 4
mark "[E] pilot_v2 start (steps=$PILOT_STEPS)"
SCALESEEK_RL_CONFIG=grpo_trainer_v2 RL_RUN_NAME=pilot_v2 RL_TOTAL_STEPS="$PILOT_STEPS" RL_BATCH=48 \
RL_SAVE_FREQ=10 RL_TEST_FREQ=20 RL_VAL_BEFORE_TRAIN=true RL_MAX_ATTEMPTS=4 \
SCALESEEK_MODEL_PATH="$HF_CKPT" SCALESEEK_TRAIN_FILES="$V2/train.jsonl" SCALESEEK_VAL_FILES="$V2/dev.jsonl" \
bash scripts/run_rl_local.sh > logs/phase2_E_pilot.log 2>&1; RC_E=$?
mark "[E] done rc=$RC_E ckpt=$(cat "$RL_CKPT_DIR/pilot_v2/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
{
  echo "== pilot_v2 progress"; bash scripts/rl_progress.sh pilot_v2 60
  echo; echo "== behaviour audit (last 3 train rollout dumps)"
  ls -t "$RL_CKPT_DIR"/pilot_v2/train_rollouts/*.jsonl 2>/dev/null | head -3 | xargs -r "$PY" scripts/analyze_rollouts.py --show 0
  for f in "$RL_CKPT_DIR"/pilot_v2/val_rollouts/*.jsonl; do echo; echo "== val $f"; "$PY" scripts/analyze_rollouts.py "$f" --show 0 | head -12; done
} > logs/phase2_F_summary.txt 2>&1
mark "chain3 finished; summary in logs/phase2_F_summary.txt"
