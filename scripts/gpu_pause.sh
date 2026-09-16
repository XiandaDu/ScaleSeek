#!/usr/bin/env bash
# 把 GPU 让出来：先停编排链（免得它接着起下一阶段），再用 SIGINT 让生成/训练进程正常收尾
# （generate_sft_data 每条轨迹已 flush，SIGINT 走 with 块正常关文件），最后清 vLLM/Ray。
#   bash scripts/gpu_pause.sh [到时刻再停，如 "2026-09-05 18:00"]
set -u
if [ -n "${1:-}" ]; then
  t=$(date -d "$1" +%s); w=$(( t - $(date +%s) )); echo "[pause] 将在 $(date -u -d @"$t" '+%F %T %Z') 停止（${w}s 后）"; [ "$w" -gt 0 ] && sleep "$w"
fi
echo "[pause] $(date -u '+%F %T %Z') 开始释放 GPU"
pkill -f "^bash scripts/phase2_(chain[0-9]*|elastic|pilot|full|switch_v21).sh" && echo "  链已停"
pkill -INT -f "scripts/generate_sft_data.py" 2>/dev/null && echo "  生成进程 SIGINT"
pkill -INT -f "scripts/screen_difficulty.py" 2>/dev/null
pkill -TERM -f "verl.trainer.main_ppo|verl.trainer.sft_trainer" 2>/dev/null && echo "  训练进程 SIGTERM"
pkill -f "^bash scripts/run_rl_local.sh|^bash scripts/sft_data_local.sh|^bash scripts/sft_train_local.sh" 2>/dev/null
sleep 20
pkill -TERM -f "scripts/generate_sft_data.py" 2>/dev/null
pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -f "^ray::" 2>/dev/null; sleep 20
pkill -KILL -f "vllm.entrypoints.openai.api_server|scripts/generate_sft_data.py|verl.trainer" 2>/dev/null
sleep 5
echo "[pause] GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' ')"
for d in v2 v2b v2c; do f=/data/xianda/scaleseek_data/data/sft/$d/trajectories.jsonl; [ -f "$f" ] && echo "  $d: $(wc -l < "$f") 行已保存"; done
echo "[pause] 恢复：source setup_env.sh; nohup bash scripts/phase2_chain3.sh > logs/phase2_chain3.log 2>&1 &"
