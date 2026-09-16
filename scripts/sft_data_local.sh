#!/usr/bin/env bash
# 本机版 sbatch/sft_data.sbatch：起 teacher vLLM（TP=卡数）→ generate_sft_data → verl parquet。
#
#   source setup_env.sh; bash scripts/sft_data_local.sh
#
# 覆盖项：
#   SFT_GPUS        [0,1]      SFT_PORT [8001]
#   SFT_TEACHER     [Qwen/Qwen3.6-35B-A3B]  SFT_TEACHER_REV [995ad96...]（下载时的 sha）
#   SFT_QUESTIONS   [$DATA/rl_data_v2/sft_questions.jsonl]（多跳 90%，prepare_rl_data --sft-questions）
#   SFT_OUT_DIR     [$SFT_TRAJ_DIR/v2]
#   SFT_PARAM_POLICY [search]  —— v1 用 heuristic（按词数套常量，学生学成常量）；search 用真实索引
#                    选出最能把目标段落排到前面的 (k1,b,top_k)，学生学到过的是 top_k 的变化
#   SFT_TOOL_TOKENS [2048]     —— v1 是 512，与 RL/eval 的 2048 不一致
#   SFT_LIMIT       [无]       冒烟用
# 产物：$SFT_OUT_DIR/{trajectories.jsonl,trajectories.summary.json,sft_train.parquet}
set -euo pipefail
: "${REPO:?先 source setup_env.sh}"; cd "$REPO"
GPUS="${SFT_GPUS:-0,1}"; TP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
PORT="${SFT_PORT:-8001}"
TEACHER="${SFT_TEACHER:-Qwen/Qwen3.6-35B-A3B}"
TEACHER_REV="${SFT_TEACHER_REV:-995ad96eacd98c81ed38be0c5b274b04031597b0}"
QUESTIONS="${SFT_QUESTIONS:-$DATA/rl_data_v2/sft_questions.jsonl}"
OUT_DIR="${SFT_OUT_DIR:-$SFT_TRAJ_DIR/v2}"
TRAJ=$OUT_DIR/trajectories.jsonl; PARQUET=$OUT_DIR/sft_train.parquet
POLICY="${SFT_PARAM_POLICY:-search}"; TOOL_TOKENS="${SFT_TOOL_TOKENS:-2048}"
mkdir -p "$OUT_DIR" logs
[ -f "$QUESTIONS" ] || { echo "FATAL: 问题文件不存在: $QUESTIONS"; exit 1; }
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: BM25 索引不存在"; exit 1; }
if [ -s "$PARQUET" ]; then echo "[sft_data] $PARQUET 已存在 — 跳过"; exit 0; fi

export HF_HUB_OFFLINE=1
# 上一个 vLLM 的 worker 进程释放显存要几十秒；不等就会死于
# "Free memory on device cuda:0 (9.1/79.25 GiB) ... less than desired" (2026-09-05 06:01)。
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") && $2>2000' | wc -l)
  [ "$used" -eq 0 ] && break
  [ "$i" -eq 1 ] && echo "[sft_data] 等待 GPU $GPUS 释放显存..."
  [ "$i" -eq 10 ] && pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null
  sleep 10
done
VLOG=logs/vllm_teacher_${PORT}.log
echo "[sft_data] teacher=$TEACHER gpus=$GPUS tp=$TP port=$PORT  questions=$QUESTIONS ($(wc -l < "$QUESTIONS") 条)"
CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$TEACHER" --revision "$TEACHER_REV" --dtype bfloat16 \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" \
    --served-model-name teacher --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --trust-remote-code \
    > "$VLOG" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT
t=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 "$VPID" 2>/dev/null || { echo "FATAL: teacher vLLM 启动失败"; tail -30 "$VLOG"; exit 1; }
  sleep 10; t=$((t+10)); [ "$t" -ge 3600 ] && { echo "FATAL: teacher vLLM 3600s 未就绪"; exit 1; }
done
echo "[sft_data] teacher up after ${t}s  $(date -u)"

"$PY" scripts/generate_sft_data.py \
    --questions "$QUESTIONS" --out "$TRAJ" \
    --teacher "openai:http://127.0.0.1:$PORT/v1" --teacher-model teacher \
    --index-dir "$BM25_INDEX_DIR" --param-policy "$POLICY" \
    --tool-response-tokens "$TOOL_TOKENS" --student-tokenizer Qwen/Qwen3.5-9B \
    --strict --resume ${SFT_LIMIT:+--limit "$SFT_LIMIT"} >> "logs/sft_data_$(basename "$OUT_DIR").log" 2>&1 \
  || { echo "FATAL: generate_sft_data 失败，见 logs/sft_data_$(basename "$OUT_DIR").log"; tail -20 "logs/sft_data_$(basename "$OUT_DIR").log"; exit 1; }
kill "$VPID" 2>/dev/null || true; wait "$VPID" 2>/dev/null || true; trap - EXIT

"$PY" -m train.sft_dataset --in "$TRAJ" --out "$PARQUET"
"$PY" - "$TRAJ" <<'PYEOF'
import collections, json, sys
n_ok = n_rec = n_multi = 0; calls = collections.Counter(); vals = {"top_k": collections.Counter(), "k1": collections.Counter(), "b": collections.Counter()}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("status") != "ok": continue
    n_ok += 1
    if (r.get("meta") or {}).get("n_failure_turns", 0) > 0: n_rec += 1
    nb = 0
    for m in r["messages"]:
        if m["role"] != "assistant" or "<tool_call>" not in m["content"]: continue
        seg = m["content"].split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0]
        try: tc = json.loads(seg)
        except Exception: continue
        calls[tc.get("name")] += 1
        if tc.get("name") == "bm25_retrieve":
            nb += 1
            for k in vals: vals[k][str((tc.get("arguments") or {}).get(k, "<unset>"))] += 1
    if nb >= 2: n_multi += 1
s = json.load(open(sys.argv[1].replace(".jsonl", ".summary.json"))) if __import__("os").path.exists(sys.argv[1].replace(".jsonl", ".summary.json")) else {}
print(f"[sft_data] ok={n_ok} (summary: {s.get('kept_for_sft')}/{s.get('questions')}, counts={s.get('counts')})")
print(f"[sft_data] multi-bm25 trajectories: {n_multi}/{n_ok}; with failure->recovery turns: {n_rec}/{n_ok}")
print(f"[sft_data] tools: {dict(calls)}"); [print(f"[sft_data] {k}: {dict(v.most_common(6))}") for k, v in vals.items()]
PYEOF
ls -lh "$PARQUET"; echo "[sft_data] 完成 $(date -u)"
