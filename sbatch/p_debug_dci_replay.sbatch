#!/usr/bin/env bash
#SBATCH -J p_debug_dci_replay
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# Debug round 2: Qwen3.5's chat template mandates the XML function-call format
# (<tool_call>\n<function=name>...), which `--tool-call-parser hermes` can never
# parse (round 1: 32/32 replays emitted tool calls as plain content). Try the
# qwen3_xml and qwen3_coder parsers in BOTH stream modes (Pi streams).
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'stop_vllm; cleanup_tmpdir' EXIT

for PARSER in qwen3_xml qwen3_coder; do
  PORT=$((11000 + SLURM_JOB_ID % 20000 + ${#PARSER}))
  echo "===== parser: $PARSER ====="
  start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 1 Qwen/Qwen3.5-9B \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "$PARSER"
  waitp "$PORT" 3600 || { echo "FATAL: server with $PARSER did not start"; stop_vllm; continue; }

  "$PY" - "$PORT" <<'PYEOF'
import json, sys, collections
from openai import OpenAI
port = sys.argv[1]
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
cap = json.load(open("results/phase3/dr_dci_raw/triviaqa_test_3/latest_model_context.json"))
p = cap["latest"]["payload"]
msgs, tools = p["messages"], p["tools"]
N = 6
for mode in ("nonstream", "stream"):
    outcome = collections.Counter(); sample = ""
    for i in range(N):
        try:
            if mode == "nonstream":
                r = client.chat.completions.create(model="Qwen/Qwen3.5-9B", messages=msgs,
                        tools=tools, max_completion_tokens=16384, stream=False)
                m = r.choices[0].message
                has_tc = bool(getattr(m, "tool_calls", None))
                content = (m.content or "").strip()
                name = m.tool_calls[0].function.name if has_tc else ""
                args = (m.tool_calls[0].function.arguments or "")[:100] if has_tc else ""
            else:
                s = client.chat.completions.create(model="Qwen/Qwen3.5-9B", messages=msgs,
                        tools=tools, max_completion_tokens=16384, stream=True)
                content = ""; name = ""; args = ""
                for ch in s:
                    d = ch.choices[0].delta if ch.choices else None
                    if not d: continue
                    if d.content: content += d.content
                    for tc in (d.tool_calls or []):
                        if tc.function and tc.function.name: name = tc.function.name
                        if tc.function and tc.function.arguments: args += tc.function.arguments
                has_tc = bool(name)
                content = content.strip(); args = args[:100]
            if has_tc:
                outcome["TOOL_CALL"] += 1
                if not sample: sample = f"tool={name} args={args}"
            elif content:
                outcome["CONTENT"] += 1
                if not sample: sample = f"content={content[:100]}"
            else:
                outcome["EMPTY"] += 1
        except Exception as e:
            outcome[f"ERROR:{type(e).__name__}"] += 1
            if not sample: sample = str(e)[:120]
    print(f"  {mode:10s} {dict(outcome)}  | {sample[:130]}", flush=True)
PYEOF
  stop_vllm
done
echo "DEBUG_REPLAY2_DONE"
