#!/usr/bin/env bash
#SBATCH -J p_debug_dci_replay
#SBATCH --account=def-amirhk_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /home/a32du/ScaleSeek/logs/%x-%j.out
# One-off debug: replay the EXACT captured DR-DCI request (tools + benchmark
# prompt) against a vLLM started with the same serving flags as p2_dr_dci, and
# measure how often the model emits content / tool_calls vs the observed
# "brief <think> then EOS" (1495/1500 blank, zero tool calls, job 18756906).
source /home/a32du/ScaleSeek/sbatch/common.sh
trap 'stop_vllm; cleanup_tmpdir' EXIT
PORT=$((10000 + SLURM_JOB_ID % 20000))

start_vllm Qwen/Qwen3.5-9B "$GEN_REV" "$PORT" 1 Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
waitp "$PORT" 3600

"$PY" - "$PORT" <<'PYEOF'
import json, sys, collections
from openai import OpenAI
port = sys.argv[1]
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
cap = json.load(open("results/phase3/dr_dci_raw/triviaqa_test_3/latest_model_context.json"))
p = cap["latest"]["payload"]
msgs, tools = p["messages"], p["tools"]

variants = {
    "as-is (pi payload)": {},
    "temp0.6/top_p0.95": {"temperature": 0.6, "top_p": 0.95},
    "temp0.7/top_p0.8": {"temperature": 0.7, "top_p": 0.8},
    "nothink (chat_template_kwargs)": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}
N = 8
for name, kw in variants.items():
    outcome = collections.Counter()
    sample = ""
    for i in range(N):
        try:
            r = client.chat.completions.create(
                model="Qwen/Qwen3.5-9B", messages=msgs, tools=tools,
                max_completion_tokens=16384, stream=False, **kw)
            m = r.choices[0].message
            fin = r.choices[0].finish_reason
            has_tc = bool(getattr(m, "tool_calls", None))
            content = (m.content or "").strip()
            rc = (getattr(m, "reasoning_content", "") or "")
            if has_tc:
                outcome["TOOL_CALL"] += 1
                if not sample: sample = f"tool={m.tool_calls[0].function.name} args={m.tool_calls[0].function.arguments[:120]}"
            elif content:
                outcome["CONTENT"] += 1
                if not sample: sample = f"content={content[:120]}"
            else:
                outcome[f"EMPTY(fin={fin},rtok~{len(rc)//4})"] += 1
        except Exception as e:
            outcome[f"ERROR:{type(e).__name__}"] += 1
            if not sample: sample = str(e)[:150]
    print(f"{name:34s} {dict(outcome)}  | {sample[:150]}", flush=True)
PYEOF
echo "DEBUG_REPLAY_DONE"
