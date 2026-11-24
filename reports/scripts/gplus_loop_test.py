"""The official loop with the GrepSeek+ interception installed, driven by a scripted fake LLM."""
import json, sys
from types import SimpleNamespace as NS
sys.path.insert(0, "/scratch/a32du/scaleseek_v2/repos/grepseek")
sys.path.insert(1, "/home/a32du/ScaleSeek/scaleseek")
import inference.run as official
import grepseek_plus as gp
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/scratch/a32du/scaleseek_v2/hf/hub/models--alireza7--GrepSeek-Qwen3.5-9B-GRPO/snapshots/a79563970cfdd2ced3cc5fde481737d0ebea6fa4")
call = lambda c: '<tool_call>\n' + json.dumps({"name": "shell", "arguments": {"command": c}}) + '\n</tool_call>'
BIG = 'rg -F "Charles Darwin" corpus.jsonl | head -n 40'


class Fake:
    def __init__(self, script):
        self.script, self.i = script, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **request):
        n = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) + 4 for m in request["messages"])
        text = self.script[self.i]; self.i += 1
        return NS(choices=[NS(message=NS(content=text, reasoning_content="thinking"), finish_reason="stop")],
                  usage=NS(prompt_tokens=n, completion_tokens=60))


def trial(variants, script):
    make = gp.install(official, lambda: Fake(script), {"variants": variants, "context": 16384})
    rec = official._run_example_safe({"id": "x", "question": "q", "golden_answers": ["a"]}, client=make(), model="m", tokenizer=tok,
                                     agent_kwargs=dict(corpus_dir=sys.argv[1], max_assistant_turns=6, max_tokens_per_turn=0,
                                                       tool_max_tokens=2048, tool_timeout=60.0, temperature=0.6, top_p=1.0, engine=None))
    print(f"== {variants}: finish {rec['finish_reason']}  prediction {rec['prediction']}  turns {rec['n_assistant_turns']}  tool calls {rec['n_tool_calls']}  error {str(rec.get('error'))[:300]}")
    used = 0
    for t in rec["turns"]:
        if t["role"] == "tool":
            p = json.loads(t["content"])
            print(f"   tool: {len(tok(t['content'], add_special_tokens=False)['input_ids']):5d} tokens  truncated {p['truncated']}  keys {list(p)[6:]}  notice: {p.get('notice')}")
    print("   total context tokens:", sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) for m in rec["messages"]) if rec["messages"] else None)


trial(["budget"], [call(BIG)] * 5 + ["<answer>\nx\n</answer>"])
trial(["budget", "dedup"], [call(BIG)] * 5 + ["<answer>\nx\n</answer>"])
trial(["budget", "dedup", "fold", "relax", "title"], [call('rg -F "Jonas Salk" corpus.jsonl | rg -i -F "religion" | head -n 3'), "no action here",
                                                      call('rg -F "Valentín" corpus.jsonl | head -n 2'), call("rm -rf /"), call(BIG), call(BIG)])
