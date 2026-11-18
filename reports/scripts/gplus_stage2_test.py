"""verify and inject, with a scripted fake LLM (no GPU)."""
import json, sys, tempfile
from types import SimpleNamespace as NS
sys.path.insert(0, "/scratch/a32du/scaleseek_v2/repos/grepseek")
sys.path.insert(1, "/home/a32du/ScaleSeek/scaleseek")
import inference.run as official
import inference.agent as agent
import grepseek_plus as gp
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/scratch/a32du/scaleseek_v2/hf/hub/models--alireza7--GrepSeek-Qwen3.5-9B-GRPO/snapshots/a79563970cfdd2ced3cc5fde481737d0ebea6fa4")
call = lambda c: '<tool_call>\n' + json.dumps({"name": "shell", "arguments": {"command": c}}) + '\n</tool_call>'
ans = lambda a: f"<answer>\n{a}\n</answer>"
CMD = 'rg -F "Evan Morris" corpus.jsonl | head -n 2'
ORIG = (official.run_agent_on_example, agent.run_tool, agent._chat_completion)


class Fake:
    def __init__(self, script):
        self.script, self.i, self.seen = script, 0, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **request):
        self.seen.append([m["role"] for m in request["messages"]])
        text = self.script[self.i]; self.i += 1
        n = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) + 4 for m in request["messages"])
        return NS(choices=[NS(message=NS(content=text, reasoning_content="thinking"), finish_reason="stop")],
                  usage=NS(prompt_tokens=n, completion_tokens=60))


def trial(tag, variants, script, turns=6, inject=None):
    official.run_agent_on_example, agent.run_tool, agent._chat_completion = ORIG
    settings = {"variants": variants, "context": 16384}
    if inject:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump({"x": inject}, f); f.close()
        settings["inject"] = f.name
    fake = Fake(script)
    make = gp.install(official, lambda: fake, settings)
    rec = official._run_example_safe({"id": "x", "question": "q", "golden_answers": ["a"]}, client=make(), model="m", tokenizer=tok,
                                     agent_kwargs=dict(corpus_dir=sys.argv[1], max_assistant_turns=turns, max_tokens_per_turn=0,
                                                       tool_max_tokens=2048, tool_timeout=60.0, temperature=0.6, top_p=1.0, engine=None))
    print(f"== {tag}: finish {rec['finish_reason']}  prediction {rec['prediction']}  loop turns {rec['n_assistant_turns']}  generations {fake.i}  error {str(rec.get('error'))[:200]}")
    for m in rec["messages"][2:]:
        c = m["content"]
        note = json.loads(c).get("notice") if m["role"] == "tool" else None
        print(f"   {m['role']:9s} {c[:70]!r}" + (f"  NOTICE: {note[:110]}" if note else ""))


V = ["budget", "dedup", "fold", "relax", "title", "verify"]
trial("ungrounded answer -> challenged, searches, answers", V, [call(CMD), "<think>x</think>" + ans("Zzz Qqq"), call(CMD), ans("Genentech")])
trial("grounded answer -> accepted", V, [call(CMD), ans("Genentech")])
trial("ungrounded but only one turn left -> accepted", V, [call(CMD)] * 4 + [ans("Zzz Qqq")], turns=5)
trial("challenged, then keeps searching to the end", V, [call(CMD), ans("Zzz Qqq")] + [call(CMD)] * 4)
inject = [{"role": "assistant", "content": call(CMD)},
          {"role": "tool", "content": json.dumps({"command": CMD, "stdout": "line one\n", "stderr": "", "exit_code": 0, "timed_out": False, "truncated": False, "notice": "These are all the search results."})}]
trial("inject, one turn", ["budget", "dedup", "fold", "relax", "title"], [ans("Genentech")], turns=1, inject=inject)
