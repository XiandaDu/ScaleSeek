"""commit, without a GPU: (1) GrepSeek+ sends the answer regex with the generation that follows an "answer now" notice;
(2) our own agent's last turn is forced into an answer call."""
import json, re, sys
from types import SimpleNamespace as NS
sys.path.insert(0, "/scratch/a32du/scaleseek_v2/repos/grepseek")
sys.path.insert(1, "/home/a32du/ScaleSeek/scaleseek")
sys.path.insert(2, "/home/a32du/ScaleSeek")
import inference.run as official
import grepseek_plus as gp
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/scratch/a32du/scaleseek_v2/hf/hub/models--alireza7--GrepSeek-Qwen3.5-9B-GRPO/snapshots/a79563970cfdd2ced3cc5fde481737d0ebea6fa4")
call = lambda c: '<tool_call>\n' + json.dumps({"name": "shell", "arguments": {"command": c}}) + '\n</tool_call>'
CMD = 'rg -F "Evan Morris" corpus.jsonl | head -n 2'
assert re.fullmatch(gp.ANSWER, "<answer>\nGenentech\n</answer>") and not re.fullmatch(gp.ANSWER, call(CMD))


class Fake:
    def __init__(self, script):
        self.script, self.i, self.constrained = script, 0, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **request):
        self.constrained.append("extra_body" in request)
        text = self.script[self.i]; self.i += 1
        n = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) + 4 for m in request["messages"])
        return NS(choices=[NS(message=NS(content=text, reasoning_content="t"), finish_reason="stop")], usage=NS(prompt_tokens=n, completion_tokens=60))


fake = Fake([call(CMD)] * 5 + ["<answer>\nGenentech\n</answer>"])
make = gp.install(official, lambda: fake, {"variants": ["budget", "dedup", "fold", "relax", "title", "commit"], "context": 16384})
rec = official._run_example_safe({"id": "x", "question": "q", "golden_answers": ["a"]}, client=make(), model="m", tokenizer=tok,
                                 agent_kwargs=dict(corpus_dir=sys.argv[1], max_assistant_turns=6, max_tokens_per_turn=0, tool_max_tokens=2048,
                                                   tool_timeout=60.0, temperature=0.6, top_p=1.0, engine=None))
print("GrepSeek+ commit: finish", rec["finish_reason"], "| generations constrained:", fake.constrained, "| error", rec.get("error"))

# ---- our own agent
from scaleseek import agent
from scaleseek.config import load_base, resolve_run
from scaleseek.methods.scaleseek import ScaleSeek
from scaleseek.methods.base import new_record
base = load_base()
spec = resolve_run(base, "scaleseek_prompt", "hotpotqa", "e5", "formal")
tok2 = AutoTokenizer.from_pretrained(str(base["paths"]["hf_home"]) + "/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
search = "ok\n</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nq\n</parameter>\n<parameter=current_answer>\nunknown\n</parameter>\n</function>\n</tool_call>"


class LLM:
    def __init__(self): self.calls = []
    def room(self, prompt): return 10000
    def complete(self, prompt, *, stop=None, max_tokens=None, **kw):
        self.calls.append((stop, max_tokens))
        if stop == ["</think>"]:
            return {"text": "I have read enough.\n", "finish_reason": "stop", "completion_tokens": 5}
        if stop == ["</parameter>"]:
            assert prompt.endswith(agent.COMMIT)
            return {"text": "Paris\n", "finish_reason": "stop", "completion_tokens": 2}
        return {"text": search, "finish_reason": "stop", "completion_tokens": 9}


class Dense:
    def search(self, q, k): return [{"id": str(i), "score": 0.0, "contents": f'"T{i}"\nbody {i}'} for i in range(k)]


for variants in (("budget",), ("budget", "commit")):
    llm = LLM()
    m = ScaleSeek(NS(spec=spec, llm=llm, retrieval=Dense(), exact=None, tokenizer=tok2, variants=variants))
    rec = new_record({"id": "t", "question": "Capital of France?", "golden_answers": ["Paris"]})
    m.run_example({"question": rec["question"]}, rec)
    print(f"own agent {variants}: finish {rec['finish_reason']}  prediction {rec['prediction']!r}  answered {rec['answered']}  turns {rec['n_turns']}  llm calls {len(llm.calls)}  last two stops {llm.calls[-2:]}")
    print("   tail of the sequence:", repr(rec["trajectory"][-1]["text"][-190:]))
