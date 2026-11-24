"""The agent loop without a GPU (scripted LLM, fake retrieval): the three protocols of the
ablation.  v1 and v2 (budget) end at the turn limit without an answer when the model keeps
searching; v3 (budget + commit) forces the last turn into an answer call.
Prints a hash of each full sequence, to compare two versions of the code."""
import hashlib
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "/home/a32du/ScaleSeek")
from transformers import AutoTokenizer

from scaleseek import agent
from scaleseek.config import load_base, resolve_run
from scaleseek.methods.base import new_record
from scaleseek.methods.scaleseek import ScaleSeek

base = load_base()
spec = resolve_run(base, "scaleseek_v3_prompt", "hotpotqa", "e5", "formal")
g = spec.method["generator"]
tok = AutoTokenizer.from_pretrained(f"{base['paths']['hf_home']}/hub/models--{g['model'].replace('/', '--')}/snapshots/{g['revision']}")
call = lambda tool, key, value: (f"<tool_call>\n<function={tool}>\n<parameter={key}>\n{value}\n</parameter>\n"
                                 "<parameter=current_answer>\nunknown\n</parameter>\n</function>\n</tool_call>")
TURNS = ["ok\n</think>\n\n" + call("find", "phrase", "Evan Morris") + "\n" + call("search", "query", "q"),
         "ok\n</think>\n\n" + call("search", "query", "q")]
passages = lambda k: [{"id": str(i), "score": 0.0, "contents": f'"T{i}"\nbody {i}'} for i in range(k)]


class LLM:
    def __init__(self):
        self.calls = []

    def room(self, prompt):
        return 10000

    def complete(self, prompt, *, stop=None, max_tokens=None, **kw):
        self.calls.append((stop, max_tokens))
        if stop == ["</think>"]:
            return {"text": "I have read enough.\n", "finish_reason": "stop", "completion_tokens": 5}
        if stop == ["</parameter>"]:
            assert prompt.endswith(agent.COMMIT)
            return {"text": "Paris\n", "finish_reason": "stop", "completion_tokens": 2}
        return {"text": TURNS[len(self.calls) > 1], "finish_reason": "stop", "completion_tokens": 9}


class Dense:
    def search(self, q, k):
        return passages(k)


class Exact:
    def find(self, q, k, *a):
        return {"count": 7, "phrase_counts": None, "relaxed_count": None, "hits": passages(k)}


for variants in ((), ("budget",), ("budget", "commit")):
    llm = LLM()
    m = ScaleSeek(NS(spec=spec, llm=llm, retrieval=Dense(), exact=Exact(), tokenizer=tok, variants=variants))
    rec = new_record({"id": "t", "question": "Capital of France?", "golden_answers": ["Paris"]})
    m.run_example({"question": rec["question"]}, rec)
    text = rec["trajectory"][-1]["text"]
    print(f"{variants}: finish {rec['finish_reason']}  prediction {rec['prediction']!r}  answered {rec['answered']}  turns {rec['n_turns']}"
          f"  retrievals {rec['n_retrievals']}  llm calls {len(llm.calls)}  sequence sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    print("   tail:", repr(text[-150:]))
