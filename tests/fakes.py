"""Fake LLM / retrieval / tokenizer for exercising method loops without a GPU."""
from types import SimpleNamespace


class FakeTokenizer:
    chat_template = "fake"
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")


class FakeLLM:
    """Returns scripted completions; records every prompt it saw."""

    def __init__(self, script, window=200_000):
        self.script = list(script)
        self.prompts = []
        self.window = window

    def room(self, prompt):
        return self.window - len(prompt) // 4

    def complete(self, prompt, **kw):
        self.prompts.append((prompt, kw))
        text, finish, stop_reason = self.script.pop(0)
        return {"text": text, "finish_reason": finish, "stop_reason": stop_reason,
                "prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}


class FakeRetrieval:
    def __init__(self):
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return [{"id": str(i), "score": 1.0 / (i + 1), "contents": f'"Title {i}"\nText about {query} {i}.'}
                for i in range(k)]


def make_ctx(method_name, prompts, script, dataset="popqa"):
    from scaleseek.config import load_base, resolve_run
    base = load_base()
    retr = {"search_r1_7b": "bm25", "search_o1": "bm25", "rag": "bm25"}.get(method_name)
    spec = resolve_run(base, method_name, dataset, retr, "smoke")
    llm = FakeLLM(script)
    ctx = SimpleNamespace(spec=spec, prompts=prompts, llm=llm, retrieval=FakeRetrieval(),
                          tokenizer=FakeTokenizer(), title_rule="first_line")
    return ctx
