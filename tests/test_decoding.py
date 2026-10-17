"""Every method sends the same decoding: our client and the GrepSeek wrapper (and AgentIR's
entry, which sets the same values in its request override)."""
from types import SimpleNamespace as NS


def test_llm_client_sends_presence_penalty():
    from scaleseek.llm import LLMClient
    client = LLMClient(NS(served_name="m"), {})
    sent = []
    client._post = lambda path, payload: sent.append(payload) or {
        "choices": [{"text": "x", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    client.complete("p", temperature=0.6, top_p=1.0, presence_penalty=2.0, max_tokens=8)
    assert (sent[0]["temperature"], sent[0]["top_p"], sent[0]["presence_penalty"]) == (0.6, 1.0, 2.0)


def test_grepseek_wrapper_adds_presence_penalty():
    from scaleseek.grepseek_entry import with_presence_penalty
    seen = []
    client = NS(chat=NS(completions=NS(create=lambda **request: seen.append(request))))
    with_presence_penalty(client, 2.0).chat.completions.create(model="m", temperature=0.6, top_p=1.0)
    assert seen == [{"model": "m", "temperature": 0.6, "top_p": 1.0, "presence_penalty": 2.0}]
