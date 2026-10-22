"""GrepSeek entry point -- runs the OFFICIAL harness `inference/run.py` main() of the pinned
checkout, exactly as `python -m inference.run` would, with two overrides that give its
OpenAI client the same settings as every other method's client (docs/deviations.md):

    (cwd and PYTHONPATH = the checkout)
    python scaleseek/grepseek_entry.py '<settings json>' -- <official inference.run args>

  * every chat-completion request also carries the profile's presence_penalty (the harness
    has no flag for it, and vLLM cannot default it server-side);
  * the client's per-request timeout and retries are `request_timeout_s` / `max_retries`
    (the SDK default 600 s / 2 cut long BCP turns short).
"""

from __future__ import annotations

import json
import os
import sys


def with_presence_penalty(client, penalty: float):
    """The client, with presence_penalty added to every chat-completion request."""
    create = client.chat.completions.create
    client.chat.completions.create = lambda **request: create(**request, presence_penalty=penalty)
    return client


def main() -> None:
    settings = json.loads(sys.argv[1])
    assert sys.argv[2] == "--"
    sys.path[0] = os.getcwd()            # as under `python -m`: the checkout, not this directory
    import inference.run as official

    real_openai = official.OpenAI

    def make_client(*args, **kw):
        kw.update(timeout=settings["request_timeout_s"], max_retries=settings["max_retries"])
        return with_presence_penalty(real_openai(*args, **kw), settings["presence_penalty"])

    official.OpenAI = make_client
    sys.argv = ["inference/run.py"] + sys.argv[3:]
    official.main()


if __name__ == "__main__":
    main()
