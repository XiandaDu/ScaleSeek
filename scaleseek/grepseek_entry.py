"""GrepSeek entry point -- runs the OFFICIAL harness `inference/run.py` main() of the pinned
checkout, exactly as `python -m inference.run` would, with one override (docs/deviations.md):

    (cwd and PYTHONPATH = the checkout)
    python scaleseek/grepseek_entry.py <presence_penalty> -- <official inference.run args>

Every chat-completion request of the harness's OpenAI client also carries the profile's
presence_penalty, as every other method's generation requests do.  The harness has no flag
for it, and vLLM cannot set a default presence_penalty server-side.
"""

from __future__ import annotations

import os
import sys


def with_presence_penalty(client, penalty: float):
    """The client, with presence_penalty added to every chat-completion request."""
    create = client.chat.completions.create
    client.chat.completions.create = lambda **request: create(**request, presence_penalty=penalty)
    return client


def main() -> None:
    penalty = float(sys.argv[1])
    assert sys.argv[2] == "--"
    sys.path[0] = os.getcwd()            # as under `python -m`: the checkout, not this directory
    import inference.run as official

    real_openai = official.OpenAI
    official.OpenAI = lambda *args, **kw: with_presence_penalty(real_openai(*args, **kw), penalty)
    sys.argv = ["inference/run.py"] + sys.argv[3:]
    official.main()


if __name__ == "__main__":
    main()
