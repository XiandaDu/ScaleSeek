"""Project-defined answer-extraction prompt for scoring DCI free-prose output.

The official DCI-Lite benchmark prompt requests no answer format (its official
metric is an LLM judge), so strict EM over the prose is 0 by construction.
This prompt turns each already-produced final report into the concise span the
agent committed to, after which the SAME EM/F1 scorer as every other method
applies. The extractor never sees gold answers, must copy from the report
rather than answer from parametric knowledge, and must refuse when the report
does not commit to an answer -- so it can locate an answer but never create
one. Recorded per TASK.md as a project-defined scoring adapter, not part of
the official DCI protocol.
"""

PROMPT = """You will be given a question and the final report a research agent wrote after searching a corpus. Your only job is to extract the report's final answer to the question.

Rules:
- Copy the answer the REPORT commits to. Never answer from your own knowledge, and never "improve" the report's answer.
- The extracted answer must be concise: a noun phrase, name, date, number, or yes/no.
- If the report states the answer could not be found, is uncertain without committing, or contains no answer to the question, output exactly NO_ANSWER.

End your response with exactly one <answer></answer> block containing only the extracted answer or NO_ANSWER. Put nothing else inside the tags."""
