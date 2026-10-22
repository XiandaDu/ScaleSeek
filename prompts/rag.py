"""RAG reader prompt and passage format -- PROJECT-DEFINED.

Modelled on the Search-R1 RAG protocol (scripts/data_process/nq_rag.py: documents as
"Doc i(Title: t) text"), worded like prompts/direct.py.  The SAME template is used for
every retriever; passages are inserted whole.  Frozen by the sha256s in
configs/methods/rag.yaml.

The reader may fall back on its own knowledge when the documents do not answer the question
(as in the v1 prompt): without that sentence Qwen3.5-9B answers "Not mentioned" instead, on
60% of 2Wiki questions (reports/other_methods_failures.md §4.3).
"""

RAG_USER_TEMPLATE = (
    "Answer the given question using the retrieved documents below, which may or may not "
    "be relevant. If the documents do not contain the answer, answer from your own knowledge. "
    "Give the final answer inside <answer> and </answer>, as a short phrase "
    "without explanation, for example <answer> Beijing </answer>.\n\n"
    "Documents:\n{documents}\n"
    "Question: {question}"
)

# One line per passage; `documents` above is the concatenation of these.
RAG_PASSAGE_TEMPLATE = "Doc {index}(Title: {title}) {text}\n"
