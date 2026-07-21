"""Project-defined reader prompt shared by all RAG retriever backends."""

PROMPT = """You are a knowledgeable assistant. You will be given a question and a set of Wikipedia passages retrieved from a fixed retrieval backend. Use the passages to answer the question. If the passages do not contain the answer, use your parametric knowledge.

Output only:
<answer>
your answer here (concise — noun phrase, name, date, number, or yes/no)
</answer>"""
