"""Prompt templates for ScaleSeek SFT cold-start data generation.

Two LLM roles (mirrors GrepSeek's Tutor/Planner structure):

- Tutor: sees gold answer + decomposition + downstream passages. Used for
    sub-question decomposition, bridge-entity extraction, backward tool-trace
    discovery, judging whether a tool output confirms an expected answer, and
    editing the planner's reasoning to align with the known-correct trace.

- Planner: sees only the original question and tool responses already in the
    trace. Generates (reasoning, tool_call) pairs forward without seeing the
    gold answer, decomposition, or downstream passages.

Key ScaleSeek adaptation from GrepSeek:
    GrepSeek backward: discover one shell command (rg/grep on corpus.jsonl)
    ScaleSeek backward: discover bm25_retrieve call(s) + optional grep_workspace
    - bm25_retrieve query must NOT contain the answer (ANSWER-LEAK rule)
    - grep_workspace CAN reference the answer — it runs after BM25 to verify
      the correct passage landed in the workspace
"""

# ============================================================
# Corpus description used wherever a prompt needs to describe the tools.
# ============================================================

CORPUS_DESCRIPTION = """Tools available to the agent:
  bm25_retrieve(query, top_k, k1, b, mode)  — BM25 search against 21M Wikipedia passages → loads hits into workspace
  grep_workspace(pattern, case_insensitive)  — regex search within workspace passages
  read_doc(doc_id)                           — read full passage by doc_id

The corpus has 21 million Wikipedia passages and cannot be accessed directly. The agent must first call bm25_retrieve to load relevant passages into a bounded workspace, then use grep_workspace or read_doc to search within them."""


# ============================================================
# Phase A: Decomposition (tutor)
# Identical to GrepSeek — no tool-specific content.
# ============================================================

DECOMPOSE_PROMPT = """You are decomposing a multi-hop question into the ordered chain of single-hop sub-questions an agent would need to solve in order to reach the answer.

Question: {question}
Final answer: {answer}

Rules:
- Output 1 to 3 ordered sub-questions; each one's answer is what an agent would need before it can solve the next one.
- Order MUST match the order an agent would solve the question (sub_q[i+1] depends on sub_q[i]'s answer).
- The LAST sub-question's answer is the final answer above.
- Do NOT include expected_answers — only the sub-questions themselves. The intermediate answers will be derived later.
- Do NOT make a sub-question whose answer is already given in the question.

Output ONLY a JSON array, no prose:
[{{"sub_question": "..."}}, {{"sub_question": "..."}}, ...]"""


# ============================================================
# Phase B: Bridge-entity extraction (tutor)
# Identical to GrepSeek — no tool-specific content.
# ============================================================

BRIDGE_EXTRACT_PROMPT = """You are reading a Wikipedia passage that was retrieved to help answer a multi-hop question. Your job is to identify the entity the passage uses as the bridge — the answer to the previous sub-question.

Original multi-hop question: {question}

Earlier sub-question (whose answer we are trying to identify): {sub_q_prev}
Later sub-question (already solved): {sub_q_next}
Answer to the later sub-question: {expected_next}

Passage retrieved for the later sub-question:
---
{doc_next}
---

Read the passage. The passage answers the later sub-question; somewhere in it, the entity named in the answer to the EARLIER sub-question appears (since the later sub-question depends on it). Extract that entity exactly.

Also collect:
- "aliases" — alternative names of the SAME entity that the passage explicitly mentions. Only include forms that literally appear in the passage. Do NOT invent aliases.
- "alternates" — at most 2 OTHER plausible candidate entities from the passage that could also be the bridge if your primary pick is wrong.

Output a JSON object, nothing else:
{{"bridge_entity": "<short noun phrase>", "aliases": ["<alt-name-1>", ...], "alternates": ["<other-candidate-1>", ...], "evidence": "<the exact short snippet from the passage that names the primary entity>"}}

If you cannot extract a clear bridge entity, output:
{{"bridge_entity": null, "aliases": [], "alternates": [], "evidence": null}}"""


# ============================================================
# Phase B: Backward tool-trace discovery (tutor)
# Adapted from GrepSeek's backward shell-command discovery.
# Key differences:
#   - Output is a bm25_retrieve call (+ optional grep_workspace), not a shell command.
#   - ANSWER-LEAK rule applies to the bm25_retrieve query only.
#   - grep_workspace may use the answer to verify retrieval success.
# ============================================================

BACKWARD_TOOL_SYSTEM = """You are an expert at constructing BM25 retrieval tool calls to surface specific Wikipedia passages.

{corpus_description}

You are working BACKWARD from a known answer to construct a tool-call trace that, when executed, retrieves a passage confirming the answer to a sub-question.

A backward trace for one sub-question consists of:
  1. One bm25_retrieve call — loads candidate passages into the workspace.
  2. Optionally: one grep_workspace call — verifies the answer appears in the workspace.

CRITICAL RULES:

(1) BM25 ANSWER-LEAK RULE.
The bm25_retrieve query must NOT contain the expected answer string (or a near-identical paraphrase). The agent at inference time does not know the answer; it must EMERGE from the retrieved passages.

You ARE encouraged to use:
- Terms and phrases from the sub-question itself.
- Synonyms and paraphrases of those terms.
- Related concepts an agent could plausibly infer from the sub-question alone.

Example (sub_question = "Which hotel company is the Oberoi family part of?", expected_answer = "The Oberoi Group"):
  WRONG (answer in query):     bm25_retrieve(query="Oberoi Group hotel company")
  WRONG (near-paraphrase):     bm25_retrieve(query="The Oberoi Group hospitality")
  RIGHT (question terms):      bm25_retrieve(query="Oberoi family hotel company")
  RIGHT (synonym):             bm25_retrieve(query="Oberoi family hospitality business")
  RIGHT (broader + narrower):  bm25_retrieve(query="Oberoi family", top_k=10)

If a query keeps failing to retrieve the right passage, try a different paraphrase of the sub-question's key concept rather than reaching for the answer string.

(2) grep_workspace MAY reference the answer.
grep_workspace runs AFTER bm25_retrieve and is used to confirm the answer appears in a retrieved passage. Using the answer term as the grep pattern is acceptable and expected.

(3) BM25 PARAMETER CHOICE.
Choose parameters deliberately:
- top_k: 5 is usually sufficient; use 10–20 for rare or ambiguous entities.
- k1=2.0, b=0.5: better for rare named-entity queries (boosts exact-term matches).
- k1=1.2, b=0.9: better for long descriptive queries (penalizes length more).
- mode="replace" for the first call in a trace; mode="merge" only if a second entity must be added.

(4) TRACE LENGTH.
Keep the trace short — typically 1 bm25_retrieve call, optionally 1 grep_workspace. Avoid unnecessary steps."""

BACKWARD_TOOL_USER_INITIAL = """Sub-question to find supporting evidence for: {sub_question}

Expected answer (known to you for verification only — do NOT use in the bm25_retrieve query):
- primary: {expected_answer}
- alternative forms (also forbidden in bm25_retrieve query): {forbidden_forms}

Other passages already retrieved for later sub-questions in this chain (for context only):
{downstream_docs}

Reason briefly about which Wikipedia article would most directly answer this sub-question and what distinctive terms from the SUB-QUESTION (not the answer) you would use to find it. Then output a short tool-call trace.

Output exactly:
<reasoning>
your reasoning (2–4 sentences); explicitly confirm your bm25_retrieve query does NOT include the expected answer or any of its alternative forms
</reasoning>
<tool_trace>
[
  {{"name": "bm25_retrieve", "arguments": {{"query": "...", "top_k": 5, "k1": 1.5, "b": 0.75, "mode": "replace"}}}},
  {{"name": "grep_workspace", "arguments": {{"pattern": "...", "case_insensitive": true}}}}
]
</tool_trace>

The grep_workspace step is optional — omit it if the bm25_retrieve result alone should be sufficient to confirm the answer."""

BACKWARD_TOOL_USER_REFINE = """Your previous tool trace did not retrieve a passage that confirms the expected answer.

Sub-question: {sub_question}
Expected answer (do NOT use in bm25_retrieve query): {expected_answer}
Alternative forms (also forbidden in bm25_retrieve query): {forbidden_forms}

Previous attempts:
{prior_attempts}

Most recent attempt:
Tool trace: {last_tool_trace}
BM25 output:
---
{last_output}
---
Judge said it does NOT confirm the answer because: {judge_reasoning}

Propose a different bm25_retrieve query. Common fixes:
- The key phrase may not appear verbatim — try a shorter or differently-worded substring from the sub-question.
- Try synonym variants ("headquartered in" vs "head office", "founded" vs "established").
- If the query is too narrow, drop a term or broaden the phrasing.
- If the query is too broad, add a second distinctive term from the sub-question.
- Try increasing top_k if the right passage might be ranked lower.

Same hard rules as before (bm25_retrieve query must NOT use the expected answer string).

Output exactly:
<reasoning>
why the previous query failed and what is different about your new approach
</reasoning>
<tool_trace>
[
  {{"name": "bm25_retrieve", "arguments": {{"query": "...", "top_k": 5, "k1": 1.5, "b": 0.75, "mode": "replace"}}}},
  {{"name": "grep_workspace", "arguments": {{"pattern": "...", "case_insensitive": true}}}}
]
</tool_trace>"""


# ============================================================
# Phase B: Judge (tutor)
# Adapted from GrepSeek: checks bm25_retrieve output (not shell output).
# ============================================================

JUDGE_PROMPT = """You are judging whether a BM25 retrieval result contains a passage that confirms the expected answer to a sub-question.

Sub-question: {sub_question}
Expected answer (primary): {expected_answer}
Acceptable alternative forms (any of these counts as confirming): {acceptable_forms}

BM25 retrieval result:
---
{tool_output}
---

Does the retrieval result contain a passage that clearly answers the sub-question with the primary expected answer OR any of the acceptable alternative forms? Be strict — a tangential mention or a passage about a different entity does NOT count. A direct statement of the answer in a retrieved passage DOES count.

Output exactly one JSON object:
{{"verdict": "YES" or "NO", "reasoning": "<one short sentence>"}}"""


# ============================================================
# Phase C: Planner (forward generation)
# PLANNER_SYSTEM is the ScaleSeek system prompt (same as prompts/system_prompt.txt).
# PLANNER_USER is identical to GrepSeek.
# ============================================================

PLANNER_SYSTEM = """You are a research agent that answers questions by searching a large Wikipedia corpus through two-stage adaptive retrieval.

{corpus_description}

For every turn, write 2–4 sentences of reasoning first: what you have learned from prior results, what entity or fact is still missing, and what you plan to do next. Then output exactly one action block.

Tool call:
<tool_call>
{{"name": "bm25_retrieve", "arguments": {{"query": "...", "top_k": 5, "k1": 1.5, "b": 0.75, "mode": "replace"}}}}
</tool_call>

<tool_call>
{{"name": "grep_workspace", "arguments": {{"pattern": "...", "case_insensitive": true}}}}
</tool_call>

<tool_call>
{{"name": "read_doc", "arguments": {{"doc_id": "..."}}}}
</tool_call>

Final answer (only when confident):
<answer>
concise answer — typically a noun phrase, name, date, number, or yes/no
</answer>

Always reason first, then exactly one action block. Never skip the reasoning."""

PLANNER_USER = """Question: {question}

Trace so far:
{history}

Produce the next step."""


# ============================================================
# Phase C: Tutor edits the planner's draft reasoning (tutor)
# Adapted from GrepSeek: references "tool call" instead of "shell command".
# Information-frontier rules are identical.
# ============================================================

TUTOR_EDIT_PROMPT = """You are editing a research agent's draft reasoning so that it leads naturally to a known-correct next tool call. Your goal is to produce a final reasoning paragraph that:

1. Is grounded ONLY in what the agent has actually observed so far (the trace below). Do NOT add facts the agent could not know at this point.
2. Concludes with a clear motivation for the target tool call.
3. Reads as natural forward reasoning by a careful agent — not as a justification written after the fact. Hedge where genuine uncertainty exists.
4. Stays close to the agent's draft when the draft is already on track; rewrite freely when the draft is going in the wrong direction.

Original question (the agent knows this): {question}

Trace observed so far by the agent:
{history}

Agent's draft reasoning:
---
{draft_think}
---

Agent's draft tool call (may be discarded):
---
{draft_tool_call}
---

The known-correct next tool call (this is what the trace will use):
---
{target_tool_call}
---

What this tool call will return when executed (for your context only — the agent only sees this AFTER the call, NOT while reasoning):
---
{target_output_preview}
---

FORBIDDEN — your edited reasoning must NOT contain:

(a) The expected answer of the current step, the final answer, or any answer to be discovered in a future step. The agent should not know these yet.

(b) Any factual claim the agent could not derive from (i) the user's question, (ii) prior tool responses in the trace, or (iii) common knowledge.

ALLOWED — the agent's reasoning may freely include:
- Words from the question and synonyms/paraphrases ("hotel" → "hospitality", "head office" → "headquarters").
- Tentative hypotheses about WHERE the answer might be found, phrased as guesses.
- Common-knowledge inference (e.g., "Delhi is the capital of India" once Delhi has appeared).
- BM25 parameter reasoning ("I'll use k1=2.0 for this rare named-entity query").

Concrete pass/fail examples:

  Question: "The Oberoi family is part of a hotel company that has a head office in what city?"
  Trace so far: (no tool calls yet)
  Target tool call: bm25_retrieve(query="Oberoi family hotel company", top_k=5)

    LEAK (names the bridge answer "Oberoi Group" before it has been seen):
      "I need to find the Oberoi Group's headquarters city."

    OK (uses question terms, makes a hypothesis):
      "I need to find which hotel company the Oberoi family is associated with. The corpus likely contains a passage mentioning 'Oberoi family' alongside hotel-business context. I'll retrieve with those terms and a moderate top_k."

  Question: same
  Trace so far: bm25_retrieve returned "...the Oberoi family is the majority shareholder in EIH Ltd, parent of The Oberoi Group..."
  Target tool call: grep_workspace(pattern="head office|headquartered", case_insensitive=true)

    OK ("Oberoi Group" is now derivable from the prior tool response):
      "The previous result identifies the hotel company as The Oberoi Group. To find its head office, I'll grep for 'head office' or 'headquartered' within the current workspace."

When the planner's draft is already valid, keep it close to the draft. When the draft proposes a substantively different direction, rewrite so it naturally leads to the target tool call — using hypotheses, synonyms, and BM25 reasoning, never by naming entities the agent shouldn't know yet.

Output exactly:
<edited_reasoning>
the final reasoning paragraph (2–4 sentences)
</edited_reasoning>"""


# ============================================================
# Phase D: Final answer (planner)
# Identical to GrepSeek.
# ============================================================

FINAL_ANSWER_USER = """Question: {question}

Trace so far:
{history}

You now have enough information. Produce a brief reasoning paragraph synthesizing the answer from the trace, then output exactly:

<answer>
the final answer (concise — just a name, date, or short noun phrase)
</answer>"""


# ============================================================
# Phase E: Quality judge (post-hoc filter on assembled trajectories)
# Adapted from GrepSeek: CHECK 2 targets bm25_retrieve query (not shell command).
# grep_workspace patterns referencing the answer are NOT a failure.
# ============================================================

QUALITY_JUDGE_PROMPT = """You are an expert reviewer of multi-hop QA trajectories produced by a research agent. The agent answers a question by calling bm25_retrieve, grep_workspace, and read_doc against a Wikipedia corpus, reasoning about the results across multiple turns. Your job is NOT to solve the question. Your job is to verify that the trajectory is INTERNALLY COHERENT.

INFORMATION FRONTIER

At any TURN k, the agent could legitimately know only:
  (a) the text of the original QUESTION,
  (b) anything that appeared in the OUTPUT of any earlier turn (1 ... k-1),
  (c) generic common knowledge that any educated reader would have without consulting Wikipedia. Examples of (c):
        - "Delhi is in India", "Brazil is a country", "WW II ended in 1945".
        - "Headquarters" / "head office" / "based in" mean roughly the same.
        - Synonyms and paraphrases of words from the question.
      Counter-examples (NOT (c)):
        - "Cafu was Brazil's 2002 World Cup captain" - too specific.
        - "The Oberoi Group is based in Delhi" - too specific.
        - The bridge answer of any sub-question.

CHECKS

The trajectory FAILS if ANY of the following holds for ANY turn. Report the FIRST turn where it fails and which check failed.

CHECK 1 - REASONING leaks future facts.
  TURN k's REASONING names a specific entity, number, date, or fact not derivable from the information frontier.
  FAIL example: TURN 1 REASONING (no prior turns) says "I'll look up the Oberoi Group's headquarters" when the question only mentions "Oberoi family" — the company name has not been observed yet.

CHECK 2 - bm25_retrieve QUERY leaks future facts.
  TURN k's bm25_retrieve query uses a term that is the expected answer (or near-identical paraphrase) to a not-yet-answered sub-question.
  Note: grep_workspace patterns may reference the answer — that is expected behavior, not a failure.
  FAIL example: the answer to the current sub-question is "The Oberoi Group" and the bm25_retrieve query is "Oberoi Group headquarters" before any tool response has named "Oberoi Group".

CHECK 3 - ACTION does not match REASONING.
  TURN k's REASONING explicitly states the question is answered and no further search is needed, but the action on this turn is a tool call rather than a final answer.
  FAIL example: REASONING ends with "the answer is clearly Amy Jo Johnson, no further search needed" and the action is a bm25_retrieve call.

CHECK 4 - FINAL ANSWER not supported.
  The final turn's answer is not stated in, or clearly inferrable from, any earlier tool output.
  FAIL example: FINAL ANSWER is "Roseau, Minnesota" but no earlier tool output contains "Roseau" or supports that location.

If none of CHECK 1–4 fails on any turn, the trajectory PASSES.

When in doubt, prefer FAIL over PASS — we want clean training data, not high yield.

OUTPUT

Return EXACTLY one JSON object, no surrounding prose, no markdown fences:
{{"verdict": "PASS" or "FAIL", "failing_check": 1 or 2 or 3 or 4 or null, "first_failing_turn": <int or null>, "reasoning": "<one short sentence, max ~30 words>"}}

QUESTION
{question}

TRAJECTORY
{trajectory_text}"""
