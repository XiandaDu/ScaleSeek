# Deviations, adaptations and project-defined choices

Everything in this file differs from an official implementation, or is a choice that
BASELINES.md leaves open. Code that is copied verbatim is byte-diffed against the pinned
checkout before every run (`python run.py check`; `scaleseek/official.py`). Code that is
**re-implemented** is marked as such below.

## All methods

| Item | Official | Here | Why |
|---|---|---|---|
| Decoding | per repository | temperature 0.6, top_p 1.0 (§4, GrepSeek Table 5) **+ presence_penalty 2.0**, nothing else sent | §4; the presence penalty is a decision of 2026-10-15, see below |
| How decoding reaches the server | — | every generation request of every method carries the same three values: our loops (`LLMClient.complete`), GrepSeek (`scaleseek/grepseek_entry.py` wraps the harness's OpenAI client), AgentIR (`agentir_entry.py`, Responses API `extra_body`). vLLM cannot default presence_penalty server-side (`--override-generation-config` only takes temperature / top_p / top_k / min_p / repetition_penalty). The BCP judge keeps its own §4 decoding. | one decoding for every method |
| Per-turn generation cap | per repository | none: `max_tokens` = what is left of `max_model_len`, counted by the server's `/tokenize` | §4 |
| Assistant turns | per repository | 6 (FlashRAG) / 16 (BCP) | §4 |
| Server sampling defaults | vLLM reads `generation_config.json` | `--generation-config vllm` | a checkpoint must not inject top_k / penalties we did not set |
| Context window | per repository | 16,384 / 131,072 | §4 |
| Seed | — | vLLM `--seed 0` (server); subset seed in base.yaml | provenance |
| Per-request timeout | official clients: OpenAI SDK default 600 s, 2 retries | 7,200 s and 5 retries for every method's client (ours, and the GrepSeek / AgentIR clients through their entry wrappers) | 600 s cut long BCP turns short for GrepSeek / AgentIR only: a hidden per-turn cap (5 GrepSeek BCP timeouts, decision 2026-10-17) |
| Scored answer | per repository | only a record that **ended** with an answer (`finish_reason == "answer"`) has one; EM/F1 and the BCP judge use it, nothing else (`metrics.final_answer`) | one rule for every method (decision 2026-10-14) |
| BCP judge input | BrowseComp-Plus judges the whole response | the final answer string only, for every method (AgentIR: its `Exact Answer:` value) | §4: the judge sees the final answer; the same evidence for every method |
| Searches per model call | AgentIR's OSS client runs every tool call of a response | at most one, as every other method | AgentIR README: parallel search off for a fair comparison with the BCP baselines |

### Presence penalty 2.0 (decisions 2026-10-15 and 2026-10-17)

Under Table 5 alone (temperature 0.6, top_p 1.0, no penalty), Qwen3.5-9B in thinking mode
looped on one line until the window ran out: in the first formal Direct runs (superseded),
269 / 500 PopQA, 113 TriviaQA and 159 NQ questions ended with the 16K window full, and
253 / 111 / 155 of those ended in a line repeated at least five times; on BCP 462 / 500
hit the 10K cap. Qwen3.5-9B ships no generation_config.json, so no server default was
missing. Settings tried on Direct PopQA (500 questions):

| decoding | window full | of which looping | EM | answered |
|---|---:|---:|---:|---:|
| Table 5 | 269 | 253 | 17.2 ± 3.3 | 46% |
| Table 5 + presence_penalty 1.5 | 128 | 99 | 20.6 ± 3.5 | 74% |
| **Table 5 + presence_penalty 2.0 (chosen)** | 93 | 53 | 20.8 ± 3.6 | 81% |
| model-card thinking (T 1.0, top_p 0.95, top_k 20, pp 1.5) | 12 | 5 | 19.8 ± 3.5 | 98% |

2.0 keeps Table 5's temperature and top_p and changes only the penalty; the model-card
setting nearly ends the loops but replaces three Table 5 values for every method. It is
added for every method alike (Search-R1's Qwen2.5 and GrepSeek's GRPO checkpoint included),
so the decoding is identical across methods. About 10% of Direct PopQA questions still
loop; the report states the share of window-full records per run.

## Direct and RAG (project-defined)

* Prompts are ours: `prompts/direct.py`, `prompts/rag.py` (frozen by sha256 in the yaml).
  The RAG reader is modelled on Search-R1's `scripts/data_process/nq_rag.py` (documents as
  `Doc i(Title: t) text`) and shares its wording with Direct.
* Qwen3.5-9B runs in its default thinking mode; the answer is the last
  `<answer>…</answer>` **after** `</think>`. A thinking block that never closes yields no
  answer.
* RAG passes the 3 passages whole, identical template for all retrievers.

## Search-R1 (`scaleseek/methods/search_r1.py`, loop re-implemented)

Copied and byte-diffed from `infer.py`: the prompt, `curr_search_template`,
`target_sequences`, `curr_eos`, `get_query`, `_passages2string`.

* infer.py loops until EOS; here at most `max_assistant_turns` generation calls.
* `max_new_tokens=1024` → no cap; `temperature=0.7` → 0.6; top-k = 3.
* vLLM matches stop strings on text, not token-id tails. When `</search>` and a following
  newline are one token, infer.py keeps that newline and we do not.
* The EOS ids `[151645, 151643]` are passed as `stop_token_ids`: the checkpoints'
  `generation_config.json` lists only 151643, so without this `<|im_end|>` would not stop
  generation (infer.py checks both ids explicitly).
* The answer is the last `<answer>` of the generated text (= Search-R1's `extract_solution`,
  which skips the prompt's own example).
* Official checkpoints are Qwen2.5 7B/14B; the GrepSeek paper's Search-R1 was retrained on
  Qwen3.5-9B and is not public (§4).

## Search-o1 (`scaleseek/methods/search_o1.py`, loop re-implemented)

Copied and byte-diffed: `get_singleqa_search_o1_instruction`,
`get_multiqa_search_o1_instruction`, `get_task_instruction_openqa`,
`get_webpage_to_reasonchain_instruction` (scripts/prompts.py), `extract_answer`
(scripts/evaluate.py), `extract_between` and the four tags and the inline f-string
messages (scripts/run_search_o1.py).

* The official script batches all questions per vLLM call; here each question runs its own
  loop (batching does not change any sequence's prompts).
* `max_turn` 15 → 6 / 16; the search limit is passed as `MAX_SEARCH_LIMIT` = turns − 1
  (5 / 15). The multi-hop prompt therefore says 5 instead of 10.
* temperature 0.7 / top_p 0.8 / top_k 20 / repetition 1.05 / max_tokens 8192 → profile
  (main generation **and** Reason-in-Documents).
* Bing top-10 → local corpus top-3. A document is shown as
  `**Web Page i:**` + `json.dumps({"id", "title", "context"}, indent=2)`; Bing's `url`,
  `site_name`, `date` and `snippet` do not exist. With no snippet, the official fallback
  `context = raw_context[:max_doc_len * 2]` (max_doc_len = 3000) applies.
* PopQA is not in the official script's dataset list; BASELINES.md assigns it the
  single-hop instruction. BCP is not covered by BASELINES.md; we use the multi-hop one.
* Answer: the official `extract_answer(·, mode='qa')` (last `\boxed{}`) is applied only to
  the text after the last `</think>`, the same rule as Direct/RAG (decision 2026-10-14).
  Officially it reads the whole output, thinking included, so an unfinished reasoning chain
  could still yield an answer.

## GrepSeek (official harness, unmodified)

`python -m inference.run` of the pinned checkout, with `--max_tokens_per_turn 0` (the
harness's own "no per-turn cap"), `--tool_max_tokens 2048 / 6000`,
`--max_assistant_turns 6 / 16`, temperature 0.6, top_p 1.0 (+ presence_penalty 2.0 on
every request, and a 7,200 s / 5-retry client as for every method, both set by
`scaleseek/grepseek_entry.py`, which otherwise runs the harness exactly as
`python -m inference.run`), server flags of
`rl/serve_rl.sh` (`--reasoning-parser qwen3`, `--language-model-only`) **without**
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`.

* Why: the harness sends no `tools` and reads `<tool_call>` from the message text. With the
  paper's vLLM 0.17 the tool parser only ran for requests carrying tools, so the text
  arrived intact. vLLM 0.25 parses tool calls whenever auto tool choice is enabled, strips
  them from `content`, and then drops them because the request has no tool choice, so every
  turn became a parse_error (smoke test, 2026-10-13). Without the parser the harness gets
  exactly the text it got under 0.17.

* Server sampling defaults disabled (`--generation-config vllm`), as for every method.
* The corpus is copied to node-local disk before the run (same bytes, faster reads).
* The harness tries each call once; an `api_error` row counts as a failure. If the vLLM
  server died, the run aborts and resumes later from the harness's own `--resume`.
* Observation budget per tool call: GrepSeek's tool output is capped at 2,048 / 6,000
  tokens (BASELINES.md §4, the checkpoint's setting); a retrieval method sees 3 documents
  per search (about 450 tokens on Wiki-18, at most 3 × 512 on BCP). Kept as specified
  (decision 2026-10-14) -- grep output is matching lines, not documents -- and stated as a
  caveat in the report.
* The official harness has no BCP mode (its system prompt describes a 21-million-line
  Wikipedia file); it runs on the BCP corpus as is -- accepted 2026-10-14.

## AgentIR (official OSS-client scaffold, through `scaleseek/agentir_entry.py`)

Decision (2026-10-14): the official `evaluation/search_agent/oss_client.py` loop, not the
Tongyi one. Qwen3.5-9B is not Tongyi-DeepResearch, so there is no reason for the Tongyi
prompt; the OSS client uses the official BrowseComp-Plus `QUERY_TEMPLATE`
(Explanation / Exact Answer / Confidence) unmodified, with native function calling
(Responses API; vLLM `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`) and the
official reasoning-aware AgentIR search tool. The official `main()` runs; overrides:

* every `responses.create` carries temperature / top_p / presence_penalty from the profile
  (the official client sends none) and `max_output_tokens=None` (official 20000): no
  per-turn cap;
* `--max-iterations` = turns, plus a hard cap of `turns` model calls per question: the
  official loop re-calls without counting when an output ends in a reasoning item;
* `--k 3` (official 5); our per-call log is saved next to the official `run_*.json`.
* One search per model call: tool calls after the first of a response are dropped (count
  in the per-call log). The official OSS loop executes every call of a response; the
  AgentIR README turns parallel search off "for fair comparison against the official
  baselines in BrowseComp-Plus", and every other method searches once per turn.
* EM/F1 and the BCP judge use the last `Exact Answer:` line of the final response
  (project-defined parse of the official format); the judge sees only that value, as it
  sees only the final answer of every other method.
* The task prefix is byte-identical to the official scripts, which pass it through bash
  double quotes: its `\n` is a literal backslash + `n`, in training and evaluation alike.
* The official searcher builds a **flat** inner-product index from tevatron pickles and
  cannot load HNSW, so AgentIR searches a flat index (BASELINES.md §3 allows this).
  On Wiki-18 that is 21M x 2560 float32 = 215 GB of RAM: AgentIR jobs take a whole node.
* The AgentIR environment is the official `uv.lock` minus `vllm`/`deepspeed` (unused by
  the client), see "Environments".

## BrowseComp-Plus: shared adaptations

* **Documents shown to agents are cut to 512 tokens** of the BCP tokenizer
  (`Qwen/Qwen3-0.6B`, pinned) with the official BrowseComp-Plus / AgentIR rule (encode without
  special tokens, keep 512, decode). Applies to RAG, Search-R1, Search-o1 (retrieval service)
  and AgentIR (official searcher, `snippet_max_tokens=512`; the config check forces the two
  to agree). Indexing uses the full page. top-3 is unchanged. Decision 2026-10-14: uncut BCP
  pages (tens of thousands of tokens each) filled the 131K window after one or two
  searches; 512 is the official BCP budget.
* **Single-call methods (Direct, RAG) on BCP: at most 10,000 tokens per response**, the
  BrowseComp-Plus official Qwen client default. Uncapped, Direct thought to the 131K window
  on 3 of 4 smoke questions. Finish reason `max_tokens_cap` when the cap ends a response.
  Multi-turn methods keep the no-cap rule; GrepSeek keeps its own settings.
* **GrepSeek on BCP**: the official harness as is (its prompt describes the Wikipedia
  corpus). Accepted, 2026-10-14.
* **Judge**: verdicts go to `judge__<judge config>.jsonl` with the raw judge text, the judge
  prompt and the agent response, so a different judge (e.g. a 32B model) can re-judge
  stored answers without overwriting. A verdict the official parser cannot read (e.g. the
  judge omits the `correct:` line) is a judge failure and scores as incorrect.

## Indexes

* bm25: Lucene via Pyserini, `k1 = 1.2`, `b = 0.75` (Pyserini's default is 0.9 / 0.4).
* e5 / qwen3_emb_4b: FAISS `IndexHNSWFlat`, inner product on L2-normalised vectors,
  M 32 / efConstruction 128 / efSearch 128. Shard vectors are stored as float16 and added
  as float32. HNSW construction is multi-threaded and therefore not bit-reproducible;
  search on a built index is deterministic (checked by `check-retrievers`).
* qwen3_emb_4b documents are truncated at 4096 tokens (the BrowseComp-Plus / AgentIR
  value; Wiki-18 passages are far shorter). The count of documents at the cap is in the
  index manifest.
* agentir_4b: the official `tevatron.retriever.driver.encode` with the arguments of
  `embed_bcp.sh`; each job encodes a contiguous slice instead of using
  `--dataset_number_of_shards` (the lookup ids are checked against the corpus).

## BrowseComp-Plus judge

* **Judge model: Qwen3-32B** (pinned 9216db57, bf16, TP=4 on one 4 x A100-40GB node), not
  the Qwen3.5-9B of §4. Decision 2026-10-14. Qwen3-32B is the default of the official
  open-source evaluation (`scripts_evaluation/evaluate_run.py`). In the smoke test Qwen3.5-9B
  left out the `correct:` line on all three AgentIR answers (long Explanation / Exact Answer /
  Confidence responses; all three answers matched the gold) and wrote the template's
  `[correct_answer]:` line instead, so the official parser scored them as judge failures.
  The report caveat changes accordingly: our BCP judge is Qwen3-32B, GrepSeek's paper used GPT-4.1.
* Official grader template and parser (texttron/BrowseComp-Plus, byte-diffed), thinking
  disabled as in the official script. temperature 0, top_p 1, 4096 new tokens (§4; the official
  defaults are 0.7 / 0.8 / top_k 20). Only runs that completed with a final answer are sent.
* The judge sees the agent's **final answer** only (§4), for every method: AgentIR's
  `Exact Answer:` value, not its whole Explanation / Exact Answer / Confidence response
  (which the official BrowseComp-Plus evaluation would judge). A record that did not end
  with an answer is not sent and counts as incorrect.

## Environments

* Main env: `env/main.in` → `env/main.lock.txt` (hashes). AgentIR env: the official
  `repos/AgentIR/uv.lock`, exported to `env/agentir.lock.txt` (`uv export --frozen`),
  minus `vllm`/`deepspeed` (unused by the client); `flash-attn` (prebuilt upstream wheel),
  `tevatron` (built from the locked commit) and `jieba` (sdist only) are installed from
  `env/agentir_extra.txt`.
* Jobs build both environments on node-local disk from the wheelhouse (setup_env.sh), so
  results never depend on which copy ran; the lock files are the identity.

## Harness provenance

A run is identified by the git commit of this repository. Formal runs refuse a checkout
with uncommitted changes and never resume under another commit; smoke runs may. `submit.sh`
refuses uncommitted changes and runs every job in a git worktree of the current commit
(`$SCALESEEK_ROOT/harness/<commit>`), so a job that starts hours later runs the code that
existed at submission.
