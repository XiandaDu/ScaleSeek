# ScaleSeek

**Training Adaptive Retrieval Agents for Scalable Direct Corpus Interaction**

## Motivation

Direct Corpus Interaction (DCI) allows an agent to search a raw corpus using tools such as `grep`, file reads, and shell commands.

However, full-corpus DCI becomes inefficient as the corpus grows:

* search latency increases;
* irrelevant matches increase;
* tool outputs become larger;
* answer accuracy may decrease.

ScaleSeek aims to make DCI scalable by first using BM25 to construct a smaller workspace, then allowing the agent to perform detailed search only inside that workspace.

---

## Core Idea

```text
Question
   ↓
Adaptive BM25 Retrieval
   ↓
Bounded Workspace
   ↓
grep / read / shell search
   ↓
Answer
```

The agent should learn to control:

* whether BM25 retrieval is needed;
* the BM25 query;
* `top_k`;
* BM25 parameters such as `k1` and `b`;
* whether newly retrieved documents should merge with or replace the current workspace;
* when to stop searching and answer.

The main contribution is a **trained retrieval and workspace-control policy**, rather than a fixed BM25-to-DCI pipeline.

---

# Stage 1: Evaluation Framework

The first stage builds a unified evaluation pipeline.

## Candidate Datasets

Single-hop QA:

* Natural Questions
* TriviaQA
* PopQA

Multi-hop QA:

* HotpotQA
* 2WikiMultihopQA
* MuSiQue
* Bamboogle

Complex or retrieval-intensive tasks:

* BrowseComp-Plus
* BRIGHT

The final dataset selection and corpus construction process remain **TBD**.

## Baselines

* direct answer;
* BM25 RAG;
* dense RAG;
* IRCoT;
* Search-R1;
* AgentIR;
* raw DCI;
* fixed BM25 followed by DCI;
* prompt-based dynamic workspace DCI.

Open-source baselines will be implemented first. API-based agents will be evaluated later.

## Metrics

Answer quality:

* Exact Match;
* F1.

Retrieval and workspace quality:

* gold-document or qrel recall;
* workspace size;
* relevant-document density.

Efficiency:

* latency per query;
* number of retrieval and shell calls;
* token or API cost;
* timeout rate.

The main experiment will measure how performance changes as corpus size increases.

---

# Stage 2: Training Framework

The second stage trains the ScaleSeek agent using an RL environment based on `verl`.

```text
Training Data
     ↓
Supervised Fine-Tuning
     ↓
Reinforcement Learning
     ↓
ScaleSeek Agent
```

The implementation will build on ideas from GrepSeek and s3.

The following details remain **TBD**:

* cold-start data generation;
* synthetic trajectory construction;
* SFT data format and filtering;
* RL algorithm and reward design;
* curriculum and hard-example selection;
* discrete or continuous BM25 parameter prediction.

These decisions will be finalized after reproducing the evaluation baselines and examining the available training frameworks.

---

## Research Question

Can a trained agent learn adaptive BM25 retrieval and workspace-management decisions that outperform fixed retrieval policies?

The target is to achieve:

* higher accuracy under the same latency budget;
* lower latency under the same accuracy target;
* smaller workspaces;
* less performance degradation as corpus size increases.

---

## Related Work
Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction
https://arxiv.org/abs/2605.05242
GrepSeek: Training Search Agents for Direct Corpus Interaction
https://arxiv.org/abs/2605.29307
DR-DCI: Scaling Direct Corpus Interaction via Dynamic Workspace Expansion
https://arxiv.org/abs/2606.14885
Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient?
https://arxiv.org/abs/2605.10848
s3: You Don't Need That Much Data to Train a Search Agent via RL
https://arxiv.org/abs/2505.14146
Towards Retrieving Interaction Spaces for Agentic Search
https://arxiv.org/abs/2606.06880
