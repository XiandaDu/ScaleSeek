# Can SFT cold-start teach adaptive BM25 parameter control?

**Status: investigated on the local smoke/puzzle testbeds. Negative for k1/b.**
All numbers below are from SMOKE / synthetic testbeds and must never enter a
result table (TASK.md). This documents a methodology finding, not an experiment.

## Question

ScaleSeek's thesis is a *trained* policy that adaptively controls BM25
`k1 / b / top_k / mode`. Can the SFT cold-start stage teach a small student
(Qwen3-1.7B) to *adapt* `k1/b` per query — the prerequisite for RL to refine it?

## Investigation arc

1. **Parroting.** The first SFT student emitted a constant `k1=1.5, b=0.75,
   top_k=5` on every call. Root cause: the example tool-calls in
   `prompts/sft_prompts.py` and the runtime prompt literally showed those numbers,
   so the teacher copied them and the student memorised the constant.
   Fixed by making the no-parameter-bias prompt the default (commit `ccf9d8f`).

2. **De-bias → omission.** With the numeric anchors removed, both Qwen3-4B **and
   Qwen3-8B** teachers set `k1/b/top_k` on **0** of ~15–31 calls — they simply omit
   the knobs. So the degeneracy is **not a model-capacity limit**; a stronger
   teacher does not spontaneously reason about BM25 parameters.

3. **Three cold-start policies** (`scripts/compare_param_policies.py`,
   `--param-policy {teacher,heuristic,search}` in `train/sft/coldstart.py`) on a
   parameter-sensitive corpus (`make_smoke_corpus.py --hard`):

   | policy | coverage (gold in workspace) | mean workspace | notes |
   |---|---|---|---|
   | omit (teacher-native) | 0.31 | 3.0 | `top_k=3` default misses buried gold |
   | heuristic (query-length rule) | 0.75 | 9.7 | brute-force widens `top_k`, no ranking gain |
   | search-then-teach | *1.00 — see below* | 9.1 | tunes `k1/b` when it helps, else default |

   > **⚠️ 2026-07-28 correction — the `search` row is not a comparable result.**
   > `_search_params` grid-searches using the gold answer (`targets =
   > [hop.expected] + hop.forms`) and stops when the gold is ranked best, then
   > picks the smallest `top_k` that includes it. "Gold in workspace" **is its
   > objective function**, so 1.00 is guaranteed whenever *any* (k1,b) surfaces
   > the gold — it is bounded at 1.00 by construction. Read it as an **upper
   > bound on what any parameter policy could achieve on this corpus**, not as
   > evidence that this policy beats the other two. Using an oracle teacher to
   > distil is legitimate; reporting the oracle's own objective as a policy
   > comparison is not. The row that actually answers "did anything transfer" is
   > the student's held-out behaviour in §4 — and there the answer is *`top_k`
   > only*. The column is renamed `coverage` in
   > `scripts/compare_param_policies.py` and the `search` row is now printed with
   > an explicit "oracle objective, upper bound" marker.

4. **Student level.** Training a student per policy and reading the params it
   *emits*: omit → none; heuristic → a single constant `1.5/0.75/5`; search →
   genuinely **varied `top_k` (1/3/5/10)** but `k1/b` still constant (the teacher's
   focused queries never needed `k1/b`, so the data carried no `k1/b` signal).

5. **Puzzle testbed (this report's core).** To force dense `k1/b` signal we built
   a synthetic testbed of controlled BM25 "parameter puzzles"
   (`scripts/make_param_puzzles.py`, rejection-sampled so each puzzle has a
   *verified* optimal knob) and feedback-driven cold-start trajectories
   (`scripts/make_puzzle_trajectories.py`): retrieve with default → observe the
   answer is absent → retry with the tuned parameter.

   Two things surfaced:
   - **k1 cannot be isolated in BM25.** A term unique to the gold has high idf and
     ranks it #1 at every k1, so k1-puzzles never pass verification. Only **b** is
     physically isolable (long gold buried by short subset distractors, rescued by
     lowering b), and only within a narrow, hand-tuned window.
   - **Negative result.** The search student did **not** reproduce the
     retry-and-lower-b behaviour — not even on the **training** puzzles
     (retry rate 0/12, EM 3/12 = only the easy controls), and identically on
     held-out puzzles (all three students: EM 3/6, retry 0, no `b` emitted).

## Competing explanation (not ruled out)

The report attributes the failure to a missing *observable* signal. A second
mechanism is equally consistent with the same evidence and was not separated:

Until 2026-07-28 the `search` policy appended a **fixed template** to the
trajectory's reasoning that quoted the gold's retrieval rank — "A default search
ranks the most on-topic passage around position {n}; with k1=… it moves to
position {m}". No student can produce that sentence at inference time, so the
data taught a phrasing rather than a policy, and the natural failure mode is to
memorise the surface form and emit constants — exactly what §4 observed. The
template is now replaced by query-shape justifications the student can actually
reach (`_search_params`), and the same leak existed on `grep_workspace` patterns
(now blocked by default; `ColdStartConfig.grep_may_leak_answer`).

**These two explanations are distinguishable and the experiment has not been
re-run since the fix.** Re-running §5 on the repaired generator is the minimum
needed before "`k1/b` are unlearnable by cold-start" can be stated as a finding
rather than a hypothesis. A third confound also stands: backward tracing only
ever emits *verified-successful* calls, so the corpus contains almost no
failure→repair pairs to imitate regardless of observability.

## Why it fails

Feedback-driven parameter adaptation needs the model to **recognise that a
retrieval failed**. With synthetic tokens there is *no observable signal*: the
query is meaningless to the model, and the first-retrieval distractors contain
2/3 of the query tokens so they *look relevant* — the model cannot tell the answer
is missing (it does not know the answer). With no observable trigger, the student
cannot learn a conditional "when the search fails, lower b" policy, so it collapses
to the simplest branch (single retrieve → answer).

## Findings

| finding | evidence |
|---|---|
| `k1/b` are weak, coupled BM25 levers | k1 un-isolable (idf-dominated); b needs precise length/idf construction |
| teachers do not self-tune params (not capacity) | Qwen3-4B and 8B both 0/N |
| **`top_k` is the observable, learnable lever** | search student learned varied `top_k` (1/3/5/10) |
| feedback-driven `k1/b` needs an observable failure signal | puzzle student: retry 0 even on train |
| small model + few examples → collapse/memorise, no generalisation | retry 0 on held-out |

## Recommendation

- **Do not rely on SFT cold-start to teach `k1/b` adaptation.** Scope the adaptive
  action space to **`top_k` / `merge`-vs-`replace` / query reformulation** — these
  are observable, learnable, and `top_k` was demonstrably learned.
- **`k1/b` belong to RL**, if kept at all: the RL retrieval reward (did the answer
  land in a small workspace?) supplies the failure signal the model cannot observe
  on its own. Cold-start's job is only to **seed the parameter keys** into the
  action space (search does this).
- This refines the SFT→RL prerequisite: it is not enough for SFT to give non-zero
  parameter support — the *when-to-adjust* signal must come from the reward.

## Reproduce

```bash
# policy comparison on a parameter-sensitive corpus
python scripts/make_smoke_corpus.py --out-dir .smoke_hard --hard 5 --build-index
python scripts/compare_param_policies.py --index-dir .smoke_hard/bm25_index \
    --questions .smoke_hard/questions.jsonl

# synthetic parameter-puzzle testbed + feedback-driven cold-start
python scripts/make_param_puzzles.py --out-dir .puzzles --n-keep 18 --heldout 6
for p in teacher heuristic search; do
  python scripts/make_puzzle_trajectories.py --puzzles-dir .puzzles --policy $p \
      --split train --out .puzzles/traj_$p.jsonl
  python scripts/run_sft_local.py --base Qwen/Qwen3-1.7B \
      --trajectories .puzzles/traj_$p.jsonl --out .puzzles/ckpt_$p/huggingface --epochs 4
  python scripts/run_scaleseek_smoke_eval.py --model .puzzles/ckpt_$p/huggingface \
      --index-dir .puzzles/bm25_index --questions .puzzles/questions_heldout.jsonl \
      --out .puzzles/eval_$p.jsonl
done
python scripts/analyze_puzzle_eval.py omit=.puzzles/eval_teacher.jsonl \
    heuristic=.puzzles/eval_heuristic.jsonl search=.puzzles/eval_search.jsonl
```

## Addendum (2026-07-30): outcome-level config race on the real corpus

The report above measured whether students *learn* k1/b adaptation. A stronger
test is whether k1/b variation even *changes outcomes* on wiki-18 — implemented
as `--param-policy race` (`train/sft/bm25_race.py`): backward pass shared, the
(k1,b) grid races full forward trajectories, evidence support picks the winner,
and hops whose default-param retrieval fails the judge get a config-rescue pass.

50-question probe (35B-A3B teacher, full 21M-passage index, job 18787433),
against the strict baseline (job 18771352, same questions):

| | strict/heuristic | race (width 4) |
|---|---|---|
| ok | 7 (14%) | 9 (18%) |
| skipped | 36 | 35 |
| non-default winners | — | **0 of 9** |
| gain_over_default | — | **0.0 on every arm** |
| preference pairs (margin ≥ 0.5) | — | 0 |
| cost per usable trajectory | 63.1 s | 61.9 s |

**Every race was won by the default config with zero gain.** The +2 ok is
within run-to-run variance (vLLM batching; no arm showed any gain to attribute
it to). This is the third independent confirmation that k1/b are dead levers on
wiki-18, and the first at the outcome level: ~100-word passages saturate TF so
fast that (k1,b) reordering does not change whether the gold form lands in a
top-5 workspace.

It also localizes the 70% skip wall: rescue can only reorder candidates for the
*same query*, so if the teacher's query lacks the right terms, no parameter
setting injects the missing recall. The wall is a query/decomposition problem.

Consequences:
- **wiki-18 production SFT data: use strict + heuristic.** The race costs the
  same per usable trajectory but its selection adds nothing here.
- **The race machinery stays** as dormant capability for long-document corpora
  (e.g. BrowseComp-Plus, should the experiment plan point the training track
  there — no such step is currently scheduled): full web pages
  are exactly the long-document regime where TF saturation matters. Correction
  (later same day): the teens-k1 recollection DID verify — against Pi-Serini
  (arXiv:2605.10848), not RISE. Pi-Serini runs BCP with tuned k1=25, b=1
  (grid-search optimum near k1=16, b=1.0; Anserini's default sits in a
  low-performing region; +18.0% answer accuracy from tuning alone, §6). The
  grid's long-document arms are now (16, 1.0) and (25, 1.0) accordingly —
  note high k1 pairs with HIGH b there. RISE, for contrast, uses bm25s
  defaults k1=1.5, b=0.75. Both wiki-18's parameter deadness (measured here)
  and BCP's parameter sensitivity (Pi-Serini's Table 3) are the same
  mechanism read off two corpus regimes: uniform ~102-word chunks leave b
  nothing to normalize and k1 nothing to saturate; ~2k-token documents make
  both levers live.
- Raising wiki-18 yield means improving backward queries/decomposition or
  question selection, not retrieval parameters.
