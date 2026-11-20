# Silent adapters — plan (written before any experiment, 2026-11-16)

> Note for this release: the wording of this file was tidied (the names of the methods, and who
> is speaking). The rules, the numbers, the verdicts and the times are as they were written at
> the time.

This file is written **before** running anything, so that the result cannot change what counts as
success. Do not edit the "Success and failure" section after the first run; add notes below it instead.

## The question

Can a LoRA adapter be trained to **change nothing on inputs that are not its task**, using only
stand-in inputs (no old-task data, no old-task statistics, no gate, no router), well enough that many
adapters can be kept switched on and added together without forgetting?

## Why this is a real gap (literature read on 2026-11-16)

| Paper | How it protects old tasks | What it must keep from old tasks | At inference |
|---|---|---|---|
| GainLoRA (NeurIPS 2025, arXiv 2505.15424) | A gate per new branch, built and constrained so its output stays 0 on old-task inputs | Subspace bases computed from old-task gate inputs, and old gate layers | No task ID; all branches on, weighted by gates |
| SPACE-LoRA (arXiv 2609.34453) | Projects the adapter so it gives zero response on protected input directions | Per-module activation and Fisher statistics computed from old-task data | No gate or router; adapters summed |
| Latent-LoRA (Findings of EMNLP 2026, arXiv 2607.23837) | Penalty that keeps new adapters orthogonal to old ones, plus a training-free selector | Old adapters, and a Gaussian mixture of each task's embeddings | No task ID; selector blends adapters |
| InfLoRA, O-LoRA | Keep the new adapter in directions old tasks do not use | Old-task subspaces or old adapters | Adapters summed |

The key sentence is in GainLoRA. It wants the new branch to contribute zero on old-task inputs, and
says that optimising this directly is "impractical" because no real or synthetic old samples are
available. Every method above therefore keeps something computed from old tasks.

**Our claim to test:** direct optimisation becomes practical if the "not my task" inputs are
stand-ins that need no old task at all. None of the three papers tries this (checked in the parts of
each paper that were read; GainLoRA Appendix C, Latent-LoRA Appendix B and SPACE-LoRA sections 5.3–5.4
were not read in full and must be checked before any claim of novelty is written down).

## The method, exactly

For task t, with the base model and all earlier adapters frozen:

1. Add one LoRA adapter (same rank and modules as the baselines).
2. Task loss: the usual loss on task t's data.
3. Silence loss: on stand-in inputs, the model's output with the new adapter must equal its output
   without it (KL between the two next-token distributions, gradients only through the new adapter).
4. Stand-in inputs, none of which use any other task of the benchmark:
   a. task t's own inputs with the task definition replaced by a definition from a public pool of
      unrelated tasks;
   b. unrelated text with an unrelated definition.
5. At inference all adapters are on and added. No gate, no router, no task ID.

Nothing is stored about old tasks except their frozen adapters.

## Stages

**Stage 0 — can we run the published method at all?** Run the official GainLoRA code
(github.com/liangyanshuo/gainlora) on one task order. Its own scripts use T5-large, 100 epochs on
SuperNI and an 80 GB A100; this does not fit a free Colab T4. So stage 0 uses a reduced setting
(smaller T5, fewer epochs) and compares methods only against **our own reruns in the same setting**,
never against the published numbers.

**Stage 1 — first comparison (reduced setting, Colab T4).** SuperNI order 1, 15 tasks. Methods, all
rerun by us: IncLoRA (adapters added, no protection), InfLoRA, GainLoRA (InfLoRA), ours. 3 seeds.
Metrics as in the papers: AP (average final score over tasks) and FT (average drop from best to
final).

**Stage 2 — the published setting.** Only if stage 1 passes. T5-large, all four orders, then
Llama-2-7B. This needs rented GPUs (roughly hundreds of GPU-hours); free Colab cannot do it.

## Success and failure (fixed now)

Stage 1, mean of 3 seeds, same setting for every method:

* **Pass:** ours is within 1.0 AP of GainLoRA **and** its FT is no more than 1.0 worse.
  That would mean stored old-task information and the gate are not needed.
* **Strong:** ours beats GainLoRA by at least 1.0 AP with FT no worse.
* **Partial:** ours beats InfLoRA by at least 1.0 AP but is more than 1.0 AP below GainLoRA.
  Worth a note, not a paper.
* **Fail:** ours is not better than InfLoRA by 1.0 AP. Stop; write down why.

One ablation is fixed in advance: ours with only stand-in (a), and with only stand-in (b).

## Known risks

1. Adapters add inside the weights, so two adapters that are each silent alone can still disturb each
   other where both act.
2. Several benchmark tasks are close to each other (several sentiment tasks). Stand-ins may not teach
   silence on a near-identical task.
3. Silence may cost accuracy on the adapter's own task.
4. The current best published method (Latent-LoRA) reports almost no forgetting on SuperNI
   (0.01). Matching GainLoRA would not be enough to beat the state of the art.
5. Compute: stage 2 cannot be done on free Colab.

## Numbers to compare against later (not yet verified against the PDFs)

T5-large, SuperNI order 1 / order 2, AP: O-LoRA about 26–28 / 33; InfLoRA about 39–40 / 40;
GainLoRA about 46–48 / 46–47; Latent-LoRA 48.60 / 49.75. Two papers report slightly different numbers
for the same baselines, because each reran them.

---

## Notes added after the plan was fixed

### 2026-11-17 — an exploratory short test, written down before it is run

Purpose: a quick look at whether silence training does anything on a real model, before spending
about 40 GPU-hours on stage 1. **It is not stage 1.** It cannot pass or fail the rules above: it has
one seed and five tasks.

Fixed before running:

* **Tasks, in this order:** relation extraction (task1510), sst2 (task363), glucose (task748),
  sentiment140 (task1687), emotion (task875). All have short inputs. The first three are the
  short-input tasks InfLoRA forgot most in the 15-task run (section 4I of the main README); the last
  two are of the kind after which that forgetting happened.
* **Setting:** as stage 0 (t5-base, 10 epochs, seed 42, effective batch 32, rank 4).
* **Methods:** `plain` (the official plain-adapter mode `olora` with both of its penalties set to 0),
  `silent` (ours: `plain` plus the silence loss), `inflora`, `gainlora_inflora`.
* **Silence loss:** weight 1.0; 400 stand-in inputs per task, half of each kind, cut to 256 tokens; 8
  per training step; the target is the model's own greedy answer before the adapter is trained; KL
  from the model without the new adapter to the model with it, dropout off in both passes.
* **Unrelated tasks for the stand-ins:** the Long_Sequence benchmark in the same repository, without
  its sentiment tasks (yelp, amazon, imdb, sst2) and without copa (it has no definition): mnli, cb,
  qqp, rte, dbpedia, agnews, yahoo, multirc, boolq, wic.
* **What will be read from it:** `silent` against `plain` (does silence reduce forgetting, and what
  does it cost in learning?), then both against the two published methods. Any outcome is a hint
  only.

Known weak points of this test: one seed; test sets of about 100 examples; the setting is
under-trained (section 4I); two of the five tasks are sentiment-like, which is the hardest case for
stand-ins that contain no sentiment task.

### 2026-11-17 — two changes after the short test, written down before they are run

The short test (main README, section 4J) showed that silence training costs no learning and reduces
forgetting but does not stop it. Two changes follow. Both are exploratory, as before.

**`locked`: an exact switch.** Training is exactly `plain`. Every input of this benchmark starts with
its task definition. When a task is learned, the tokens of its definition are stored as a key (in
`saved_weights/task_key.pt`). An input whose first tokens equal a stored key is answered with every
adapter added after that task switched off; an input that matches no key is answered with all
adapters on. For inputs with a known definition the model therefore computes what it computed right
after that task was learned, so forgetting is zero by construction, for any number of later tasks.

* What must hold in the run, or the implementation is wrong: every task scores the same at the end
  as right after it was learned; no test input is left unmatched; scores right after learning equal
  those of `plain` (same seed, same training).
* What it is not: it is a switch keyed on the task description, the thing the earlier steps tried to
  avoid; using the task's identity is a known upper bound, not a new method; and it does nothing for
  an input whose definition is worded differently. It is not a candidate for the claim in this plan.
  Its use here is to show the ceiling (`plain`'s learning with no forgetting) and as a base for a
  later test with reworded definitions.

**`silent2`: more varied stand-ins.** Same as `silent`, with 800 stand-in inputs instead of 400 and a
third kind: the task's own input under a "definition" that is a random piece (8 to 40 words) of
unrelated text. Half of the stand-ins are of this kind. Reason: `silent` saw only 10 different wrong
definitions and may have learned to be silent on those and not in general. Not yet run on a GPU.

### 2026-11-17 — `certified`: a switch that does not need the exact wording, written down before the GPU run

`locked` (above) forgets nothing, but it recognises a task only from the exact text of its
definition. `certified` replaces the exact match by a rule on frozen word-vectors and keeps the
guarantee where it can be proved.

**The rule.** Word-vectors are rows of the base model's embedding table, which is frozen. For an
input, two averages are taken: over its first *n* tokens and over all its tokens. Both are centred and
rescaled once, with numbers computed from 4,000 unrelated inputs (the Long_Sequence tasks already used
for the stand-ins), so that unrelated inputs vary by about the same amount in every direction.

* When task *k* is learned, it stores: the length *n_k* of its fixed opening (its definition); its
  **signature**, the average word-vector of that opening; a **radius**, 0.5 times the distance from
  the signature to the nearest signature already known (earlier tasks and the ten unrelated tasks);
  and its **centre**, the average over its training inputs of the whole-input average. The ball
  around the signature is the task's **protected zone**. Nothing stored is ever changed.
* An input is given to the **oldest** task whose protected zone contains the average of its first
  *n_k* tokens. If no zone contains it, it is given to the task with the nearest centre.
* An input given to task *k* is answered with the adapters of tasks 1 to *k*; later adapters are
  multiplied by zero, as in `locked`. Training is exactly `plain`.

**What is guaranteed, and why.** If an input is inside some zone when *T* tasks exist, the task it is
given to is the oldest zone containing it. Later tasks only add newer zones, so that task never
changes, the adapters used never change, and they are frozen: the model's answer on that input is the
same for ever, whatever is learned later. This holds for any model and any later tasks, with no
assumption about them. It does **not** hold for inputs outside every zone: the nearest centre can
change when a task is added. The price is visible at once and not later: an input of a new task that
falls inside an older zone is answered by the older adapters, so the new task is not learned on it.
The code counts such training inputs when each task is added.

**What was known before the GPU run (CPU only, real benchmark inputs, t5-base word-vectors;
`routing_check.py`).** Where the inputs go, not what the model answers:

| Sequence, wording | inside own zone | given to own task | inside a wrong zone | route changed later |
|---|---|---|---|---|
| five tasks, original | 100% | 100% | 0 | 0 of 500 |
| five tasks, light rewording | 99.8% | 100% | 0 | 0 of 500 |
| five tasks, medium rewording | 20% | 100% | 0 | 0 of 500 |
| five tasks, heavy rewording | 20% | 97.4% | 0 | 1 of 500 |
| fifteen tasks (SuperNI order 1), original | 100% | 100% | 0 | 0 of 1,197 |

The factor 0.5 was chosen after seeing these routes (0.3 certifies fewer reworded inputs, 0.7 more;
none of the three put any input in a wrong zone). No model answer was looked at to choose it. The
reworded definitions (three per task: light, medium, heavy) were written by machine for
this test and keep the answer format of the original.

**The GPU test (five tasks, t5-base, 10 epochs, seed 42, as before).** One training run. After every
task, each test input is answered again under four rules with the same trained adapters, with the
original definition and the three rewordings: `certified`; `nearest` (nearest centre only, no zones);
`key` (the exact switch of `locked`); `all` (every adapter on, which is `plain`). Scores in this
re-test use every reference answer, so they do not have the evaluation noise found in 4K.

What must hold, or the implementation is wrong:

1. Original wording, `certified`: 0 answers changed between "right after learning" and the end;
   every input inside a zone; same answers as `key`; the re-test reproduces the official test.
2. Any wording: no answer changes for an input that was inside a zone.

What is expected but not certain (written before the run):

3. Under light, medium and heavy rewording, `key` matches nothing and behaves like `all`: forgetting
   close to `plain`'s (more than 30 points).
4. Under the same rewordings `certified` gives forgetting of at most 5 points, and at least 95% of
   inputs go to their own task.
5. `nearest` gives the same routes as `certified` on these inputs. If so, the zones were never
   needed here and their value in this test is only the guarantee, not a better score.

What would count against the idea: forgetting under rewording of more than half of `plain`'s; or
scores right after learning that collapse under rewording (the adapter itself depends on the exact
wording), which would mean that recognising the task is not the bottleneck.

**What it is not.** Recognising the task from frozen pooled word-vectors, with no trained router, is
published (Latent-LoRA, arXiv:2607.23837, fits a mixture model to them and blends adapters).
Recognising a task from out-of-distribution detectors is an established line of work (Kim et al.,
2022). What two literature searches on 2026-11-17 did not find is the combination used here: zones
that are never changed, the oldest zone wins, a proof that answers inside a zone cannot change, and
a flag on every answer saying whether it is covered. That is a small addition to known parts, not a
breakthrough, and the searches may have missed a paper. Open weaknesses: one seed, five tasks,
rewordings written by machine, no comparison with InfLoRA, GainLoRA or Latent-LoRA on these tasks,
no test with a later task that is deliberately close to an old one (the case the zones are for), and
no second model.

### 2026-11-17 — result of the GPU test of `certified`, read against the rules above

Colab T4, five tasks, t5-base, 10 epochs, seed 42, radius factor 0.5; numbers and table in the main
README, section 4L; raw output in `results/2026-11-17_certified_five_tasks_colab_T4.txt`.

| Rule written before the run | Result |
|---|---|
| 1. Original wording: 0 answers changed; every input inside a zone; same answers as `key`; the re-test reproduces the official test | **held**: 0 of 400; 500 of 500; same scores and same routes as `key`; 100/100 up to 500/500 |
| 2. No answer changes for an input that was inside a zone, in any wording | **held**: no answer changed at all (0 of 400 in each of the four wordings) |
| 3. Expected: under rewording `key` behaves like `all`, forgetting above 30 | **as expected**: identical to `all`; FT 57.90 / 56.11 / 55.61 |
| 4. Expected: under rewording `certified` has forgetting of at most 5 and at least 95% of inputs reach their own task | **as expected**: FT 0.00 three times; 100%, 100%, 97.4% |
| 5. Expected: `nearest` gives the same routes as `certified` | **as expected**: identical scores in every row |
| Against the idea: forgetting under rewording above half of `plain`'s | did not happen (0.00 against about 57) |
| Against the idea: scores right after learning collapse under rewording | did not happen (79.33 → 79.15 / 77.61 / 76.72 with the right adapters) |

What this result is and is not. It is one seed on five tasks, so it is an early result that has to be
repeated. By rule 5 the gain over plain adapters on this sequence comes from two known ingredients,
frozen adapters and recognising the task from frozen word-vectors; the protected zones contributed
the proof and the flag, not score. Nothing was compared with InfLoRA, GainLoRA or Latent-LoRA on these
tasks yet.

**Next runs.** (This paragraph first said that nothing in the method would be changed for them. That
was written before the CPU check in the next section, which led to one change of the rule for inputs
outside the zones. The next runs use the final rule, `certified2`.)

* *Published methods on the same five tasks* (`quick_test_colab.ipynb`): InfLoRA and GainLoRA,
  official code unchanged (`SILENT_BASELINE=1` only notes the test set and repeats the test with the
  reworded definitions afterwards; the re-test must reproduce the official test on every input).
  Read-out: learned, AP, FT and answers changed for each wording, next to `certified`. A fair
  statement can only be about this reduced setting; 4I showed that it under-trains both methods.
* *`certified2` on the 15 tasks of SuperNI order 1* (`certified15_colab.ipynb`), to set next to 4I
  (InfLoRA: AP 25.85, FT 5.66; GainLoRA after ten tasks: AP 28.48, FT 1.24). The CPU check predicts
  that all 1,197 test inputs are inside their own zone. What must hold, or the implementation is
  wrong: 0 answers changed for inputs inside a zone; the re-test reproduces the official test. What
  is expected: FT 0.00 and AP equal to the average score right after learning. What is not known:
  how well plain adapters learn the ten tasks not seen so far. Counting against the idea: any
  training input inside an older zone (a blocked task), or a route that changes for more than 1% of
  test inputs.

### 2026-11-17 — the rule for inputs outside every zone was changed after a CPU check, before any further GPU run

**What the check showed.** The first GPU run used "outside every zone: nearest centre of the whole
input", with every token counting equally. Its routes had been checked with reworded definitions on
the five short tasks only. The same check on the 15-task sequence (routes only, CPU, t5-base
word-vectors) showed that this fallback fails when tasks with long inputs are present: of the 500
reworded test inputs only 70.2% (medium) and 60.8% (heavy) reached their own task, and 72 and 100
routes changed as later tasks were added. Reason: when the definition of a short input changes, the
average over the whole input moves towards the centres of long-text tasks. Inputs inside a zone were
not affected (original and light wording: 100% right in both sequences).

**The final rule (`certified2`).** The zones, "oldest zone wins" and the proof are unchanged. Two
things changed:

1. An input outside every zone goes to the **nearest zone**: the task with the smallest distance
   between the input's opening average and the task's signature, divided by that zone's radius. The
   whole-input centre is no longer part of the method. (It is still computed, as rule `centre` in the
   re-test, so that the two fallbacks can be compared.)
2. In every average, a token counts with weight a / (a + f), where f is its share among the tokens
   of the 4,000 unrelated inputs and a = 0.001, so that common words count less. This is the usual
   weighting for averaged word-vectors (Arora, Liang and Ma, ICLR 2017) with its usual constant;
   a = 0.0001 gave nearly the same routes and nothing else was tried.

**Routes with the final rule** (CPU, `routing_check.py`, radius factor 0.5). "Centre" is the old
fallback with the new weights:

| Sequence, wording | inside own zone | right task: `certified2` | routes changed later | right task: `centre` | routes changed later |
|---|---|---|---|---|---|
| five tasks, original | 100% | 100% | 0 | 100% | 0 |
| five tasks, light | 100% | 100% | 0 | 100% | 0 |
| five tasks, medium | 20% | 100% | 0 | 99.2% | 0 |
| five tasks, heavy | 19.6% | 93.8% | 0 | 72.2% | 58 |
| fifteen tasks, original (1,197 inputs) | 100% | 100% | 0 | 99.5% | 2 |
| fifteen tasks, light (500 inputs) | 100% | 100% | 0 | 100% | 0 |
| fifteen tasks, medium | 20% | 100% | 0 | 85.2% | 29 |
| fifteen tasks, heavy | 19.6% | 93.8% | 0 | 59.2% | 100 |

No input was inside the zone of another task in any row. "Nearest zone only" (no priority for
older zones) gave the same routes as `certified2` in every row.

**What this means for the claims.**

* The guarantee is untouched. For the original wording nothing changes either: every input is inside
  its own zone under both versions.
* The five-task GPU result above was obtained with the first version and stays as reported. On five
  tasks the final rule is expected to be slightly worse under heavy rewording (93.8% of inputs to
  the right task instead of 97.4%) and equal elsewhere. The five-task run is repeated with the final
  rule (method `certified2`, about 40 minutes) so that all reported numbers come from one rule.
* The radius factor and these two changes were all chosen while looking at the routes of the same
  test inputs and the same three machine-written rewordings (never at an answer of the model). **The rule
  is frozen from here.** A claim about rewording in a paper needs rewordings written by someone else
  after this date, and they must not be used to adjust anything.

**Rules for the `certified2` runs** (five tasks and fifteen tasks). Must hold, or the implementation
is wrong: with the original wording 0 answers changed and every input inside its own zone; no answer
changes for an input inside a zone, in any wording; the re-test reproduces the official test.
Expected: with reworded definitions forgetting of at most 5 points; 100%, 100% and about 94% of
inputs to the right task (light, medium, heavy); `key` equal to `all`; `centre` clearly worse than
`certified2` on the fifteen-task sequence under medium and heavy rewording. Counting against the
idea: any training input inside an older zone; forgetting under rewording above half of `plain`'s;
on fifteen tasks, an average score right after learning for plain adapters that is not above
InfLoRA's 30.21 (then the comparison with 4I says nothing in favour of this approach).

### 2026-11-17 — result of the second GPU run (`certified2` and the published methods, five tasks), read against the rules

Kaggle T4, t5-base, 10 epochs, seed 42; numbers in the main README, section 4L; raw log in
`results/2026-11-17_certified2_and_published_five_tasks_kaggle_T4.txt`.

| Rule for `certified2` written before the run | Result |
|---|---|
| Original wording: 0 answers changed, every input inside its own zone | **held**: 0 of 400; 500 of 500 |
| No answer changes for an input inside a zone, in any wording | **held**: no answer changed at all (0 of 400 in each wording) |
| The re-test reproduces the official test | **held**: 100/100 up to 500/500 |
| Expected: forgetting of at most 5 under rewording | **as expected**: 0.00 three times |
| Expected: 100%, 100%, about 94% of inputs to the right task (light, medium, heavy) | **as expected**: 100%, 100%, 93.8% |
| Expected: `key` equal to `all` under rewording | **as expected** |
| Against: a training input inside an older zone | did not happen (0 of 800 for every task) |
| Against: forgetting under rewording above half of `plain`'s | did not happen |

The published methods on the same five tasks: InfLoRA learned 56.69, AP 58.37, FT −1.10; GainLoRA
learned 60.30, AP 61.05, FT −0.69; `certified2` learned 78.70, AP 78.88, FT 0.00. Their re-test
reproduced the official test on 500 of 500 inputs (InfLoRA) and 498 of 500 (GainLoRA, batch size 32
against 16; fixed afterwards). **Reading fixed here, before more runs:** the difference to the
published methods on these tasks is a difference in learning within ten epochs, not in forgetting;
no run so far says anything about the published setting.

**The 15-task run, fixed before it is started** (`certified15_colab.ipynb`, three methods through the
same code): rules for `certified2` as written above. Checks on the published methods: InfLoRA should
come out close to its Colab run of 4I (official scores: learned 30.21, AP 25.85, FT 5.66; same
code, seed and kind of GPU), otherwise the two platforms differ and that has to be understood
first; their re-tests must reproduce their official tests. The comparison to report: learned, AP
and FT of the three methods side by side, with the statement of where a gap comes from. Counting
against the certified switch: an AP for `certified2` that is not clearly above InfLoRA's and
GainLoRA's; any blocked training input; any changed answer inside a zone.

### 2026-11-17 (evening) — a mechanical edit test of the routes, fixed before it is run

**Why.** The three rewordings per task were written by machine and were looked at while the
rule was designed, so they are a development set. Until a person writes new ones, a second view
that needs nobody's wording: edit the instruction mechanically (typos, dropped words, replaced
words, lower case, an extra sentence in front, a sentence removed, sentences shuffled) and look at
the routes. Routes only, CPU, T5-base word-vectors, the frozen rule (`certified2`, radius factor
0.5, word weights 0.001), the fifteen tasks in official order 1, every test input of every task.
Nothing of the rule may be changed after this test; it is reported as it comes out.

**Edits** (of the definition only; the instance is not touched; each test input gets its own random
edit, random seed 0):

- `typo-k`, k = 1, 2, 4, 8, 16: in k different words of at least four letters, two neighbouring
  inner letters are swapped.
- `drop-k`, k = 1, 2, 4, 8, 16: k different words are deleted.
- `replace-k`, k = 1, 2, 4, 8, 16: k different words are replaced by words drawn at random from the
  definitions of all fifteen tasks.
- `lower`: the definition in lower case.
- `polite`: the sentence "Please help me with this task." in front of the whole input.
- `no-first`, `no-last`: the first or the last sentence of the definition removed (tasks whose
  definition has one sentence are left out, and the number of tasks is reported).
- `shuffle`: the sentences of the definition in a random order (same tasks).

**Measured** for every edit: share of inputs inside their own zone (certified), number of inputs
inside the zone of another task (certified for the wrong task), share given to the right task,
and the worst task.

**Must hold, or the implementation is wrong:** without an edit, every input inside its own zone.

**Expected** (a guess written before the run): with one typo or one dropped word at least 90% of
the inputs stay certified; the right task for at least 99% of the inputs for every edit with
k up to 4, and for at least 95% for every other edit.

**Counting against the switch:** any edited input inside the zone of another task; a right-task
share under 95% for an edit with k up to 4; a right-task share under 80% for any edit.

**Also computed (no rule, a number for the paper):** the worst-case bound on edits. If an edit
changes the tokens that carry a share `l` of the weight of the opening before the edit and `l'`
after it, the input moves by at most `(l + l') D`, where `D` is the largest distance between two
rows of the rescaled word-vector table. So an edit with `l + l' <= R_k / D` can never leave the
zone. `D` and `R_k / D` are reported next to the measured shares; the bound is expected to be far
more cautious than the measurement.

**Result of the mechanical edit test, read against the rules above** (run once, right after the rules
were pushed; `paper_certified/edit_analysis.py`, numbers in `paper_certified/routing/edits.json`;
1,197 test inputs per edit, 23,640 edited inputs in total).

| Written before the run | Result |
|---|---|
| Must hold: without an edit every input inside its own zone | **held**: 1,197 of 1,197 |
| Expected: one typo or one dropped word leaves at least 90% certified | **not met for typos**: 87.1% (one typo); 93.7% (one dropped word) |
| Expected: right task for at least 99% for every edit with k up to 4 | **not met for four typos** (98.1%); all others 99.6% to 100% |
| Expected: right task for at least 95% for every other edit | **not met** for 8 typos (82.7%), 16 typos (45.4%), 16 dropped words (87.0%), 16 replaced words (85.9%), last sentence removed (81.5%); met for 8 dropped (95.5%), 8 replaced (95.7%), lower case, sentence in front, first sentence removed, sentences shuffled (100% each) |
| Against: an edited input inside the zone of another task | **did not happen**: 0 of 23,640 |
| Against: right-task share under 95% for an edit with k up to 4 | did not happen (lowest 98.1%) |
| Against: right-task share under 80% for any edit | **happened**: 16 typos, 45.4% |

Share of inputs still certified (k = 1, 2, 4, 8, 16): typos 87.1 / 74.4 / 37.8 / 10.7 / 0.3; dropped
words 93.7 / 87.6 / 73.1 / 47.5 / 17.0; replaced words 94.6 / 89.0 / 77.4 / 49.3 / 16.5. Lower
case 91.4, sentence in front 91.6, first sentence removed 24.7, last sentence removed 32.8,
sentences shuffled 100.

**Reading.** (1) The certificate was never given for a wrong task. (2) The route of uncertified
inputs is not robust to heavy damage: with sixteen typos fewer than half of the inputs reach
their task, and this counts against relying on the nearest-zone route for badly damaged
instructions. (3) Robustness follows the length of the instruction: tasks with openings of at
least 90 tokens keep 98% to 100% right even with sixteen typos, the two shortest (sciq with 26
tokens, a ten-word definition, and xsum with 34) fail first. (4) Removing the last sentence
breaks sst2 (2% right) and emotion (8%): for both, the removed sentence is the one that states
the task. (5) Shuffling sentences changes nothing, because the feature is an average over the
first tokens and ignores their order; that is a property to state as a limit, not a strength.
(6) Worst-case bound: the largest distance between two rows of the rescaled table is 509.6, so
edits that touch 2.0% to 4.2% of the weight of an opening (before plus after) can never leave
the zone; as expected this is far more cautious than the measurement. **Nothing of the rule is
changed.**

### 2026-11-17 (evening) — how much do the routes depend on the unrelated inputs? Fixed before it is run

The frozen rule uses unrelated inputs in three places: how common each token is (word weights),
the scale of distances, and the signatures of the unrelated tasks, which set most radii (13 of 15
in the order analysis). A reader will ask whether the results depend on this particular choice. A
sensitivity check, routes only, CPU, T5-base, both sequences, **nothing of the rule is changed by
it, whatever comes out**:

- the 4,000 unrelated inputs drawn with random seeds 1 and 2 instead of 0;
- 500, 1,000 and 2,000 unrelated inputs instead of 4,000;
- only five of the ten unrelated tasks (the first five, the last five), so that five of the
  unrelated signatures are missing and radii can become larger.

Reported for each: share of original inputs inside their own zone, inputs in a wrong zone,
blocked training inputs, share of inputs to the right task for the four wordings, and the
smallest and largest radius. Expected: seeds and sizes change little; with half of the unrelated
tasks the radii grow and more reworded inputs are certified. Counting against the method: any
input in a wrong zone or any blocked training input in one of these settings (then the choice of
unrelated tasks is a hidden tuning knob and has to be said so).

**Result of the check of the unrelated inputs, read against the rules above** (run once;
`paper_certified/background_analysis.py`, numbers in `paper_certified/routing/background.json`).

| Unrelated inputs | Five tasks: certified, % (original / light / medium / heavy) | Right task under heavy rewording, % (five / fifteen tasks) | Wrong zone | Blocked | Radii (fifteen tasks) |
|---|---|---|---|---|---|
| as in the paper (seed 0, 4,000, ten tasks) | 100 / 100 / 20 / 20 | 93.8 / 93.8 | 0 | 0 | 10.1 to 21.3 |
| another draw (seed 1) | 100 / 100 / 20 / 19 | 94.2 / 94.2 | 0 | 0 | 10.0 to 21.2 |
| another draw (seed 2) | 100 / 100 / 20 / 20 | 93.0 / 93.0 | 0 | 0 | 10.0 to 21.3 |
| 500 inputs | 100 / 100 / 20 / 20 | 90.0 / 90.0 | 0 | 0 | 11.0 to 23.1 |
| 1,000 inputs | 100 / 100 / 20 / 18 | 91.8 / 91.8 | 0 | 0 | 10.6 to 22.4 |
| 2,000 inputs | 100 / 100 / 20 / 20 | 90.8 / 90.6 | 0 | 0 | 10.3 to 21.6 |
| only the first five unrelated tasks | 100 / 100 / 20 / 20 | 90.6 / 90.2 | 0 | 0 | 10.0 to 20.7 |
| only the last five unrelated tasks | 100 / 100 / 20 / 20 | 97.8 / 97.6 | 0 | 0 | 11.1 to 20.1 |

In every variant all inputs with the original and the light wording are certified, all inputs with
the original, light and medium wording reach their own task, no input is in a wrong zone and no
training input is blocked. **Counting against the method: did not happen.** Expected and not
found: with half of the unrelated tasks the radii did not grow and no more reworded inputs were
certified (the scale of distances changes together with the signatures). Expected and only
partly found: other draws change little (93.0 to 94.2), but fewer inputs or other unrelated tasks
move the share of heavily reworded inputs that reach their task between 90.0% and 97.8%.
**Reading:** the certified part does not depend on the choice of unrelated inputs in these
variants; the number for heavy rewording does, by several points, and 93.8% must be reported as
one value from this range. Nothing of the rule is changed.

### 2026-11-17 (evening) — the zones on many real tasks. Fixed before it is run

**Why.** Fifteen tasks say little about a system that keeps learning. The stress test built one
look-alike by hand. A large collection of real task instructions contains look-alikes by itself,
so it shows (a) how often a new task collides with an older zone, the price of the method, and
(b) how often a nearest-match rule would silently re-route inputs of older tasks, the case the
guarantee is for. Routes only, CPU, T5-base word-vectors, the frozen rule (`certified2`, radius
factor 0.5, word weights 0.001, the same unrelated inputs). **Nothing of the rule is changed by
this test, whatever comes out.**

**Data.** The tasks of the default English split of Super-NaturalInstructions
(github.com/allenai/natural-instructions, `splits/default/train_tasks.txt` and `test_tasks.txt`,
875 tasks), fetched at one fixed commit. For every task: its definition and its first inputs (up
to six instances; positive examples if no instance is in the fetched part of the file). Tasks with
fewer than two inputs are left out and counted. Inputs are built with the template of the
benchmark. The fifteen tasks of this project are part of that collection and are not treated
differently.

**Procedure.** The tasks arrive in a random order (seed 0); as a second view, in the order of their
task numbers; and in four more random orders (seeds 1 to 4) for the spread. Each task gets its zone
when it arrives, as in the method. Measured after 50, 100, 200, 400 and all tasks:

- tasks that arrive with no input inside an older zone (admitted without a collision), tasks with
  some, and tasks with all inputs inside an older zone;
- inputs given to their own task at the end, under `certified` and under `nearest`;
- inputs whose route changed after their task had arrived, under both rules, and how many of
  them were certified at the time (must be 0);
- for every collision: the older task that holds the inputs, and how alike the two definitions
  are (share of common words, counted as bags of lower-case words: common / larger bag).

**Must hold, or the implementation is wrong:** the routes of this implementation equal those of
`silent_run.py` on the fifteen tasks; no certified input changes its route.

**Expected** (a guess written before the run): at least 80% of the tasks arrive without a
collision; most collisions are between tasks whose definitions are nearly the same text (at least
80% common words); under `nearest` some inputs of older tasks are re-routed by later tasks.

**Counting against the switch:** more than 10% of the colliding tasks collide with a task whose
definition shares less than half of its words (then the zones are too greedy for a large
collection); fewer than 80% of all inputs given to their own task under `certified`.

**What this test cannot say:** nothing about answers (no training); the inputs are the first few
of each task, not a random sample; reworded instructions are not part of it.

**Result of the test on many real tasks, read against the rules above** (run once;
`paper_certified/many_tasks_fetch.py` and `many_tasks_analysis.py`, numbers in
`paper_certified/routing/many_tasks.json`; dataset commit `55a3656`; 875 tasks, all with at least two
inputs, 5,237 inputs; openings of 25 to 512 tokens, middle value 80; 13 pairs of tasks have exactly
the same definition text).

| Written before the run | Result |
|---|---|
| Must hold: routes equal to `silent_run.py` on the fifteen tasks | **held**: 294 of 294 routes, radii equal |
| Must hold: no certified input changes its route | **held** in all six orders and at every count of tasks |
| Expected: at least 80% of the tasks arrive without a collision | **as expected, narrowly**: 724 of 875 (82.7%) in the main order; 718 to 724 in the four other random orders; 777 (88.8%) in the order of the task numbers |
| Expected: most collisions are between tasks with nearly the same definition (80% common words or more) | **as expected**: of 151 colliding tasks, 9 have the same text as the older task, 114 share 80% or more of the words, 28 share 50% to 80%; lowest 63% |
| Expected: under `nearest` some inputs of older tasks are re-routed by later tasks | **wrong**: 0 of 5,237 |
| Against: more than 10% of the colliding tasks share less than half of their words with the task that holds them | did not happen (none) |
| Against: fewer than 80% of all inputs given to their own task under `certified` | did not happen, narrowly: 83.8% (82.6% to 83.8% over the random orders) |

Main order (random, seed 0), after 50 / 100 / 200 / 400 / 875 tasks: tasks without a collision
49 / 96 / 178 / 337 / 724; tasks with all inputs inside an older zone 0 / 1 / 15 / 50 / 129; inputs
given to their own task under `certified` 98.7 / 96.7 / 90.4 / 85.7 / 83.8%, under `nearest`
100 / 100 / 98.5 / 99.0 / 98.3%.

**Reading, including what the rules did not ask for.**
(1) The guarantee held everywhere.
(2) The price of the guarantee is real in a large collection: about one task in six (17%) arrives
inside an older zone, and 129 of 875 tasks cannot be learned at all under their given instruction.
The collisions come from families of tasks whose definitions differ in a word or two: 35 of the
151 are subject variants of one multiple-choice task (MMLU), 17 are relation variants of one
classification task (ATOMIC).
(3) **On unedited instructions the nearest-zone rule is the better rule.** An unedited input lies
exactly on its own signature, so no later task can be nearer; `nearest` gives 98.3% of the inputs
to their own task against 83.8%, and re-routes nothing. Our guess that it would re-route inputs here
was wrong, and for a reason we should have seen. Its weakness needs edited inputs (the stress test:
100 of 100 lightly reworded inputs taken by a later look-alike), and this test has none.
(4) So the honest statement of the trade is: the zones protect the neighbourhood of an old
instruction (its reworded forms) and take it away from later tasks that live there. Whether that
is the right trade depends on whether look-alike tasks or reworded instructions are the more
common event.

**Added after the run, not part of the rules** (same main order): other radius factors. Tasks
without a collision, and inputs given to their own task under `certified`: 0.1 → 856 of 875,
97.7%; 0.2 → 843, 96.3%; 0.3 → 827, 94.6%; 0.4 → 795, 91.0%; 0.5 → 724, 83.8%. A smaller factor
buys fewer collisions with smaller zones (the factor 0.3 certified only 59% of the lightly reworded
inputs of the five tasks, against 100% for 0.5). **The rule stays as frozen (0.5);** the table is
reported as the trade-off it is.

### 2026-11-17 (evening) — a remedy for the collisions, designed after the 875-task test. Written before it is run

**Status of this idea.** It was thought of *after* seeing that 17% of 875 real tasks collide with an
older zone. It is therefore not part of the frozen rule (`certified2`), and its numbers on the same
875 tasks are not an independent test. It is recorded here as a variant, and tested on routes only.

**The variant ("exact claims first").** An input that carries the opening of a task token for token
goes to that task; if several tasks have that same opening, to the oldest. Otherwise as before: the
oldest zone that contains the input, else the nearest zone. A task whose instruction lies inside an
older zone can then be learned: its own inputs carry its instruction exactly and go to it.

**What it guarantees, worked out before any run.**
(a) An input that carries a registered opening exactly keeps its route for ever, without
conditions: later tasks with the same opening are newer, and the oldest wins.
(b) An input inside a zone that carries no registered opening keeps its route for ever *unless*
a task is admitted later whose opening the input carries exactly. This is weaker than the frozen
rule (which has no "unless"), and it is the price of serving sibling tasks.
(c) The only tasks that still cannot be learned are those whose opening is identical to an older
task's opening.

**Checks (routes only, CPU, T5-base).** Must hold, or the implementation is wrong: no input with
an exact claim changes its route when later tasks arrive; on the five and on the fifteen tasks the
variant gives the same route as the frozen rule for every test input in every wording (then every
GPU result of the frozen rule holds for the variant unchanged). Measured on the 875 tasks, six
orders: tasks whose inputs go to themselves when they arrive; inputs given to their own task at
the end; routes changed later, split into inputs that had an exact claim (must be 0) and inputs
that were inside a zone without one. Stress test (a sixth task that looks like sst2): routes
changed, next to the frozen rule and the nearest-zone rule.

**Expected:** on the 875 tasks, about as many inputs to their own task as the nearest-zone rule
(98%); every task learnable except those with an identical opening; in the stress test the same
re-routing as the nearest-zone rule for the old inputs that carry exactly the new task's
instruction, and none otherwise. **Counting against the variant:** any failed check above; more
than a handful of zone inputs taken by later exact claims among the unedited inputs of the 875
tasks (that would mean real tasks overwrite each other through it).

**Result of the checks of the variant "exact claims first"** (run once;
`paper_certified/exact_first_analysis.py`, numbers in `paper_certified/routing/exact_first.json`).

| Written before the run | Result |
|---|---|
| Must hold: no input with an exact claim changes its route | **held**: 0 in all six orders and at every count of tasks |
| Must hold: same routes as the frozen rule on the five and on the fifteen tasks, every wording | **held**: 2,000 of 2,000 and 2,697 of 2,697 test inputs; only the original wording has an exact claim (500 and 1,197 inputs) |
| Expected: about as many inputs to their own task as the nearest-zone rule (98%) | **as expected**: 5,153 of 5,237 (98.4%) in every order; nearest zone 5,147; frozen rule 4,327 to 4,661 |
| Expected: every task learnable except those with an identical opening | **as expected**: 861 of 875 tasks get all their own inputs when they arrive; the other 14 have the same opening as an older task |
| Expected: in the stress test the same re-routing as the nearest-zone rule for old inputs that carry exactly the new instruction, none otherwise | **as expected**: light look-alike 0 / 100 / 0 / 0 (frozen rule 0 / 0 / 0 / 0); medium and heavy look-alike identical for the three rules. The look-alike's own 800 inputs go to it (frozen rule: 1 of 800) |
| Against: a failed check | none |
| Against: more than a handful of zone inputs taken by later exact claims among the unedited inputs of the 875 tasks | did not happen: 0 |

**Reading.** The variant does what it was designed for: sibling tasks are served, unedited inputs
never change their route, and on sequences without collisions it is the frozen rule. What it gives
up is exactly what the stress test shows: old inputs that carry, word for word, the instruction of
a task that is learned later go to that task (100 of 100 in the light case). In these tests its
routes are almost those of the nearest-zone rule; the difference is in what can be promised: the
nearest-zone rule promises nothing for an edited input, the variant promises that an edited input
inside a zone changes hands only when a later task registers exactly its text. **Not shown by
any test here:** a case where the variant and the nearest-zone rule differ on real inputs; the
variant with trained answers on a GPU. The paper's main method stays the frozen rule; the variant
is reported as an option for collections with sibling tasks, with its weaker guarantee.

### 2026-11-17 — result of the three-seed run (five tasks, seeds 42 to 44), read against the rules

Kaggle T4 x2, t5-base, 10 epochs, two runs at a time; fetched with `kaggle_fetch.py` into
`results/2026-11-17_kaggle_five_tasks_seeds42-44/`. Official scores, mean over the three seeds
(lowest to highest seed): `certified2` learned 79.13 (78.54 to 80.21), AP 79.28 (78.75 to 80.35),
FT 0.00 (0.00 to 0.00); InfLoRA learned 59.57 (56.29 to 62.73), AP 62.49 (58.43 to 64.99), FT −3.15
(−5.45 to −1.43); GainLoRA learned 61.02 (58.68 to 63.96), AP 61.55 (60.42 to 63.46), FT −0.47
(−2.63 to 0.71).

| Rule for `certified2` written before the runs | Result over three seeds |
|---|---|
| Original wording: 0 answers changed, every input inside its own zone | **held** in every seed: 0 of 1,200 answers; 1,500 of 1,500 inputs |
| No answer changes for an input inside a zone, in any wording | **held**: no answer changed at all (0 of 1,200 in each of the four wordings) |
| The re-test reproduces the official test | **held**: 4,500 of 4,500 answers for each of the three methods (batch sizes now equal) |
| Expected: forgetting of at most 5 under rewording | **as expected**: 0.00 in every wording and seed |
| Expected: 100%, 100%, about 94% of inputs to the right task (light, medium, heavy) | **as expected**: 100%, 100%, 93.8% (routes do not depend on the seed) |
| Expected: `key` equal to `all` under rewording | **as expected** (36.4 / 36.0 / 35.2 for both) |
| Against: a training input inside an older zone | did not happen (0 of 12,000) |
| Against: forgetting under rewording above half of `plain`'s | did not happen (0.00 against 54) |

Re-test with every reference answer, mean over seeds (AP; answers changed of 1,200): certified
79.74 / 79.59 / 78.98 / 74.07 with 0 changed in each wording; plain 36.08 with 994 changed; nearest
centre under heavy rewording 59.38 with 171 changed; InfLoRA 62.93 / 64.32 / 57.77 / 54.33 with
370 / 296 / 320 / 470 changed; GainLoRA 61.96 / 64.65 / 58.75 / 51.63 with 65 / 58 / 122 / 178
changed.

**Seed 42 was run twice** (the single-seed run of the afternoon and this run). Same code, same
seed, same kind of GPU, and not the same run: for all three methods the first task was
reproduced answer for answer, and from the third task on the answers differ (certified switch:
427 of 500 final answers the same; average 79.33 against 79.23; InfLoRA 58.89 against 58.98;
GainLoRA 61.31 against 60.86). Both runs used batches of 16 throughout. So training on these
GPUs is not reproducible bit for bit, and a difference of a few tenths of a point between two
runs means nothing. The count that matters did not vary: 0 answers changed for the certified
switch in both runs.

**Reading, unchanged by the two new seeds:** the published methods do not forget on these five
tasks in this setting (their forgetting is negative on average); they learn less in ten epochs
(about 60 right after learning against 79.7); the certified switch has the learning of plain
adapters and no change of any answer. With three seeds the ranges do not overlap: 79.2 to 80.8
against 59.0 to 65.4 and 60.9 to 63.9 (AP, re-test).

### 2026-11-17 (evening) — an independent review of the paper draft, and what it changed

A second agent that had not written any of the text read the draft against `facts.txt`, the tables,
`routing/*.json` and this plan, and reported 22 findings. The ones that change a reading above:

* **The bound on edits was looser than the proof gives.** The rules above state it as `(l + l') D`.
  The same argument gives `max(l, l') D` (split the difference of the two averages into
  `l (m+ − m−) + (l' − l)(m+ − mc)` for `l' >= l`). Checked numerically on 2,250 edited inputs
  (`paper_certified/bound_check.py`): the tighter bound held for every one, the inputs moved at most
  41% of it (15% in the middle), and it alone certifies 164 of the 1,324 inputs that stayed inside
  their zone. In tokens of average weight it covers the replacement of one to four tokens,
  depending on the task.
* **"The certificate was never wrong" was too wide.** It is true for the 23,640 edited inputs of the
  fifteen tasks. In the 875-task collection 849 unedited inputs are certified for an older sibling
  task. A certificate promises a stable answer, not the right task. The paper now says so.
* **"Radius zero: 8 tasks" was a rounding artefact** of how the fast analysis computed distances
  between signatures. Computed directly, 14 tasks have an opening that is token for token the
  opening of an earlier task (8 definition texts are shared by 21 tasks, and one pair differs by a
  single blank). All other numbers of the 875-task test and of the variant are unchanged by the
  rerun.
* **The comparison with the published methods says less than the earlier readings above.** Right
  after learning, InfLoRA and GainLoRA are at the level of guessing on the two tasks with two labels
  (SST-2 55.3 and 57.7; Sentiment140 43.7 and 50.7, where always answering "negative" gives 63).
  So in these runs they had not learned three of the five tasks, and one step of their code ran with
  a substitute (PyTorch instead of CuPy for an SVD) that was never compared number for number.
  "They learn less" stands as an observation; "their protection is paid for in learning" was a
  causal claim without a test and has been removed from the paper.
* **The exact key is not the small-radius limit of the certified switch** (the radius factor
  cancels in the nearest-zone rule; the limit is the nearest-zone rule with certificates for
  unedited inputs). The remark in the paper was wrong and has been replaced.
* **On the benchmark sequences the scores do not separate the certified switch from simpler rules**
  (nearest zone; with unedited instructions even the exact key). The paper now says in the
  abstract that the switch adds the proof and the certificate, not points.
* **Three things were in this plan and not in the paper:** how the five tasks were chosen (those
  InfLoRA had forgotten most in the 15-task run), that the unrelated tasks leave out the sentiment
  tasks on purpose, and the substitute in the baselines' code. All three are now in the paper's
  protocol and limitations.
* Smaller corrections: ties go to the older task (now part of the definitions); a zone of radius
  zero is defined; blocked inputs are counted on the training set only, so unseen inputs can be
  blocked unreported; under the variant a task whose opening begins with an older opening is also
  blocked; the stress test is one constructed case about routes.

### 2026-11-17 (night) — next idea, fixed before any run: zones in the frozen model's own hidden states

**Why.** The aim here is research that lifts this work clearly above where it stands. Its two
real weaknesses are measured above: zones built from word-vectors lose the certificate as soon as
an instruction is paraphrased, and zones large enough to help collide with sibling tasks. The
proof never used the word-vectors: it holds for any feature that is frozen. So the open question
is whether a frozen feature that reads meaning gives a better trade between the two.

**Idea.** For a decoder-only model, the hidden states of the frozen base model (no adapters) at
the first n positions depend only on the first n tokens. One pass of the base model therefore
gives, for every prefix length at once, a feature of the instruction in the model's own reading.
Zones, radii, routes and the theorem stay as they are; only the feature changes (layer 0 with the
weighted mean is exactly the feature of the paper). If lower layers are enough, adapters can be
put on the layers above and routing costs nothing extra.

**Test (routes only, CPU, SmolLM2-135M first).**
- *Rewordings that nobody on this project wrote:* every definition of the 875 real tasks is
  translated to German and back, and to French and back, sentence by sentence, with public
  translation models (deterministic). These are machine paraphrases; some will be poor, and they
  are used as they come.
- *Features:* layers 0, 2, 4, 8, 16 and 30 of the frozen model; pooled over the first n positions
  as a weighted mean (word weights as in the paper), or the state at position n. Scale of
  distances from the same 4,000 unrelated inputs as before.
- *Radius:* the rule of the paper (a share of the distance to the nearest known signature), and a
  second rule: one radius for all tasks, set so that a chosen share of machine paraphrases of
  *other* tasks' instructions falls inside (calibrated on tasks that are then not used again).
- *Split:* 175 of the 875 tasks (random, seed 0) are for choosing layer, pooling and radius; the
  other 700 are used once, with the chosen setting.
- *Measured on the 700:* tasks that collide when they arrive (unedited inputs); share of
  paraphrased inputs that are certified for their own task; share certified for another task;
  share given to the right task.
- *Anchor that must hold, or the code is wrong:* layer 0 with the weighted mean and the paper's
  radius rule must give the routes already published for this model on the five and fifteen tasks
  (`paper_certified/routing/models.json`).

**What would count as success:** on the 700 tasks, a hidden-state feature certifies at least 20
percentage points more of the paraphrased inputs than the word-vector feature at the same or a
lower collision rate, without more inputs certified for another task. **What would count
against the idea:** no setting gains 5 points at a matched collision rate (then meaning-level
features do not escape the trade and the idea is dropped), or wrong certificates go up.
**Whatever comes out is reported.** Answers are not tested here; that needs a GPU and approval, because it uses our Kaggle hours.

### 2026-11-17 (night) — result of the 15-task run, read against the rules

Kaggle T4 x2, t5-base, 10 epochs, seed 42, official order 1 of SuperNI (15 tasks), three methods
through the same code and re-test, two at a time; 3.1 to 3.5 hours per method. Raw files in
`results/2026-11-17_kaggle_fifteen_tasks_seed42/` (`log.txt` has the notebook's own tables). Numbers
are from the re-test with all reference answers unless said otherwise.

| Rule written before the run (paragraph "The 15-task run, fixed before it is started" and "Rules for the `certified2` runs") | Result |
|---|---|
| Original wording: 0 answers changed, every input inside its own zone | **held**: 0 of 1,097 answers changed; 1,197 of 1,197 test inputs inside their own zone |
| No answer changes for an input inside a zone, in any wording | **held**: 0 of 400 changed in each of the three wordings |
| The re-test reproduces the official test | **held**: 8,874 of 8,874 answers the same, for each of the three methods |
| Expected: forgetting of at most 5 under rewording | **as expected**: 0.00 in every wording |
| Expected: 100%, 100% and about 94% of inputs to the right task (light, medium, heavy) | **as expected**: 100%, 100%, 93.8% |
| Expected: `key` equal to `all` under rewording | **as expected**: 39.6 / 40.4 / 36.1 for both |
| Expected: `centre` clearly worse than `certified2` under medium and heavy rewording | **as expected**: AP 72.0 and 57.5 against 80.2 and 76.0; right task 85.2% and 59.2%; 29 and 100 of 400 answers changed |
| InfLoRA close to its Colab run (official scores 30.21 / 25.85 / 5.66) | **held**: exactly the same, 30.21 / 25.85 / 5.66 |
| Against: any blocked training input | did not happen (0 of 9,391) |
| Against: any changed answer inside a zone | did not happen |
| Against: AP of `certified2` not clearly above InfLoRA's and GainLoRA's | did not happen: 42.77 against 26.13 and 28.62 |
| Against: plain adapters right after learning not above InfLoRA's 30.21 | did not happen: 42.77 |

Scores of the re-test (Learned / AP / FT, answers changed of 1,097): certified switch 42.77 / 42.77 /
0.00, 0 changed; nearest zone and exact key the same; nearest centre 42.77 / 42.63 / 0.15, 2 changed;
plain adapters (same adapters, all on) 42.77 / 14.88 / 30.67, 973 changed; InfLoRA 30.46 / 26.13 /
5.69, 661 changed; GainLoRA 29.08 / 28.62 / 1.09, 215 changed. Scores of the official script
(Learned / AP / FT): certified switch 42.28 / 42.26 / 0.19; InfLoRA 30.21 / 25.85 / 5.66; GainLoRA
28.73 / 28.21 / 1.11.

Reworded instructions (the five tasks that have rewordings, inside the 15-task sequence; light /
medium / heavy): certified switch AP 80.71 / 80.20 / 76.00 with 0 of 400 answers changed in each;
certified inputs 500 / 100 / 98 of 500; right task 100% / 100% / 93.8%. Nearest centre: right task
85.2% (medium) and 59.2% (heavy), 29 and 100 of 400 answers changed. InfLoRA: FT 10.5 (light) and
12.6 (heavy) on these five tasks; GainLoRA 0.9 and 3.4.

**Reading.**

* Every rule written before the run held. The outcome of the certified switch is the one the theorem
  guarantees. Simpler rules (nearest zone, exact key with the original instructions) reach the same
  score with the original instructions, so the switch adds the proof and the certificate, not points.
* On fifteen tasks, unlike on five, InfLoRA forgets (FT 5.7) and GainLoRA forgets a little (FT 1.1).
  Both also learned less right after each task (30.5 and 29.1 against 42.8). We do not say why they
  learned less. The caveats of the five-task runs stay: a tenth of the published budget, a smaller
  model, and one step of their code computed with a substitute. One seed.
* The official script gives the certified switch a forgetting of 0.19. This is not forgetting. The
  official script scores each answer against one randomly picked reference. The re-test shows that
  all 8,874 answers are identical to those of the official test and that none of 1,097 answers
  changed. When the claim is that nothing changed, compare the answers, not the scores.
* InfLoRA's official scores are exactly those of the earlier run of the same method on Google Colab,
  so the two platforms agree for this method.
* The reworded instructions are the machine-written development set; their results do not become an
  independent test by being run inside a longer sequence.

**Result of the hidden-state test, read against the rules above** (run once by a smaller model;
`paper_certified/paraphrase_make.py`, `feature_study.py`, numbers in
`paper_certified/routing/feature_study.json`; SmolLM2-135M; 875 definitions translated to German
and back and to French and back, 83% of the words kept on average, 6 of 1,750 unchanged).

- Anchor: **held.** Layer 0 with the weighted mean and the paper's radius rule gives the published
  routes of this model on the five and the fifteen tasks.
- Chosen on the 175 calibration tasks: last layer, weighted mean, one radius for all tasks.
- On the 700 evaluation tasks (4,198 paraphrased inputs):

| | Tasks that collide | Paraphrased inputs certified for their own task | certified for another task | given to the right task | Unedited inputs to their own task |
|---|---|---|---|---|---|
| word-vectors, rule of the paper | 18.9% | 42.1% | 8.9% | 75.6% | 81.3% |
| last hidden layer, chosen setting | 25.7% | 64.9% | 13.7% | 83.2% | 74.5% |

- **Success condition: not met.** The gain is 22.7 points, but with more collisions and more wrong
  certificates, not with the same or fewer.
- **Counting against the idea: yes, by the second clause** (wrong certificates went up). The first
  clause could not be judged as written: the collision rate chosen on 175 tasks does not carry
  over to 700 tasks, because a longer sequence has more collisions. That is a flaw of our rule, not
  a reason to read the result more kindly.
- **Reading.** Hidden states are more tolerant of paraphrase and less able to tell sibling tasks
  apart; they move along the trade, they do not escape it. The idea is dropped as tested, and no
  GPU run is started for it.
- **A by-product that belongs in the paper as a limit:** this is the first rewording test that
  nobody on the project wrote. With the paper's own kind of feature, in a collection of 700 real
  tasks, only 42% of machine-paraphrased inputs keep their certificate, 9% are certified for
  another task and 76% reach their own task. The 100% / 100% / 94% on the five tasks were measured
  on tasks that are far apart and on rewordings written by machine.

### 2026-11-17 (night) — second idea, fixed before any run: measure distance in units of what paraphrasing does

**Why the first idea failed, as we understand it.** In any feature that measures how alike two
texts are, a sibling task (one or two words changed) is *closer* to the old instruction than a
paraphrase of it (about a sixth of the words changed). A ball that holds the paraphrases must
then hold the sibling. What differs is not how far the two move, but in which direction:
paraphrasing changes common words and word forms, a sibling changes a content word.

**Idea.** Keep the word-vector feature and the zones, and change only the ruler. Today distances
are rescaled by how unrelated inputs vary in every direction. Instead, estimate from unrelated
tasks how a paraphrase moves the feature (the covariance of "feature of the paraphrased
instruction minus feature of the original"), and rescale by that: directions in which paraphrases
typically move count little, all other directions count fully. This is the standard step of
removing within-class variation (as in linear discriminant analysis). The feature map stays
fixed after set-up, so the theorem and all propositions hold unchanged. It needs machine
paraphrases of unrelated instructions once, at set-up, and nothing per task.

**Test (routes only, CPU, word-vectors of SmolLM2-135M and of T5-base, minutes).**
- Paraphrase covariance from the 175 calibration tasks, **German pivot only**, shrunk towards the
  identity (primary: 10% identity; reported also: 50%).
- Evaluation on the 700 evaluation tasks with the **French-pivot** paraphrases (a style and tasks
  not used for the estimate), and with both pivots as a second view.
- One radius for all tasks, swept over a grid, so that the whole curve is seen: tasks that
  collide (C), paraphrased inputs certified for their own task (P), certified for another task
  (M), given to the right task (R), unedited inputs to their own task (U). The same curve for the
  old ruler. No layer, pooling or other choice is made on the evaluation tasks.
- **Summary fixed now:** read both curves at the radius where C equals the collision rate of the
  paper's rule on these 700 tasks (interpolated on the evaluation set, for both rulers).
  **Success:** at that point the new ruler certifies at least 20 points more paraphrased inputs
  for their own task, with no more inputs certified for another task. **Against the idea:** a
  gain under 5 points at that point, or more inputs certified for another task.
- Secondary: the five and fifteen tasks with the machine-written rewordings under the new ruler.

### 2026-11-17 (night) — result of the second idea (paraphrase ruler), read against its rule

Numbers: `paper_certified/routing/ruler_study.json`, script `paper_certified/ruler_study.py`. The
reference run reproduced the first study exactly (SmolLM2: C 18.9%, P 42.1%, M 8.9%).

At the pre-registered point (one radius for all tasks, read where the collision rate equals that of
the paper's rule; 700 evaluation tasks, French-pivot paraphrases):

| Word-vectors of | Ruler | P (certified, own task) | M (certified, other task) | R (right task) |
|---|---|---|---|---|
| SmolLM2-135M | old (unrelated inputs) | 41.6% | 12.3% | 81.8% |
| SmolLM2-135M | new (paraphrase, 10% identity) | 50.1% | 13.1% | 83.2% |
| T5-base | old | 31.8% | 9.5% | 83.0% |
| T5-base | new | 42.0% | 10.7% | 84.4% |

**Verdict by the rule written before the run: against the idea, on both tables.** The new ruler
certifies more paraphrased inputs for their own task (+8.5 and +10.2 points), but the gain is far
below the 20 points asked for, and it certifies *more* inputs for another task, which the rule
forbade. With the paper's own radius rule the gain on T5-base is zero (39.5% against 39.8%). On the
five benchmark tasks with the machine-written rewordings the new ruler is worse than the old one (T5-base,
heavy rewording: 66% of inputs to the right task against 94%). The idea is dropped; nothing of it
goes into the method. No GPU run.

**What we take from it.** The movement of a paraphrase is not confined to a few directions (the ten
largest directions hold 35% of it for T5-base, 52% for SmolLM2), so no linear ruler can separate
"paraphrase" from "sibling task" in an average of word-vectors. Two ideas that kept the word-vector
average have now failed for the same reason.

**A number the paper has to report.** The reference run is the first test of the paper's rule with
rewordings other than the machine-written development rewordings, at scale (T5-base word-vectors, 700 real tasks, 4,198
machine-paraphrased inputs): 40.5% are certified for their own task, 7.6% are certified for
*another* task (a stable answer from the wrong adapters), 75.4% reach the right task with or without
a certificate, and 16.9% of the tasks collide. This goes into the limitations of the paper.

### 2026-11-17 (night) — third idea, fixed before any run: a small frozen sentence encoder as the feature, and a radius with a stated coverage

**Why.** The zones of the paper are narrow: on 700 real tasks only about four in ten machine-paraphrased
inputs are certified for their own task (first idea's reference numbers). The theorem does not care
which frozen feature the zones are built from. Word-vector averages are free, but they were not made
to bring paraphrases together. A small sentence encoder was made for exactly that. And the radius can
be set by a rule with a guarantee of its own instead of by a hand-picked factor.

**Idea.** (1) Feature: a small frozen sentence encoder (22 million weights), applied to the first
tokens of the input, in place of the weighted word-vector average. It is fixed before the first task,
so Theorem 1 and its corollary hold unchanged (the edit bound of Proposition 2 does not carry over:
it needs an average). (2) Radius: one radius for all tasks, the split-conformal quantile of
"distance of a paraphrased input from its own signature" on unrelated calibration tasks. If a new
task and its rewording are drawn like the calibration tasks, at least a share 1 − α of its reworded
inputs lie inside its zone (a standard result). Whether an older zone claims them first is not
covered by this and is measured. (3) Optional: the paraphrase ruler of the second idea on top.

**Fixed now (nothing below is chosen on the 700 evaluation tasks).**
- Encoder (primary, chosen before any run because it is the most used small one):
  `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, unit length, Euclidean distance.
  Reported too, not primary: `BAAI/bge-small-en-v1.5`.
- Prefix: a task's opening length n_k (in the encoder's tokens) is rounded down to the grid
  8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 254; the feature of an input for task k is the encoding
  of its first that-many tokens; the signature is the encoding of the same number of tokens of the
  opening. An unedited input is then at distance 0 from its own signature, as before.
- The 175 calibration tasks are split in two halves by their position in the sorted list (even: A,
  odd: B). Ruler (if used): from half A, German pivot, 10% identity. Radius: from half B, German
  pivot, first input of each task, r_α = the ⌈(1 − α)(m + 1)⌉-th smallest distance; α = 0.1
  (primary), 0.05 and 0.2 reported.
- Plain or with ruler: decided on half B only, by the larger ratio (median distance of a task's
  signature to the nearest other half-B signature) / r_0.1.
- Route rule as in the paper: oldest zone that contains the input, else the nearest signature.
- Evaluation: the 700 evaluation tasks in the arrival order of the first study, **French-pivot**
  paraphrases (primary), German and both as second views; C, P, M, R, U as before, and Z = share
  of paraphrased inputs inside their own zone (to be compared with 1 − α).
- **Success:** at α = 0.1, French pivot: P ≥ 75%, M ≤ 5% and C ≤ 19% (the paper's rule has about
  42%, 9% and 19%). **Against the idea:** P < 55%, or M > 9%, or C > 25%. Anything between is
  "partial" and is reported as such. A GPU run is started only after a success.
- Second views, reported whatever they show: the curve over a grid of radii and its point at the
  collision rate of the paper's rule (as in the second idea); the same test when the instruction is
  a separate field of the input (encoding of the whole definition, no prefix); the five and fifteen
  benchmark tasks with the machine-written rewordings.
- If it succeeds, one more check before it enters the paper as a result: paraphrases through a
  third language that no step above has seen.

**What would count against using it even after a success:** it needs a second small model at
inference (up to eleven short passes per input), and it gives up the edit bound.


---
### 2026-11-17 (night) — the Latent-LoRA comparison run (not pre-registered), what it shows and what it does not

No rule for this run was written in this plan before it was started; it is therefore read without
"held" or "failed". Kaggle T4 x2, t5-base, five tasks, 800 training and 100 test inputs per task, 10
epochs, seeds 42 to 44; the certified switch (`certified_switch/`, rank 4, batch 32, rules
certified / nearest / all) and our re-implementation of Latent-LoRA (rules soft and sum), both in
`latent_lora/run_five.py`. Latent-LoRA has no public code; it was written from the paper's text
(`latent_lora/README.md`: the method, and 20 details the paper does not give). Differences from the
paper: 10 epochs instead of 30, t5-base instead of T5-large, five tasks, our own Rouge-L scorer. Raw
files in `results/2026-11-17_kaggle_latent_lora_five_tasks/`. Original instructions, mean over three
seeds (Learned / AP / FT, answers changed of 1,200, right task):

| Method | Learned | AP | FT | Changed | Right task |
|---|---|---|---|---|---|
| certified switch | 76.49 | 76.49 | 0.00 | 0 | 100.0% |
| same adapters, all on | 76.49 | 36.04 | 50.56 | 970 | |
| Latent-LoRA soft | 42.55 | 42.55 | 0.00 | 0 | 100.0% |
| Latent-LoRA sum | 48.71 | 47.66 | 1.31 | 429 | |

Wordings (light / medium / heavy): certified switch AP 75.87 / 75.91 / 71.74, 0 changed in each,
right task 100% / 100% / 94.0%. Latent-LoRA soft AP 48.71 / 32.53 / 27.75, 0 changed in each, right
task 100% in each.

**Reading.**

1. The re-implemented Latent-LoRA learned the first task and almost nothing after it (seed 42, soft,
   right after learning: relation 90.8, SST-2 56.0, Glucose 2.2, Sentiment140 37.0, emotion 27.0; the
   other seeds are alike; training loss at the end of the Glucose task about 2.0 against about 0.6 for
   the certified switch). We do not know whether this is the reduced budget (10 of 30 epochs, smaller
   model) or a mistake in our re-implementation; without public code we cannot check. The run says
   nothing about how well Latent-LoRA learns or retains, and no such claim is made. A sign that its
   adapters were barely trained: its average with the light rewording (48.71) is higher than with the
   original instructions (42.55), because the Sentiment140 score moves from 37 to 65 to 69.
2. Its router gave every test input to the right task in all four wordings (ours 1.000 / 1.000 / 1.000
   / 0.940) and with the soft rule none of 1,200 answers changed in any wording. On these five tasks
   its router is as stable as ours in practice and recognises the heavy rewording better. The certified
   switch adds the proof and the certificate on each answer, not better recognition. Latent-LoRA was
   not run in the stress test with a look-alike task: how its router behaves there is unknown.
3. Sum rule (no router): AP 47.66, FT 1.31, 429 of 1,200 answers changed.
4. The stand-alone code of the certified switch reproduces the main result on t5-base in a second,
   independent code base (AP 76.49, FT 0.00, 0 of 1,200 changed in every wording; all adapters on:
   AP 36.04, FT 50.56, 970 changed). The main experiments use other code and rank (AP 79.74).

**What it does not show:** anything about Latent-LoRA as its authors built it; its router outside
these five tasks; anything beyond one benchmark order; rewordings are machine-written. Written into
both papers (appendix `sec_appendix_latent.tex`) with these limits.

### 2026-11-17 (night) — fourth idea, fixed before any run (and before the third idea's result is in): a zone decided by a frozen judge of "same instruction"

**Why.** The first two ideas failed for one reason: in any measure of how alike two texts are, a
sibling task (one or two words changed) is as close as a paraphrase. The question the zone has to
answer is not "how alike" but "does this instruction ask for the same thing". Small frozen models
exist that were trained for exactly that question (natural-language inference: does text A imply
text B). The theorem needs no distance: a zone may be *any* set that is fixed when the task arrives.

**Idea.** Assume the instruction is a separate field of the input (as the system prompt is in a chat
interface, and as the definition is in our benchmark). The zone of task k is the set of instructions
u with (1) the sentence-encoder distance between u and the task's instruction at most r_w (a wide,
cheap first filter), and (2) both directions of implication between u and the task's instruction at
least τ according to a small frozen inference model. Route: the oldest task whose zone holds the
instruction, else the nearest by the encoder, without certificate. Theorem 1 holds unchanged.

**Fixed now.**
- Encoder for the filter: `sentence-transformers/all-MiniLM-L6-v2` on the whole instruction (cut at
  254 tokens). Judge (primary, chosen before any run because it is the smallest of its family):
  `cross-encoder/nli-deberta-v3-xsmall`, score E(u, o) = the smaller of the two probabilities of
  "entailment" (u given o, o given u), each text cut at 128 tokens; E = 1 for identical texts.
  A second judge that costs nothing is computed next to it: 1 − (the largest weight of a word
  that is in one of the two texts and not in the other), with the paper's word weights. Which of
  the two is used is decided on half B of the calibration tasks only, by how well it tells a task's
  German paraphrase (same task) from the nearest other calibration instruction (area under the
  curve); the inference model is primary unless the free judge is better by more than 0.02.
- Thresholds from half B of the calibration tasks, German pivot, by split-conformal quantiles:
  r_w at α = 0.02 and τ at α = 0.08, so that together at least 90% of reworded instructions of a new
  task lie inside its zone if they are drawn like the calibration ones.
- Evaluation: the 700 evaluation tasks in arrival order, one unit per task and paraphrase,
  **French pivot** primary: C (the task's own instruction lies in an older zone), Z (paraphrase in
  its own zone), P (certified for its own task), M (certified for another task), R (right task with
  or without certificate). To keep the cost down, at most the 20 nearest older tasks inside r_w are
  judged per text; how often that limit was reached is reported.
- **Success and failure: the same numbers as for the third idea** (success: P ≥ 75%, M ≤ 5%,
  C ≤ 19%; against: P < 55%, or M > 9%, or C > 25%).
- Reported whatever it shows: the same with the German pivot; how the judge scores the pairs that
  collide under the paper's rule; the five benchmark tasks with the machine-written rewordings; the
  number of judged pairs per input and the time per input.

**What would count against using it even after a success:** two extra small models at inference, a
cost that grows with the number of look-alike tasks, the assumption that the instruction can be
separated from the rest of the input, and no edit bound.

### 2026-11-17 (night) — result of the third idea (sentence encoder, radius with stated coverage), read against its rule

Numbers: `paper_certified/routing/encoder_study.json`, script `paper_certified/encoder_study.py`.
(The section on the fourth idea above was written and pushed before this result was read.)

Set-up as fixed: half A 88 tasks, half B 87. The ruler variant had the larger ratio on half B (1.73
against 1.50), so it is the primary variant. All 4,193 unedited inputs are at distance 0 from their
own signature. Evaluation: 700 tasks, 2,099 French-pivot and 2,099 German-pivot paraphrased inputs.

| MiniLM encoder, α = 0.1, French pivot | C | P | M | R | Z |
|---|---|---|---|---|---|
| paper's rule on T5-base word-vectors (for comparison, both pivots) | 16.9% | 40.5% | 7.6% | 75.4% | |
| encoder with ruler (primary) | 16.6% | 71.2% | 11.6% | 86.1% | 82.0% |
| encoder, plain | 22.9% | 73.1% | 17.1% | 82.0% | 89.8% |

**Verdict by the rule written before the run: against the idea** (M = 11.6% is above the 9% limit;
P = 71.2% misses the 75% asked for; C is fine). The second encoder (bge-small) gives the same
picture (P 71.2%, M 12.7%, C 19.9%). With the instruction taken as a separate field: P 74 to 75%,
M 12 to 15% for three encoders. No GPU run.

**What it shows.** (1) A sentence encoder does what it was chosen for: it certifies about seven in ten
machine-paraphrased inputs for their own task where the word-vector average certifies four in ten,
at the same collision rate, and it sends 86% to the right task. (2) The coverage promise holds
for the plain metric (Z 89.8% French, 92.0% German, promised 90%) but not for the ruler variant
on the other pivot (82.0%): the ruler was fitted on German-pivot paraphrases and does not carry
over fully. (3) It makes wrong certificates *more* frequent, not less: the tasks that collide are
siblings that differ in one decisive word (date format day-month against month-day; closest
integers against farthest integers), and to an encoder those are nearly the same text. (4) On the
five benchmark tasks the machine-written heavier rewordings are *outside* the zones (38% inside for the
medium, 1% for the heavy rewording): those rewordings move further than a back-translation does. They
are still sent to the right task (100%).

**Consequence.** Three features (word-vector average with two rulers, hidden states, sentence
encoder) all fail on the same cases: sibling tasks. Better "likeness" does not solve it. The fourth
idea, written before this result, asks a different question (same request or not); its test is
running. If it fails too, the honest end is: the guarantee stays as proved, and the paper reports
the measured price at scale, including these four attempts.

### 2026-11-18 (00:12 UTC) — result of the fourth idea (zone decided by a frozen judge), read against its rule

Numbers: `paper_certified/routing/judge_study.json`, script `paper_certified/judge_study.py`. The
run took 91 minutes on four CPU cores.

Calibration as fixed (half B, 87 tasks, German pivot). The filter radius came out as r_w = 1.125,
which is the largest calibration distance (with 87 tasks the quantile at α = 0.02 is the maximum),
so the filter lets almost every pair through. Area under the curve on half B: 0.969 for the
inference judge and 0.677 for the free word-difference judge, whose threshold came out as 0 (it is
useless here). So the inference judge was used, as planned, with τ = 0.199. At this τ, 93.1% of the
paraphrase pairs and 11.5% of the nearest-other-task pairs of half B count as "the same request".

| 700 evaluation tasks | C | P | M | R | Z |
|---|---|---|---|---|---|
| paper's rule on T5-base word-vectors (for comparison, both pivots) | 16.9% | 40.5% | 7.6% | 75.4% | |
| judge-based zone, French pivot (primary) | 36.0% | 58.9% | 34.7% | 64.7% | 90.6% |
| judge-based zone, German pivot | 36.0% | 56.3% | 41.1% | 58.7% | 95.0% |

**Verdict by the rule written before the run: against the idea** (M = 34.7% is far above the 9%
limit, C = 36.0% is above the 25% limit, and P = 58.9% misses the 75% asked for). No GPU run.

**What it shows.** (1) The coverage promise held (Z 90.6% on the French pivot, promised 90%).
(2) The judge is good on single pairs and bad as a zone. A text is compared with up to 20 older
tasks, and one wrong "same" among them is enough for a wrong certificate; 11.5% wrong "same" on
the nearest pair becomes 35% wrong certificates. (The limit of 20 was reached for 594 of the 700
French texts, because the filter is so wide.) (3) A stricter threshold does not rescue it. Curve
over 12 thresholds, looked at after the run, nothing chosen from it: τ = 0.73 gives C 19.0%,
P 69.3%, M 17.7%; τ = 0.91 gives C 10.4%, P 70.4%, M 10.6%, R 88.3%; P never reaches 75%, and M
is below 9% only at the last threshold, where almost nothing is covered (P 0.6%). At its best the
judge is where the sentence encoder alone already was. (4) The wrong "same" verdicts are the same
sibling tasks as before: remove decimal numbers against remove odd numbers (score 0.65), the
ATOMIC tasks "reaction" against "intent" (0.69), Amazon reviews by category against by polarity
(0.49). Of the 151 pairs that collide under the paper's rule, the judge calls 60 "the same".
(5) On the five benchmark tasks the judge handles free rewording of *different* tasks better than
any distance did: 14 of the 15 rewordings written by machine are inside their own zone (the
heavy rewording of the sentiment task is not), all are sent to the right task, and none is
certified for a wrong task, among the five and among the fifteen tasks. (6) Cost: 18 judged pairs
and 3.5 seconds per text on two CPU threads.

**Consequence, as written before the run.** Four ideas, four features, the same end: when tasks are
siblings that differ in one decisive word, a zone wide enough for rewordings also takes the
sibling's rewordings. No more ideas of this kind. The guarantee stays as proved, and the paper
reports the price measured at scale, including these four attempts. Not tested, and a possible
next step for someone with a GPU: a much larger frozen judge (for example the frozen base model
itself, asked whether two instructions ask for the same thing).

### 2026-11-18 (00:12 UTC) — one confirmation check, fixed before the Spanish paraphrases are read

**Why.** Reading the table of the third idea we noticed that a setting that was *not* the primary
one (MiniLM encoder, ruler metric, α = 0.2) was better than the paper's rule on all four measures
on the French pivot: C 12.1% against 16.9%, P 62.1% against 40.5%, M 6.0% against 7.6%, R 90.0%
against 75.4% (German pivot: P 72.6%, M 8.1%, R 89.7%). That comparison was chosen after seeing
the results, so by itself it proves nothing. The paraphrases through Spanish
(`paper_certified/routing/paraphrases_es.json`) have not been read by any study. They are used
once, for this.

**Fixed now.**

- The same 700 evaluation tasks in the same order, the same calibration (ruler from half A, radius
  from half B at α = 0.2, both from German-pivot paraphrases). Nothing is fitted or chosen again.
  The script has to reproduce the stored radius (49.99) and the stored French numbers before it
  reads the Spanish file.
- On the Spanish-pivot paraphrased inputs: P, M, R and Z for (a) the paper's rule with T5-base
  word-vectors and (b) the encoder zone with the ruler metric at α = 0.2. C and U do not depend on
  the paraphrases and are known (16.9% and 12.1%).
- **Confirmed** if on the Spanish pivot (b) has P at least 15 points above (a), R at least 8 points
  above (a), and M not above (a). **Not confirmed** otherwise.
- What we expect: (a) P about 40%, M 7 to 8%, R about 75%; (b) P about 60%, M about 6%, R 88 to 90%.
- Reported whatever it shows: the other five encoder settings on the Spanish pivot; the paper's
  rule on each pivot separately; the same comparison on five other random splits of the 875 tasks
  into 175 and 700 (seeds 1 to 5, ruler and radius fitted again on each split's calibration tasks,
  French and Spanish pivots), to see how much the numbers depend on the split; the five and the
  fifteen benchmark tasks under setting (b).

**What a confirmation would and would not mean.** The tasks are the same 700, so it would show that
the advantage holds for new paraphrases, not for new tasks. The paraphrases are made by a
translation model, not by people. The setting stays one that was chosen after the fact, and it does
not meet the bar set for the third idea (P ≥ 75%, M ≤ 5%). The coverage promise does not hold for
the ruler metric on another pivot (Z 68.0% on the French pivot where 80% was promised), so P is a
measured number and not a promised one. The costs stay: an encoder with 22 million weights run at
several prefix lengths per input, and no edit bound. No GPU run follows from it: routes are
computed on CPU and no adapter changes.

### 2026-11-18 (00:32 UTC) — result of the confirmation check on the Spanish paraphrases, read against its rule

Numbers: `paper_certified/routing/spanish_check.json`, script `paper_certified/spanish_check.py`.
Before the Spanish file was opened the script reproduced the stored radius (49.988), the stored
French and German numbers of the encoder setting, and the numbers of the paper's rule.

| Spanish pivot, 700 tasks, 2,099 paraphrased inputs | P | M | R | Z |
|---|---|---|---|---|
| (a) paper's rule, T5-base word-vectors | 50.4% | 9.7% | 77.5% | |
| (b) encoder zone, ruler metric, α = 0.2 | 63.8% | 5.3% | 91.7% | 68.6% |

**Verdict by the rule written before the run: not confirmed.** The gain in P is 13.4 points and
the rule asked for 15. The other two conditions hold (R gains 14.2 points where 8 were asked; M is
lower, 5.3% against 9.7%).

**What it shows.** (1) The encoder setting is ahead of the paper's rule on each measure on
paraphrases that nobody had looked at, but by less than we asked for. (2) Our guess for the paper's
rule was wrong: we expected P about 40% and it is 50.4%. The Spanish paraphrases keep more of the
original words (86% against 83 to 84%; 15 of 875 are identical to the original, against 5 and 1),
so the paper's rule covers more of them. By pivot the paper's rule has P 39.8% (French), 41.2%
(German), 50.4% (Spanish); M 9.2%, 5.9%, 9.7%; R 75.4%, 75.5%, 77.5%. So "four in ten" becomes
"four to five in ten, depending on the pivot". (3) Five other random splits of the 875 tasks
(seeds 1 to 5, everything fitted again on each split's calibration tasks), Spanish pivot, mean and
range: paper's rule C 16.0% (14.3 to 17.3), P 49.4% (48.3 to 50.7), M 7.9% (4.2 to 10.4), R 76.9%;
encoder setting C 12.7% (9.1 to 16.6), P 63.1% (55.5 to 70.0), M 6.0% (2.4 to 9.7), R 90.4%
(87.7 to 93.3). None of the five splits passes the rule by itself. The encoder setting depends
more on the split than the paper's rule does. (4) The other five encoder settings on the Spanish
pivot repeat the French picture (ruler α = 0.1: P 69.2%, M 10.2%; plain α = 0.2: P 65.3%,
M 7.9%). (5) The coverage promise does not carry over for the ruler metric: Z is 68.6% (Spanish)
and 68.0% (French) where 80% was promised, and 80.4% on the German pivot it was fitted on.
(6) On the five benchmark tasks under this setting the original and the light rewording are
inside their own zones, and none of the medium and heavy rewordings written by machine is
(0%); all are still sent to the right task and none is certified for a wrong one.

**Consequence.** The paper does not claim that the encoder zone is an improvement by the margin
asked for. It reports the encoder zone as a variant with these numbers and its costs (a second
model at several prefix lengths per input, no edit bound, no coverage promise across pivots, no
certificate for free rewordings of the benchmark tasks). The main method stays as it is. No GPU
run. The research part ends here; what remains is the paper and its final check.

### 2026-11-18 (00:41 UTC) — what the zones add, at scale: do later tasks re-route reworded inputs? Fixed before it is run

**Why.** The paper has one constructed case in which a best-match rule silently re-routes old
inputs when a look-alike task arrives (the stress test), and it says of the 875 real tasks: "The
weakness of `nearest` needs edited inputs, and this collection contains none." Since the tests
above there *is* a collection with edited inputs: machine paraphrases of 700 real tasks. So the
central claim of the paper can be tested at scale, on tasks nobody constructed: how often does the
arrival of later tasks change the route of a reworded input, under the best-match rule and under
the certified switch? This is not a new idea for wider zones; it uses the paper's rule as it is.
We had written above that the research part ends; there was time left, and this test belongs to the
paper's main claim, so it is run, with its rule written first as always.

**Fixed now.**

- The 700 evaluation tasks in the same arrival order as in the paraphrase tests, T5-base
  word-vectors, the paper's rule unchanged (`certified`), the nearest-zone rule (`nearest`) and
  the variant (`exactfirst`) exactly as in `paper_certified/many_tasks_analysis.py`.
- Inputs: the paraphrased inputs of every task (French and German pivot together as the primary
  set, 4,198 inputs; the Spanish pivot separately), and the unedited inputs.
- For every input and rule: the route right after the input's own task has arrived, and the route
  after all 700 tasks. **G** = share of paraphrased inputs whose route at the end differs from the
  route at arrival (a silent change: the same text would get its answer from other adapters).
  **L** = share that their own task served at arrival and another task serves at the end.
- The script has to reproduce the stored numbers of the paper's rule (C 16.9%, P 40.5%, M 7.6%,
  R 75.4%) before anything else, and it has to check the theorem: no input that is inside a zone
  at arrival may change its route under `certified`.
- **Reading, fixed now.** "The zones add something at scale" if G under `nearest` is at least 5%
  and G under `certified` is at most half of it. "They add little here" if G under `nearest` is
  below 2%, or G under `certified` is more than three quarters of it. Anything else: "something,
  but less than half of the changes are removed".
- What we expect: G under `nearest` between 8% and 20%; under `certified` about half of that, all
  of it among uncertified inputs. We also expect `nearest` to give *more* paraphrased inputs to
  their own task at the end than `certified` does (the price), and we will report that next to G.
- Reported whatever it shows: own-task share at arrival and at the end for each rule; among the
  inputs that `certified` certifies for their own task, the share that `nearest` gives to another
  task at the end; the number of tasks with a changed input; examples; the same for the unedited
  inputs (the paper says none changes) and for the Spanish pivot.

**What would count against the paper's claim.** If silent re-routing under `nearest` is rare even
with paraphrases (below 2%), the case the zones exist for is rare in this collection, and the
paper has to say so. If `certified` removes less than a quarter of the changes, the zones are too
narrow to matter where it counts.

### 2026-11-18 (00:50 UTC) — result of the re-routing test at scale, read against its rule

Numbers: `paper_certified/routing/reroute_study.json`, script `paper_certified/reroute_study.py`.
The script first reproduced the stored numbers of the paper's rule (C 16.9%, P 40.5%, M 7.6%,
R 75.4%).

| 4,198 paraphrased inputs (French and German pivot), 700 tasks | own task at arrival | own task at the end | route changed after arrival (G) |
|---|---|---|---|
| `certified` | 75.46% | 75.42% | 0.31% (13 inputs, 7 tasks) |
| `nearest` (nearest zone) | 75.46% | 75.42% | 0.31% (the same 13 inputs) |
| `exactfirst` | 75.46% | 75.42% | 0.31% (the same 13 inputs) |

**Verdict by the rule written before the run: "the zones add little here".** G under `nearest` is
0.31%, far below the 2% line, and `certified` removes none of it (ratio 1.0). Our guess (8 to 20%
under `nearest`) was wrong.

**What it shows.** (1) The theorem holds, as it must: of the 48.1% of the paraphrased inputs that
are inside a zone when their task arrives, none changed its route. All 13 changes are inputs
outside every zone. (2) But the nearest-zone rule, which has no guarantee, changed exactly the
same 13 inputs and no others. Of the 1,701 inputs that `certified` certifies for their own task,
`nearest` gives every one to the same task at the end. On the Spanish pivot: 7 of 2,099 changed
under every rule. On unedited inputs nothing changed under any rule, and `nearest` serves more of
them (98.4% against 83.1%). (3) So in this collection of real tasks, with machine paraphrases, the
case the frozen zones exist for (the constructed stress test) does not occur. What the certified
switch adds over the nearest-zone rule here is the proof for the inputs inside a zone, not any
measured stability, and it costs the unedited inputs of the tasks that collide.

**A possible reason, not yet tested.** `nearest` is the nearest *zone*: it compares distances
divided by each zone's radius, and a radius is fixed at arrival as half the distance to the nearest
older signature. A sibling that arrives close to an old task therefore gets a tiny radius, and for
every input that is not exactly on its signature the old, larger zone is relatively nearer. If that
is the reason, then what keeps routes stable in practice is the radius that is fixed at arrival,
and a plain nearest-signature rule without radii (the usual "nearest prototype" router) should
re-route many more inputs. That can be checked on the same data.

### 2026-11-18 (00:50 UTC) — follow-up, fixed before it is run: the plain nearest-signature rule

- The same 700 tasks, inputs and measures as in the test above. One more rule, `signature`: the
  input goes to the task whose signature is nearest in plain distance (the same whitened features,
  each task at its own prefix length, no radius; ties to the older task).
- **Reading, fixed now.** "The radius fixed at arrival is what keeps routes stable" if G under
  `signature` is at least 5% on the French and German pivot together. "Re-routing is rare in this
  collection under any best-match rule" if it is below 2%. In between: "some".
- What we expect: G between 5% and 20%. We do not know whether `signature` gives more or fewer
  paraphrased inputs to their own task at the end than `certified` (75.4%); both are reported,
  with the unedited inputs.
- This follow-up was thought of after the result above was seen. Whatever it shows, the verdict
  above stays: against the nearest-zone rule the certified switch adds no measured stability here.

### 2026-11-18 (00:56 UTC) — result of the follow-up (plain nearest-signature rule), read against its rule

Numbers: `paper_certified/routing/reroute_study.json` (rule `signature`).

| 4,198 paraphrased inputs (French and German pivot), 700 tasks | own task at arrival | own task at the end | lost (own task at arrival, another at the end) | route changed after arrival (G) |
|---|---|---|---|---|
| `certified` (and `nearest`, `exactfirst`: the same numbers) | 75.46% | 75.42% | 0.05% (2) | 0.31% (13 inputs, 7 tasks) |
| `signature` (plain nearest signature, no radius) | 92.35% | 91.31% | 1.05% (44) | 2.22% (93 inputs, 41 tasks) |

**Verdict by the rule written before the run: "some"** (G is between 2% and 5%; we had expected 5 to
20%). So the radius fixed at arrival does keep routes more stable (0.31% against 2.22%), but
re-routing is not frequent in this collection even without it.

**What we did not expect, and what matters more.** The plain nearest-signature rule gives **91.3%**
of the paraphrased inputs to their own task at the end, where the paper's rule (and the
nearest-zone rule) gives **75.4%**. On the Spanish pivot it is 93.0% against 77.5%. On unedited
inputs it is 98.4% against 83.1%. Of the 1,701 inputs that `certified` certifies for their own
task, `signature` loses 11 (0.65%) to another task later. Of the 93 inputs whose route changed
under `signature`, 34 end at a task whose definition shares at least 80% of its words with their
own.

**Reading.** At scale the simplest rule serves reworded inputs much better than the paper's rule,
and it pays for that with about two silent changes in a hundred inputs (one in a hundred loses its
own task). The paper's rule changes three in a thousand and can prove for 48% of the inputs that
they will never change, but it sends 16 points fewer to the right task. The fallback of the
paper's rule (nearest zone, distance divided by the radius) was chosen on the benchmark tasks and
their machine-written rewordings (entry of 2026-11-17 above); on real tasks with machine paraphrases it
is the weak part: dividing by the radius favours old, large zones, and the paraphrases of a later
sibling go to the older task. This goes into the paper as measured, with the verdicts above.

### 2026-11-18 (00:56 UTC) — one more rule on the same data, written before it is computed: zones first, then the plain nearest signature

- This is a design made after seeing the numbers above, on data that have been looked at. It has
  no free setting, so nothing can be tuned, but it is not an independent test and the paper has to
  say so.
- Rule `hybrid`: an input inside a zone goes to the oldest zone that holds it, exactly as in the
  paper (same certificates, same theorem). An input outside every zone goes to the task with the
  nearest signature in plain distance, not divided by the radius. Also `exactfirst_hybrid`: the
  same, with an exact match of a known opening taking priority, as in the variant of the paper.
- Same 700 tasks, inputs and measures. Also the five and the fifteen benchmark tasks with the four
  wordings, routes only, for `signature` and `hybrid` (share given to the right task; changes
  after arrival), to see whether the fallback chosen on the benchmark was the better one there.
- What we expect on the French and German pivot: `hybrid` gives 84 to 88% of the paraphrased inputs
  to their own task at the end (certified 75.4%, signature 91.3%); G between 1.4% and 2.0%
  (certified 0.31%, signature 2.22%); the certified share stays 48.1% and none of those inputs
  changes. On the benchmark tasks we expect `signature` and `hybrid` to be no better than the
  paper's rule under heavy rewording (93.8%), because that is where the fallback was chosen.
- Whatever comes out is reported next to the paper's rule. The paper's rule stays the rule of all
  GPU runs; a changed fallback would be a variant that has not been run with trained answers.

### 2026-11-18 (00:58 UTC) — the rule "zones first, then the plain nearest signature": computed and read

Numbers: `paper_certified/routing/reroute_study.json` (rules `hybrid`, `exactfirst_hybrid`, key
`benchmark`). No input that is inside a zone at arrival changed under `hybrid` (checked).

| 4,198 paraphrased inputs (French and German pivot), 700 tasks | own task at the end | route changed after arrival | lost its own task | inside a zone at arrival |
|---|---|---|---|---|
| `signature` (plain nearest signature) | 91.3% | 2.22% | 1.05% | no zones |
| `nearest` (nearest zone) | 75.4% | 0.31% | 0.05% | no certificate |
| `certified` (the paper's rule) | 75.4% | 0.31% | 0.05% | 48.1% |
| `hybrid` (zones first, then plain nearest signature) | 85.6% | 1.95% (82 inputs, 37 tasks) | 0.79% | 48.1% |

Spanish pivot: `hybrid` 84.4% and 2.29% changed; `signature` 93.0% and 2.33%; `certified` 77.5%
and 0.33%. Unedited inputs: `hybrid` 83.1% (as `certified`); `signature`, `nearest` and
`exactfirst_hybrid` 98.4%; no change under any rule.

The same rules on the benchmark tasks with the rewordings written by machine (routes only;
share given to the own task after the last task, changed routes in brackets, 500 inputs):

| wording | `certified` and `nearest` | `signature` and `hybrid` |
|---|---|---|
| original, light | 100% (0) | 100% (0) |
| medium | 100% (0) | 76.8% (55) on five tasks, 71.6% (74) on fifteen |
| heavy | 93.8% (0) | 60.0% (100) |

**Our guesses.** `hybrid` at scale: 84 to 88% expected, 85.6% found; G 1.4 to 2.0% expected, 1.95%
found. On the benchmark we expected the plain rule to be no better under heavy rewording; it is far
worse.

**Reading.** The two fallbacks trade places. On real tasks with machine paraphrases the plain
nearest signature is the better fallback (85.6% against 75.4% with the same certificates; 91.3%
without zones). On the benchmark tasks with the machine-written heavier rewordings it is much worse
(60.0% against 93.8%, and 100 of 500 routes change when later tasks arrive). So "zones first, then
nearest signature" is not a repair: it gains ten points in one test and loses thirty-four in the
other. Why the plain rule fails on the benchmark was not tested. No fallback we have is right in
both places.

**What the paper has to say after these three computations.** (1) At scale the certified switch
showed no measured stability beyond the nearest-zone rule: both changed the same 13 of 4,198
routes. What it adds there is the proof for the inputs inside a zone, about half of the
machine-paraphrased inputs. (2) The plain nearest-signature rule gives more machine-paraphrased
inputs to their own task than the paper's rule (91.3% against 75.4%) and silently changes about
two routes in a hundred. (3) The fallback of the paper's rule was chosen on the benchmark tasks and
does not carry over to the large collection; which fallback is better depends on the collection.
(4) The certificates do not depend on the fallback. None of this was run with trained answers.

### 2026-11-18 (01:28 UTC) — a fallback normalised for the length of the opening. Fixed before it is run

**Why.** The two fallbacks trade places (entry above): dividing the distance by the radius is right
on the benchmark tasks (93.8% under heavy rewording) and wrong at scale (75.4%); the plain distance
is right at scale (91.3%) and wrong on the benchmark (60.0%). A reading that fits both, not tested
yet: the radius does two things at once. (a) It puts tasks with openings of different lengths on
one scale: an average over many tokens lies closer to everything, so in plain distance a task with
a long opening looks near to every input. That is needed on the benchmark, where the openings have
very different lengths. (b) It shrinks the zone of a task that arrives next to a sibling, so that
every reworded input of the newer sibling looks far from it. That is what goes wrong at scale. If
this is right, a scale that depends only on the length of the opening should keep (a) and drop (b).

**The rule, fixed now.** For an opening length n let σ(n) be the root-mean-square distance of the
4,000 unrelated inputs from their own mean in the feature ψ_n (the same weights and rescaling as
everywhere; an input shorter than n counts with all its tokens). σ(n) depends only on the unrelated
inputs and on n, so it is fixed before the first task. Rule `normalised`: an input goes to the task
with the smallest distance to its signature divided by σ(n_k) (ties to the older task). Rule
`certified3`: the oldest zone that holds the input, exactly as in the paper (same zones, same
certificates, same theorem), else `normalised`. There is no free setting.

**The test.** Routes only, T5-base word-vectors.

- Benchmark: the five and the fifteen tasks with the wordings original, light, medium, heavy (the
  rewordings written by machine; they have been looked at many times, also for this design).
- Scale: the 700 evaluation tasks. The French, German and Spanish paraphrases have been looked at
  for this design. So one more set is made now and used once: paraphrases through **Italian**
  (same procedure, `paraphrase_make.py it paraphrases_it.json`). The verdict at scale is read on
  the Italian pivot only.
- **Success ("one fallback that is right in both places")** if all of these hold for `normalised`:
  at least 90% of the inputs to their own task under heavy rewording, on five and on fifteen tasks;
  at least 97% under medium rewording; and at least 88% of the Italian-pivot paraphrased inputs to
  their own task after all 700 tasks. **Against** if heavy rewording is below 80% in either
  sequence, or the Italian pivot is below 85%. Anything else: partial.
- What we expect: benchmark heavy 90 to 96%, medium 100%; Italian pivot 89 to 93% for `normalised`
  and 84 to 88% for `certified3`; routes changed after arrival 1.5 to 2.5% for both.
- Reported whatever it shows: the same on the French, German and Spanish pivots; `certified3` on
  everything (own task at the end, routes changed after arrival, share inside a zone, none of the
  inputs inside a zone may change); the paper's rule and the plain nearest signature on the Italian
  pivot for comparison; unedited inputs; the routes of `certified3` in the stress test.

**What a success would and would not mean.** It would give a fallback for the inputs outside the
zones that works on the benchmark tasks and on the large collection, and a reason why the old one
did not. It would not change a certificate, would not be an independent test on the benchmark side
(there the only rewordings are the development set), and would not have been run with trained
answers: the GPU runs of the paper used the old fallback, and their scores under medium and heavy
rewording hold for the new rule only where the routes are the same. No GPU run is started for it.

### 2026-11-18 (02:05 UTC) — a look at the words, on the calibration tasks only (an exploration, not a test)

More ideas, faster, towards a stronger paper were asked for (01:45 UTC). Before designing
another one we looked at *why* every "likeness" idea failed, on the 175 calibration tasks and their
German-pivot paraphrases only (script kept in the scratch folder, `explore/lex.py`; nothing from the
700 evaluation tasks was read).

- A paraphrase drops 13.7% of the words of a definition. It drops rare words *more* often than
  common ones: 33% of the words that occur in only one calibration definition, 13% of the words
  that occur in more than 60.
- Sibling tasks (43 pairs of calibration tasks with different texts that share at least 60% of their
  words) differ in the same kind of words: 31% of the words in which they differ occur in only one
  definition, and 55% in at most three.
- So a rule "the rare words of the instruction must all be present" does not work: only 49% of the
  paraphrases keep all words that occur in one definition only (37% for words in at most two), and
  12% (5%) of the sibling pairs have the same such words.

**Reading.** Rewording and "being a sibling task" change the same words. No count of words, no
weighting by rarity and no distance between averages can tell the two apart; ideas built on key
words or on weights learned from what rewording changes are dropped without a test. What separates
"a synonym was used" from "a different thing is asked" is meaning, sentence by sentence. The fourth
idea asked a judge of meaning, but about whole instructions of up to 128 tokens (where one decisive
word is easily lost) and against twenty candidates at once. The next test asks it sentence by
sentence and only for close candidates.

### 2026-11-18 (02:05 UTC) — sixth idea, fixed before any run: a zone decided sentence by sentence

**The zone of task k:** all inputs whose instruction u (taken as a separate field, as in the fourth
idea) passes two frozen tests against the stored instruction o_k. (1) Filter: the distance between
u and o_k in the sentence encoder (all-MiniLM-L6-v2, whole text) is at most r. (2) Judge, sentence
by sentence: u and o_k are split into sentences; every sentence of u is paired with its most
similar sentence of o_k (encoder cosine) and every sentence of o_k with its most similar sentence
of u; a pair of identical sentences scores 1; any other pair gets the smaller of the two
probabilities of "entailment" (each direction) from a frozen inference model; the score of (u, o_k)
is the smallest score over all pairs, and it has to be at least τ. Every task within r is judged
(no limit on their number, so the zone is a fixed set and Theorem 1 applies). Route: the oldest
zone that holds the input; else not certified.

**Fixed now.**

- Thresholds from half B of the calibration tasks, German pivot, by split-conformal quantiles:
  r at α = 0.10 and τ at α = 0.05 (together at least 85% of the rewordings of a new task inside its
  zone, if they are drawn like the calibration ones).
- Judge: chosen on half B only, between `cross-encoder/nli-deberta-v3-xsmall`,
  `cross-encoder/nli-deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, by how
  well the sentence-by-sentence score tells a task's German paraphrase (same task) from the
  calibration instructions of *other* tasks within r (area under the curve); ties to the smaller.
- Evaluation: the 700 evaluation tasks in arrival order, one unit per task and paraphrase. The
  French, German and Spanish paraphrases have been used before, so the verdict is read on the
  **Italian pivot** (being generated; untouched). Measures as before: C, Z, P, M, R (R with the
  plain nearest encoder signature for inputs outside every zone).
- **Success and failure: the same numbers as for the third and fourth idea** (success: P ≥ 75%,
  M ≤ 5%, C ≤ 19%; against: P < 55%, or M > 9%, or C > 25%; otherwise partial).
- What we expect: P 62 to 74%, M 3 to 7%, C 5 to 10%. So most likely "partial"; a success needs the
  judge to pass about nine in ten true rewordings and to stop about four in five siblings.
- Reported whatever it shows: the French, German and Spanish pivots; the judges that were not
  chosen (on calibration only); how many tasks lie within r per text and how many sentence pairs
  are judged; the time per text; how the pairs that collide under the paper's rule are judged; the
  five benchmark tasks with the machine-written rewordings; twelve wrong certificates as examples.

**What would count against using it even after a success:** a second and a third frozen model at
inference, the instruction as a separate field, a cost that grows with the number of close tasks,
no edit bound, and a judge whose answer on a border case depends on reproducible arithmetic.

### 2026-11-18 (02:18 UTC) — a second scale for the fallback: the distance to generic text only. Fixed before it is run

**Where this comes from.** The test of the fallback normalised for the length of the opening is
still running; its numbers on the data that had been looked at before are in (they have been read;
no number of the Italian pivot has been read): on the 700 tasks it is as good as the plain rule
(91.8% on the French and German pivot), and on the benchmark tasks it repairs most of the medium
rewording (94.0% and 93.4%, plain rule 76.8% and 71.6%) but not the heavy one (63.0%, where the
rule of the paper has 93.8%). So that test will end as "against" whatever the Italian pivot shows,
and our reading (a) above was only partly right: the correlation between σ(n_k) and the radius is
0.06, so the radius is not mainly a scale for the length of the opening.

**A better reading, and the rule it leads to.** The radius of the paper is half the distance to the
nearest *known* signature, and "known" means the unrelated tasks *and the earlier tasks*. It
measures how isolated an instruction is. On the benchmark the tasks are far apart, the nearest
known signature is mostly an unrelated one, and the scale helps. At scale a sibling's nearest known
signature is its older sibling, its radius becomes tiny, and its reworded inputs look far away.
So: take the scale from the unrelated tasks only. For task k let B_k be the distance from its
signature to the nearest signature of the unrelated tasks (the ones known before the first task;
earlier tasks do not count). Rule `background`: an input goes to the task with the smallest
distance divided by B_k (ties to the older task). Rule `certified4`: the zones and certificates of
the paper unchanged, and `background` for an input outside every zone. No free setting. B_k is
fixed when task k arrives.

**The test.** The same as for the length-normalised fallback, with the same lines: **success** if
`background` gives at least 90% of the inputs to their own task under heavy rewording on five and
on fifteen benchmark tasks, at least 97% under medium rewording, and at least 88% of the
Italian-pivot paraphrased inputs after all 700 tasks; **against** if heavy rewording is below 80%
in either sequence or the Italian pivot is below 85%; otherwise partial. The Italian pivot will
also have been read for the length-normalised rule by then; for this rule it is read once, and its
rule is written here before any Italian number has been seen.

**What we expect.** Benchmark: heavy 88 to 94%, medium 97 to 100% (close to the rule of the paper,
because there the nearest known signature is mostly an unrelated one). Italian pivot: 86 to 92%
for `background`, 82 to 87% for `certified4`. We are less sure about the 700 tasks: B_k differs from
task to task, and an instruction that looks like generic text gets a small scale.

### 2026-11-18 (02:50 UTC) — results of the two fallback tests, read against their rules

Numbers: `paper_certified/routing/reroute_study.json` (rules `normalised`, `certified3`,
`background`, `certified4`; keys `verdict_normalised`, `verdict_background`, `sigma`, `stress`).
The Italian paraphrases are in `paper_certified/routing/paraphrases_it.json` (7 of 875 identical to
the original; 86.1% of the words kept).

**The fallback normalised for the length of the opening: against.** Share of inputs given to their
own task: heavy rewording 63.0% on five and on fifteen benchmark tasks (the line was 90%, and below
80% counts against), medium 94.0% and 93.4%, Italian pivot 93.9%. As written above, the scale σ(n)
is not what the radius provides (correlation 0.06).

**The fallback scaled by the distance to the unrelated tasks (`background`): success**, by the rule
written before it was run and before any Italian number had been seen.

| share of inputs given to their own task | `background` | plain nearest signature | nearest zone (the fallback of the paper) |
|---|---|---|---|
| benchmark, heavy rewording (five and fifteen tasks) | 93.8% | 60.0% | 93.8% |
| benchmark, medium rewording | 100% | 76.8% and 71.6% | 100% |
| 700 tasks, Italian pivot (untouched until now; the verdict) | 95.9% | 93.1% | 77.0% |
| 700 tasks, French and German pivot | 94.4% | 91.3% | 75.4% |
| 700 tasks, Spanish pivot | 95.3% | 93.0% | 77.6% |
| routes changed after arrival, French and German (Italian) | 1.60% (1.29%) | 2.22% (2.24%) | 0.31% (0.10%) |
| unedited inputs | 98.4% | 98.4% | 98.4% |

With the zones of the paper in front (`certified4`: same zones, same certificates, the new fallback
outside them): 88.6% of the French- and German-pivot inputs to their own task (the paper's rule:
75.4%), 86.6% on the Spanish pivot (77.5%), 91.3% on the Italian pivot (76.8%); routes changed
after arrival 1.33% (0.31%); 48.1% of the inputs inside a zone at arrival and none of them changed;
unedited inputs 83.1%, as before, because the zones and their collisions are the same. On the
benchmark tasks the shares are those of the paper's rule in every wording (100%, 100%, 100%, 93.8%;
no route changed).

**Our guesses.** Benchmark as expected. On the 700 tasks we had expected 86 to 92% for `background`
and 82 to 87% for `certified4`; both are higher (95.9% and 91.3% on the Italian pivot).

**Why it works, as far as the numbers show.** B_k has a correlation of 0.52 with the radius. On the
benchmark the radius *is* half of B_k (for all five tasks and for 14 of the fifteen), so the two
rules agree there. At scale the radius is half of B_k for only 28.6% of the 700 tasks: for the
others an earlier task is nearer than every unrelated one, the radius shrinks, and the old fallback
turns away the reworded inputs of exactly those tasks. Taking the scale from the unrelated tasks
only keeps the first effect and drops the second.

**What it costs.** (1) Stability outside the zones: 1.3 to 1.6% of the paraphrased inputs change
their route after their task has arrived (0.3% with the old fallback). (2) In the stress test a
look-alike task takes some uncertified old inputs that the old fallback kept: with the light
rewording as the look-alike's definition, 2 of the 100 old inputs with the medium wording and 20
with the heavy wording change (0 and 0 with the old fallback); the 200 certified old inputs
(original and light wording) do not change under either. (3) It is a variant: every GPU run and
every other route test of the paper used the old fallback. Where the routes are the same the
scores are the same; a check of that, input by input, is being made for the benchmark sequences.

**The Italian pivot for the other rules** (own task at the end; routes changed): the paper's rule
76.8% and 0.10% (P 47.0%, M 5.9%, R 76.8%); nearest zone 77.0%, 0.10%; plain nearest signature
93.1%, 2.24%; zones then plain signature 88.5%, 2.24%; length-normalised 93.9%, 1.67%.

**Consequence.** This goes into the paper as measured: a fallback for the inputs outside the zones
that is right on the benchmark and at scale, found at the end, with its rule fixed in advance and
its verdict on paraphrases that had not been read; the zones, the certificates and the theorem are
untouched. The statement "we have no fallback that is right in both places" is replaced. The
sentence "at scale the switch adds the proof, not better routing" stays true for the comparison
with the best rule without zones (94.4 to 95.9% against 88.6 to 91.3% with zones): the zones cost
about five points of right routes and give a proof for about half of the inputs.

### 2026-11-18 (02:58 UTC) — a new direction: no instruction at all, and a bound on forgetting. Fixed before any run

**The aim set at 02:51 UTC.** Find a big flaw in the field: show with human-written rewordings that most published continual-learning methods only look good because the instruction gives away the task, or another flaw if that is the better one. Fix it: build a method that reaches near-zero forgetting without any instruction and comes with a non-trivial guarantee, and prove it.

**What can and cannot be done before the deadline (12:00 India time), honestly.**

- Rewordings written by people do not exist in this project, so none can be called
  "human-written". Running the published methods without instructions needs their official code on a GPU for
  several hours and at least one retry; that does not fit before the deadline. So a claim about
  "most published methods" cannot be shown tonight. What the existing runs show stays as it is in
  the paper: on these benchmarks a rule without any training recognises the task from the
  instruction for every test input, and the two published methods we ran lose 8.6 and 10.4 points
  under the heavy rewording where the certified switch loses 5.6.
- The second half can be attempted now: a version of the method that does not use the instruction,
  with a guarantee that is more than "inside a zone nothing changes".

**The guarantee (to be written into the paper with its proof).** When task k arrives, hold out m
of its training inputs (calibration inputs). Fix a score s_k(x), "how unlike task k does x look",
computed from frozen parts and the other training inputs of the task, not from the calibration
inputs. Let q_k be the ⌈(1−α)(m+1)⌉-th smallest calibration score and Z_k = {x : s_k(x) ≤ q_k};
the zone is never changed, and an input goes to the oldest zone that holds it.
*Claim.* If a test input X of task k is exchangeable with the calibration inputs of task k (for
example drawn from the same distribution), then for any model, any later tasks and any way of
training them: the probability that the answer to X changes at any time after task k has been
learned is at most α. *Proof.* X lies in Z_k with probability at least 1−α (split-conformal
argument, as in the coverage proposition of the paper). If X lies in Z_k, the oldest zone that
holds X belongs to a task ≤ k and can never change, so by the write-once theorem the answer never
changes. *Consequence.* For any fixed score of an answer with values between 0 and 100, the expected
forgetting of task k is at most 100·α points. This holds for all test inputs of the task, not only
for a marked subset, and it needs no instruction. *What it does not say:* nothing about whether the
task could be learned (inputs of task k inside older zones are answered by older adapters; their
share is measured and reported at admission); the probability is over calibration and test input
together; the test inputs must be like the task's training inputs.

**Pilot A, routes only, CPU (`paper_certified/noinstruction_study.py`).**

- Data: the official benchmark files, instances only, the definition is never read. Two sequences:
  the 15 tasks of SuperNI order 1, and the 15 tasks of the Long Sequence benchmark in its official
  order 3; and the two together as one sequence of 30 tasks. Training inputs (800 per task) for the
  zone, the official test inputs (never used for the zone) for the measures.
- Frozen feature, primary: the weighted, rescaled average of the T5-base word-vectors of all tokens
  of the instance (cut at 256 tokens), as in the rule `centre` of the paper. Secondary: the frozen
  sentence encoder all-MiniLM-L6-v2 on the instance.
- Score, primary: the distance to the 10th nearest of the task's training features (the
  calibration inputs excluded). Secondary: the distance to the mean of the task's training features.
- m = 200 calibration inputs per task (random, seed 0); α = 0.05 primary, 0.1 and 0.2 reported.
  Fallback for inputs outside every zone: the task with the smallest score divided by its q_k.
- Measures per task and averaged: test inputs inside the own zone (coverage); test inputs inside an
  older zone when the task arrives (blocked); test inputs given to their own task after the last
  task; test inputs whose route after the last task differs from the route when their task arrived
  (changed).
- **Lines, fixed now.** (1) The claim: on each sequence the average changed share must be at most α
  plus two standard errors, and the average coverage at least 1−α minus two standard errors; if
  not, the assumption (test inputs like training inputs) fails on these files and the paper has to
  say so. (2) Use: "works without instructions" on a sequence if on average at least 90% of the test
  inputs are given to their own task and at most 10% are blocked; "does not" if fewer than 75% or
  more than 25% blocked; otherwise partial.
- What we expect: the claim holds (coverage near 95%, changed below 5%). SuperNI order 1: works
  (90 to 98% to their own task). Long Sequence: partial at best, because several of its tasks have
  the same kind of input with different answers (review sentiment in four tasks, sentence pairs in
  four): without an instruction those cannot be told apart by anyone, and they should show up as
  blocked.

**Pilot B, with trained answers on a small model, CPU (`certified_switch/`).** The stand-alone code
with `t5-small`, the five tasks of the paper, the definition removed from every input (training and
test), 400 training and 50 test inputs per task, 5 epochs, seed 0. Rules compared on the same
adapters: all adapters on; the zones of Pilot A's primary setting (α = 0.05) with their fallback.
**Lines:** "works" if with the zones at most 10% of the 200 answers to earlier tasks change and
forgetting is at most 3 points, while with all adapters on more than half change; "does not" if
more than 20% change. One seed and a small model: a demonstration, not a result about scale. If it
is not finished by 04:30 UTC it is dropped and the paper says that no trained answers exist for
this variant.

**No GPU run** is started for this tonight: there is no time to verify one. The notebook for the
published methods without instructions is the next step and is described in the log.

### 2026-11-18 (02:57 UTC) — result of the sixth idea (zone decided sentence by sentence), read against its rule

Numbers: `paper_certified/routing/sentence_judge_study.json`, script
`paper_certified/sentence_judge_study.py` (about 47 minutes on CPU).

Calibration as fixed (half B, 87 tasks, German pivot): r = 0.588; the smallest judge was chosen
(area under the curve 0.965 for `nli-deberta-v3-xsmall`, 0.964 and 0.954 for the two base-size
models). The threshold came out as τ = 0.0017: for about 4% of the true rewordings one sentence
pair gets almost no "entailment", so a threshold that keeps 95% of them has to be nearly zero, and
at that threshold 24% of the nearest other instructions still pass.

| 700 evaluation tasks | C | Z | P | M | R |
|---|---|---|---|---|---|
| **Italian pivot (untouched; the verdict)** | 12.4% | 87.0% | 76.1% | 10.9% | 88.7% |
| French pivot | 12.4% | 85.3% | 75.4% | 10.0% | 89.1% |
| German pivot | 12.4% | 83.1% | 74.3% | 9.0% | 89.7% |
| Spanish pivot | 12.4% | 81.7% | 73.9% | 8.0% | 89.9% |

**Verdict by the rule written before the run: against the idea** (M = 10.9% on the Italian pivot is
above the 9% limit). P (76.1%) and C (12.4%) are on the right side of the success lines; the wrong
certificates are not.

**What it shows.** (1) It is the best point so far: three in four rewordings certified for their
own task with half the wrong certificates and half the collisions of the sentence encoder alone
(73.1%, 17.1%, 22.9%). (2) The wrong certificates are the same siblings as always: the date formats
day-month and month-day, the ATOMIC tasks "intent" and "reaction", "all elements except the first
i" and "all elements from index i to j". Of the 142 colliding pairs with different texts the judge
still passes 50. (3) Judging sentence by sentence did not make the judge stricter where it counts:
the threshold that keeps the rewordings is so low that it stops only a part of the siblings. A
small inference model does not know that "reaction" and "intent" ask for different things.
(4) On the five benchmark tasks the rewordings written by machine are mostly outside these
zones (4 of 5 light, 0 of 5 medium, 1 of 5 heavy inside), all routed to the right task. (5) Cost:
0.84 seconds per text on three CPU threads; a text has 1.6 tasks within r on average (14 at most).

**A second exploration, on the calibration tasks only (not a test):** a round trip through a frozen
translator as a "normal form" (English, German, English). After it, 55% of the sentences of a
French-pivot paraphrase are identical to the sentences of the original's round trip, and the share
of sibling comparisons in which the sibling is farther than the own paraphrase (word-level measure)
rises from 42% to 73%. Not enough for a zone; not tested as one.

**Where the six ideas leave the zones.** None met its bar. The frontier moved (from 40% certified
with 8% wrong, to 76% certified with 11% wrong), and the wall did not: what separates a rewording
from a sibling task is the meaning of one or two words, and nothing frozen and small that we tried
knows it. A large frozen judge (the base model itself, asked directly) is the open experiment.

### 2026-11-18 (03:07 UTC) — first numbers of Pilot A, and a second score, fixed before it is computed

**Pilot A, primary setting (word-vector average of the instance, distance to the 10th neighbour,
level 0.05): the claim holds, the use fails.** On the three sequences (15 SuperNI tasks, 15 Long
Sequence tasks, all 30): coverage 96.1%, 94.9%, 95.5% (at least 95% promised on average); test
inputs whose route changed after their task arrived 0.7%, 1.1%, 0.5% (at most 5% promised). But
86.5%, 88.3% and 93.2% of the test inputs lie inside the zone of an older task when their task
arrives, and only 12.7%, 10.6% and 6.3% are given to their own task. Verdict by the lines fixed
before: claim "holds", use "does not". Pilot B shows the same on the small model: 377 of the 400
training inputs of the second task lie inside the zone of the first. With this score a zone is "as
close to the task's inputs as the task's own outliers are", and in this feature other tasks'
inputs are that close too. (The sentence-encoder feature of the same run is still being computed.)

**A second score, designed after these numbers, on the same sequences.** What worked for the
fallback an hour ago was to measure everything in units of the distance to unrelated text. The
same idea for the zones: an input belongs to a task if it is clearly closer to the task's inputs
than to *unrelated* inputs. Score: a_k(x) = d_k(x) / d_0(x), where d_k is the distance to the 10th
nearest fit input of task k as before and d_0 is the distance to the 10th nearest input of a fixed
pool of unrelated inputs (about 5,000 inputs, a few from each of the real tasks of the collection
of Section "875 real tasks" that are not in the sequences; instances only, no definitions; fixed
before the first task). Everything else as before: threshold by the split-conformal quantile on
the task's own held-out inputs, level 0.05, the oldest zone wins, fallback to the smallest
a_k / q_k. The theorem covers this score as well (it depends only on frozen things and on the fit
inputs). No free setting.

- Same sequences, same measures, same lines ("works": at least 90% of the test inputs to their own
  task and at most 10% blocked; "does not": below 75% or above 25% blocked). Both features are
  reported; the word-vector feature stays the primary one for the verdict, because it needs no
  second model.
- It is a second attempt on data that the first attempt has already been run on; it has no free
  setting, but it is not an independent test, and the paper has to say so. A test on tasks that
  have not been used is the next step and is not possible tonight.
- What we expect: far fewer blocked inputs (10 to 40%), still not "works" on the Long Sequence
  tasks, where four tasks are review sentiment and four are sentence pairs.
- Pilot B is repeated once with this score (same settings, same lines). Only if that run "works"
  is a GPU run of the same script started (T5-base, the budget of the paper, two seeds), as the
  earlier rule allows.

### 2026-11-18 (03:13 UTC) — Pilot B, first run (plain score): read, and its lines corrected before the second run

Numbers: `certified_switch/demo_results/no_instruction_t5-small_seed0.json` (t5-small, five tasks,
no definition in any input, 13 minutes on CPU).

The file says "works", and by the lines we wrote it is: with the zones no answer to an earlier task
changed and forgetting is zero. But the lines were badly chosen, and the result is empty: 377 of
the 400 training inputs of the second task lie inside the zone of the first task, and so do nearly
all inputs of the later tasks; only 4% of the test inputs of the second task and none of the third
to fifth are given to their own task. The scores right after learning are 98.1 for the first task
and 4.0, 18.0, 0.0 and 0.5 for the others: with these zones the later tasks were never served by
their own adapters, so there was nothing to forget. The bound did what it says (nothing changed);
the method did not work. **This run counts as "does not work".**

**Correction of the lines, written before the second run (ratio score) is read.** "Works" needs
both: (1) as before, with the zones at most 20 of the 200 answers to earlier tasks change and
forgetting is at most 3 points, while with all adapters on more than 100 change; and (2) new: the
tasks are learned, that is, the average score right after learning with the zones is at least 90%
of the average score right after learning with all adapters on (the adapter of a task alone on its
own task), and on average at least 80% of the test inputs are given to their own task. A GPU run is
started only if both hold.

### 2026-11-18 (03:25 UTC) — the second score (ratio to unrelated inputs) and Pilot B, read against their lines

Numbers: `paper_certified/routing/noinstruction_study.json` (score `ratio10`, key `verdict_ratio`;
pool of 5,165 unrelated inputs), `certified_switch/demo_results/no_instruction_t5-small_seed0.json`
and `..._seed0_ratio.json`.

**Pilot A, word-vector feature, level 0.05, averages over the tasks of each sequence:**

| sequence | score | inside own zone | route changed after arrival | blocked by an older zone | given to own task |
|---|---|---|---|---|---|
| 15 SuperNI tasks | distance to the 10th neighbour | 96.1% | 0.7% | 86.5% | 12.7% |
| 15 SuperNI tasks | ratio to unrelated inputs | 94.3% | 0.0% | 65.5% | 33.7% |
| 15 Long Sequence tasks | distance to the 10th neighbour | 94.9% | 1.1% | 88.3% | 10.6% |
| 15 Long Sequence tasks | ratio to unrelated inputs | 94.3% | 1.1% | 81.0% | 17.8% |
| all 30 | distance to the 10th neighbour | 95.5% | 0.5% | 93.2% | 6.3% |
| all 30 | ratio to unrelated inputs | 94.3% | 1.5% | 78.1% | 20.3% |

**Verdicts by the lines fixed before.** The claim holds in every row (at most 5% changed promised,
0.0 to 1.5% found; at least 95% inside the own zone promised on average, 94.3 to 96.1% found,
within two standard errors). The use fails in every row ("does not": far more than 25% blocked).
The ratio score helps (on the SuperNI tasks a third of the inputs reach their own task instead of
an eighth; five of the fifteen tasks are told apart almost completely: the three summarisation
tasks, the dialogue task, the reading-comprehension task), and it is not enough: the short-input
tasks (the two sentiment tasks, emotion, relation extraction, the question tasks) lie inside older
zones.

**Pilot B (t5-small, five tasks, no definition, trained answers).** With all adapters on: average
score 64.4 right after learning, 9.6 at the end; all 200 answers to earlier tasks changed. With the
zones and the ratio score: 36.7 right after learning, 33.6 at the end, forgetting 3.8 points, 8 of
200 answers changed (4%; the bound allows 5% on average), none of the 188 inputs inside their own
zone changed; but only the first two tasks were learned (blocked test inputs: 0, 0, 23, 48 and 45
of 50). By the corrected lines: **does not work** (the tasks are not learned: 36.7 against 64.4;
half of the inputs reach their own task). With the plain score (first run) only the first task was
learned. Also: without its definition the small model learned two of the five tasks only to the
level of guessing even with all adapters on (48.0), so the instruction carries part of the task
itself, not only its name.

**What this direction has reached tonight, and what it has not.** Reached: a guarantee that needs
no instruction and is more than "inside a zone nothing changes": at most a chosen share of all
inputs of a task can ever change, for any model and any later tasks; proved; and in every run the
measured share was below the promise. Not reached: a method that *works* without instructions.
With frozen word-vector features the zones of early tasks swallow later tasks, which is reported
when they arrive (nothing is silent), but then those tasks are not learned. What is missing is one
thing, and the bound makes it the only thing: a frozen feature in which the inputs of different
tasks are apart when no instruction is given. The frozen model's own hidden states and a sentence
encoder are the candidates (the sentence encoder is being computed on the same sequences). No GPU
run was started, because no pilot "worked".

**On the first half of what was asked.** Not shown, and not claimed: that published methods only
look good because the instruction gives the task away. What the paper can say from its own runs is
stated in its introduction: with unedited instructions these benchmarks cannot tell retention from
recognition. The decisive experiment (the published methods without instructions, and rewordings
written by people) is described in the log as the next step.

### 2026-11-18 (03:37 UTC) — a correction to the entry on the fallback scaled by the unrelated tasks: one cost was missing

The last careful read of the paper found that our entry above gave the stress test only for the
look-alike whose definition is the light rewording (2 and 20 uncertified old inputs lost). The same
file has the other two cases. With the medium rewording as the look-alike's definition, the new
fallback lets it take 80 of the 100 old inputs with the heavy wording, where the fallback of the
paper lets it take 2 (the 100 inputs that carry the look-alike's own wording go to it under both).
With the heavy rewording as its definition both lose the 100 inputs with that wording. No certified
input changes in any case. So the new fallback is clearly less stable for uncertified inputs when a
look-alike task arrives; the paper now says so next to the gain at scale. Also made precise: "the
same routes on the benchmark" means the same route for every test input from the arrival of its
own task on, which is everywhere a score is taken; before an input's own task exists the two
fallbacks can differ (110, 223 and 7 inputs of the fifteen-task sequence).

### 2026-11-18 (03:55 UTC) — the sentence-encoder feature of Pilot A (the secondary feature of the plan)

Numbers: `paper_certified/routing/noinstruction_study.json`, feature `minilm`. Level 0.05, averages
over the tasks; SuperNI 15 / Long Sequence 15 / all 30.

| score | inside own zone | route changed | blocked by an older zone | given to own task |
|---|---|---|---|---|
| distance to the 10th neighbour | 94.7 / 93.7 / 94.2% | 1.4 / 3.0 / 2.3% | 46.6 / 53.2 / 63.5% | 51.0 / 43.5 / 34.1% |
| ratio to unrelated inputs | 95.9 / 94.4 / 95.2% | 1.8 / 1.4 / 1.5% | 39.9 / 54.3 / 62.6% | 58.0 / 43.8 / 35.7% |

The claim holds again in every row (changed routes below 5%). The use still fails by the lines
fixed in advance (more than 25% blocked everywhere), but the sentence encoder tells the tasks apart
clearly better than the average of word-vectors: on the SuperNI tasks 40% of the test inputs are
blocked instead of 65%, and 58% are given to their own task instead of 34%. Still blocked with it:
the two tweet tasks (sentiment, emotion), the dialogue-message task, part of the reading and
question tasks. This feature was the secondary one, so it changes no verdict; it points to where
the next step is: a stronger frozen feature of the whole input (the frozen model's own hidden
states have not been tried). A note on the set-up that the run brought out: four tasks have fewer
than 400 training inputs (the smallest 100), so their zones were calibrated on half of them instead
of 200; the three very small early tasks get loose zones that hold many later inputs.

### 2026-11-18 (05:43 UTC) — one more feature for the zones without instructions: the frozen model's own hidden states. Fixed before it is run

**Why now.** The question at 05:40 UTC was why the search for ideas had stopped. Part of the answer is
that it should not have: after the final papers were sent (03:56 UTC) there was time, and the entry
above names this feature as the next step. It is run now. The papers that were sent stay as they
are until this result has been read and checked.

**Fixed now.**

- Feature: the frozen T5-base encoder without any adapter; the mean of its last-layer states over
  the tokens of the instance (cut at 256 tokens). No second model is needed: it is the base model
  that the adapters sit on. Everything else as in Pilot A: instances only, the definition never
  read; scores "distance to the 10th neighbour" and "ratio to unrelated inputs"; level 0.05; the
  oldest zone wins; fallback to the smallest score divided by its threshold.
- To fit the time on a CPU: at most 400 training inputs per task (random, seed 0), of which 200
  are calibration inputs (half of the task's training inputs if it has fewer than 400), and a
  random 2,000 of the 5,165 unrelated inputs as the pool (seed 0). The official test inputs as
  before. Primary sequence: the 15 SuperNI tasks of order 1. The Long Sequence tasks and all 30
  only if time allows; they are reported, not part of the verdict.
- **Lines, the same as before.** The claim: average share of changed routes at most 5% plus two
  standard errors, average coverage at least 95% minus two standard errors. The use, with the
  ratio score: "works" if on average at least 90% of the test inputs are given to their own task
  and at most 10% are blocked; "does not" if fewer than 75% or more than 25% blocked; otherwise
  partial.
- What we expect: fewer blocked inputs than with the sentence encoder (40% on these tasks), perhaps
  20 to 35%; "partial" or "does not". The two tweet tasks ask different things about the same kind
  of input and should stay blocked whatever the feature is, so "works" would surprise us.
- If it "works": the small trained model is run with this feature (CPU), and only if that works
  too is the prepared GPU notebook started. If it does not: one more row in the table of the paper.

### 2026-11-18 (06:05 UTC) — a GPU run approved: the published methods without instructions. Fixed before it is started

**Approval.** Asked at 05:44 UTC whether to start a Kaggle run of about two hours for this, the answer
was "Yes, start it"; results arriving after 12:00 India time may still go into the
paper. The papers sent at 03:56 UTC stay the safe version until then.

**The question.** On these benchmarks the instruction names the task. Do the two published methods
that we ran (InfLoRA and GainLoRA, official code) still hold their answers when the instruction is
taken away? This is the first half of what was asked for on 2026-11-18 ("show that published
methods only look good because the instruction gives away the task"), for the two methods and the
reduced setting that we have, not for "most published methods".

**The run** (notebook `silent_adapters/five_tasks_noinstruction_colab.ipynb`, being built and
checked). Exactly the five-task run of the paper (T5-base, 10 epochs per task, seed 42, the same
code and patches), with one change: the definition of every task is replaced by the empty string in
every data file, for training and for testing, so that every input is the same fixed text followed
by the instance. Three jobs: InfLoRA, GainLoRA, and the project's own job, whose rule "all" gives
plain adapters and whose rule "centre" gives a router that looks at the whole input (the rules that
need an instruction are expected to fail and are reported as they come out). The re-test counts, as
before, the answers to the 400 test inputs of the first four tasks that differ at the end from the
answer right after their task was learned.

**Reading, fixed now.** For each published method, compared with its own run *with* instructions
(three seeds: InfLoRA learned 60.0, forgetting −3.1, 370 of 1,200 answers changed, that is 123 per
400; GainLoRA learned 61.4, forgetting −0.5, 65 of 1,200, that is 22 per 400):

1. First: was anything learned? If the average score right after learning is below 60% of the
   value with instructions, the run says "this method could not learn these tasks without their
   instructions in this budget", and nothing about forgetting (the mistake of Pilot B is not
   repeated: nothing learned means nothing to forget).
2. If it learned: "holds its answers less well without instructions" if forgetting rises by at
   least 5 points or the number of changed answers is at least twice the per-seed number with
   instructions and at least 60 of 400 higher. "No sign of that" if forgetting rises by less than
   2 points and the changed answers by less than a quarter. Anything else: unclear.
3. It is one seed, five tasks, a small model and a tenth of the published budget. Whatever comes
   out is a first look at two methods, and the paper will call it that.

**What we expect.** We do not know. GainLoRA decides how much each adapter contributes from the
input, and without the instruction its inputs carry less about the task, so we expect it to change
more answers (perhaps 60 to 150 of 400). InfLoRA keeps new updates away from the directions of old
inputs; without the shared instruction tokens those directions overlap more, so more interference
is plausible, but its answers already change often with instructions. Learning itself may suffer
for all methods: on the small model two of the five tasks reached only the level of guessing
without their definition. Plain adapters should forget as badly as before.

### 2026-11-18 (06:10 UTC) — the target for the work from here on, and the next idea. Fixed before any run

**The target set at 06:08 UTC:** instruction-free task recognition that works; full learning with no blocked tasks; a non-trivial forgetting guarantee; to beat the state of the art. Experiments and new ideas go on until there are signs of this, and a GPU run is started only when the method is thought to work. The papers sent at 03:56 UTC stay as the safe version; this is new work.

**What "signs" will mean, fixed now so that we cannot move it later.** On routes, without any
instruction, on the 15 SuperNI tasks: on average at least 90% of the test inputs given to their own
task and at most 10% blocked by an older zone, with the forgetting bound met (at most 5% of the
routes changed). Then the same on tasks that no test has used (below). Then with trained answers
on the small model: the corrected lines of Pilot B (tasks learned: average score right after
learning at least 90% of that with all adapters on; at most 20 of 200 answers changed; forgetting
at most 3 points). Only then a GPU run of the method. "Beating the state of the art" can only be
measured on a GPU and only against the published methods run in the same setting without
instructions; that baseline run is approved and is being prepared.

**Why the zones failed so far, and the idea.** A zone was "everything about as close to the task's
inputs as the task's own outliers". That is a test with one side only: it knows what the task looks
like and nothing about what else exists. Every later task that is not far away falls inside. What
is known when task k arrives is more than the task itself: a pool of unrelated inputs, and the
stored features of all earlier tasks. So the zone can be a *boundary*: train, on frozen features, a
small classifier that tells the inputs of task k from everything known before (pool and earlier
tasks), freeze it, and let the zone be "on the task's side of that boundary, and close to the
task's inputs". Both thresholds are set on held-out inputs of the task, so the bound on forgetting
still holds (the zone depends only on things that are fixed when the task arrives and not on the
held-out inputs).

**The rule (`boundary`), fixed now.**

- Frozen feature: the sentence encoder (all-MiniLM-L6-v2 on the instance), because its features of
  all 30 tasks and of the pool are already computed; the frozen T5-base encoder as a second
  feature when its run is finished.
- When task k arrives: positives are its fit inputs; negatives are the pool of unrelated inputs
  (5,165) and the fit inputs of all earlier tasks. A logistic regression with L2 penalty (strength
  1, classes weighted to balance) is trained on the features and frozen. Score one: minus its
  decision value. Score two: the distance to the 10th nearest fit input of the task, as before.
  Thresholds: for each score the split-conformal quantile at level 0.025 on the task's calibration
  inputs, so that both tests together keep at least 95% of the task's inputs on average. Zone: both
  tests passed. Oldest zone wins. Outside every zone: the task with the largest margin of score one
  below its threshold.
- Same three sequences, same measures, same lines as before. These sequences have been used by the
  earlier scores, so this is development, and it is said so. The verdict that counts is read on
  **tasks that no test has used**: 30 tasks drawn at random (seed 0) from the collection of 875
  real tasks, without the 15 benchmark tasks, with at least 1,000 instances each (800 for the
  zone, 200 for the test), instances only, in a random order; they are being fetched now. Lines on
  the fresh tasks: "works" if on average at least 85% of the test inputs are given to their own
  task and at most 15% are blocked (the bar is lower than on the benchmark because a random draw
  from this collection contains sibling tasks that share their inputs and cannot be told apart by
  anyone without an instruction; those are reported by name); "does not" below 70% or above 30%.
- What we expect on the benchmark sequences: a large step (blocked from 40% to below 15% on the
  SuperNI tasks). On the fresh tasks we do not know; the sibling tasks will decide it.

### 2026-11-18 (06:20 UTC) — a second new idea, written while the first is running: the newest claim wins, but it may not take the held-out inputs of older tasks. Fixed before any run

**Why.** With "the oldest zone wins" an old task is perfectly protected inside its zone and a new
task is blocked wherever an old zone is too wide. That is where "no blocked tasks" fails. The other
order, "the newest zone wins", never blocks a new task, but then old inputs can be taken away. The
idea: let the newest zone win, and make every new zone *prove at admission* that it takes almost
nothing from the older tasks. The proof uses what is stored anyway: the held-out (calibration)
inputs of each older task.

**The rule (`claims`), fixed now.** Frozen feature as before (sentence encoder). When task k
arrives, the same classifier as in the rule `boundary` is trained (inputs of task k against the
pool and the *fit* inputs of all earlier tasks) and frozen; score a_k(x) = minus its decision
value. The threshold q_k is the smaller of two numbers: (1) the split-conformal quantile at level
0.025 on the task's own calibration inputs; (2) the smallest score a_k of any stored calibration
input of any earlier task. Zone: a_k(x) < q_k (strictly below). So a new zone contains none of the
stored held-out inputs of the older tasks. Route when tasks 1..T have arrived: the *newest* zone
that holds the input; outside every zone, the task with the largest margin q_k − a_k(x). What is
stored per task: the frozen classifier, q_k, the features of its fit inputs and of its m
calibration inputs.

**The guarantee (to be written out and checked before it goes into any paper).** Let X be a test
input of task j, exchangeable with the m stored calibration inputs of task j, and let the later
tasks not depend on these held-out inputs. For each later task k, the chance that X lies in its
zone is at most 1/(m+1): the zone lies strictly below the smallest of m calibration scores, and X's
score is equally likely to have any rank among the m+1. If X is in its own zone when task j
arrives and in no later zone, its route never changes (the newest zone that holds it stays j). So
the chance that the route of X ever changes after task j is at most
(1 − coverage of zone j) + (number of later tasks)/(m+1), where the coverage of zone j is measured
on its own held-out inputs when it arrives and reported. With m = 200 and 14 later tasks that is
at most 7 points on top of the uncovered share; the bound gets better with more stored held-out
inputs. And no task is blocked by an older one: at its arrival every input inside its zone is its
own. What can go wrong instead: if a new task looks like an old one, the second threshold makes its
zone small, and the admission report shows a low coverage for the newcomer.

**The test.** In the same script as the rule `boundary`, on the same data: the three benchmark
sequences (development) and the 30 fresh tasks (the verdict). Measures: coverage of the own zone
at arrival; given to the own task at arrival and after the last task; route changed after
arrival; "taken": share of a task's test inputs that lie in the zone of a *later* task after the
last task; and the bound above per task next to the measured changed share.
**Lines:** the guarantee is "met" if on average the changed share is at most the average bound
(plus two standard errors) on each sequence. Use on the fresh tasks: "works" if on average at
least 85% of the test inputs are given to their own task after the last task *and* at least 85%
right after arrival (so that the newcomers are learned), "does not" below 70%; on the SuperNI
sequence 90% and 75%. There is no "blocked" measure here by construction; what replaces it is the
coverage at arrival, which is reported per task.
**What we expect.** Given to the own task: 90% or more on the SuperNI tasks, 80 to 90% on the fresh
tasks; changed routes a few per cent, below the bound; a low coverage for a few newcomers that
share their inputs with an older task.

### 2026-11-18 (07:00 UTC) — the rules `boundary` and `claims` on the benchmark sequences (development), and how the next version is chosen

Numbers: `paper_certified/routing/boundary_study.json` (part 1; the part on the 30 fresh tasks is
still being computed). Sentence-encoder feature, no instructions, averages over the tasks.

| sequence | rule | given to own task after the last task | route changed after arrival | blocked / coverage at arrival |
|---|---|---|---|---|
| 15 SuperNI | distance to 10th neighbour (oldest wins) | 51.0% | 1.4% | 46.6% blocked |
| 15 SuperNI | ratio to unrelated inputs (oldest wins) | 58.0% | 1.8% | 39.9% blocked |
| 15 SuperNI | `boundary` (oldest wins) | 67.2% | 1.1% | 31.2% blocked |
| 15 SuperNI | `claims` (newest wins) | **87.7%** | 1.3% | 81.2% of a newcomer's inputs inside its zone |
| 15 Long Sequence | `boundary` | 49.5% | 1.0% | 48.8% blocked |
| 15 Long Sequence | `claims` | 70.6% | 0.8% | 57.6% inside |
| all 30 | `claims` | 70.9% | 1.0% | 62.5% inside |

**Reading against the lines (development only).** `boundary`: the bound holds, the use "does not"
(31% blocked). `claims`: the guarantee is met (1.3% changed against a bound of 24.6%, which is
loose because it counts every input outside a newcomer's zone as lost); the use is "partial" on the
SuperNI tasks (87.7%, the line is 90%) and "does not" reach 75% on the Long Sequence tasks. Letting
the newest zone win removes the blocking, as intended: twelve of the fifteen SuperNI tasks have 92
to 100% of their inputs at their own adapters. What is left is recognition itself: the classifier
alone, without zones, gives 92.4% (SuperNI) and 83.3% (Long Sequence) to the right task, and the
weak tasks are the ones that share their kind of input with an older task (tweets: sentiment and
emotion; sentence pairs: MNLI, CB, RTE; reviews: IMDB and SST-2; passage questions: BoolQ and
MultiRC; Quoref after the news task). For those the second threshold makes the newcomer's zone
small (coverage 8 to 49%).

**How the next version is chosen (fixed now).** The benchmark sequences are development data from
here on: variants may be compared on them. The variants, all inside the rule `claims` (the newest
zone wins; a zone may not hold stored held-out inputs of older tasks; the same guarantee):
(a) one tailored boundary against each older task (a classifier of task k against task j alone,
with its own threshold just beyond the held-out inputs of task j) in place of one boundary against
everything; (b) the same feature with 15 simple numbers that describe the shape of the text added
(length, word length, shares of capitals, digits, punctuation, line breaks, and whether it has @,
# or a web address), scaled with the pool's mean and spread; (c) allowing r − 1 held-out inputs of
an older task inside a new zone, for r = 1, 2, 3 (the bound then has r/(m+1) per later task);
(d) the strength of the penalty of the classifier (1, 10, 100). The version with the highest
average share given to the own task on the 15 SuperNI tasks, among those with at most 3% changed
routes, is frozen. **Its verdict is then read once on the second batch of 30 fresh tasks** (fetched
at 06:14 UTC, not read by anything), with the lines already fixed for fresh tasks: "works" if on
average at least 85% of the test inputs are given to their own task after the last task and at
least 85% right after arrival; "does not" below 70%. A third batch is kept in reserve.

### 2026-11-18 (07:01 UTC) — the approved GPU run of the published methods without instructions is started

Notebook `silent_adapters/five_tasks_noinstruction_published.ipynb` (made by
`make_notebook_noinstruction.py`; InfLoRA and GainLoRA only, one per GPU; the definition of every
task replaced by the empty string in all data files; seed 42; everything else as in the five-task
run of the paper), uploaded to our Kaggle account as
`bammidishivkumar/no-instruction-published-five-tasks`, version 1, private, at 07:01 UTC. Tested
before on a CPU in the notebook's self-test mode (the definition removal on a copy of the data, the
prompt builders, and the start of training of both methods). Expected to take 45 to 50 minutes.
Its reading was fixed above (entry of 06:05 UTC) before it was started. The project's own job is
not in this run: it needs a training mode without blocking and the routes of the new rules, which
are being prepared, and it will run in a second notebook only when the CPU tests say it works.

### 2026-11-18 (07:07 UTC) — two more shapes added to the development grid, before it is run

Other shapes were asked for again. Added to the grid of the entry above, before any variant
has been computed, inside the same rule `claims` (the newest zone wins; a zone may not hold more
than r − 1 stored held-out inputs of an older task; same guarantee):
- **balls around the task's own inputs ("pair-knn"):** an input is on task k's side against an
  older task j if it is closer to its 10th nearest stored input of task k than to its 10th nearest
  stored input of task j; the threshold again lies just beyond the held-out inputs of task j. No
  classifier is trained; the zone is a union of balls cut by the same kind of union of the other
  task.
- **curved boundaries ("pair-rff"):** the features are first passed through 2,048 random
  cosine waves (random Fourier features, one fixed draw), and the pairwise classifier is trained
  there; in the original feature space its boundary is curved.
So four shapes are compared on the development sequences: one flat boundary against everything, one
flat boundary per older task, balls, curved boundaries; each with and without the 15 numbers that
describe the shape of the text. The choice rule and the fresh batch for the verdict are unchanged.

### 2026-11-18 (07:12 UTC) — a different kind of answer, not a shape: one joint recogniser, and a forgetting certificate from held-out inputs. Fixed before any run

**Why.** The aim is not to stay with shapes of zones. Stepping back: with one frozen adapter per
task, the answer to an input changes only if its *route* changes. So two things are needed, and
they can be separated. (1) A recogniser that is as good as possible. (2) A statement about how
often a route can change. So far both were forced into one object, the zone. They need not be.

**The recogniser.** At any time, one classifier over all tasks learned so far, on frozen features
of the whole input. Two forms: (a) a multinomial logistic regression retrained at every arrival on
the stored features of the fit inputs of all tasks (the same memory as the rule `claims`); (b) a
ridge regression on random cosine features of the frozen feature, whose two sums (the matrix of
second moments and the sum per task) are simply added up when a task arrives, so that after any
number of tasks, in any order, it is exactly the classifier one would get from all tasks at once,
and no input and no feature has to be stored (this is the known "analytic" way of continual
classification; it is used here as a strong recogniser and as a baseline, and is not claimed as
new). No task can be blocked: every task is a class.

**The certificate (the new part).** Each task keeps m held-out inputs that the recogniser never
trains on. At any later time, count how many of the m held-out inputs of task j are now routed
differently from when task j arrived; call it c. Because the held-out inputs and a future test
input of task j are exchangeable and neither recogniser depends on them, the chance that a test
input of task j has changed its route is at most (c + 1)/(m + 1) on average. This holds for *any*
recogniser, covers every input of the task (not only a zone), costs nothing at training time, and
can be read off at every arrival; a system can refuse or flag a newcomer that would push the
certificate of an older task above a limit. It is a measured guarantee, not one given in advance;
the one given in advance is the weaker statement that a route cannot change unless the recogniser
is wrong at one of the two times, so the share of changed routes is at most the sum of the two
error rates.

**The test** (`paper_certified/audit_study.py`). Development on the 15 SuperNI and the 15 Long
Sequence tasks, same data and splits as before: features (sentence encoder; with the 15 shape
numbers; with hashed character 1-to-3-grams, tf-idf, 16,384 dimensions), recogniser (a) with
penalty strength 1, 10, 100 and (b) with 4,096 random cosine features and ridge 0.01, 0.1, 1 (not
with the character features). Measures on the test inputs: given to the own task at arrival and
after the last task, route changed; the certificate (c + 1)/(m + 1) per task next to the measured
changed share. The version with the highest share given to the own task on the SuperNI tasks
(among those with at most 3% changed routes) is frozen, and **its verdict is read once on the
third batch of 30 fresh tasks**: "works" if on average at least 85% of the test inputs are given to
their own task after the last task and right after arrival, "does not" below 70%; the certificate
is "met" if the average changed share is at most the average certificate plus two standard errors.
**What we expect:** on the SuperNI tasks 93 to 98% with the character features; a few per cent of
changed routes; the certificate a little above the measured share. Whether it beats the rule
`claims` on fresh tasks we do not know; both verdicts will stand next to each other.

### 2026-11-18 (07:20 UTC) — a note on theory while the tests compute: the price of having no instruction

No test is attached to this note; it is a statement and its proof, written down so that it can be
checked before it goes anywhere near the paper.

**Why it is needed.** The target has two parts that pull against each other: "no blocked
tasks" and "a guarantee against forgetting". The results so far (oldest zone wins: no forgetting,
later tasks blocked; newest zone wins: all tasks learned, some routes change) look like two ends of
one trade. The statement below says that this trade is not a weakness of our rules. It is forced
on every method that does not read an instruction, and its size can be written down.

**Statement.** Two tasks j < k have inputs with densities p_j and p_k (for texts: probabilities of
each text). An answer y to an input x has the score s_j(x, y) under task j and s_k(x, y) under task
k, both between 0 and M. A system without instructions answers at time T with y_T(x), which depends
on the input alone. Write best_i(x) for the best score any answer can get under task i, and
loss_i(T) for the average, over the inputs of task i, of best_i(x) - s_i(x, y_T(x)). Call

  conflict(x) = the smallest value, over all answers y, of
                (best_j(x) - s_j(x, y)) + (best_k(x) - s_k(x, y)).

Then for every time T >= k and every system without instructions

  loss_j(T) + loss_k(T)  >=  Omega_jk  :=  sum over x of  min(p_j(x), p_k(x)) * conflict(x).

**Proof.** Fix x and let y = y_T(x), a = best_j(x) - s_j(x, y) >= 0, b = best_k(x) - s_k(x, y) >= 0.
Then p_j(x) a + p_k(x) b >= min(p_j(x), p_k(x)) (a + b) >= min(p_j(x), p_k(x)) conflict(x). Sum over
x. (For a system that answers at random, a and b are averages over its answers and the same lines
hold.)

**What follows.** (i) If every input of a task begins with the task's own instruction, no text
belongs to two tasks, Omega is zero, and nothing is forced: this is the precise sense in which
"the instruction gives the task away". (ii) With exact-match scores and tasks whose right answers
differ, conflict(x) = 1 and Omega is the overlap of the two input distributions, 1 minus their
total-variation distance. (iii) Forgetting: the forgetting of task j between its arrival and time
T is loss_j(T) - loss_j(j), so  forgetting_j(T) + loss_k(T) >= Omega_jk - loss_j(j):  what is
forgotten of the older task plus what is not learned of the newer one is at least the conflict
mass (minus what the older task never had). (iv) The bound is reached by a switch with one frozen
module per task that sends each input to the task under which it is more probable. *How* the mass
is split is then a choice: "oldest zone wins" puts all of it on the newer task (nothing forgotten,
Theorem "Bounded forgetting rate"); the rule `claims` with its number r puts at most
(r + 1)/(m + 1) per later task on the older one. A method with shared weights pays the same price
but does not say where.

**How Omega can be measured (later, with rules written first).** From below: texts that occur
word for word in the data of two tasks with different reference answers. From above: twice the
error of the best classifier that tells the inputs of the two tasks apart (its error is at least
half the overlap). The two numbers bracket the price for every pair of tasks, and can be shown as
a table next to what each method actually loses.

**Not claimed as deep.** The proof is three lines, and that task recognition limits learning
without task labels is known (class-incremental learning = prediction within a task plus
recognition of the task). What may be worth a place in the paper is the form: a lower bound for
*forgetting plus not-learning* that is zero with instructions and positive without, next to a
rule that lets the user choose the split and proves it.

### 2026-11-18 (07:23 UTC) — the frozen model's own hidden states as the feature: read against its rule; and the next test, fixed before any run

**The result** (`paper_certified/routing/noinstruction_study.json`, keys `verdict_t5hidden` and
`results_reduced`; 15 SuperNI tasks of order 1, at most 400 training inputs per task, pool of 2,000
unrelated inputs, level 0.05, the oldest zone wins, instances only):

| feature, score | inside own zone | blocked | given to own task at the end | route changed |
|---|---|---|---|---|
| word vectors, ratio | 95.4% | 65.1% | 34.0% | 0.0% |
| sentence encoder, ratio | 96.1% | 40.5% | 58.3% | 0.9% |
| **frozen T5-base encoder states, ratio** | 94.0% | **10.5%** | **87.9%** | 1.1% |
| frozen T5-base encoder states, 10th neighbour | 94.9% | 64.7% | 34.2% | 1.0% |

**Read against the rule of 05:43 UTC.** The claim holds (changed 1.1% against a limit of 6.6%;
coverage 94.0% against a limit of 93.1%). The use is **"partial"**: 87.9% is below the line of 90%
and 10.5% blocked is above the line of 10%, both narrowly. So by its own rule this is not "works",
and the small trained model is not run on it. **Our guess was wrong in the good direction:** we
expected 20 to 35% blocked and wrote that "works" would surprise us; it came within two points of
it. Where it fails is where we said it would: emotion (48% blocked by the other tweet task),
evalution (46% blocked by sst2), reddit (22% by quoref), sentiment140 (21%), sciq (13%); all other
tasks are at 6% or below. Caveats: one seed, reduced data, a second look at tasks already used,
and only the SuperNI sequence was computed (the first part took 1.5 hours on the loaded machine).

**What it says.** The feature matters more than the shape of the zone. The sentence encoder was
built to ignore how a text is written and keep what it is about; the model's own states keep both,
and "how it is written" is most of what separates one task's inputs from another's.

**The next test, fixed now: the same feature under the two rules that block nothing.**
(`paper_certified/hidden_rules_study.py`; result in
`paper_certified/routing/hidden_rules_study.json`.)

- Feature: as above (mean of the last-layer states of the frozen T5-base encoder over the tokens of
  the instance, 256 tokens), standardised per dimension with the mean and standard deviation of the
  pool of unrelated inputs. Data: as above (at most 400 training inputs per task: 200 held out,
  the rest for fitting; official test inputs).
- Rule J (joint recogniser, plan entry of 07:12): a multinomial logistic regression over all tasks
  learned so far, retrained at every arrival on the stored fit features, strength C in
  {0.1, 1, 10}; and rule L, its form without stored inputs: a ridge regression on the features
  themselves (sums added up task by task; ridge in {1, 10, 100} times the number of dimensions).
  Certificate (c + 1)/(m + 1) from the held-out inputs, which neither rule sees.
- Rule `claims` (plan entry of 06:20) with r in {1, 2, 3} and C = 1 (r is the rank of the smallest held-out score of an older task that a newcomer may not pass; r = 1 means that none can be taken. An earlier wording of this line listed r = 0, which does not exist; corrected before any run).
- **Development** on the 15 SuperNI tasks. Chosen: the variant with the highest share of test
  inputs given to their own task after the last task, among those with at most 3% changed routes
  (ties: rule L, then J, then the smaller number).
- **Verdict, read once, on a fourth batch of 30 fresh tasks** (fetched now, opened only after the
  choice is written to the JSON): "works" if on average at least 85% of the test inputs are given
  to their own task both after the last task and right after arrival; "does not" below 70%;
  certificate "met" if the average changed share is at most the average certificate plus two
  standard errors. Fresh tasks: the first 400 of the first 800 instances after a seed-0 shuffle
  are the training inputs (200 held out, 200 fit), the last 200 instances the test inputs.
- **What we expect:** rule J at 95 to 99% on the 15 SuperNI tasks, with 1 to 3% changed routes;
  on fresh tasks 80 to 90%, because random task collections contain pairs of tasks built on the
  same texts, which nothing can tell apart without the instruction.
- **If it passes** (90% on the 15 tasks was the first "sign" fixed at 06:10, then the fresh
  tasks): the routes of the chosen rule for the five-task sequence are written to
  `silent_adapters/noinstruction_routes.json`, and the GPU run of our method is prepared against
  the published methods without instructions.

### 2026-11-18 (07:29 UTC) — two results read against their rules: the published methods without instructions (GPU), and the frozen model's states under the rules that block nothing (development)

**1. The GPU run of the published methods without instructions** (Kaggle, T4, 22 minutes; five
tasks, T5-base, 10 epochs, seed 42; the definition of every task replaced by the empty string for
training and testing; files in `silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/`).
Compared with the same seed with definitions (run of 2026-11-17). Re-test on the 400 test inputs
of the first four tasks:

| method | run | learned | score at end (AP) | forgetting (FT) | answers changed |
|---|---|---|---|---|---|
| InfLoRA | with definitions | 56.96 | 58.89 | -1.41 | 98 of 400 |
| InfLoRA | **no definition** | 51.46 | **37.45** | **18.01** | **217 of 400** |
| GainLoRA | with definitions | 60.85 | 61.31 | -0.32 | 17 of 400 |
| GainLoRA | **no definition** | 56.08 | **51.94** | **6.51** | **129 of 400** |

(Official test, one reference answer per example: InfLoRA 51.08 / 37.28 / 17.75, GainLoRA
55.82 / 51.64 / 6.55 without definitions.)

**Read against the rule of 06:05 UTC.** Step 1, was anything learned: yes for both (InfLoRA 51.5
against 60.0 with instructions, 86%; GainLoRA 56.1 against 61.4, 91%; the line was 60%). Step 2:
**both "hold their answers less well without instructions"**. InfLoRA's forgetting rises by 21
points against its three-seed value with instructions (line: 5), and 217 answers change against
123 per 400. GainLoRA's forgetting rises by 7.0 points and 129 answers change against 22 per 400
(line: twice as many and 60 more). Our guess for GainLoRA was 60 to 150 changed answers; it is 129.
Step 3 stands: one seed, five tasks, a small model, a tenth of the published budget, two methods.
It is a first look, and the paper will call it that. Two more things have to be said with it.
(a) Without definitions the tasks are also learned less well by every method (two of the five,
sst2 and emotion, stay near guessing: 43 to 48 and 12 to 28). (b) The job with our own rules was
not part of this notebook; only the two published methods were run.

**What it means for the target.** In the setting without instructions the numbers to beat are a
final score of 51.9 (GainLoRA) with 6.5 points of forgetting and 129 of 400 answers changed.

**2. The frozen T5-base states under the rules that block nothing** (development, the 15 SuperNI
tasks, reduced data; `paper_certified/routing/hidden_rules_study.json`):

| rule | own task right after arrival | own task after the last task | route changed | certificate (held-out) | 95% bound |
|---|---|---|---|---|---|
| J, joint logistic regression, C = 1 | 99.7% | 99.0% | 0.7% | 1.2% | 2.9% |
| **L, ridge from added-up sums, 1** | 99.9% | **99.0%** | 0.8% | 1.5% | 3.4% |
| L, 10 / 100 | 99.8 / 99.5% | 98.5 / 97.6% | 1.3 / 1.9% | 1.5 / 2.2% | 3.4 / 4.2% |
| `claims`, r = 1 / 2 / 3 | 98.9 / 99.0 / 99.2% | 98.2 / 97.7 / 97.6% | 0.7 / 1.3 / 1.6% | not valid | — |

**Chosen by the rule written before the run: L with ridge 1** (tie with J at 99.03%, and the rule
prefers L). It stores no input of any task, only one matrix and one vector per task, and its
weights after all 15 tasks are the same whatever the order of arrival (largest difference with the
order reversed: 5e-17). No task is blocked by construction. The first "sign" fixed at 06:10 UTC
(at least 90% on the 15 SuperNI tasks) is met. **Our guess (95 to 99%) held.** Weak spots: the
tweet task falls from 98% to 92% when the emotion task arrives (15 of its 200 held-out inputs
change route; its certificate says 8.0%, and 6% of its test inputs did change), and the dialogue
task falls to 94%. Everything else stays at 99 to 100%. The certificate is above the measured
share in the average (1.5% against 0.8%). Caveats: development data, one split, reduced data;
the verdict is the fourth fresh batch, whose features are being computed.

**3. The GPU run of our method, fixed now, before any route is written.** The order fixed at
06:10 UTC stays: the fresh tasks first, then the small trained model, then the GPU. To lose no
time the three are prepared together, and the GPU run is started only if the fresh tasks do not
say "does not".

- The run: the same five tasks, seed, model, epochs and empty definitions as the run above. Plain
  adapters (nothing blocked: while task t is learned, adapters 1 to t are on for all its inputs).
  At test time an input is answered by the adapters that existed when the task it is routed to
  was learned. Routes come from rule L with ridge 1 on the frozen T5-base states of the instance
  text (computed here on the CPU from the same cached features as the development table: 200 fit
  and 200 held-out inputs per task, the definition never read, test inputs never used for
  fitting), written to `silent_adapters/noinstruction_routes.json` before the run. For comparison
  the table also holds rule J, `claims` with r = 1, and "own" (every input to its own task: what
  perfect recognition would give).
- **Reading.** "Beats the published methods in this setting" if all three hold: the final score
  (AP, official test) is at least 2 points above the better published method without definitions
  (51.64, so at least 53.6); forgetting is below 2 points; at most 32 of the 400 compared answers
  change (a quarter of GainLoRA's 129). "Does not" if the final score is below 51.64 or at least
  129 answers change. Anything else: mixed. Reported next to it: the score right after learning
  (plain adapters), the row "own", the certificate of each task next to its measured share of
  changed answers, and GainLoRA *with* definitions (61.05), which we expect to stay ahead, because
  every method learns these tasks less well without their definitions.
- **What we expect:** a learned score of 54 to 58, a final score within one point of it,
  forgetting below 1 point, 5 to 15 changed answers of 400, most of them tweets that move to the
  emotion task.
- One seed and five tasks are a first sign and no more. If the reading is "beats", the next GPU
  hours go to seeds 43 and 44 and to the 15-task sequence, for the published methods without
  definitions and for ours.

### 2026-11-18 (07:34 UTC) — rule H: the joint recogniser with a handicap for the newcomer (a bound known in advance, without zones). Fixed before any run

**Why.** Rule L (chosen above) blocks nothing and recognises well, but its guarantee is measured
after the fact (the certificate from held-out inputs). The target asks for a guarantee;
one that is known before the later tasks arrive is the stronger kind. The rule `claims` has one,
but with a weaker classifier and with a term for the inputs outside the task's zone. Rule H puts
the same protection on the joint recogniser and needs no zone.

**The rule.** When task l arrives, the joint recogniser W_l is computed as in rule L (ridge 1).
For an input x let margin_l(x) = (score of task l) minus (the largest score of an older task),
both under W_l. The newcomer takes x if margin_l(x) > theta_l, where theta_l is the largest, over
the older tasks j, of the (t+1)-th largest margin_l among the held-out inputs of task j (so the
newcomer takes at most t held-out inputs of every older task; t = 0: none). Otherwise the route
of x stays what it was before task l arrived. The first task takes everything. So the route after
T tasks is the newest task l <= T that took x. The recognisers W_l and thresholds are stored
(one small matrix per task) and never changed.

**The statement.** Let X be a test input of task j, exchangeable with the m held-out inputs of
task j, and let the later tasks not depend on them. Then the probability that the route of X
ever changes after the arrival of task j is at most (number of later tasks) x (t + 1)/(m + 1).
Proof: the route of X changes at the arrival of task l only if margin_l(X) > theta_l. The function
margin_l does not depend on the held-out inputs or on X, and theta_l is at least the (t+1)-th
largest of the m held-out margins of task j, so by exchangeability this has probability at most
(t + 1)/(m + 1); sum over the later tasks. No part of the statement depends on a zone, on the
feature, or on how good the recogniser is. The price is paid by the newcomer and is visible when
it arrives: the share of its own held-out inputs that it does not take.

**The test.** Development on the 15 SuperNI tasks (same features and split as rule L), with
t in {0, 1, 2}: given to own task at arrival and at the end, route changed, the bound, and for
every newcomer the share of its own inputs it takes. On the fourth fresh batch H with t = 0 is
computed in the same run as the verdict of rule L and reported next to it (a second result on
the same data, not a second verdict): **"keeps the recognition"** if its share given to the own
task after the last task is within 2 points of rule L's, and **"bound met"** if the average share
of changed routes is at most the average bound plus two standard errors.
**What we expect:** with t = 0 on the 15 tasks, 97 to 99% to the own task and at most 0.5% changed
routes, against a bound of 14/201 = 7% for the first task and 0 for the last. The bound is far
above what happens because most later tasks take nothing from most older ones; the bound cannot
know that, the certificate can.

### 2026-11-18 (07:35 UTC) — rule H on the development tasks: a flaw found and corrected (rule H'), numbers, and what goes into the GPU table

**Rule H as written** (15 SuperNI tasks, ridge 1; `routing/hidden_rules_study.json`, key
`dev_rule_H`):

| t (held-out inputs of each older task a newcomer may take) | own task at arrival | at the end | route changed | bound (average) |
|---|---|---|---|---|
| 0 | 95.1% | 94.7% | 0.5% | 5.8% |
| 1 | 98.7% | 97.7% | 1.3% | 11.6% |
| 2 | 99.7% | 97.3% | 2.4% | 17.4% |
| rule L, for comparison | 99.9% | 99.0% | 0.8% | (certificate 1.5%) |

**Read against what we wrote:** with t = 0 we expected 97 to 99% and at most 0.5% changed. The
changed share is as guessed; the 94.7% is below our guess. One task pays: quoref takes only 36% of
its own inputs, because a single held-out input of the news task looks like a quoref passage and
sets the threshold. And with t = 1 or 2 the rule changes *more* routes than rule L does with no
protection at all. That is a flaw in the rule as we wrote it: when no held-out input of an older
task is near the newcomer, the threshold is negative, and the newcomer then takes inputs that
even the joint recogniser gives to an older task.

**Rule H' (the correction): the threshold is never below zero.** A newcomer takes an input only
if the joint recogniser prefers it *and* the margin is above the protection threshold. The
statement and its proof are unchanged (the threshold only became larger).

| t | own task at arrival | at the end | route changed | bound (average) |
|---|---|---|---|---|
| 0 | 95.1% | 95.0% | 0.07% | 5.8% |
| 1 | 98.7% | 98.2% | 0.5% | 11.6% |
| **2** | 99.7% | **99.2%** | **0.5%** | 17.4% |
| 5 | 99.8% | 99.3% | 0.5% | 34.7% |
| no protection (the cascade alone) | 99.9% | 99.1% | 0.8% | none |

With t = 2, H' gives more inputs to their own task than rule L (99.2 against 99.0%) and changes
fewer routes (0.5 against 0.8%); the tweet task keeps 97% when the emotion task arrives (93 to
94% without protection). **What has to be said with it:** (a) H' was written after seeing the
development numbers of H, on the same 15 tasks, so these numbers are development numbers and no
more; (b) the bound known in advance is loose: for a task with 200 held-out inputs and 14 later
tasks it is 3 x 14/201 = 21%, and for the three small tasks (50 to 64 held-out inputs) it says
nothing; what actually changed is 0.5%. A tighter bound in advance needs more held-out inputs
(they cost nothing in training: the adapter may still learn from them). The certificate of rule
L remains the tight statement.

**Fixed now.** (1) On the fourth fresh batch, H' with t = 2 is the second rule reported next to
the verdict of rule L, with the two lines written at 07:34 ("keeps the recognition" within 2
points of rule L; "bound met"). t = 2 is the smallest t within 0.2 points of the best on the
development tasks. (2) The route table of the GPU run gets a fifth rule, "handicap" (H' with
t = 2), added before the run: on the five-task sequence it gives 497 of 500 test inputs to their
own task after the last task and changes 1 route on the 400 compared inputs (rule L: 492 and 6);
its bounds known in advance are 6.0, 4.5, 3.0 and 1.5% for the first four tasks. The reading of
the GPU run fixed at 07:29 stays on rule L ("ridge"); the row "handicap" is read by the same
three lines and reported next to it.

### 2026-11-18 (07:39 UTC) — the joint recogniser with the certificate, on the sentence-encoder features: read against its rule (third fresh batch)

Files: `paper_certified/audit_study.py`, `paper_certified/routing/audit_study.json` (rule and
lines: entry of 07:12 UTC; this test does not use the frozen model's states, which were found
later).

**Development** (share of test inputs given to their own task after the last task, 15 SuperNI /
15 Long Sequence tasks; route changed; certificate):

| recogniser, feature | SuperNI | Long Sequence | changed (SuperNI / Long) | certificate (SuperNI / Long) |
|---|---|---|---|---|
| logistic regression, sentence encoder, C = 10 | 96.5% | 90.4% | 2.0 / 5.3% | 3.7 / 6.4% |
| **logistic regression, + 16 shape numbers, C = 10** | **99.3%** | 95.4% | 0.6 / 2.8% | 2.2 / 2.8% |
| logistic regression, + character n-grams, C = 10 | 98.4% | 95.6% | 1.1 / 2.3% | 2.5 / 2.7% |
| sums of random cosine features, sentence encoder, 0.1 | 94.9% | 86.3% | 3.8 / 5.9% | 6.3 / 6.1% |
| sums of random cosine features, + shape numbers, 0.01 | 98.6% | 92.2% | 1.4 / 4.8% | 3.0 / 5.1% |

Chosen by the rule written before the run: logistic regression on the sentence encoder plus the
shape numbers, C = 10. (The plan said "15 shape numbers" and listed 16; all 16 were used.) The
form without stored inputs is the same for every order of arrival (largest difference of the
weights with the order reversed: 3e-12).

**Verdict, read once, on the third batch of 30 fresh tasks:** on average 96.4% of the test inputs
are given to their own task right after arrival and **93.8%** after the last task; 3.0% of the
routes changed; the average certificate is 3.8%. By the lines of 07:12: **use "works"** (both
shares above 85%), **certificate "met"** (3.0% is below 3.8% plus two standard errors). The plain
sentence encoder alone, for comparison: 89.5% and 5.1% changed.

**Read against what we expected.** We expected 93 to 98% on the SuperNI tasks with the character
features (98.4%: at the top end) and did not know what the fresh tasks would give. We did not
expect the handful of shape numbers to do as well as the character n-grams; they do, and both
say the same thing as the frozen model's states did an hour later: how an input is written
tells the tasks apart. **Where it fails, and why this failure belongs in the paper:** two of the
30 fresh tasks are built on the same texts (answers for a question, and a question for an
answer, from one dataset). They get 45% and 49%: each input of the pair is given to one of the
two tasks about as often as to the other. Nothing can do better without the instruction; this
is the conflict of the note of 07:20 UTC, seen in the data. The other 28 tasks are fine.

**What it means for the order of work.** The joint recogniser has now passed on fresh tasks with
the weaker feature. The check of rule L on the frozen model's states (fourth batch) is still
computing; the machine is slow, three tasks of thirty in ten minutes.

### 2026-11-18 (07:42 UTC) — one more use of the recogniser, fixed before any run: with instructions, route on the example alone

**The idea.** The zones of the paper read the instruction, and that is where they are weak: a
reworded instruction leaves the zone (at scale, half of the reworded inputs or fewer were
certified for the right task, and five attempts to widen the zones failed). The recogniser found
today never reads the instruction. So in the normal setting, *with* instructions, it can set the
route from the example text alone, while the adapters still read the whole input, instruction
included. Then (a) no rewording of the instruction can change a route, because the route does
not look at it; (b) the two statements of today (the audit, and the bound of rule H') hold for
every wording of the instruction, and for none: for a fixed input the answer still changes only
if the route changes, and the route depends on the example alone; (c) the tasks are learned with
their instructions, which every method needs here (without them all methods learn less). What is
given up: exactly zero changed answers on unedited inputs, which only the zones give. A system
can keep both (the zone first, the recogniser for inputs outside every zone); that mixture has
no clean statement for reworded inputs and is not tested now.

**Why it is not a trick.** The benchmark gives the instruction and the example as two fields;
the recogniser reads the second. A system that receives one string must find where the example
begins; that is assumed here and will be said.

**The GPU run (run U), after the run without instructions.** The five tasks, seed 42, T5-base,
10 epochs, **with** the definitions, plain adapters (nothing blocked), the same route table as
the run without instructions (the routes do not depend on the definition, so the file is the
same). Re-test at the four wordings of the definition (original, light, medium, heavy), rules
"ridge" and "handicap", next to the numbers of the same seed from the run of 2026-11-17:

| seed 42, with definitions | original | light | medium | heavy |
|---|---|---|---|---|
| certified switch of the paper: final score | 79.33 | 79.15 | 77.61 | 72.12 |
| certified switch: answers changed of 400 / right task | 0 / 100% | 0 / 100% | 0 / 100% | 0 / 93.8% |
| GainLoRA: final score / answers changed of 400 | 61.31 / 17 | 59.49 / 15 | 58.61 / 32 | 52.01 / 49 |
| InfLoRA: final score / answers changed of 400 | 58.89 / 98 | 59.87 / 85 | 60.45 / 91 | 56.37 / 151 |
| every input answered by its own task's adapters (the ceiling for any route) | 79.33 | 79.15 | 77.61 | 76.72 |

**Reading.** "Repairs the weak spot" if, for the rule "ridge": at the heavy wording the final
score is at least 74.1 (2 points above the certified switch), at the original wording it is at
least 77.3 (within 2 points of the certified switch), and at every wording at most 8 of the 400
compared answers change. "Does not" if the heavy score is below 72.12 or the original score
below 75.3. Anything else: mixed. **What we expect:** original 78.5 to 79.3 with about 6 changed
answers (rule "ridge") or 1 to 3 (rule "handicap"); heavy 75.5 to 76.7. So a gain of about 4
points at the heavy wording, a loss of up to 1 point and a few changed answers at the original
wording, where the certified switch is exact. One seed, five tasks, three rewordings written by
machine (not by people): a first look.

### 2026-11-18 (07:45 UTC) — the order of the last steps is changed, and why (written before the GPU run of our method is started)

At 06:10 and 07:29 UTC we wrote that the GPU run of our method starts only after the check on
fresh tasks. We change this, and say so here before doing it:

- The check of rule L on the fourth fresh batch needs the frozen model's states of 18,000
  inputs. On this machine, with the other tests running, it has done 6 of 30 tasks in 15
  minutes; it will take about one more hour. The small trained model (CPU) needs about half an
  hour more.
- What the gate was for: not to spend GPU hours on a recogniser that only works on the 15 tasks
  it was chosen on. Since then the joint recogniser has passed its check on fresh tasks with a
  weaker feature (third batch, entry of 07:39 UTC: 93.8%, "works"), and it gives the same 99.0%
  for every order of the 15 tasks (second official order and five random orders; development
  data, `routing/hidden_rules_study.json`, key `dev_other_orders`: 0.4 to 0.9% of routes
  changed, audit 0.9 to 1.5%; rule H' with t = 2: 99.0 to 99.5%, 0 to 0.4% changed).
- The run costs about half an hour of the roughly 15 GPU hours left this week.
- The readings of the GPU run (07:29 UTC), of the fresh batch (07:23 and 07:34 UTC) and of the
  small model (corrected lines of 03:13 UTC) are all fixed already and do not depend on each
  other, so the order in which they are read cannot bend any of them. All three will be read
  and reported, whatever they say. If the fresh batch says "does not", that is reported next to
  the GPU result as a limit of the recogniser, and the GPU result is not dropped or hidden.

So: the GPU run of our method without instructions is started as soon as the notebook has
passed its test on the CPU.

### 2026-11-18 (07:48 UTC) — the recogniser on the second benchmark and on many tasks, with the features computed on the GPU. Fixed before any run

**Why.** The frozen model's states are slow to compute on this CPU (the fourth fresh batch
alone takes more than an hour), and two things are still missing for the recogniser: the second
standard benchmark (the 15 Long Sequence tasks), and what happens when there are many tasks. A
small GPU job computes the states in minutes. Only routes are measured; nothing is trained.

**The job** (a stand-alone notebook, `silent_adapters/hidden_states_scale.ipynb`; about 15
minutes on a T4). Frozen T5-base encoder states of the instance text (256 tokens), standardised
with the mean and standard deviation of the pool computed here on the CPU. Rules L (ridge 1),
H' (t = 2) and J (C = 1), exactly as in `paper_certified/hidden_rules_study.py`. Per task at
most 400 training inputs (seed-0 shuffle; 200 held out, 200 fit), as before. Sequences:

1. the 15 SuperNI tasks, orders 1 and 2 (a check: must agree with the CPU numbers above up to
   rounding of the features);
2. **the 15 Long Sequence tasks, order 3 and order 4, all official test inputs;**
3. the three batches of fresh tasks that other tests have already used (30, 60 and 90 tasks),
   and everything together (120 tasks: 15 + 15 + 90).

**Reading.** Long Sequence: the same line as the first "sign" (at least 90% of the test inputs
given to their own task after the last task, rule L) = "works on the second benchmark"; below
75% = "does not"; the audit "met" if the average changed share is at most the average audit
plus two standard errors. Parts 1 and 3 are descriptions, not verdicts: part 1 is a check of the
code, part 3 uses tasks that earlier tests have seen (other rules, other features), and is shown
as a curve of the share given to the own task against the number of tasks. The untouched fourth
batch stays the verdict for fresh tasks and is not part of this job.
**What we expect:** Long Sequence 94 to 98% (the sentence encoder with shape numbers reached
95.4%; the entailment tasks cb, rte and mnli will be the weak ones); with 120 tasks 90 to 95%,
falling slowly with the number of tasks, with the pairs of tasks built on the same texts at
about 50%.

### 2026-11-18 (07:51 UTC) — a count, not a test: texts that occur word for word in two tasks

`paper_certified/conflict_count.py`, `paper_certified/routing/conflict_count.json`. For the
note of 07:20 UTC (the price of having no instruction) the part of the conflict that can be
counted directly is the share of inputs that occur, word for word, in two tasks.

| collection | tasks | inputs | inputs that also occur in another task |
|---|---|---|---|
| 15 SuperNI tasks of the benchmark (training and test) | 15 | 10,588 | 0 |
| 15 Long Sequence tasks of the benchmark (training and test) | 15 | 115,381 | 0 |
| fresh batch 1 | 30 | 30,000 | 0 |
| fresh batch 2 | 30 | 30,000 | 0 |
| fresh batch 3 | 30 | 30,000 | 586 (2.0%), all in one pair of tasks |

On the two standard benchmarks no input belongs to two tasks. So nothing forces a method to
forget there when the instruction is removed: what the two published methods lost without
instructions (entry of 07:29 UTC) is not a price that had to be paid. In the third fresh batch
one pair of tasks is built on the same dataset (29% of the inputs of each occur word for word
in the other, and the rest are the same kind of text); this is the pair on which the recogniser
gives 45% and 49% to the right task (entry of 07:39 UTC). The count is a lower limit of the
conflict: tasks can also overlap without sharing exact texts. The fourth batch was not opened.

### 2026-11-18 (07:53 UTC) — the GPU run of our method without instructions is started

Notebook `silent_adapters/five_tasks_noinstruction_ours.ipynb`, uploaded as
`bammidishivkumar/no-instruction-ours-five-tasks` (private, T4, started 07:52 UTC). Five tasks,
seed 42, T5-base, 10 epochs, every definition replaced by the empty string, plain adapters with
nothing blocked, the re-test with the rules "all" (plain adapters, all on), "centre" and the five
table rules (ridge, joint, claims, handicap, own). The embedded route table was compared with
`silent_adapters/noinstruction_routes.json` as pushed: identical. Tests before the start: the
whole path (training, re-test with table rules, final tables, saving) ran to the end on the CPU
with a small model and two other tables (one sending every input to its own task, one sending
everything to the first task; the second gave the expected lower score and 33% right task); with
the real five tables the check of the tables and the first two tasks had passed, and the third
task was still running when the notebook was uploaded. Reading: as fixed at 07:29 UTC.

### 2026-11-18 (07:55 UTC) — the rules `boundary` and `claims` (sentence encoder) on the first fresh batch: read against their lines

`paper_certified/routing/boundary_study.json`, keys `part2`, `verdict_fresh`,
`verdict_claims_fresh` (30 fresh tasks, sentence-encoder feature; lines of 06:10 and 06:20 UTC).

| rule | own task at arrival | own task after the last task | blocked | route changed | verdict |
|---|---|---|---|---|---|
| `boundary` (oldest zone wins) | 55.3% | 54.4% | 44.2% | 1.1% | claim holds; use **"does not"** (below 70%) |
| `claims` (newest zone wins, held-out inputs protected) | 72.4% | 72.2% | none | 0.4% | guarantee met; use **"partial"** (between 70 and 85%) |

Both do what they promise about change (1.1% and 0.4% of routes changed), and neither
recognises well enough on fresh tasks. The classifier of these rules alone, with no zone, gives
83.3% of the test inputs to their own task here. This is the same feature that reached 89.5%
with the joint recogniser on the third batch and 93.8% with the shape numbers added (entry of
07:39 UTC); with the frozen model's states the check is still running. So the order of what
mattered today stays: first the feature, then one recogniser over all tasks instead of one zone
per task, and only then the shape of a zone. The grid of four zone shapes on this feature was
cut short to save the CPU (decided before any grid number was read); what was computed of it
will be reported as it is.

### 2026-11-18 (08:02 UTC) — an independent check of the three new statements and of the code (a second, stronger model read them; nothing was run)

Draft: `paper_certified/sec_audit.tex`. What the check found, and what was changed:

- **The bound in advance (rule H') is correct as stated and as coded** (the rule, the proof line
  by line, ties, the union bound, the constant). One condition was too loose and is now written
  exactly: the later tasks and their fit inputs must be independent of the held-out inputs and
  of the test input (for example fixed in advance); they may not be chosen by looking at
  anything computed from the held-out inputs.
- **The audit was stated more weakly and more clumsily than it is.** By exchangeability the
  share of held-out inputs whose route changed is an *unbiased estimate* of the share of the
  task's inputs whose route changed (an equality in expectation); the number (c + 1)/(m + 1)
  that the plan called "the certificate" is a slightly larger number with no extra meaning.
  The actual guarantee is the confidence bound (one-sided Clopper-Pearson, which the code
  already computes as `upper95`). The paper will report the estimate c/m and the 95% bound. The
  lines fixed in advance compared the measured change with (c + 1)/(m + 1); they stay as they
  were read. Also said now: each bound is for one task and one pair of times; read at every
  arrival and for all tasks they do not hold all at once at the same confidence.
- **The price of having no instruction**: true; wording tightened (answers at random, scores as
  expectations, when exactly the overlap formula holds). "A method with shared weights pays the
  same sum" was an overstatement and now reads: every method is bound by the same inequality,
  but does not say how the sum is split.
- **The code matches the text** (rule L, rule H' with the floor, the audit counts, the bound;
  no test input reaches any fit; the held-out inputs are never used by rules L and J and only
  in the allowed way by H').
- **A weakness of the fresh-task checks, found by the check and not yet fixed:** in the fresh
  batches the test inputs are the last 200 instances of the part of the file that was read,
  while the held-out inputs are a random part of the first 800. If a file is ordered in some
  way, test and held-out inputs are not the same kind of draw, and the statements about change
  (not the shares given to the own task) rest on an assumption that is not ensured by
  construction. This applies to the verdicts on the first, third and fourth batch as they were
  fixed; they stay as fixed, and the caveat is reported with them. The feature job on the GPU
  gets a second version of the fresh sequences with one random split of all 1,000 instances
  (seed 0: 200 held out, 200 fit, 200 test), so that the size of the effect can be seen.
- Symbols that collided with earlier sections of the paper were renamed in the draft.

A slip in the entry of 07:45 UTC is corrected there: the audit of rule L over the other orders
is 0.9 to 1.5%, not 1.0 to 1.5%.

### 2026-11-18 (08:13 UTC) — the four zone shapes (sentence encoder), and the verdict of the chosen one on the second fresh batch

`paper_certified/routing/boundary_study.json`, keys `dev_grid`, `chosen`, `fresh2`,
`verdict_fresh2` (rule `claims`: the newest zone wins, held-out inputs of older tasks are
protected; rules of 07:00 and 07:07 UTC). 44 variants on the 15 SuperNI tasks; the Long Sequence
part of the grid was not computed (cut for CPU time before any grid number was read).

**Development: share of test inputs given to their own task after the last task (best penalty,
no held-out input of an older task may be inside a new zone):**

| shape of the zone | sentence encoder alone | with the 16 shape numbers (best weight) |
|---|---|---|
| one flat boundary against everything | 87.8% | 93.2% |
| **one flat boundary per older task** | 92.0% | **98.7%** |
| balls around the task's own inputs | 87.2% | 77.2% (worse with the numbers) |
| curved boundaries (random cosine waves) | 90.7% | 98.3% |

Chosen by the rule written before the grid: one flat boundary per older task, shape numbers with
weight 1, penalty strength 10, r = 1 (98.7% own task, 0.3% of routes changed, bound in advance
11.9% on average).

**Verdict, read once, on the second batch of 30 fresh tasks:** 94.1% of the test inputs go to
their own task right after arrival and **93.6%** after the last task; 0.57% of the routes
changed; the bound known in advance is 17.7% on average (94.15% and 17.75% before rounding;
an earlier version of this line rounded both up). By the lines fixed before: **use
"works", guarantee "met".** The rule `claims` as first written (one boundary against
everything, no shape numbers) gives 80.2% on the same batch. One weak task (49%), most of whose
misrouted inputs go to one older task.

**Read against what was expected.** No number was written down for the shapes, only the choice
rule, so there is no guess to check. What the grid says: the shape matters (a boundary per older
task instead of one boundary against everything: +4 to +6 points; balls are the worst and get
worse with more features), but less than the feature (the shape numbers: +5 to +7 points for
three of the four shapes), and the frozen model's states, found later, do better than all of
them with the plainest rule. The curved boundary is not better than the flat one.

**Where today's results stand on fresh tasks** (each rule chosen on the 15 benchmark tasks, each
read once on a batch nothing else had used):

| rule | feature | fresh batch | own task after the last task | changed | verdict |
|---|---|---|---|---|---|
| zone, oldest wins (`boundary`) | sentence encoder | 1 | 54.4% | 1.1% | does not |
| zone, newest wins (`claims`) | sentence encoder | 1 | 72.2% | 0.4% | partial |
| zone, newest wins, one boundary per older task | + shape numbers | 2 | 93.6% | 0.6% | works |
| joint recogniser with the audit | + shape numbers | 3 | 93.8% | 3.0% | works |
| joint recogniser (rule L) and rule H' | frozen model's states | 4 | still computing | | |

Two rules of different kinds pass on fresh tasks with the same feature: zones with a bound known
in advance (fewer changed routes: 0.6%), and the joint recogniser with the audit (no stored
inputs, the same for every order). The batches differ, so the two numbers are not a comparison
of the rules.

### 2026-11-18 (08:15 UTC) — the GPU run of our method without instructions: read against its rule

Kaggle, T4, 20 minutes of training and re-tests; five tasks, seed 42, T5-base, 10 epochs, every
definition replaced by the empty string; plain adapters, nothing blocked; routes from the table
written and pushed before the run. Files:
`silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks/`. Re-test scores (every
reference answer; the same scoring for all rows):

| no definition anywhere | learned | final score (AP) | forgetting (FT) | answers changed of 400 | right task |
|---|---|---|---|---|---|
| **ours: the recogniser (rule L)** | 77.70 | **77.50** | **0.50** | **6** | 98.4% |
| **ours: the recogniser with the handicap (rule H')** | 77.70 | **77.90** | **0.00** | **1** | 99.4% |
| every input to its own task (the ceiling for any route) | 78.10 | 78.10 | 0.00 | 0 | 100% |
| rule J (logistic regression) / zones, newest wins | 77.70 / 77.10 | 77.50 / 76.70 | 0.50 / 0.50 | 6 / 5 | 98.6 / 97.6% |
| nearest centre of word vectors (rule "centre") | 76.30 | 74.74 | 1.94 | 11 | 92.8% |
| plain adapters, all on | 78.10 | 14.18 | 79.90 | 400 | 20% |
| GainLoRA (published), same setting | 56.08 | 51.94 | 6.51 | 129 | |
| InfLoRA (published), same setting | 51.46 | 37.45 | 18.01 | 217 | |
| for comparison, **with** definitions (run of 2026-11-17): certified switch / GainLoRA / InfLoRA | 79.33 / 60.85 / 56.96 | 79.33 / 61.31 / 58.89 | 0.00 / -0.32 / -1.41 | 0 / 17 / 98 | |

**Read against the rule of 07:29 UTC: "beats the published methods in this setting".** All three
lines hold for rule L: the final score is 25.6 points above the better published method without
definitions (the line was 2 points); forgetting is 0.50 (line: below 2); 6 of 400 answers changed
(line: at most 32). Rule H', read by the same lines: 26.0 points above, forgetting 0.00, 1 answer
changed. **One thing in the rule was badly worded and is read here in the only way that
compares like with like:** the rule said "AP, official test", but the official test of this run
scores the plain adapters with all of them on (14.17); the routed rules exist only in the
re-test. So the comparison above is re-test against re-test (GainLoRA 51.94). The published
methods' official-test numbers differ from their re-test numbers by 0.3 points (51.64 against
51.94), which changes nothing here.

**Read against what we expected.** Changed answers (5 to 15 expected: 6) and forgetting (below 1:
0.50) as guessed. **The score right after learning was far above our guess** (54 to 58 expected;
77.7). We had assumed that without definitions our adapters would learn as poorly as the
published methods did (56). They do not: plain adapters learn the five tasks almost as well
without the definitions as with them (78.1 against 79.3), sst2 at 85 and emotion at 70, where
the two published methods reach 43 to 48 and 12 to 28 in this budget. **So a sentence of the
entry of 07:29 UTC was wrong as written and is corrected here:** it is not "every method" that
learns two of the five tasks poorly without definitions; the two published methods do, in this
budget of 10 epochs. This also means that most of the 25 points is learning, not routing: the
published methods restrict what a new adapter may change in order to protect old tasks, and
with a tenth of their training budget they learn less. The part that is about forgetting is
the other two columns (0.5 against 6.5 and 18.0 points; 6 against 129 and 217 changed answers),
and that part does not depend on how much was learned.

**The audit next to what happened** (rule L; held-out inputs per task: 200): the estimate says
0%, 0%, 0% and 4.0% of the routes of the first four tasks changed (95% bounds 1.5%, 1.5%, 1.5%
and 7.1%); on the test inputs 0, 1, 0 and 5 of 100 routes changed, and the 6 changed routes are
the 6 changed answers.

**What this is and is not.** It is the first GPU evidence for all four points of the target at
once: the task is recognised without an instruction (98.4% and 99.4% of test inputs), no task
is blocked, the guarantees were met, and the two published methods are beaten in this setting,
also when they are given their definitions (61.3 and 58.9) and ours is not. It is one seed,
five tasks, a small model and 10 epochs. It is not the published setting, and the published
methods might close the learning gap with their full budget. Next, as fixed at 07:29 UTC:
seeds 43 and 44, the 15-task sequence, and (new, because of the learning gap) one run of the
published methods with their full number of epochs, so that the comparison of forgetting does
not rest on methods that were stopped early.

### 2026-11-18 (08:16 UTC) — seeds 43 and 44 started on the GPU (both without instructions)

As fixed at 07:29 UTC for the case that the first seed says "beats". Two notebooks, uploaded and
started at 08:16 UTC: `bammidishivkumar/no-instruction-published-five-tasks-seeds-43-44`
(InfLoRA and GainLoRA, seeds 43 and 44, definitions removed; about 45 minutes) and
`bammidishivkumar/no-instruction-ours-five-tasks-seeds-43-44` (our job, plain adapters, the same
route table, which does not depend on the seed; about 25 minutes). Everything else as for seed
42. **Reading, fixed now:** the three-seed means and the range over seeds are reported for both;
the statement "beats the published methods in this setting" stands for three seeds if the
three lines of 07:29 UTC hold for the mean of the three seeds and the final score of rule L is
above the better published method in every single seed; otherwise the paper reports the seeds
as they are and says "in one of three seeds" or whatever is true. What we expect: the same
picture; our score between 76 and 79 in each seed, 4 to 9 changed answers of 400 for rule L and
0 to 3 for rule H' (the routes are the same in every seed, only the adapters differ).

### 2026-11-18 (08:21 UTC) — the next GPU runs and how each will be read. Fixed before any of them is started

The first seed says "beats" (entry above), and it also says where the comparison is weak: most
of the gap in the final score is learning, in a budget of 10 epochs where the published methods
learn less. The remaining GPU hours of this week (about 13 by our count; the exact number is only
visible in the Kaggle editor) go to these runs, in this order of value. If the hours run out, the
later ones are dropped and that is said.

**D. The published methods with their published number of epochs** (`five_tasks_noinstruction_published_e100.ipynb`;
five tasks, seed 42, no definitions, InfLoRA and GainLoRA, 100 epochs instead of 10; about 3.5
to 4 hours). Question: with their full training, do they learn the tasks without definitions,
and do they then keep them? **Reading:** (1) learning: "they close the learning gap" if the
score right after learning reaches at least 73 (within 5 points of our 77.7 at 10 epochs);
"they do not" if it stays below 65. (2) retention at 100 epochs, by the same lines as at 10:
"hold their answers less well than ours" if forgetting is at least 5 points above ours (0.5)
or at least 60 more of the 400 answers change than ours (6); "no such sign" if forgetting is
within 2 points of ours and at most 25 more answers change. (3) The headline number: our final
score at 10 epochs (77.5, rule L) against theirs at 100 epochs; whichever is higher is
reported as it is. **What we expect:** they learn much more (68 to 78 right after learning),
and they still change many answers (60 to 200 of 400), because nothing in them replaces the
instruction as the sign of the task; forgetting 4 to 15 points. We are not sure of the second
part: a method that has learned more may also separate the tasks better.

**U. With instructions, routes from the example alone** (entry of 07:42 UTC; about one hour):
reading as fixed there.

**The feature job** (entry of 07:48 UTC; about 15 minutes; no training): reading as fixed
there, with the second, random split of the fresh tasks added (entry of 08:01 UTC).

**C. The 15 tasks of the benchmark without definitions** (order 1, seed 42, 10 epochs):
`fifteen_tasks_noinstruction_published.ipynb` (InfLoRA, GainLoRA) and
`fifteen_tasks_noinstruction_ours.ipynb` (plain adapters, the route table
`silent_adapters/noinstruction_routes_15.json`, written at 07:46 UTC from the cached features:
rule L gives 1,187 of 1,197 test inputs to their own task after the last task and changes 8
of 1,097 routes; rule H' 1,190 and 3). For reference, with definitions (run of 2026-11-17):
certified switch learned 42.77 = final 42.77, 0 of 1,097 answers changed; GainLoRA 29.08 /
28.62, 215 changed; InfLoRA 30.46 / 26.13, 661 changed. **Reading:** "holds on fifteen tasks"
if, for rule L, the final score (re-test) is at least 2 points above the better published
method without definitions, forgetting is below 2 points, and the changed answers are at most
a quarter of the better published method's count. "Does not" if the final score is below the
better published method or at least as many answers change. Anything else: mixed. And for the
published methods the reading of 06:05 UTC is applied to fifteen tasks (forgetting up by at
least 5 points against their run with definitions, or twice as many changed answers and at
least 165 more of 1,097 (the same share as 60 of 400)). **What we expect:** ours learns 39 to
43 and ends within half a point of it, with 8 to 20 changed answers; the published methods
learn 22 to 29 without definitions and change 300 to 800 answers.

**Not planned for lack of hours:** our method at 100 epochs; trained answers on the second
benchmark; a larger model. They stay open and the paper will say so.

### 2026-11-18 (08:25 UTC) — the feature job is ready and waits for a free GPU slot

`silent_adapters/hidden_states_scale.ipynb` (made by `make_notebook_hidden.py`; reading fixed at
07:48 and 08:01 UTC). Tested by its builder on the CPU: its copy of the rules reproduces the CPU
numbers on the cached features of the 15 SuperNI tasks (rule L 99.03% and 0.84% changed; rule H'
99.23% and 0.50%; second order 99.03 / 0.90 and 99.40 / 0.13), the embedded standardiser is
identical bit for bit, the download and parsing of two fresh tasks give the same texts as the CPU
files (fingerprints), and a whole run with a small model and tiny samples went to the end. Kaggle
allows two GPU sessions at a time and both are in use (the two seed runs), so the job is started
automatically as soon as one is free. The fourth fresh batch is not part of it.

### 2026-11-18 (08:40 UTC) — seeds 43 and 44 of our method without instructions are in; the published methods' two seeds are still running

`silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks_seeds43_44/` (the same
route table as seed 42; only the adapters differ). Re-test:

| seed | rule L: learned / final / forgetting / changed of 400 | rule H': learned / final / forgetting / changed | every input to its own task | plain, all on |
|---|---|---|---|---|
| 42 | 77.70 / 77.50 / 0.50 / 6 | 77.70 / 77.90 / 0.00 / 1 | 78.10 | 14.18 |
| 43 | 77.43 / 76.83 / 0.75 / 6 | 77.43 / 77.43 / 0.00 / 1 | 77.83 | 15.35 |
| 44 | 77.73 / 77.33 / 0.50 / 6 | 77.73 / 77.73 / 0.00 / 1 | 78.13 | 14.78 |
| mean of three | 77.62 / 77.22 / 0.58 / 18 of 1,200 | 77.62 / 77.69 / 0.00 / 3 of 1,200 | 78.02 | 14.77 |

Our guess (entry of 08:16 UTC: 76 to 79 in each seed; 4 to 9 changed answers for rule L and 0 to 3 for
rule H') held. The three seeds differ very little, as they must: the routes are the same in
every seed, so the only thing that varies is how well the adapters were trained. The reading
for three seeds (fixed at 08:16 UTC) needs the published methods' seeds 43 and 44, which are
still running. The feature job started on the freed GPU slot at 08:38 UTC.

### 2026-11-18 (08:46 UTC) — two checks that were still open, read against their lines: the recogniser on the fourth fresh batch, and the small trained model

**1. Rule L on the frozen T5-base states, fourth batch of 30 fresh tasks (read once;
`routing/hidden_rules_study.json`, key `verdict_fresh4`; lines of 07:23 and 07:34 UTC).**

| rule | own task at arrival | own task after the last task | route changed (ever) | audit c/m, (c+1)/(m+1), 95% bound | verdict |
|---|---|---|---|---|---|
| **rule L** | 97.5% | **94.6%** | 3.4% (3.8%) | 3.3%, 3.7%, 5.3% | **use "works"; audit "met"** |
| rule H' (t = 2), reported next to it | 93.9% | 93.6% | 0.5% (0.5%) | bound in advance 21.6% | "keeps the recognition" (within 2 points of rule L); "bound met" |
| rule J (logistic regression), for comparison | 97.4% | 94.5% | 3.2% | | |

**Read against what we expected** (80 to 90% on fresh tasks, "because random task collections
contain pairs of tasks built on the same texts"): the share is higher than we guessed, and the
pairs are there as guessed. Two tasks are built on the same reading passages (an answer for a
question, and a question for the passage): the older one falls from 100% to 53% when the newer
one arrives and the newer one gets 52%; 47% of the older one's routes change, and its audit says
so (102 of its 200 held-out inputs changed). Two tasks on the same kind of product reviews end
at 68% and 84%. The other 26 tasks are at 95 to 100%. Under rule H' the newer task of the first
pair is kept out instead (it takes few inputs) and the older one keeps its routes: 0.5% changed
on average against 3.4%; the price moves from the older task to the newer one, as the note of
07:20 UTC says it must. The caveat of the independent check applies here too: test inputs are
the last 200 instances of a file, held-out inputs a random part of the first 800.

**So all four fresh batches are read:** the two zone rules on the sentence encoder alone do
not pass (54.4%, 72.2%); with the shape numbers a zone rule passes (93.6%) and so does the
joint recogniser (93.8%); and the rule that went to the GPU, on the frozen model's states,
passes (94.6%).

**2. The small trained model on the CPU** (`certified_switch/demo_results/no_instruction_t5-small_seed0_ridge.json`;
t5-small, five tasks, no definition anywhere, 400 training and 50 test inputs per task, 5
epochs, one seed; routes by rule L on the frozen t5-small states; corrected lines of 03:13 UTC).

| | learned | at the end | forgetting | answers changed of 200 | own task at the end |
|---|---|---|---|---|---|
| all adapters on | 64.35 | 9.60 | 68.44 | 200 | |
| routes of rule L | 63.95 | 62.75 | 1.50 | 6 | 97.2% |

**Verdict "works"**, both parts: (a) 6 of 200 answers changed (line: at most 20), forgetting
1.5 points (at most 3), 200 changed with all adapters on (more than 100); (b) the learned score
with the routes is 99% of that with all adapters on (line: 90%) and 97.2% of the test inputs go
to their own task (line: 80%). Two of the five tasks are learned poorly by this small model
with either rule (48.0 on both), so they add nothing to the comparison. The audit per task was
0, 1, 0 and 6 of 200 held-out inputs changed; on the 50 test inputs per task 0, 2, 0 and 4
routes changed, and each changed route changed its answer. By the order fixed at 06:10 UTC
this check was to come before the GPU run; it came after it (entry of 07:46 UTC) and agrees
with it.

### 2026-11-18 (08:47 UTC) — run U (with instructions, routes from the example alone) is tested and queued for the next free GPU slot

Notebook `silent_adapters/five_tasks_instance_routes_with_definitions.ipynb` (reading fixed at
07:42 UTC). Definitions kept; plain training; the re-test at four wordings with the rules of the
paper, the five table rules (the same embedded table as in the run without instructions,
compared again with the pushed file: identical) and two mixed rules (the certified route where
an input is inside a zone, otherwise the table route). Tested by its builder on the CPU with a
small model and three tasks, to the end: the certified switch still gives every input to its
task at the original wording; the table rules give the same routes at all four wordings; the
mixed rules equal the certified switch wherever an input is certified (1,198 checks). It starts
automatically when one of the two GPU sessions ends. Expected: about one hour.

### 2026-11-18 (09:02 UTC) — three seeds without instructions: read against the rule of 08:16 UTC

`silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_seeds43_44/` (InfLoRA
and GainLoRA, seeds 43 and 44) with the two runs of seed 42 and our seeds 43 and 44. Re-test,
five tasks, T5-base, 10 epochs, no definition anywhere:

| method | seed 42: learned / final / forgetting / changed of 400 | seed 43 | seed 44 | mean of three; changed of 1,200 |
|---|---|---|---|---|
| InfLoRA | 51.46 / 37.45 / 18.01 / 217 | 48.89 / 36.07 / 16.27 / 278 | 54.34 / 43.23 / 14.63 / 251 | 51.56 / 38.92 / 16.31 / **746** |
| GainLoRA | 56.08 / 51.94 / 6.51 / 129 | 50.88 / 48.67 / 4.27 / 147 | 56.53 / 52.45 / 6.10 / 138 | 54.50 / 51.02 / 5.63 / **414** |
| ours, rule L | 77.70 / 77.50 / 0.50 / 6 | 77.43 / 76.83 / 0.75 / 6 | 77.73 / 77.33 / 0.50 / 6 | 77.62 / **77.22** / 0.58 / **18** |
| ours, rule H' | 77.70 / 77.90 / 0.00 / 1 | 77.43 / 77.43 / 0.00 / 1 | 77.73 / 77.73 / 0.00 / 1 | 77.62 / **77.68** / 0.00 / **3** |
| nearest centre of word vectors | 76.30 / 74.74 / 1.94 / 11 | 75.63 / 73.88 / 2.18 / 11 | 76.53 / 74.99 / 1.93 / 11 | 76.15 / 74.54 / 2.02 / 33 |
| with definitions, three seeds (paper): certified switch / GainLoRA / InfLoRA | | | | final 79.7 / 62.0 / 62.9; changed 0 / 65 / 370 |

**Reading.** (1) Ours against the published methods: the three lines of 07:29 UTC hold for the
means (final score 26.2 points above the better published method; forgetting 0.58; 18 changed
answers against a quarter of GainLoRA's 414, that is 103), and rule L is above the better
published method in each of the three seeds. So **"beats the published methods in this setting"
stands for three seeds**, for rule L and for rule H', with the same things to be said as for
one seed (most of the gap in the final score is learning; a tenth of the published epochs; two
methods; one sequence of five tasks; the five tasks are among the 15 the rule was chosen on).
(2) The published methods against themselves with definitions (line of 06:05 UTC): InfLoRA's
forgetting rises from -3.1 to 16.3 points and its changed answers from 370 to 746 of 1,200;
GainLoRA's from -0.5 to 5.6 points and from 65 to 414. **Both "hold their answers less well
without instructions" in every seed.** (3) Our guesses of 08:16 UTC (76 to 79 per seed; 4 to 9
and 0 to 3 changed answers) held. One more thing the three seeds show: the simple rule "nearest
centre of word vectors" already ends at 74.5 with 33 changed answers; any rule that recognises
the task goes most of the way in this setting, and rules L and H' add the last three points,
fewer changed answers, and the statements about them.

### 2026-11-18 (09:04 UTC) — the feature job on the GPU: the second benchmark and up to 120 tasks, read against its lines

Kaggle, T4, about 25 minutes, nothing trained; `silent_adapters/results/2026-11-18_kaggle_hidden_states_scale/`
(lines of 07:48 and 08:01 UTC). Frozen T5-base states of the instance text; rules exactly as on
the CPU. Checks inside the job: all 90 downloaded fresh tasks have the same texts as the CPU
files (fingerprints), and the 15 SuperNI tasks reproduce the CPU numbers (rule L 99.03% and
0.84% changed; rule H' 99.23% and 0.50%; rule J 99.03%).

| sequence | tasks | rule L: own task at arrival / at the end | changed | audit (c+1)/(m+1); 95% bound | rule H' (t = 2): at the end / changed | rule J at the end |
|---|---|---|---|---|---|---|
| SuperNI order 1 | 15 | 99.9 / 99.0% | 0.8% | 1.5%; 3.4% | 99.2% / 0.5% | 99.0% |
| SuperNI order 2 | 15 | 99.7 / 99.0% | 0.9% | 1.5%; 3.3% | 99.4% / 0.1% | 99.0% |
| **Long Sequence order 3** (all official test inputs) | 15 | 99.3 / **97.8%** | 1.6% | 2.7%; 4.4% | 96.7% / 0.4% | 98.0% |
| **Long Sequence order 4** | 15 | 99.0 / **97.8%** | 1.6% | 2.6%; 4.3% | 96.3% / 0.3% | 98.0% |
| fresh batch 1 | 30 | 99.2 / 98.2% | 1.4% | 1.8%; 3.2% | 97.5% / 0.5% | 98.8% |
| fresh batches 1+2 | 60 | 97.5 / 93.7% | 4.6% | 5.2%; 7.1% | 91.8% / 0.7% | |
| fresh batches 1+2+3 | 90 | 96.4 / 91.7% | 5.4% | 6.2%; 8.3% | 89.3% / 0.9% | |
| everything (15 + 15 + 90) | 120 | 95.6 / **90.5%** | 6.0% | 6.7%; 9.0% | 88.2% / 1.1% | |

**Read against the lines.** Long Sequence, rule L: **"works on the second benchmark"** (97.8%;
the line was 90%), audit **"met"**, for both official orders (the share at the end is the same
for both, as it must be for this rule). The weak tasks are the ones we named: of the three
entailment tasks, mnli falls from 100% to 84% when rte arrives (its audit says so: 38 of its 200
held-out inputs changed route) and rte gets 90%; cb is not weak (100%). With rule H' the older
task is protected and the newer one pays: mnli keeps 96.5%, rte takes only 59% of its own
inputs. The other parts are descriptions, as said in advance: the code check agrees with the
CPU; and **the share given to the own task falls slowly with the number of tasks: 99.0% with
15, 98.2% with 30, 93.7% with 60, 91.7% with 90, 90.5% with 120** (rule L), while the changed
routes rise from 0.8% to 6.0% and the audit follows them (6.7%). Rule H' changes about 1% of
the routes at every size and gives 2 to 3 points fewer inputs to their own task; its bound in
advance says nothing at this size (above 100% on average for 120 tasks with 200 held-out
inputs). **Our guesses held** (Long Sequence 94 to 98%; 120 tasks 90 to 95%, falling slowly).

**The weakness of the fresh-task split, measured.** With one random split of all 1,000
instances instead of "last 200 as test", the numbers move by 0.4 points or less (own task at
the end: at most 0.25 points for rule L and 0.41 for rule H'; changed routes -0.2 to +0.1
points). So the split that does not ensure exchangeability mattered little on these tasks.
(Corrected at 09:16 UTC: this line first said "0.2 points or less", and the table above said
90.6% for 120 tasks where the file has 90.55%.)

**A second look at an old result.** On the first fresh batch the two zone rules on the
sentence encoder gave 54.4% and 72.2%. On the same 30 tasks, rule L on the frozen model's
states gives 98.2%. (That batch was used before, so this is a description, not a verdict; the
verdict for this rule is the fourth batch, 94.6%.)

### 2026-11-18 (09:06 UTC) — run D started; the long paper updated and checked a second time

**Run D** (the two published methods without definitions, 100 epochs instead of 10; reading
fixed at 08:21 UTC): notebook `silent_adapters/five_tasks_noinstruction_published_e100.ipynb`,
uploaded as `bammidishivkumar/no-instruction-published-five-tasks-100-epochs` at 09:05 UTC. Its
builder tested the whole path on the CPU (small model, 1 epoch in place of 100; the final table
with the 10-epoch numbers next to it printed without error). Results are saved after every task.
Expected: 3.5 to 5 hours. Run U (uploaded 08:59 UTC) and run D were both in Kaggle's queue at
09:05 UTC.

**The long paper** (`paper_certified/main.pdf`, 74 pages) now has: a new abstract; the new
method subsection (three statements with proofs); the new results subsection with three seeds,
all four fresh batches and the small trained model; an updated introduction, related work,
limits and conclusion. A second independent reading of the whole long version (after the first
one of the results subsection) found twelve points; all but two were changed in the text
(statements narrowed: "inputs that occur in two tasks can at best be split", the abstract's
sentence on what is proved, the scope of the GPU comparison, the fourth batch's shared texts
were not counted, stale "not reached" items). Left as they are, and to be told to the authors:
the order of the sections (the introduction comes third, by the authors' choice), and the
number of figures in the abstract. Not yet in the paper: the feature job (second benchmark,
120 tasks), runs U and D, the 15 tasks. The short version and the Word files are being updated.

### 2026-11-18 (09:17 UTC) — both versions of the paper rebuilt with today's results; what is in them and what is not

- **Long version** (`paper_certified/main.pdf`, 75 pages; `main.docx` validated): new abstract; new
  method subsection (`sec_audit.tex`); new results subsection (`sec_results_recogniser.tex`) with
  the published methods without definitions (three seeds), the features, the rules that block
  nothing, the zone shapes, all four fresh batches, the second benchmark and up to 120 tasks
  (table and figure), the small trained model, and the GPU comparison (one seed in detail, then
  three seeds); introduction, related work, limits and conclusion brought in line. Two independent
  readings of the new parts and one of the whole long version; every point they raised was
  changed in the text except the order of the sections (the authors' choice) and the number of
  figures in the abstract.
- **Short version** (`short.pdf`, 75 pages: 9 pages of main text, references to page 13, appendix
  from page 14; `short_main_only.pdf` 13 pages, `short_appendix_only.pdf` 62 pages; `short.docx`
  validated): new abstract; one paragraph and one contribution item in the introduction; one
  paragraph in the method section; a subsection "Without instructions" with the three-seed
  table; limits and conclusion. To make room, the variant for sibling tasks, most of the
  mechanical edits and most of "other models" moved to the appendix; nothing was deleted and no
  spacing or font size was changed. One independent reading of the nine pages; its eight points
  were changed in the text.
- **Numbers:** 1,074 macros from the result files, each compared with its source by the script
  (`paper_certified/audit_results.py`).
- **Not in the paper yet:** run U (with instructions, routes from the example alone; running
  since 09:13 UTC), run D (the published methods with 100 epochs; in Kaggle's queue since 09:05
  UTC), the 15 tasks without definitions, the second feature job with other frozen models.

### 2026-11-18 (09:18 UTC) — the order of the GPU runs from here, and the hours

Kaggle runs two GPU sessions at a time. Run U is running (since 09:13 UTC, about one hour).
Run D (100 epochs, about 5 hours) was accepted at 09:05 UTC and is still in Kaggle's queue; a
queued run holds one of the two places. When run U ends, the 15-task run of our method without
definitions starts automatically (`fifteen_tasks_noinstruction_ours.ipynb`, about 3.3 hours; its
embedded route table was compared with `silent_adapters/noinstruction_routes_15.json`:
identical). After it, the 15-task run of the two published methods without definitions (about
3.5 hours), and last, if hours remain, the feature job with other frozen models (about half an
hour). Reading of all of them: as fixed at 08:21 UTC.

**Hours.** By our count about 16.5 of this week's 30 GPU hours are used (we cannot see the exact
number; only the Kaggle editor shows it). The runs above need about 13.3 more. That leaves
almost nothing. If the hours end while a run is going, Kaggle stops it and its results may be
lost; the order above therefore puts first what matters most, and the exact number is to be
looked at in the Kaggle editor.

### 2026-11-18 (09:50 UTC) — the paper is restructured: the best method first (as asked for)

The paper was to be rewritten (asked for between 09:30 and 09:50 UTC) with all tests in it, the method
found last put first and the others integrated, the structure being left open. Decided:

- **Order of the paper:** introduction, related work, method, results, discussion, limitations,
  conclusion, appendix (the long version had the introduction third; that was the authors' earlier
  choice and is given up).
- **Method:** (1) the setting, with one lemma that carries everything: with one frozen set of
  adapters per task, an answer changes only if its route changes; (2) without instructions: what
  must be paid, the recogniser, the audit, the handicap; (3) with instructions: the zones on the
  instruction and the certified switch, with the write-once theorem and its propositions; (4) the
  zones from the inputs of a task (the bound that does not grow with later tasks, and that blocked
  tasks in our tests). The statements and proofs are the ones checked earlier; only their place
  and the linking sentences changed, and the conditions that the audit and the handicap need on
  the adapters are now stated once, in the lemma.
- **Results, same order:** the published methods without their instructions; recognising the task
  without an instruction (features, rules, fresh tasks, the second benchmark, many tasks);
  trained answers without instructions (small model, GPU, three seeds); then the certified
  switch with instructions (five and fifteen tasks, rewordings, what the zones add and cost,
  edits, orders, other models, rewording at scale); last the first test of the zones from inputs.
  No test is dropped.
- **Two versions as before:** the long one, and the 9-page one with the appendix, in the same
  order. The method part of the long version is done; the results and the introduction,
  discussion and conclusion are being reordered in two parts; then the 9 pages; then an
  independent reading of both, the Word files, and the research log.

### 2026-11-18 (09:52 UTC) — run U (with instructions, routes from the example alone): read against its rule

Kaggle, T4, 36 minutes of training and re-tests; five tasks, seed 42, T5-base, 10 epochs, with
the definitions; plain adapters; the same route table as in the runs without instructions.
Files: `silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions/`. Re-test at the
four wordings of the definition (final score / answers changed of 400 / share given to the own
task):

| rule | original | light | medium | heavy |
|---|---|---|---|---|
| certified switch (zones on the instruction), this run | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 77.61 / 0 / 100% | 72.12 / 0 / 93.8% |
| **recogniser, routes from the example alone (rule L)** | 78.73 / 3 / 98.4% | 78.35 / 3 / 98.4% | 76.81 / 3 / 98.4% | **76.12** / 3 / 98.4% |
| **the same with the handicap (rule H')** | 79.13 / 1 / 99.4% | 78.75 / 1 / 99.4% | 77.21 / 1 / 99.4% | **76.52** / 1 / 99.4% |
| every input to its own task (ceiling) | 79.33 | 79.15 | 77.61 | 76.72 |
| zone first, else rule L | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 76.81 / 3 / 98.4% | 76.12 / 3 / 98.4% |
| zone first, else rule H' | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 77.21 / 1 / 99.4% | 76.52 / 1 / 99.4% |
| nearest centre of word vectors of the whole input | 79.33 / 0 | 79.15 / 0 | 77.21 / 0 | 58.91 / 58 / 72.2% |
| GainLoRA, same seed (run of 2026-11-17) | 61.31 / 17 | 59.49 / 15 | 58.61 / 32 | 52.01 / 49 |
| InfLoRA, same seed | 58.89 / 98 | 59.87 / 85 | 60.45 / 91 | 56.37 / 151 |

The certified switch's numbers are the same as in the run of 2026-11-17 to the last digit, so
plain training without blocking gave the same adapters here, as expected.

**Read against the rule of 07:42 UTC: "repairs the weak spot".** For rule L the heavy wording
ends at 76.12 (line: at least 74.1; the certified switch has 72.12), the original wording at
78.73 (line: at least 77.3), and 3 of 400 answers change at every wording (line: at most 8).
Rule H': 76.52, 79.13 and 1. **Our guesses held** (original 78.5 to 79.3; heavy 75.5 to 76.7; a
gain of about 4 points at the heavy wording and a loss of up to 1 point at the original), except
one in the good direction: we expected about 6 changed answers for rule L and there are 3. Six
routes change, as in the run without definitions, but with the definition in front of it the
model gives the same answer for three of those inputs whichever of the two tasks' adapters
answer.

**What it shows.** The routes of the recogniser are the same at every wording, because it never
reads the instruction; so the score stays close to the ceiling at every wording (rule L 0.6 to 0.8
points below it, rule H' 0.2 to 0.4 points; corrected at 10:00 UTC, we had first written "within
0.2 to 0.6 points"), and what is lost under rewording is only what the adapters themselves lose when their
instruction is reworded (79.3 to 76.7), not misrouting. The zones are exact where the
instruction is unedited (0 changed answers, where the recogniser changes 1 to 3 of 400) and lose
4.6 points at the heavy wording through misrouted inputs (76.72 against 72.12). The mixed rule (the zone where an input
is certified, the recogniser elsewhere) has the best of both at the original, light and heavy
wordings; at the medium wording it is 0.4 to 0.8 points below the certified switch, whose own
fallback happens to be right for all inputs there. One seed, five tasks, three rewordings written
by machine and not by people: a first look, and it repairs on the benchmark what five attempts
to widen the zones did not.

**Also started:** the 15-task run of our method without definitions, automatically, when the
GPU slot of run U became free.

### 2026-11-18 (12:24 UTC) — a pause, and run D (the two published methods without definitions, at their published 100 epochs) read against its rule

**The pause.** From about 10:22 to 12:20 UTC nothing was done. The fixes of the two independent readings of the restructured paper were half
applied at that moment (the short version was finished, the long version was cut off part of the
way) and are being finished now. The 15-task run of our method started on Kaggle some time
after 10:15 UTC and is still running.

**Run D.** Kaggle T4 x2, both methods at the same time, 128 minutes. Five tasks, seed 42, T5-base,
no definitions, 100 epochs per task, which is the published number and ten times what all
earlier runs here had. Files:
`silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_e100/`. Re-test
(every reference answer); the official test (one random reference) in brackets.

| five tasks, no definitions, seed 42 | score right after learning | final score | forgetting | answers changed of 400 |
|---|---|---|---|---|
| InfLoRA, 100 epochs | 78.37 (77.72) | 25.37 (25.13) | 66.25 (65.74) | 382 |
| GainLoRA, 100 epochs | 76.29 (75.66) | 56.41 (55.88) | 25.35 (25.22) | 178 |
| InfLoRA, 10 epochs (run of this morning) | 51.46 | 37.45 | 18.01 | 217 |
| GainLoRA, 10 epochs | 56.08 | 51.94 | 6.51 | 129 |
| ours, 10 epochs: plain adapters behind the recogniser (rule L) | 77.70 | 77.50 | 0.50 | 6 |
| ours, 10 epochs: with the handicap (rule H') | 77.70 | 77.90 | 0.00 | 1 |

Per task, official test, right after learning -> at the end. InfLoRA: 99.2 -> 11.0, 83.0 -> 3.0,
62.4 -> 36.6, 76.0 -> 7.0, 68.0 -> 68.0. GainLoRA: 99.7 -> 65.8, 72.0 -> 31.0, 62.6 -> 60.6,
74.0 -> 52.0, 70.0 -> 70.0.

**Read against the rule of 08:21 UTC.**

1. *Learning:* **"they close the learning gap"**. Both reach at least 73 right after learning
   (78.37 and 76.29; ours has 77.70 with 10 epochs). With their full number of epochs they learn
   sst2 (83 and 72) and emotion (68 and 70), the two tasks they had learned poorly with 10 epochs.
2. *Retention:* **"hold their answers less well than ours"**. Forgetting is 66.25 and 25.35
   points against 0.50 (the line was 5 points above ours), and 382 and 178 of 400 answers change
   against 6 (the line was 60 more).
3. *The headline number:* ours with 10 epochs ends at 77.50; they end at 56.41 (GainLoRA) and
   25.37 (InfLoRA) with 100 epochs.

**Our guesses.** Learning 68 to 78: right for GainLoRA, and InfLoRA is just above it (78.37).
Changed answers 60 to 200: right for GainLoRA (178), far too low for InfLoRA (382). Forgetting 4
to 15 points: wrong for both, it is 25 and 66. Our doubt ("a method that has learned more may
also separate the tasks better") did not come true: they learned more and lost more.

**What this changes.** The sentence "most of the gap in the final score is learning, not
retention" belongs to the 10-epoch comparison. With their published number of epochs the two
methods learn the five tasks as well as plain adapters do, and the gap in the final score (77.5
against 56.4 and 25.4) is then forgetting. Forgetting grew with what was learned: from 10 to 100
epochs it went from 18.0 to 66.3 points (InfLoRA) and from 6.5 to 25.4 (GainLoRA), and the
changed answers from 217 to 382 and from 129 to 178.

**What it does not show.** One seed, five tasks, T5-base (the published setting uses a larger
model), and our adapters had a tenth of the epochs (which is against us, not for us). And one
control is missing: **the same two methods at 100 epochs WITH their definitions, in this
setting.** Their papers report little forgetting with definitions, on a larger model; here, with
definitions and 10 epochs, their forgetting was near zero, but then they had learned little.
Without that control the loss at 100 epochs cannot be put on the missing instruction alone: it
could also come from 100 epochs on this smaller model. So the control is the next GPU run
(entry below), before the 15-task run of the published methods.

### 2026-11-18 (12:25 UTC) — run E, the control for run D: the two published methods WITH their definitions at 100 epochs. Rule fixed before the run is started

**Why.** Run D says: without definitions and with their full number of epochs, the two published
methods learn the five tasks and then lose them. To put this loss on the missing instruction,
the same run is needed with the definitions left in: same five tasks, seed 42, T5-base, 100
epochs, InfLoRA and GainLoRA, official code, both at the same time on two GPUs. The re-test is
done at the four wordings of the definition (original, light, medium, heavy), as in the 10-epoch
run with definitions. Notebook: `silent_adapters/five_tasks_published_e100_with_definitions.ipynb`
(being built from the notebook of run D; the only intended difference is that the definitions
are not removed). About 2.5 hours.

**Numbers it will be read against** (seed 42, re-test). Without definitions, 100 epochs: InfLoRA
forgetting 66.25, 382 of 400 answers changed, final 25.37; GainLoRA 25.35, 178, final 56.41. With
definitions, 10 epochs: InfLoRA forgetting -1.41, 98 changed, final 58.89; GainLoRA -0.32, 17,
61.31. Ours with definitions, 10 epochs: certified switch final 79.33 with 0 changed; recogniser
with routes from the example alone 78.73 with 3 changed; under the heavy rewording 72.12 and
76.12.

**Reading, for each method by itself.**

1. *Was it the instruction?* **"The instruction was what kept the answers"** if, with
   definitions, forgetting is at most half of its forgetting without definitions AND at most half
   as many answers change (GainLoRA: forgetting at most 12.7 and at most 89 of 400; InfLoRA: at
   most 33.1 and at most 191). **"It was not the instruction"** if forgetting with definitions is
   at least 80% of the forgetting without (GainLoRA at least 20.3; InfLoRA at least 53.0).
   Anything else: mixed.
2. *Against ours, with definitions.* Reported as it is: their final score at the original wording
   with 100 epochs next to ours with 10 epochs (79.33 and 78.73), and their changed answers next
   to ours (0 and 3). If their final score is above 79.33, the paper says that with their
   instructions and ten times our epochs they end above our runs, and that our method was not run
   with 100 epochs.
3. *Reworded definitions.* Their final score and changed answers at the heavy wording next to
   ours (76.12 and 3 for the recogniser, 72.12 and 0 for the certified switch); reported as it is.

**What we expect.** They learn 79 to 84 right after learning. GainLoRA: forgetting 0 to 6 points,
15 to 80 changed answers, final score 75 to 83. InfLoRA: forgetting 1 to 12 points, 60 to 200
changed answers, final 68 to 82. So "the instruction was what kept the answers" for both, less
sure for InfLoRA. We think it is more likely than not that GainLoRA ends above our 10-epoch score
at the original wording, and that both lose several points under the heavy rewording.

**Hours.** Run D took 128 minutes for both methods together (one session with two GPUs), less than
the 4.5 to 5 hours we had counted. By our count about 19.5 of the week's 30 GPU hours were used
when run D ended, the running 15-task run of our method will take about 3 more, and run E about
2.5. Then about 5 hours would be left for the 15-task run of the published methods. The exact
number is only visible in the Kaggle editor.

### 2026-11-18 (12:42 UTC) — the restructured paper is finished and sent; run E is started

**The paper, restructured.** Both versions now have the
method found last first. Long version (`paper_certified/main.pdf`, 80 pages): abstract,
introduction, related work, method (setting and the lemma "an answer changes only with its
route"; the recogniser with its audit and its variant with the handicap; then word-vector
features, zones, the certified switch and its guarantee, the variant for sibling tasks, zones
from the inputs of a task; compared methods; protocol), results (4.1 the published methods
without instructions; 4.2 recognition; 4.3 trained answers without instructions, with the
100-epoch run; 4.4 five tasks with instructions; 4.5 reworded instructions; 4.6 routes from the
example alone; 4.7 fifteen tasks; 4.8 where the gap comes from; then the stress test, 875 tasks,
edits, orders, unrelated inputs, other models, rewording at scale, zones from inputs),
discussion, limitations, conclusion, appendix. Short version (`paper_certified/short.pdf`, 78
pages; the main text ends on page 9): the same order. No test was dropped.

**Two independent readings** (one per version, by a reader that had not written the text) found
about 45 points in the long version and about 35 in the short one. The serious ones were
leftovers of the old order and missing caveats: "the rule of the paper" still meant the zones; a
limitation said "tasks that differ only in their data were not tested", which the main method
now tests; the related-work paragraph argued against best-match rules although the recogniser is
one; the new result on routes from the example alone was reported without its seed-42 caveat
(the certified switch has 72.1 at the heavy wording in seed 42 and 74.1 on average over three
seeds); the feature symbol clashed with the gate symbol; "pairs of tasks built on the same texts
got about half each, which no method can do better than" was too strong and was about a related
rule (now: under the recogniser the older of the two tasks falls from 100.0% to 37.5% and the
newer gets 65.0%; under the handicap the older keeps 97.5% and the newer gets 0.5%; 29.3% of
their texts are the same word for word, and only there no rule can do better). All points were
applied or answered; both versions build with no overfull box and no undefined reference; all
1,292 numbers are checked against the result files by the script.

**Run D is in both versions** (abstract, introduction, results, discussion, limitations,
conclusion), with its limits and with the sentence that the control with definitions was not
run. That sentence will be replaced by the result of run E.

**Run E is started** (12:42 UTC, Kaggle, two GPUs; the rule was fixed and pushed at 12:26 UTC):
`silent_adapters/five_tasks_published_e100_with_definitions.ipynb`,
kernel `bammidishivkumar/published-five-tasks-with-definitions-100-epochs`. It is the notebook
of run D with two changes: the definitions are not removed, and the re-test uses the four
wordings again. A dry run on the CPU (small model, three tasks, one epoch) went through all
cells. Expected: about 2.5 hours. The 15-task run of our method is still running.

### 2026-11-18 (12:49 UTC) — run F: our method with 100 epochs, without definitions. Rule fixed before the run is started

**Why.** The question was whether our method was run with 100 epochs. It was not: run D gave the
published methods their 100 epochs, and ours is compared with 10. The fair comparison at equal
training is missing, and a reader will ask for it. It replaces the 15-task run of the published
methods in the order of runs, because the GPU hours of this week will not carry both (that run
would again be at 10 epochs, where the published methods learn little).

**The run.** `silent_adapters/five_tasks_noinstruction_ours_e100.ipynb`: the notebook of this
morning's run of our method (five tasks, seed 42, T5-base, no definitions, plain adapters with
nothing blocked, the same route table) with one change, 100 epochs per task instead of 10. The
10-epoch run took 20 minutes of training and tests; we expect 2 to 3.5 hours.

**What the lemma already says.** The routes do not depend on the adapters, so they are the same
as with 10 epochs: 6 of the 400 compared routes change under the recogniser (rule L) and 1 under
the handicap (rule H'). An answer can change only if its route changes. So at most 6 answers can
change under rule L and at most 1 under rule H', however long the adapters are trained. More
than that would mean an error in the code, not a result.

**Reading, for rule L.** "Beats the published methods at equal epochs, in this setting" if all
three hold: the final score (re-test) is at least 2 points above the better published method at
100 epochs (56.41, so at least 58.41); forgetting is below 2 points; at most 44 of 400 answers
change (a quarter of GainLoRA's 178). "Does not" if the final score is below 56.41. Anything
else: mixed. Also reported as it is: the score right after learning with 100 epochs next to
theirs (78.37 and 76.29) and next to ours with 10 (77.70).

**What we expect.** Score right after learning 77 to 82 (a little more than with 10 epochs, or
the same: these tasks were already learned); final score for rule L within 1 point of it;
forgetting 0.2 to 1.5 points; 3 to 6 changed answers for rule L and 0 or 1 for rule H'. We are not
sure that 100 epochs help: plain adapters may also fit the training data too closely and lose a
point or two.

**Hours.** By our count about 25 of the week's 30 GPU hours are used or promised once the running
15-task run and run E are done. Run F then leaves about 2. The exact number is only visible in
the Kaggle editor; if the hours end during a run, Kaggle stops it and its results may be lost.
It is started when one of the two GPU places is free.

### 2026-11-18 (12:52 UTC) — 9 GPU hours are left; one more run is planned (two more seeds of run U). Rule fixed before it is started

**Hours.** In Kaggle, 9 GPU hours of this week are left (about 12:50 UTC). Our
own count had been lower. Still to pay from them: the rest of the running 15-task run of our
method (about 1 hour), run E (about 2.5), run F (2 to 3.5). That leaves 2 to 3.5 hours.

**Run U2: the run with instructions and routes from the example alone, seeds 43 and 44**
(`five_tasks_instance_routes_with_definitions_seeds43_44.ipynb`, being built from the notebook
of run U; both seeds at the same time, about 45 minutes). **Why:** the result "the recogniser
repairs the weak spot of the zones under rewording" rests on seed 42, and an independent reader
of the paper pointed out that seed 42 is the seed in which the zones lose most at the heavy
wording (72.1, where the mean over three seeds is 74.1). So the gain of 4 points may be smaller
on average.

**Reading, for the means over seeds 42, 43 and 44, rule L.** "Stands for three seeds" if the
mean final score at the heavy wording is at least 2 points above that of the certified switch,
the mean at the original wording is within 2 points of the certified switch, and in every seed
at most 8 of 400 answers change at every wording. "Does not stand" if the mean heavy score is
below that of the certified switch or the mean original score is more than 4 points below.
Anything else: mixed, and the paper then says "a gain in seed 42, smaller on average".

**What we expect.** The certified switch should have about 75 at the heavy wording in seeds 43
and 44 (from the three-seed mean of 74.1 and the 72.1 of seed 42). The recogniser should again
be about 0.6 to 0.8 points below the ceiling at every wording, so about 76 to 77 at the heavy
wording. That makes a mean gain of about 2 to 2.5 points: close to the line, and we are not sure
it holds. Changed answers: 2 to 6 of 400 per seed for rule L (6 routes change in every seed), 0
or 1 for rule H'.

**Order of the runs now:** E (in the queue), F (starts by itself when a place is free), then
U2. The 15-task run of the published methods without definitions would need about 3.5 hours
more and probably does not fit; if hours are left after U2, it is started last.

### 2026-11-18 (12:58 UTC) — names of the methods (the choice of names was left open)

The two rules without instructions get names that match the title: the recogniser is
**Finder** (it finds the task of an input) and its variant with the handicap is **Keeper** (an
older task keeps what it found). Only the two macros changed (`\recog`, `\handicap`), the
labels of one figure, and one sentence in each introduction; "recogniser", "audit" and
"handicap" stay as the ordinary words for what the rules are. The other names stay: Certified
(the certified switch, with ten figures that carry the name), Exact-first, and the comparison
rules Plain, Key, Nearest and Centre. To fit the new sentence, three short sentences that were
said twice were cut from the short version; its Conclusion still ends on page 9.

### 2026-11-18 (13:00 UTC) — the renaming is undone

We had renamed the recogniser to "Finder" and its variant to "Keeper" in both versions right
after the choice of names was left open. Suggestions were to be seen first and the names decided afterwards.
The commit is reverted: the paper, the figure and the Word files are back to
the names Recogniser and Handicap (state of 12:42 UTC, plus nothing else). Names will be changed
only after they have been picked.

### 2026-11-18 (13:17 UTC) — new title

After several rounds of suggestions the title chosen is **"Never Forget What You Learn:
FK-LoRA for Continual Learning Without Forgetting"**. It is in both versions and in the Word files. The system is now called FK-LoRA in the abstract and the
introduction ("after the rule finders keepers"). The names of the single rules (Recogniser,
Handicap, Certified) are not changed yet: FinderLoRA, KeeperLoRA and ClaimLoRA are proposed and
wait for a yes. It was noted once that "never forget" in a title is stronger than
the tables (the main rule changes 18 of 1,200 answers; only inside a zone is "never" proved);
the abstract, the limitations and the section "what we do not claim" say what is and is not
shown.

### 2026-11-18 (13:20 UTC) — the chosen method names are in the paper

These names were approved: **FinderLoRA** (was Recogniser: finds the task, reads no
instruction), **KeeperLoRA** (was Handicap: older tasks keep their inputs), **ClaimLoRA** (was
Certified: the zone method), and **FK-LoRA** for the whole system. Changed: the three macros in
both versions, one sentence in each introduction that introduces the names, the Word converter,
and the labels of seven figures (regenerated with `make_results.py` and `audit_results.py`; no
table and no number changed; `make_results.py` now skips the no-instruction demo files, which
have another format and had made it stop near its end). Not
renamed: Exact-first, Cert-scaled, and the comparison rules Plain, Key, Nearest and Centre. The
words "recogniser", "handicap" and "certified switch" stay in the text as ordinary words for
what the rules are. To pay for the new sentences, three short sentences that were said twice
were cut from the short version; its Conclusion still ends on page 9. The code folder
`certified_switch/` keeps its class names.

### 2026-11-18 (13:24 UTC) — the 15-task run of our method without definitions: read; run F is running

**Run F** (our method with 100 epochs, five tasks, no definitions) was accepted by Kaggle at
13:22 UTC and is running; run E (the published methods with definitions at 100 epochs) is
running too.

**The 15-task run of our method** (Kaggle, 176 minutes; the 15 SuperNI tasks in the official
order 1, seed 42, T5-base, 10 epochs, no definitions, plain adapters with nothing blocked, the
route table written at 07:46 UTC). Files:
`silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_fifteen_tasks/`. The log shows
that the definitions were removed in all 45 files. Re-test:

| fifteen tasks, seed 42 | learned | final | forgetting | answers changed of 1,097 | own task |
|---|---|---|---|---|---|
| all adapters on (plain), no definitions | 42.77 | 4.96 | 40.84 | 1,097 | 8.4% |
| nearest centre of word vectors, no definitions | 40.09 | 39.28 | 0.88 | 28 | 92.6% |
| **recogniser (rule L, FinderLoRA), no definitions** | 42.64 | **42.16** | 0.51 | **8** | 99.2% |
| **with the handicap (rule H', KeeperLoRA), no definitions** | 42.57 | **42.36** | 0.22 | **3** | 99.4% |
| every input to its own task (ceiling), no definitions | 42.77 | 42.77 | 0.00 | 0 | 100% |
| for reference, WITH definitions (run of 2026-11-17): certified switch | 42.77 | 42.77 | 0.00 | 0 | 100% |
| WITH definitions: GainLoRA | 29.08 | 28.62 | | 215 | |
| WITH definitions: InfLoRA | 30.46 | 26.13 | | 661 | |

**Read against the rule of 08:21 UTC.** The rule compares with the published methods without
definitions on fifteen tasks, and that run has not been made (the hours went to runs E, F and
U2). So the verdict "holds on fifteen tasks" cannot be given. What can be read: forgetting is
0.51 points (line: below 2); 8 answers change under rule L and 3 under rule H', which are
exactly the routes that the table changes (as the lemma says); and the final score is 0.6 points
below the ceiling. **Our guesses held:** learned 39 to 43 (42.64), final within half a point of
it (0.48 below), 8 to 20 changed answers (8).

**Two things to say with it.** (1) Without any definition our method ends at 42.16 on fifteen
tasks, where the two published methods WITH their definitions ended at 28.62 and 26.13 with
215 and 661 changed answers (10 epochs for all; at 10 epochs they learn less, so this is again
partly learning). (2) The score right after learning is 42.77 without definitions and 42.77
with them. That is a coincidence of the two averages at two decimals: the per-task scores
differ (samsum 35.5 against 32.3, sst2 85 against 88, outcome extraction 43.6 against 52.7, and
so on). Plain adapters learn these fifteen tasks about as well without the definitions as with
them.

### 2026-11-18 (15:39 UTC) — run F (our method with 100 epochs, no definitions): read against its rule

Kaggle, 134 minutes. Five tasks, seed 42, T5-base, no definitions, plain adapters with nothing
blocked, 100 epochs per task, the same route table as with 10 epochs. Files:
`silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks_e100/`. Re-test:

| five tasks, no definitions, seed 42, 100 epochs | learned | final | forgetting | changed of 400 | own task |
|---|---|---|---|---|---|
| all adapters on (plain) | 81.18 | 16.88 | 80.37 | 400 | 20.0% |
| nearest centre of word vectors | 78.58 | 77.03 | 1.94 | 11 | 92.8% |
| **recogniser (rule L, FinderLoRA)** | 80.78 | **80.18** | 1.00 | **6** | 98.4% |
| **with the handicap (rule H', KeeperLoRA)** | 80.78 | **80.98** | 0.00 | **1** | 99.4% |
| every input to its own task (ceiling) | 81.18 | 81.18 | 0.00 | 0 | 100% |
| InfLoRA, 100 epochs (run D) | 78.37 | 25.37 | 66.25 | 382 | |
| GainLoRA, 100 epochs (run D) | 76.29 | 56.41 | 25.35 | 178 | |
| ours with 10 epochs, rule L | 77.70 | 77.50 | 0.50 | 6 | 98.4% |

**Read against the rule of 12:5x UTC: "beats the published methods at equal epochs, in this
setting".** All three lines hold for rule L: the final score is 80.18 (line: at least 58.41, two
points above GainLoRA's 56.41); forgetting is 1.00 point (line: below 2); 6 of 400 answers
change (line: at most 44). Rule H': 80.98, 0.00 and 1.

**The lemma, checked.** The routes are the same as with 10 epochs, and exactly as many answers
change as routes: 6 under rule L and 1 under rule H'. Ten times the training did not change one
answer more.

**Our guesses held:** learned 77 to 82 (80.78), final within 1 point of it (0.60 below),
forgetting 0.2 to 1.5 (1.00), 3 to 6 changed answers (6) and 0 or 1 for rule H' (1). Our doubt
that 100 epochs might not help was wrong: the score right after learning rose by 3 points
(77.70 to 80.78), mostly on the emotion task (70 to 83).

**What it shows.** At equal training (100 epochs each, the published number), without
definitions, on five tasks and one seed: the three ways learn about the same (80.8, 78.4, 76.3
right after learning), and then our rules keep it (80.2 and 81.0 at the end) where the
published methods lose it (56.4 and 25.4). The forgetting of rule L is 1.00 point here against
0.50 with 10 epochs: the same six inputs go to another task, and a wrong route costs more when
the adapters are better. **What it does not show:** more than one seed, more than five tasks at
100 epochs, the larger model of the published setting; and the control with definitions (run E)
is still running.

### 2026-11-18 (16:16 UTC) — all tables made easier to read

The tables were hard to read, and full values and the best values were asked for, in all
tables. Done in both versions: (1) column headings are full words with an arrow for the better
direction ("score right after learning", "final score", "forgetting (points)", "answers
changed", "sent to own task"), instead of Learned, AP, FT, Changed, Own; (2) the best value is
bold wherever rows are methods or rules compared on a quality measure, computed by the scripts
from the numbers, with ceiling rows left out and ties all bold; tables whose rows are tasks,
settings or sweeps have no bold across rows. The 20 tables of today's results are written this
way by `audit_results.py`; the 38 older tables are post-processed by the new
`paper_certified/polish_tables.py` (run it after the generator scripts; it changes no number and
running it twice changes nothing more). Captions were brought in line, and one sentence in each
protocol section says what bold means. Two captions used bold for "the chosen variant"; they
now name the chosen variant in words. Lower forgetting is bold even when it is negative (the
published methods on five tasks with definitions), as the convention says. No number changed:
1,374 numbers still verify. To keep the 9-page limit, the headings of the tables in the short
main text are at most two lines, three captions were shortened, and three sentences that are
also in the appendix were cut. Long version 82 pages, short version 80.

### 2026-11-18 (16:26 UTC) — run U2 (routes from the example alone, with definitions, seeds 43 and 44): read against its rule

Kaggle, both seeds at the same time, about 45 minutes. Files:
`silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions_seeds43_44/`. Final
score (re-test) and answers changed of 400, per seed; seed 42 is run U of this morning.

| wording, rule | seed 42 | seed 43 | seed 44 | mean of three |
|---|---|---|---|---|
| original, certified switch (zones) | 79.33 / 0 | 80.80 / 0 | 79.20 / 0 | 79.78 |
| original, recogniser (rule L) | 78.73 / 3 | 80.20 / 3 | 78.40 / 3 | 79.11 |
| original, handicap (rule H') | 79.13 / 1 | 80.60 / 1 | 78.80 / 1 | 79.51 |
| heavy, certified switch (zones) | 72.12 / 0 | 75.72 / 0 | 75.03 / 0 | 74.29 |
| heavy, recogniser (rule L) | 76.12 / 3 | 78.92 / 3 | 78.63 / 2 | 77.89 |
| heavy, handicap (rule H') | 76.52 / 1 | 79.32 / 1 | 79.03 / 0 | 78.29 |
| heavy, every input to its own task (ceiling) | 76.72 | 79.72 | 79.43 | 78.62 |

At the light and the medium wording the recogniser changes 3 and 3, and 2 and 3 answers in seeds
43 and 44, the handicap 1 each time.

**Read against the rule of about 13:00 UTC: "stands for three seeds".** Mean final score at the
heavy wording: 77.89 for rule L against 74.29 for the certified switch, 3.6 points above (line:
at least 2). Mean at the original wording: 79.11 against 79.78, 0.67 below (line: within 2). In
every seed and at every wording at most 3 of 400 answers change (line: at most 8).

**Our guesses.** Right: the certified switch has about 75 at the heavy wording in seeds 43 and 44
(75.72 and 75.03). Too low: we expected the recogniser at 76 to 77 there and a mean gain of 2 to
2.5 points; it is 78.9 and 78.6 and the gain is 3.6, because in these two seeds the adapters
themselves lose less under the heavy rewording (ceiling 79.7 and 79.4, against 76.7 in seed
42). The recogniser stays 0.6 to 0.8 points below the ceiling at every wording and in every
seed, as in seed 42; the zones lose about 4 points at the heavy wording in every seed.

**So:** the result of run U does not rest on seed 42 any more. Still five tasks, and rewordings
written by machine during development.

### 2026-11-18 (16:40 UTC) — the paper after run U2, and "out of" in all tables

**Run U2 is in both versions.** The result on routes from the example alone is now stated for
three seeds everywhere (abstract, introduction, results, discussion, limitations, conclusion):
77.9 against 74.3 for the zones at the heavy wording, at most 3 of 400 answers changed in any
seed. The long version keeps the seed-42 table and adds a paragraph "Three seeds" with its line
and a three-seed table; the table in the short main text now shows the three-seed means, with
the two published methods (three seeds, with definitions) below.

**"Out of" in the tables (asked for; the form was left open).** Scores now say
"(out of 100)" or "(of 100)" in the heading, forgetting says "(points of 100)", and every count
says "of N", in the heading or, where N differs between rows, in the cell ("382 of 400"). Where
N could not be found with certainty for an older count column, the heading was left as it was
(a few "wrong zone", "blocked" and "changed later" columns in appendix tables). A first version
of the main table of the short paper came out cramped (small font, columns touching) and was
redone in the normal table font; in that redo the ceiling row became bold by mistake and was
fixed (the ceiling row is now marked by its rule in the code, not by its label). 1,553 numbers
verify. Long version 83 pages; short version 81, Conclusion on page 9.

### 2026-11-18 (16:53 UTC) — bold and "out of" taken out of the tables again

Two of the table changes of this afternoon were to be undone: the bold for the best value,
and the "out of" wording. Both are removed in all tables (the code for them is switched off in
`audit_results.py` and removed from `polish_tables.py`). The bold that was there before today
and means something else is back as it was: the variants used later in the table of rules, the
chosen feature row, the chosen zone shape, and the primary setting in one appendix table; the
two captions say so again, and the sentence "bold marks the best value" is out of the two
protocol sections. Kept: headings in full words with arrows, the roomier layout of the main
table of the short paper, and every result and table added today (15 tasks, our 100-epoch run,
the three seeds of the run with routes from the example alone). No number changed; 1,553
numbers verify. Long version 82 pages; short version: Conclusion ends on page 9.

### 2026-11-18 (17:09 UTC) — run E (the control: the two published methods WITH definitions at 100 epochs): read against its rule

Kaggle, both methods at the same time, 224 and 208 minutes (longer than the 2.5 hours we had
counted: the re-test at four wordings after every task takes time). Five tasks, seed 42,
T5-base, 100 epochs, definitions left in. Files:
`silent_adapters/results/2026-11-18_kaggle_published_five_tasks_e100_with_definitions/`. Re-test:

| seed 42, 100 epochs | learned | final | forgetting | changed of 400 |
|---|---|---|---|---|
| InfLoRA, WITH definitions (this run) | 79.74 | 76.97 | 3.47 | 103 |
| InfLoRA, no definitions (run D) | 78.37 | 25.37 | 66.25 | 382 |
| GainLoRA, WITH definitions (this run) | 73.86 | 73.70 | 0.21 | 9 |
| GainLoRA, no definitions (run D) | 76.29 | 56.41 | 25.35 | 178 |
| ours, no definitions, 100 epochs (run F): rule L / rule H' | 80.78 | 80.18 / 80.98 | 1.00 / 0.00 | 6 / 1 |
| ours, WITH definitions, 10 epochs: certified switch / rule L | 79.33 | 79.33 / 78.73 | 0.00 / 0.50 | 0 / 3 |

With reworded definitions (final score / changed of 400), this run: InfLoRA light 74.63 / 124,
medium 63.34 / 214, heavy 68.59 / 143; GainLoRA light 73.03 / 12, medium 68.29 / 48, heavy
57.58 / 71.

**Read against the rule of 12:26 UTC.**

1. *Was it the instruction?* **"The instruction was what kept the answers", for both methods.**
   GainLoRA: forgetting 0.21 with definitions against 25.35 without (line: at most 12.7) and 9
   changed answers against 178 (line: at most 89). InfLoRA: 3.47 against 66.25 (line: at most
   33.1) and 103 against 382 (line: at most 191). Same methods, same seed, same epochs, same
   model; the only difference is the definition in the input.
2. *Against ours, with definitions.* Their final scores at the original wording, 76.97 and
   73.70 with 100 epochs, are below ours with 10 epochs (79.33 and 78.73), and they change 103
   and 9 answers where ours change 0 and 3. Our method was not run with definitions at 100
   epochs.
3. *Reworded definitions.* At the heavy wording they end at 68.59 (143 changed) and 57.58 (71
   changed); ours with 10 epochs at 76.12 (3 changed, recogniser) and 72.12 (0 changed,
   certified switch).

**Our guesses.** Right: InfLoRA learns 79.74 (79 to 84), forgets 3.47 (1 to 12), changes 103
answers (60 to 200), ends at 76.97 (68 to 82); GainLoRA forgets 0.21 (0 to 6); both lose under
the heavy rewording. Wrong: GainLoRA learns only 73.86 (we said 79 to 84) and ends at 73.70 (we
said 75 to 83), changes 9 answers (we said 15 to 80, so fewer), and it does not end above our
10-epoch score, which we had thought more likely than not.

**What it settles.** Run D alone could not say why the published methods lost their answers at
100 epochs. Now it can be said for this setting: with the definition in the input they forget
0.2 and 3.5 points; without it, 25.4 and 66.3. Our rules, which never read the definition,
forget 1.0 and 0.0 points without it. One seed, five tasks, the smaller model.

**Hours.** This run used about 3.7 GPU hours. With runs F and U2 the 9 hours left
at 12:50 UTC are used up or nearly so. No further run is started.

### 2026-11-18 (17:46 UTC) — the long version goes back to the order method, results, introduction

The order asked for was "methodology first, results, then the introduction, then the
rest". The long version (`paper_certified/main.tex`) is now: abstract, 1 Method, 2 Results,
3 Introduction, 4 Related work, 5 Discussion, 6 Limitations, 7 Conclusion, appendix. Only the
order of the files changed, and the first paragraph of the method section now introduces the
names (FK-LoRA, FinderLoRA, KeeperLoRA, ClaimLoRA), because it is now the first text after the
abstract. The 9-page version keeps the usual order (introduction first), as conferences expect;
the same order can be used there too.

### 2026-11-18 (18:41 UTC) — a shortened copy of the long version

The aim: remove only what is not needed; remove no experiment, no research and no useful
information; review the document at the end and put back anything that was lost. There is no
page target.

**What was made.** `paper_certified/main_compact.tex` (it reads the new files
`paper_certified/k_*.tex`) builds `main_compact.pdf`: 66 pages against 83 for `main.pdf`
(main text to page 39, references to page 41, appendix pages 42 to 66). The long version
`main.tex` and its `sec_*.tex` files are not touched. Same order as the long version (method,
results, introduction, related work, discussion, limitations, conclusion).

**What was removed.** Sentences that said the same thing twice; the day-by-day story of how a
result was reached where the result itself is stated; numbers repeated in the text that the
table next to it already prints; side statements moved from the main text to the appendix; and
empty space (wider text block, tighter spacing around tables and figures).

**What was not removed.** Every table (63), every figure (9), every citation (56), every
theorem, lemma and proposition with its proof, every guess written before a test and whether it
held, every limitation.

**How "nothing useful is lost" was checked.** (1) A script (`kcheck.py`, kept outside the repo)
compares the old and the new text: every number command of the long version is either in the
new text or its value is printed in a table the new text includes; every table, figure and
label is there. (2) Two independent readers compared old and new side by side and listed what
was lost or made weaker; about 60 points were put back (for example a guess wrongly marked as
"held", a dropped caveat, two "by chance" remarks, the order of the method names in the
abstract). (3) Own reading of the abstract, the main results and the conclusion; two sentences
made clearer. Build: 0 overfull lines, no undefined reference. Word copy: `main_compact.docx` (the three appendix files are now read with a plain
`\input`, because the converter does not follow `\IfFileExists`); it passes the validator.

**Open.** The abstract of the shortened copy has 461 words (2,644 characters); arXiv's form
allows 1,920 characters, so the arXiv form will need a shorter abstract than the PDF.

### 2026-11-18 (19:13 UTC) — new figures, and one look for all figures

Figures from other papers were shown (a method sketch with one panel per approach, heat
maps of tasks against stages, confusion matrices) and ours were said to look generic. Only the pictures
changed: no number, no table and no result. `audit_results.py` still verifies all 1,599 numbers
and the files under `paper_certified/tables/` are the same as before.

**One look.** `paper_certified/figstyle.py`: one colour per method in every figure (ours cool,
published warm, simple baselines grey; checked for colour-blind separation), one font, one ramp
for magnitudes. All scripts that draw figures use it.

**Six new figures** (in the long and the shortened version in the text, in the 9-page version
in a new appendix section "The method and the main result in pictures"):
1. `overview` (drawn in TikZ, source in `figures_src/`): one shared adapter / branches added
   together (InfLoRA, GainLoRA) / FK-LoRA with frozen adapters and a switch. The picture follows
   Eq. (2): route 2 means adapters 1 and 2 on.
2. `switch` (TikZ): the three rules FinderLoRA, KeeperLoRA (with the exact condition of the
   handicap: margin above the (nu+1)-th largest held-out margin), ClaimLoRA.
3. `answer_grid`: the 400 test answers of the four older tasks as squares, changed or kept
   (five tasks, no definitions, 100 epochs): 382, 178, 6 and 1 changed.
4. `stage_heatmap`: score of every task after every stage, same runs (re-test scores).
5. `learn_keep`: score right after learning against score at the end, with and without
   definitions, 10 and 100 epochs; the diagonal is "nothing lost".
6. `route_matrix`: fifteen tasks without definitions, where the test inputs are sent
   (nearest centre 92.6 % to their own task, FinderLoRA 99.2 %).
Figures 3 to 6 are drawn by `paper_certified/make_figures.py` from the run archives; the
script stops if a plotted number differs from the audited one.

**The nine older figures** were redrawn in the same look (same data): `learn_forget` is now
open and filled circles joined by a line instead of hatched bars (captions and text changed to
match), the others have the shared colours and fewer legends.

**Pages.** Long version 86 (was 83), shortened version 69 (was 66), 9-page version 86 in all,
main text still ends on page 9.

### 2026-11-18 (19:17 UTC) — figure colours: no blue, no purple

Blue and purple were to be avoided in the figures. `paper_certified/figstyle.py` now uses
green (FinderLoRA), charcoal (KeeperLoRA), gold (ClaimLoRA), vermilion (InfLoRA), crimson
(GainLoRA) and greys for the simple baselines; heat maps run from near white to dark green;
"answer kept" is pale grey and "answer changed" vermilion. The four colours were checked together
for colour-blind separation of every pair. The two TikZ figures use the same colours. All
fifteen figures were redrawn; data, numbers and tables are unchanged (1,599 numbers verified,
no file under `tables/` differs). Pages unchanged: 86, 69, and the 9-page version ends on page 9.

### 2026-11-18 (19:29 UTC) — a release repository, Shiv9993/FK-LoRA

The work is released in the repository `github.com/Shiv9993/FK-LoRA`. It holds the folders
`paper_certified`, `silent_adapters`, `certified_switch` and `latent_lora`, `requirements.txt`, a
short README and `RESULTS_LOG.md` (the dated log of the runs). Earlier lines of work that are
not part of this paper are not in it. Build files of LaTeX and Python caches are left out.

**In the paper.** Appendix "Reproducibility" (long and shortened version) gives the address of
the repository, the commit of the code and the results, the plan (`silent_adapters/PLAN.md`) and
the log (`RESULTS_LOG.md`), and says that the repository is a copy of the one in which the work
was done and that the dates are those written in the files. The release repository has no
history of its own, so the dates of the plan entries cannot be checked there from the commits.

**Also.** The CPU study scripts no longer hold a path of one machine: their scratch folder is the
setting `FK_LORA_SCRATCH`.

### 2026-11-20 (08:01 UTC) — corrections after an outside reading of the paper

An outside reading of the shortened version found sentences that did not match our own results.
All were corrected in the three versions; no number, table or result changed (the audit still
verifies all 1,599 numbers).

- The run with definitions at 100 epochs exists (Table "fulldefinitions"); a sentence still called
  it missing.
- At 100 epochs the published methods learn the tasks nearly as well as plain adapters with the
  same epochs (78.37 and 76.29 against 80.78); the text had compared with the 10-epoch value 77.70.
- "With their definitions they hold their answers" is true for GainLoRA (9 of 400 changed), much
  less for InfLoRA (103 of 400); the text now gives both numbers and says the instruction held most
  of their answers.
- The ClaimLoRA numbers of the three-seed wording table and of the three-seed table of routes from
  the example come from different training runs (seed 42 was trained twice; seeds 43 and 44 again
  for run U2): 79.7 and 74.1 against 79.8 and 74.3. The text and the caption now say so.
- The Italian pivot: the second fallback rule was written while the first test was running, before
  any Italian number had been read, and the Italian numbers of both rules were read together. The
  text had said "used once" and named "the person who wrote the rule".
- Smaller points: the recogniser's features are standardised with 2,000 instances of other SuperNI
  tasks, not with the unrelated inputs; routes depend only on the tokenizer and embedding table
  for ClaimLoRA, on the frozen encoder for FinderLoRA and KeeperLoRA; Lemma 1 gives "at most as
  many answers as routes", the equality was observed; the opening sentence of the results said
  all of them carry no instruction; a broken sentence on fifteen tasks (now 661 and 215 of 1,097);
  "77.5 at 10 epochs" is seed 42; the language setting is that of the GainLoRA paper (InfLoRA's
  own paper is on images); "c" in the audit column is H of the audit proposition; the cost of the
  zones on the Spanish pivot is 8.7 points (the text said 4.6 to 5.8).
- Style: dates, code names of rules, "developed in one day of experiments", and "the method" for
  ClaimLoRA in the appendix are gone; the notation table has the symbols of FinderLoRA and
  KeeperLoRA.
- Checked and found correct: the German-pivot coverage at alpha 0.2 (80.4% against 80% promised;
  the text says the ruler misses on German only at alpha 0.1); the two values 48.71 in the
  Latent-LoRA tables (48.714 and 48.709, a coincidence); every 2025 and 2026 reference (arXiv
  identifiers, titles, authors, the ACL Anthology page and the ICML page).
- The Word copy of the shortened version now has the 6.5-inch text width of its PDF (71 pages
  instead of 77), and the Word converter no longer splits words inside narrow table columns.
- In the 9-page version four figures appeared twice in the appendix; one copy is kept.
