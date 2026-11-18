# Dated log of results

This file is the log of every run, written when the run was read. It is copied from the repository
in which the work was done, where it forms sections 4I to 4L of a longer log; the section letters
are kept because the plan (`silent_adapters/PLAN.md`) and other files refer to them as
"the README, section 4L". The wording was tidied for this release; the numbers, the verdicts and
the dates are as written at the time. Names in older entries: the recogniser is FinderLoRA, the
handicap is KeeperLoRA, the certified switch is ClaimLoRA.

### 4I. Hypothesis 5, stage 0: the published methods in a reduced setting (`silent_adapters/stage0_colab.ipynb`)

**Run on a Colab T4, 2026-11-16/17.** Official GainLoRA code (commit `e9d0d70`), SuperNI order 1, 15 tasks, **t5-base, 10 epochs** (the papers use t5-large, 100 epochs), seed 42, effective batch 32. CuPy did not load on Colab, so the SVD step used the PyTorch stand-in. Scores are Rouge-L on the official test sets (16 to 100 examples per task, so single-task numbers are noisy). These numbers must not be compared with the papers' tables.

**InfLoRA (finished, 3.0 GPU-hours): AP 25.85, FT 5.66.**

| Task (in the order learned) | Right after learning | At the end | Change |
|---|---|---|---|
| samsum summary | 23.28 | 9.07 | −14.21 |
| sst2 polarity | 61.00 | 50.00 | −11.00 |
| xsum summarization | 17.94 | 17.07 | −0.87 |
| outcome extraction | 51.16 | 51.16 | 0.00 |
| quoref answer generation | 33.23 | 24.30 | −8.93 |
| relation extraction | 87.06 | 68.23 | −18.83 |
| multi_woz utterance | 3.81 | 2.56 | −1.26 |
| personachat next line | 12.57 | 9.06 | −3.51 |
| commonsenseqa | 15.00 | 20.00 | +5.00 |
| diplomacy text | 3.86 | 4.96 | +1.09 |
| glucose cause event | 42.60 | 31.30 | −11.30 |
| reddit_tifu summarization | 13.86 | 13.49 | −0.37 |
| sciq answer generation | 11.73 | 10.57 | −1.17 |
| sentiment140 | 37.00 | 37.00 | 0.00 |
| emotion classification | 39.00 | 39.00 | 0.00 |

What this shows so far:

1. **The official code runs on a free T4** and the baseline behaves as expected: it learns, then loses part of it.
2. **Forgetting costs InfLoRA 4.36 points.** The average score right after learning is 30.21; at the end it is 25.85. So in this setting a method that forgot nothing, and learned equally well, could gain at most about 4.4 AP. That is the whole room there is to win.
3. **The setting is under-trained.** With 10 epochs several tasks are barely learned (sst2 is a two-way choice and reaches only 61, then 50; several generation tasks are below 15). Most of the gap to the papers is in learning, not forgetting. Whether 10 epochs is enough to compare methods is decided when GainLoRA finishes: it must forget clearly less than InfLoRA, as in its paper; if it does not, stage 1 needs more epochs.

**GainLoRA:** stopped after 9 of 15 tasks (it ran out of GPU memory on task 3 at batch 16 and continued at batch 8). Over those 9 tasks: AP 31.15, FT 1.30. At the same point it forgets less than InfLoRA but learns less well (relation extraction 71.6 against 87.1, sst2 52 against 61). It can be continued from task 10 by running the notebook again.

### 4J. Hypothesis 5: short first test of silent adapters on a real model (`silent_adapters/quick_test_colab.ipynb`)

**Run on a Colab T4, 2026-11-17, stopped before it finished.** Exploratory (see the note at the end of `silent_adapters/PLAN.md`): t5-base, 10 epochs, seed 42, five short SuperNI tasks in this order: relation extraction, sst2, glucose, sentiment140, emotion. Rouge-L on about 100 test examples per task. `plain` and `silent` (ours) finished all five tasks (the last `silent` task in a later session); InfLoRA and GainLoRA were **not run** on these five tasks when this was written.

Score of each earlier task after each new task was learned:

| After task | `plain` | `silent` (ours) |
|---|---|---|
| 1 | 99.2 | 99.3 |
| 2 | 58.1, 86.0 | 99.3, 85.0 |
| 3 | 74.8, 30.7, 61.3 | 85.3, 60.7, 62.5 |
| 4 | 31.9, 39.0, 11.7, 74.0 | 57.8, 74.0, 29.6, 71.0 |
| 5 | 14.5, 18.0, 6.8, 55.0, 73.0 | 38.7, 25.0, 12.9, 62.0, 77.0 |

After four tasks the average is 39.2 for `plain` and 58.1 for `silent`. At the end of five (the fifth `silent` task was run later the same day): `plain` learned 78.70, AP 33.46, FT 56.55; `silent` learned 78.96, **AP 43.12, FT 44.80**.

What this shows:

1. **Plain adapters learn these tasks well and forget almost all of it** (99, 86, 61 fall to 14.5, 18.0, 6.8). The low scores of InfLoRA in 4I (87, 61, 43 on the same three tasks) were mostly a limit of that method, not of 10 epochs.
2. **Silence training costs no learning here**: each task right after learning scores the same as `plain` (99.3 / 85.0 / 62.5 / 71.0 against 99.2 / 86.0 / 61.3 / 74.0).
3. **Silence training reduces forgetting only a little, and the protection wears off as tasks are added.** After task 2 the first task was fully kept (99.3 against 58.1). After five tasks the first three are at 38.7, 25.0 and 12.9, against 14.5, 18.0 and 6.8 for `plain`: ahead on every old task, but most of what was learned is gone. **As built, silence training is not a solution** (AP 43.12 against 33.46; the exact switch of 4K gives 78.88).
4. Task 2 (sst2) went up from 60.7 to 74.0 when task 4 (also sentiment) was learned.

**Not shown:** any comparison with InfLoRA or GainLoRA on these five tasks; more than one seed. A possible cause of the leak, not tested: the stand-in inputs use only 10 different task definitions, so the adapter may learn to be silent on those and not in general.

### 4K. Hypothesis 5: the exact switch (`locked`) on five tasks (`silent_adapters/quick_test_colab.ipynb`)

**Run on a Colab T4, 2026-11-17.** Same five tasks and setting as 4J (t5-base, 10 epochs, seed 42). `locked` trains exactly like `plain`; when answering, the task is recognised from the task definition at the start of the input and every adapter added after that task is switched off (design in `silent_adapters/PLAN.md`).

| Task | `plain`: right after → end | `locked`: right after → end |
|---|---|---|
| relation extraction | 99.2 → 14.5 | 99.2 → 99.2 |
| sst2 | 86.0 → 18.0 | 86.0 → 86.0 |
| glucose | 61.3 → 6.8 | 61.3 → 62.2 |
| sentiment140 | 74.0 → 55.0 | 74.0 → 74.0 |
| emotion | 73.0 → 73.0 | 73.0 → 73.0 |
| **learned / AP / FT** | 78.70 / 33.46 / 56.55 | 78.70 / **78.88** / **0.00** |

What this shows:

1. **With the switch, nothing is forgotten: AP 78.88 against 33.46 for `plain`, FT 0.00.** Scores right after learning are the same as `plain`'s, as they must be (same training).
2. **All test inputs were matched to their task** (0 unmatched of 2,200 seen after the last task).
3. **The answers themselves did not change at all.** An answer-by-answer comparison (right after each task was learned, against after the last task) found 0 changed answers out of 400 across the first four tasks. The glucose score still moved from 61.3 to 62.2, and the cause is the benchmark, not the model: 23 of its 100 test examples have more than one reference answer, and the official data loader picks one of them at random every time the data is loaded (`random.randint` in `src/cl_dataset.py`), so the same answers are scored against different references. The other four tasks have a single reference each.

What it is not: the switch uses the task definition to identify the task. That is a known upper bound (the model is told, in effect, which task it is doing), not a new method, and it gives no protection when the same task is asked with a differently worded definition. InfLoRA and GainLoRA have still not been run on these five tasks.


### 4L. Hypothesis 6: the certified switch (`silent_adapters/quick_test_colab.ipynb`, `routing_check.py`)

The exact switch of 4K forgets nothing but needs the exact wording of the task definition. The certified switch decides from the frozen word-vectors of the input (the base model's embedding table; nothing is trained). Each task stores a *signature* (the average word-vector of its fixed opening) with a radius, and a *centre* (the average over its whole inputs). The ball around the signature is the task's protected zone; it is never changed. An input inside a zone is given to the oldest task whose zone contains it; an input outside every zone is given to the task with the nearest centre. Training is exactly `plain`; when answering, adapters added after the chosen task are switched off, as in `locked`. Full rule, proof and the rules for reading the GPU run: `silent_adapters/PLAN.md`.

#### GPU run, Colab T4, 2026-11-17 (five tasks, t5-base, 10 epochs, seed 42)

One training run of `certified` (38 minutes). Official scores: learned 78.70, AP 78.88, FT 0.00, the same as `locked`. After every task, each test input was answered again under four rules with the same trained adapters, with the original task definition and with three rewordings of it (written by machine). Scores in this re-test use every reference answer, so they do not have the evaluation noise of 4K. Raw output: `silent_adapters/results/2026-11-17_certified_five_tasks_colab_T4.txt`.

| Wording of the definition | Rule | learned | AP | FT | answers changed (tasks 1–4) | given to own task | inside a protected zone |
|---|---|---|---|---|---|---|---|
| original | `certified` | 79.33 | **79.33** | **0.00** | **0 of 400** | 100% | 500 of 500 |
| original | exact key (`locked`) | 79.33 | 79.33 | 0.00 | 0 of 400 | 100% | |
| original | all adapters on (`plain`) | 79.33 | 33.54 | 57.23 | 331 of 400 | | |
| light rewording | `certified` | 79.15 | **79.15** | **0.00** | **0 of 400** | 100% | 499 of 500 |
| light rewording | exact key | 79.15 | 32.83 | 57.90 | 332 of 400 | | |
| medium rewording | `certified` | 77.61 | **77.61** | **0.00** | **0 of 400** | 100% | 100 of 500 |
| medium rewording | exact key | 77.61 | 32.72 | 56.11 | 334 of 400 | | |
| heavy rewording | `certified` | 75.32 | **75.32** | **0.00** | **0 of 400** | 97.4% | 100 of 500 |
| heavy rewording | exact key | 76.72 | 32.23 | 55.61 | 335 of 400 | | |

With reworded definitions the exact key matched no input, so its rows are identical to "all adapters on". The rule "nearest centre only, no zones" gave exactly the numbers of `certified` in all four wordings. The re-test reproduced the official test on every input (100 of 100 up to 500 of 500).

What this shows, read against the rules fixed before the run (`PLAN.md`):

1. **The two things that had to hold did hold.** With the original wording no answer changed (0 of 400), every input was inside a protected zone, and the answers equal those of the exact key. No answer changed for any input inside a zone in any wording.
2. **The exact switch of 4K depends on the exact text.** At the lightest rewording it protects nothing: forgetting 57.9, the same as plain adapters, 332 of 400 answers changed.
3. **The certified switch kept every answer under all three rewordings** (0 of 400 changed each time, FT 0.00), and 97.4% to 100% of inputs reached their own task. Both expectations written beforehand (forgetting of at most 5 points, at least 95% to the own task) were met.
4. **The adapters themselves tolerate rewording.** With the right adapters the score right after learning falls only from 79.33 to 79.15, 77.61 and 76.72. Under heavy rewording `certified` has 75.32, because 13 of 500 inputs were given to a wrong task from the start. That is a loss in learning, not forgetting, and those inputs were outside the zones.
5. **The guarantee covers the original wording and light rewording** (500 and 499 of 500 inputs inside a zone). With medium and heavy rewording only 100 of 500 inputs are inside a zone; the other 400 kept their answers too, but nothing guaranteed that.
6. **The zones added no score here.** The nearest-centre rule alone did equally well, as expected beforehand. What the zones add is the proof and the flag on each answer.

What it does not show: one seed, five tasks, about 100 test inputs per task. The rewordings were written by machine and share words with the originals; a rewording with no word in common may fail. InfLoRA and GainLoRA have not been run on these five tasks. The radius factor was chosen on the CPU routes of these same test inputs. A later task made deliberately close to an old one, the case the zones are for, has not been tried. The line "locked: every task scores exactly the same ...: False" in the raw output is the evaluation noise of 4K (glucose 61.3 → 62.2 with a randomly picked reference), not a changed answer.

#### After the GPU run: a CPU check on the 15-task sequence changed the rule for inputs outside the zones (2026-11-17)

The GPU run above used the first version: an input outside every zone went to the nearest centre of the *whole* input, and every token counted equally. Its routes under rewording had been checked on the five short tasks only. The same check on the 15-task sequence (routes only, no model answers) showed that this fallback fails when tasks with long inputs are present: only 70% (medium) and 61% (heavy) of the reworded inputs reached their own task, and 72 and 100 of 500 routes changed as later tasks were added. Inputs inside a zone were not affected.

**Final rule (`certified2` in the notebooks).** Zones, "oldest zone wins" and the proof are unchanged. (1) An input outside every zone goes to the nearest *zone* (distance to the signature divided by the zone's radius); the whole-input centre is no longer used. (2) In every average a common word counts less (weight a / (a + frequency in unrelated text), a = 0.001, the usual weighting for averaged word-vectors; not tuned). Routes with the final rule (`routing_check.py`, CPU):

| Sequence, wording | inside own zone | given to own task: final rule | old fallback (`centre`) | routes changed later: final rule / old |
|---|---|---|---|---|
| five tasks, original / light | 100% / 100% | 100% / 100% | 100% / 100% | 0 / 0 |
| five tasks, medium | 20% | 100% | 99.2% | 0 / 0 |
| five tasks, heavy | 19.6% | 93.8% | 72.2% | 0 / 58 |
| fifteen tasks, original (1,197 inputs) | 100% | 100% | 99.5% | 0 / 2 |
| fifteen tasks, light / medium (500 inputs each) | 100% / 20% | 100% / 100% | 100% / 85.2% | 0 / 29 |
| fifteen tasks, heavy (500 inputs) | 19.6% | 93.8% | 59.2% | 0 / 100 |

No input was inside the zone of another task. What this means: the guarantee and everything about the original wording are unchanged; the five-task GPU numbers above belong to the first version and are being repeated with the final rule; on five tasks the final rule is expected to be slightly worse under heavy rewording (93.8% against 97.4% to the right task) and it is far better on fifteen. The radius factor and both changes were chosen while looking at routes of the same test inputs and the same machine-written rewordings, never at an answer of the model. The rule is frozen from here (`silent_adapters/PLAN.md`); a claim about rewording needs rewordings written by someone else.

#### Second GPU run, Kaggle T4, 2026-11-17: the final rule and the two published methods on the five tasks

Same five tasks, t5-base, 10 epochs, seed 42. Trained in this run: `certified2` (final rule), and InfLoRA and GainLoRA with the official code unchanged (`SILENT_BASELINE=1` only repeats their test with the reworded definitions). 36 minutes per method. Raw log: `silent_adapters/results/2026-11-17_certified2_and_published_five_tasks_kaggle_T4.txt`.

Official scores (original wording, one randomly picked reference answer, as the official code does):

| Method | learned | AP ↑ | FT ↓ | relation extr. | sst2 | glucose | sentiment140 | emotion |
|---|---|---|---|---|---|---|---|---|
| `plain` (4J) | 78.70 | 33.46 | 56.55 | 99.2 → 14.5 | 86.0 → 18.0 | 61.3 → 6.8 | 74.0 → 55.0 | 73.0 → 73.0 |
| `locked`, exact switch (4K) | 78.70 | 78.88 | 0.00 | 99.2 → 99.2 | 86.0 → 86.0 | 61.3 → 62.2 | 74.0 → 74.0 | 73.0 → 73.0 |
| **`certified2`** | 78.70 | **78.88** | **0.00** | 99.2 → 99.2 | 86.0 → 86.0 | 61.3 → 62.2 | 74.0 → 74.0 | 73.0 → 73.0 |
| InfLoRA (published) | 56.69 | 58.37 | −1.10 | 99.0 → 98.2 | 55.0 → 61.0 | 50.4 → 46.7 | 40.0 → 47.0 | 39.0 → 39.0 |
| GainLoRA (published) | 60.30 | 61.05 | −0.69 | 98.9 → 98.9 | 54.0 → 55.0 | 50.7 → 49.4 | 57.0 → 61.0 | 41.0 → 41.0 |

(Each task: score right after it was learned → score at the end.)

Re-test with every reference answer, original and reworded definitions. Each cell: AP / FT / answers changed out of 400:

| Wording | `certified2` | InfLoRA | GainLoRA | all adapters on (`plain`) |
|---|---|---|---|---|
| original | **79.33** / 0.00 / **0** | 58.89 / −1.41 / 98 | 61.31 / −0.32 / 17 | 33.54 / 57.23 / 331 |
| light | **79.15** / 0.00 / **0** | 59.87 / −3.09 / 85 | 59.49 / 0.27 / 15 | 32.83 / 57.90 / 332 |
| medium | **77.61** / 0.00 / **0** | 60.45 / 1.37 / 91 | 58.61 / 0.31 / 32 | 32.72 / 56.11 / 334 |
| heavy | **72.12** / 0.00 / **0** | 56.37 / 1.31 / 151 | 52.01 / 0.15 / 49 | 32.23 / 55.61 / 335 |

For `certified2`: 500, 500, 100 and 98 of 500 inputs were inside a protected zone; 100%, 100%, 100% and 93.8% went to their own task; the exact key again equals "all adapters on" as soon as the definition is reworded; no training input was inside an older zone. The first version's fallback (`centre`, with the new word weights) fell to AP 58.91 with 58 answers changed under heavy rewording.

What this shows:

1. **The rules fixed for `certified2` held** (`PLAN.md`): 0 answers changed and every input in its own zone with the original wording; no answer changed inside a zone in any wording; forgetting 0.00 under all rewordings; 100%, 100%, 93.8% of reworded inputs to the right task, as the CPU check had predicted to the digit.
2. **Final rule against first version on five tasks:** the same for the original, light and medium wording (79.33, 79.15, 77.61) and lower under heavy rewording (72.12 against 75.32), as expected when the rule was changed. The reason for the change is the 15-task sequence, not these five tasks.
3. **In this reduced setting `certified2` ends 18 to 21 points above both published methods** (78.88 against 61.05 and 58.37), and 16 to 20 points above them under heavy rewording (72.12 against 56.37 and 52.01).
4. **The gap is in learning, not in forgetting.** InfLoRA and GainLoRA do not forget here either (FT −1.10 and −0.69; old tasks even improve a little when a related task is learned). They learn the three classification tasks poorly in ten epochs: 55 and 54 on sst2, where 50 is a constant answer, 40 and 57 on sentiment140, 39 and 41 on emotion, against 86, 74 and 73 for plain adapters. Section 4I already showed that this setting under-trains both. What the run supports: with this budget, the certified switch keeps all the learning of plain adapters and forgets nothing, while the two published methods pay for their protection with learning. What it does not support: that the published methods forget, or any statement about the published setting (T5-large, 100 epochs).
5. **At the level of single answers `certified2` is the only method that changes nothing:** 0 of 400 in every wording, against 15 to 49 for GainLoRA and 85 to 151 for InfLoRA. Some of their changes are improvements. `certified2` cannot improve an old task either: keeping answers exactly also rules out help from later tasks.
6. **Rewording costs every method something.** From the original to the heavy wording `certified2` loses 7.2 points (4.6 of them because 6% of inputs go to a wrong task), GainLoRA 9.3 and InfLoRA 2.5. The wrongly routed inputs are all of one kind (read from the saved answers): 31 of the 100 sentiment140 inputs, whose heavy rewording is one short sentence about tweets being positive or negative, are given to the other sentiment task, sst2. Every other task keeps 100% of its inputs in every wording.
7. **The re-test reproduced the official test** on every input for `certified2` and InfLoRA, and on 1,494 of 1,500 for GainLoRA. The re-test used batches of 32 and the official test batches of 16; the re-test now uses the official batch size.
8. **Colab and Kaggle gave the same numbers:** `certified2` on Kaggle has the same scores right after learning as `plain`, `locked` and the first `certified` on Colab (78.70), and the same re-test numbers for the original, light and medium wording.

What it does not show: one seed; five tasks that we picked; about 100 test inputs per task; rewordings written by machine; only the reduced setting; the plain adapters train both LoRA matrices while InfLoRA trains one, so they also have more trainable numbers.

#### Third GPU run, Kaggle T4 x2, 2026-11-17: three seeds of the five-task comparison

Seeds 42, 43 and 44, two runs at a time on the two GPUs, about 3 hours in all; fetched through Kaggle's API (`silent_adapters/results/2026-11-17_kaggle_five_tasks_seeds42-44/`). Re-test with every reference answer; mean over the three seeds, lowest and highest seed in brackets; counts are added up over the seeds.

| Method | Learned | AP | FT | Answers changed |
|---|---|---|---|---|
| certified switch (`certified2`) | 79.74 (79.2 to 80.8) | **79.74** (79.2 to 80.8) | **0.00** | **0 of 1,200** |
| plain adapters (same adapters, all on) | 79.74 | 36.08 (32.3 to 39.5) | 54.58 | 994 of 1,200 |
| InfLoRA (official code) | 60.02 (56.6 to 63.3) | 62.93 (59.0 to 65.4) | −3.14 (−5.3 to −1.7) | 370 of 1,200 |
| GainLoRA (official code) | 61.42 (59.1 to 64.4) | 61.96 (60.9 to 63.9) | −0.51 (−2.6 to 0.7) | 65 of 1,200 |

With reworded instructions (AP, light / medium / heavy; answers changed of 1,200): certified switch 79.59 / 78.98 / 74.07 with 0 / 0 / 0 changed; InfLoRA 64.32 / 57.77 / 54.33 with 296 / 320 / 470; GainLoRA 64.65 / 58.75 / 51.63 with 58 / 122 / 178; exact key 36.4 / 36.0 / 35.2 (equal to plain); nearest centre under heavy rewording 59.38 with 171 changed. Scores of the official script: 79.28, 62.49 and 61.55. The re-test gave the same answers as the official test on 4,500 of 4,500 inputs for each method. No training input lay inside an older zone (0 of 12,000). Every rule written before the runs held in every seed (`silent_adapters/PLAN.md`).

**What three seeds add.** The ranges do not overlap, so the picture of the single seed is not an accident of that seed. The reading stays the same and must stay attached to the numbers: **the published methods do not forget here either** (their forgetting is negative on average, old tasks even improve a little); **they learn less within ten epochs** (60 right after learning against 79.7; SST-2, a task with two labels, stands at 55 and 58 for them and at 86 for plain adapters). The certified switch keeps the learning of plain adapters and changes no answer. Nothing here says how the published methods do with their own budget (T5-large, 100 epochs).

**The same seed twice is not the same run.** Seed 42 was run twice (in the afternoon and in this run). The first task was reproduced answer for answer, the later tasks were not: 427 of 500 final answers of the certified switch are the same in the two runs, and the average score is 79.33 in one and 79.23 in the other. Both runs used the same batch size. So a few tenths of a point between two runs mean nothing on these GPUs. The count that the method is about did not vary: 0 answers changed in both.

#### Fourth GPU run, Kaggle T4 x2: the fifteen tasks (2026-11-17, night)

Official order 1 of SuperNI, 15 tasks, t5-base, 10 epochs, seed 42; two methods at a time, 3.1 to 3.5 hours per method. Raw files: `silent_adapters/results/2026-11-17_kaggle_fifteen_tasks_seed42/` (`log.txt` has the notebook's own tables). Re-test with every reference answer.

| Method | Learned | AP | FT | Answers changed |
|---|---|---|---|---|
| certified switch (`certified2`) | 42.77 | **42.77** | **0.00** | **0 of 1,097** |
| plain adapters (same adapters, all on) | 42.77 | 14.88 | 30.67 | 973 of 1,097 |
| InfLoRA (official code) | 30.46 | 26.13 | 5.69 | 661 of 1,097 |
| GainLoRA (official code) | 29.08 | 28.62 | 1.09 | 215 of 1,097 |

Certified switch: all 1,197 test inputs lie inside their own zone and 0 of 9,391 training inputs were blocked. The nearest zone and the exact key give the same numbers with the original instructions. This is the outcome that the theorem guarantees; simpler rules reach the same score. Scores of the official script: certified switch 42.28 / 42.26 / FT 0.19, InfLoRA 30.21 / 25.85 / 5.66, GainLoRA 28.73 / 28.21 / 1.11. The FT of 0.19 is not forgetting: the official script scores each answer against one randomly picked reference. The re-test shows that all 8,874 answers are identical to those of the official test and that none of the 1,097 answers changed. InfLoRA's official scores are exactly those of the earlier run of the same method on Google Colab, so the two platforms agree for this method.

**Reworded instructions** (the five tasks that have rewordings, inside the 15-task sequence; light / medium / heavy):

| | AP | Answers changed (of 400) | Certified inputs (of 500) | Right task |
|---|---|---|---|---|
| certified switch | 80.71 / 80.20 / 76.00 | 0 / 0 / 0 | 500 / 100 / 98 | 100% / 100% / 93.8% |
| nearest centre | 80.7 / 72.0 / 57.5 | 0 / 29 / 100 | | 100% / 85.2% / 59.2% |
| plain adapters and exact key | 39.6 / 40.4 / 36.1 | 316 / 313 / 310 | | |
| InfLoRA | 51.6 / 44.7 / 32.6 | 257 / 272 / 270 | | |
| GainLoRA | 51.1 / 47.1 / 37.0 | 69 / 85 / 109 | | |

Forgetting on these five tasks: InfLoRA 10.5 (light) and 12.6 (heavy); GainLoRA 0.9 and 3.4. Every rule written before the run held (`silent_adapters/PLAN.md`, section "result of the 15-task run, read against the rules").

**Reading.**

1. The certified switch changes no answer and keeps the learning of plain adapters, with the original and with the reworded instructions. Simpler rules reach the same score; the switch adds the proof and the certificate.
2. On fifteen tasks, unlike on five, InfLoRA does forget (5.7 points) and GainLoRA a little (1.1). Both also learned less right after each task (30.5 and 29.1 against 42.8). We do not say why they learned less. The caveats of the five-task runs stay: a tenth of the published budget, a smaller model, and one step of their code computed with a substitute.
3. The nearest centre fails under reworded instructions on this sequence, as the CPU check had said it would (right task 85.2% and 59.2% under medium and heavy rewording).

**What it does not show:** one seed; 15 tasks of one benchmark order, about 16 to 100 test inputs per task; rewordings written by machine, for five of the tasks only; only the reduced setting (a tenth of the published budget, t5-base); nothing about the published setting. The two published methods learned less in our set-up, so the gap to them is mostly a gap in learning, with a forgetting part (5.7 and 1.1 points) that we can see only now.

#### Fifth GPU run, Kaggle T4 x2: our re-implementation of Latent-LoRA next to the certified switch (2026-11-17, night)

**Not pre-registered:** no rule for this run was written in `silent_adapters/PLAN.md` before it was started. Five tasks, t5-base, 800 training and 100 test inputs per task, 10 epochs, seeds 42 to 44, both methods in the same stand-alone code (`latent_lora/run_five.py`). The certified switch is the code of `certified_switch/` (rank 4, batch 32). Latent-LoRA was written by us from the paper's text (there is no public code); `latent_lora/README.md` lists the 20 details the paper does not give and the choice made for each. Differences from the paper: 10 epochs instead of 30, t5-base instead of T5-large, five tasks, our own Rouge-L scorer. Raw files: `silent_adapters/results/2026-11-17_kaggle_latent_lora_five_tasks/` (`log.txt`, `small_files.zip`); tables and numbers for the paper from `paper_certified/latent_results.py`. Original instructions, mean over three seeds:

| Method | Learned | AP | FT | Answers changed | Right task |
|---|---|---|---|---|---|
| certified switch (stand-alone code) | 76.49 | **76.49** | **0.00** | **0 of 1,200** | 100.0% |
| same adapters, all on | 76.49 | 36.04 | 50.56 | 970 of 1,200 | no router |
| Latent-LoRA, soft rule (our re-implementation) | 42.55 | 42.55 | 0.00 | 0 of 1,200 | 100.0% |
| Latent-LoRA, sum rule (no router) | 48.71 | 47.66 | 1.31 | 429 of 1,200 | no router |

With reworded instructions (light / medium / heavy): certified switch AP 75.87 / 75.91 / 71.74, 0 of 1,200 changed in each, right task 100% / 100% / 94.0%; Latent-LoRA soft AP 48.71 / 32.53 / 27.75, 0 of 1,200 changed in each, right task 100% in each.

**Reading.**

1. The re-implemented Latent-LoRA learned the first task and almost nothing after it. Scores right after learning (seed 42, soft rule, original instructions): relation 90.8, SST-2 56.0, Glucose 2.2, Sentiment140 37.0, emotion 27.0; the other seeds are alike. Training loss at the end of the Glucose task about 2.0 against about 0.6 for the certified switch. Learned = AP = 42.55 against 76.49. We do not know whether this comes from the reduced budget (10 of 30 epochs, smaller model) or from a mistake in our re-implementation; without public code we cannot check. **So the run says nothing about how well Latent-LoRA learns or retains, and we make no such claim.** One sign that its adapters were barely trained: with the light rewording its average is higher (48.71) than with the original instructions (42.55), because the Sentiment140 score moves from 37 to 65 to 69.
2. Its router gave every test input to the right task in all four wordings (right task 1.000; the certified switch 1.000 / 1.000 / 1.000 / 0.940), and with the soft rule none of 1,200 answers changed in any wording. On these five tasks its router is as stable as ours in practice and recognises the heavy rewording better. What the certified switch adds is the proof and the certificate on each answer, not better recognition.
3. With its sum rule (no router): AP 47.66, FT 1.31, 429 of 1,200 answers changed.
4. The stand-alone code of the certified switch reproduces the main result on t5-base in a second, independent code base: AP 76.49, FT 0.00, 0 of 1,200 answers changed in each of the four wordings (AP 75.87 / 75.91 / 71.74), right task 94.0% under heavy rewording; the same adapters all on: AP 36.04, FT 50.56, 970 of 1,200 changed. (The main experiments of the paper use other code and rank; their AP is 79.74.)

**What it does not show:** anything about Latent-LoRA as its authors built it (learning, retention, or the effect of its 30 epochs and T5-large); how its router behaves in the stress test with a look-alike task (not run); anything beyond five tasks of one benchmark order; rewordings written by machine. Not pre-registered, so no rule was held or failed. Written into both papers (Appendix "A re-implementation of Latent-LoRA", label `app:latent`) and into the introduction, related work and limitations.

#### CPU work while the 15-task and the three-seed runs were going (2026-11-17, evening)

**A case where the protected zones matter** (`routing_check.py --stress`, routes only, t5-base word-vectors). After the five tasks a sixth task arrives that looks like task 2: sst2 sentences under one of the rewordings of task 2's definition. Routes of the 100 old test inputs of task 2 that change afterwards:

| Definition of the new task | Training inputs of the new task inside an older zone | `certified`: original / light / medium / heavy | `nearest`: original / light / medium / heavy |
|---|---|---|---|
| light rewording | 799 of 800 (reported when the task arrives) | 0 / 0 / 0 / 0 | 0 / **100** / 0 / 0 |
| medium rewording | 0 of 800 | 0 / 0 / 100 / 2 | 0 / 0 / 100 / 2 |
| heavy rewording | 0 of 800 | 0 / 0 / 0 / 100 | 0 / 0 / 0 / 100 |

This is the first test in which `certified` and `nearest` differ. When the look-alike falls inside the old zone, the nearest-zone rule silently hands all 100 lightly reworded old inputs to the new task; the certified switch changes none and reports the collision at once. When the look-alike is outside the old zone, both rules give it the old inputs with exactly that rewording, which were never certified. So the guarantee protects exactly the certified inputs, no more.

**Other models' word-vectors** (`routing_check.py --model ...`, routes only). The same check with the embedding tables of eight models, nothing adjusted per model:

| Embedding table of | Five tasks: certified, % (original / light / medium / heavy) | Five tasks: right task, % | Fifteen tasks: right task, % | Inputs in a wrong zone | Routes changed later |
|---|---|---|---|---|---|
| t5-small | 100 / 100 / 20 / 20 | 100 / 100 / 100 / 89 | 100 / 100 / 100 / 89 | 0 | 0 |
| t5-base | 100 / 100 / 20 / 20 | 100 / 100 / 100 / 94 | 100 / 100 / 100 / 94 | 0 | 0 |
| t5-large | 100 / 100 / 20 / 19 | 100 / 100 / 100 / 83 | 100 / 100 / 100 / 83 | 0 | 0 |
| flan-t5-base | 100 / 100 / 20 / 14 | 100 / 100 / 100 / 99 | 100 / 100 / 100 / 99 | 0 | 2 |
| gpt2 | 100 / 99 / 20 / 20 | 100 / 100 / 100 / 87 | 100 / 100 / 100 / 87 | 0 | 1 |
| SmolLM2-135M | 100 / 99 / 20 / 20 | 100 / 100 / 100 / 82 | 100 / 100 / 100 / 82 | 0 | 0 |
| Qwen2.5-0.5B | 100 / 99 / 20 / 20 | 100 / 100 / 100 / 80 | 100 / 100 / 100 / 80 | 0 | 0 |
| roberta-base | 100 / 99 / 20 / 20 | 100 / 100 / 100 / 94 | 100 / 100 / 100 / 93 | 0 | 0 |

For every model all inputs with the original wording are inside their own zone, in both sequences. The recognition step is therefore not special to T5. Heavy rewording is where the models differ (80% to 99% to the right task). This says nothing about training those models: only routes were computed.

**The method as stand-alone code** (`certified_switch/`). The rule and per-input switching of adapters were written again without any code of the GainLoRA repository, for any Hugging Face model. First run on a CPU with t5-small (five tasks, 400 training examples and 5 epochs each, 50 test inputs each): score right after learning 65.8; at the end 65.8 with the certified switch and 0 of 200 answers changed; 12.0 with all adapters on and 200 of 200 answers changed. Small model, barely trained: it shows that the code works and keeps answers, not how good a model can become. Runs with two decoder-only models (gpt2, SmolLM2-135M) were started.

**Paper draft** for the certified switch: `paper_certified/` (method with proofs, related work, discussion, limitations, appendices; the results sections are filled when the two Kaggle runs have finished). Tables and figures are made from the saved result files by `paper_certified/make_results.py`.

**Where reworded inputs lie** (`paper_certified/extra_analysis.py`, routes only, t5-base word-vectors). Distance of the test inputs from their own zone, in units of the zone's radius (1 = the border; inside = certified), middle value of 100 inputs, and how close the closest input comes to the zone of another task:

| Task | Light | Medium | Heavy | Nearest other zone |
|---|---|---|---|---|
| relation extraction | 0.34 | 0.85 | 0.94 | 2.53 |
| sst2 | 0.78 | 1.94 | 2.08 | 2.07 |
| glucose | 0.76 | 1.85 | 2.06 | 2.26 |
| sentiment140 | 0.11 | 1.84 | 2.29 (up to 2.60) | 2.05 |
| emotion | 0.53 | 1.52 | 1.73 | 2.21 |

With the original wording the distance is exactly 0. Light rewordings stay inside the zone. Medium and heavy rewordings leave it (except for relation extraction, the longest instruction), but other zones begin only 2.1 to 2.6 radii away, so most inputs are still nearest to their own zone. The certificate is lost long before the route is. The one place where the two numbers cross (sentiment140, heavy) is the place of the 31 wrongly routed inputs.

**Does the order of the tasks matter?** (`paper_certified/order_analysis.py`, routes only.) A signature depends only on the task's own instruction, and a radius on the signatures known when the task arrives, so every order can be computed without training. All 120 orders of the five tasks, the two official orders of the fifteen tasks, and 1,000 random orders of the fifteen: in every order all original inputs are certified, 0 inputs are in a wrong zone, 0 training inputs are blocked, and the right-task shares are the same to the last digit (100 / 100 / 100 / 93.8%). **Why, said plainly:** for 13 of the 15 tasks the nearest known signature is that of an unrelated task, and those are known from the start, so the radius cannot depend on the order. Only samsum and xsum have another task of the sequence (reddit_tifu) as nearest neighbour; their radius is up to 7% and 13% smaller if it comes first, and no test input is affected. So this test says "the order did not matter here", and it could not have mattered much. What does depend on the order is *when* an uncertified input goes wrong: the 31 heavily reworded tweets go to sst2 from the start if sst2 is learned first (official order 1), and are re-routed when sst2 arrives if it comes later (official order 2: 31 routes change, all uncertified).

**A mechanical edit test of the routes, with the rules written first** (`paper_certified/edit_analysis.py`; rules and reading in `silent_adapters/PLAN.md`). The rewordings so far were written by machine, so this test needs nobody's wording: every test input of the fifteen tasks (1,197 inputs) gets its own random damage of the definition. 23,640 edited inputs in total.

| Edit of the definition | Still certified, % (1 / 2 / 4 / 8 / 16 words) | Right task, % (1 / 2 / 4 / 8 / 16 words) |
|---|---|---|
| typos (two inner letters swapped) | 87.1 / 74.4 / 37.8 / 10.7 / 0.3 | 100 / 99.9 / 98.1 / 82.7 / **45.4** |
| dropped words | 93.7 / 87.6 / 73.1 / 47.5 / 17.0 | 100 / 100 / 99.6 / 95.5 / 87.0 |
| replaced words | 94.6 / 89.0 / 77.4 / 49.3 / 16.5 | 100 / 100 / 99.7 / 95.7 / 85.9 |

| Edit of the whole definition | Still certified, % | Right task, % |
|---|---|---|
| lower case | 91.4 | 100 |
| "Please help me with this task." in front | 91.6 | 100 |
| first sentence removed (14 tasks) | 24.7 | 100 |
| last sentence removed (14 tasks) | 32.8 | 81.5 (sst2 2%, emotion 8%) |
| sentences shuffled (14 tasks) | 100 | 100 |

Read against the rules written before the run: (1) **no edited input was ever inside the zone of another task** (0 of 23,640), so the certificate was never given for a wrong task; (2) two guesses were too optimistic (87.1% instead of at least 90% certified with one typo; 98.1% instead of 99% right with four typos); (3) one of the three outcomes written down as **counting against the switch happened**: with sixteen typos only 45.4% of the inputs reach their task. So the route of an uncertified input cannot be relied on when the instruction is badly damaged; what the method gives in that case is the missing certificate. (4) Long instructions are robust and short ones are not: tasks with an opening of at least 90 tokens keep 98% to 100% right even with sixteen typos; the two shortest (sciq, a ten-word definition, and xsum) fail first (3% and 18% right with eight typos). (5) Removing the last sentence breaks sst2 and emotion, where that sentence is the one that states the task. (6) Shuffling the sentences changes nothing, because the switch reads which words open an input and not their order. That is a limit of the method. The rule was not changed after this test.

**A proved bound on edits** (paper, Proposition 2). If an edit removes a share `l` and adds a share `l'` of the weight of the opening, the input moves by at most `(l + l') D`, where `D` is the largest distance between two rows of the rescaled word-vector table (509.6 for t5-base). So edits with `l + l' <= R_k / D` (2.0% to 4.2% for the fifteen tasks, about one average token replaced in an opening of a hundred tokens) can never leave the zone. This is a worst case and far more cautious than the measurement above.

**Do the results depend on which unrelated texts were used?** (`paper_certified/background_analysis.py`, routes only; rules and reading in `silent_adapters/PLAN.md`.) The rule uses unrelated inputs for the word weights, the scale of distances and most radii. Eight settings: as in the paper, two other random draws, 500 / 1,000 / 2,000 inputs instead of 4,000, and only the first or only the last five of the ten unrelated tasks. In all eight: every input with the original or the light wording is certified, every input with the original, light or medium wording reaches its task, 0 inputs in a wrong zone, 0 blocked. The share of heavily reworded inputs that reach their task moves between **90.0% and 97.8%** (93.8% in the paper's setting). So the guaranteed part did not depend on the choice; the number for heavy rewording does, by several points, and has to be reported as one value from a range. One guess written before the run was wrong: with fewer unrelated tasks the radii did not grow.

**The zones on 875 real tasks: what the guarantee costs** (`paper_certified/many_tasks_fetch.py`, `many_tasks_analysis.py`, routes only; rules and reading in `silent_adapters/PLAN.md`). All 875 tasks of the default English split of Super-NaturalInstructions (public repository, fixed commit), each with its definition and its first inputs (5,237 inputs), arrive in a random order, and each gets its zone when it arrives. No training.

| Tasks arrived | No collision | All inputs inside an older zone | Inputs given to their own task: `certified` | `nearest` | Routes changed later (`certified` / `nearest`) |
|---|---|---|---|---|---|
| 50 | 49 | 0 | 98.7% | 100% | 0 / 0 |
| 100 | 96 | 1 | 96.7% | 100% | 0 / 0 |
| 200 | 178 | 15 | 90.4% | 98.5% | 0 / 0 |
| 400 | 337 | 50 | 85.7% | 99.0% | 0 / 0 |
| 875 | 724 | 129 | 83.8% | 98.3% | 0 / 0 |

What this says, good and bad:

* **The guarantee held everywhere** (six orders, every size): no certified input ever changed its route.
* **The price is real.** After 875 tasks, 151 tasks (17%) had arrived inside an older zone, and 129 of them completely: under their given instruction they could not be learned at all. These are not accidents: 9 have the same definition text as the older task, 114 share 80% or more of its words, none shares less than 63%. They are families of tasks whose definitions differ in a word or two (35 subject variants of one multiple-choice task, 17 relation variants of one classification task).
* **With unedited instructions the simple nearest-zone rule does better**: 98.3% of inputs to their own task against 83.8%, and it re-routes nothing. An unedited input lies exactly on its own signature, so no later task can be nearer. We had guessed before the run that `nearest` would re-route inputs here; that was wrong. Its weakness needs reworded inputs (the stress test above), and this collection has none.
* **So the trade is:** the zones protect the neighbourhood of an old instruction (its reworded forms) and take it away from later tasks that live there. With frozen zones a later sibling task is refused and reported; with a best-match rule the sibling is served and the reworded inputs of the old task are at risk.
* Added after the run (not in the rules): a smaller radius factor means fewer collisions and smaller zones. Tasks without a collision: 856 (factor 0.1), 843 (0.2), 827 (0.3), 795 (0.4), 724 (0.5). The factor stays at 0.5 as frozen; 0.3 would certify only 59% of lightly reworded inputs.

**A remedy for the collisions, designed after that test: "exact claims first"** (`paper_certified/exact_first_analysis.py`, routes only; the idea, what it guarantees and the checks were written into `silent_adapters/PLAN.md` before any run). An input that carries a task's instruction token for token goes to that task (the oldest, if several tasks have the same instruction); otherwise as before (oldest zone, else nearest zone). Results: on the five and on the fifteen tasks it gives **exactly the routes of the frozen rule** for every test input in every wording (2,000 and 2,697 inputs), so every GPU result holds for it unchanged. On the 875 tasks, **861 tasks receive all their own inputs** when they arrive (the other 14 have an instruction identical to an older one), 98.4% of inputs go to their own task in every order (frozen rule 83.8%, nearest zone 98.3%), and no route changes later. In the stress test it serves the look-alike task (800 of 800 of its inputs, frozen rule 1 of 800) and pays as predicted: the 100 old inputs that carry the look-alike's instruction word for word go to it; nothing else moves. **Its guarantee is weaker and that must be said with it:** an unedited input is fixed for ever; an edited input inside a zone is fixed *unless* a later task registers exactly its text. On these tests its routes are almost those of the nearest-zone rule; the difference is what can be promised. Not tested: a case where the two differ on real inputs, and the variant with trained answers on a GPU. The paper's main method stays the frozen rule; the variant is in the paper as an option (method section, Proposition 5) and in the stand-alone code as `rule="exact_first"`.

**Stand-alone code on a decoder-only model** (`certified_switch/demo.py`, CPU, SmolLM2-135M, five tasks, 400 training examples and 5 epochs each, 30 test inputs each, 48 minutes). Score right after learning 74.34; at the end 74.34 with the certified switch, 0 of 120 answers changed, all 150 test inputs certified and given to the right task; with all adapters on 27.73 at the end and 100 of 120 answers changed. Heavy rewording: 65.08, 0 of 120 changed, 80.7% to the right task. The first gpt2 run was killed by the machine for lack of memory (two demos and an analysis at once) and was run again alone: score right after learning 67.94; at the end 67.94 with the certified switch, 0 of 120 answers changed, all 150 inputs certified; 15.80 with all adapters on and 119 of 120 answers changed; heavy rewording 54.81, 0 changed, 81.3% to the right task (70 minutes).

**References of the paper draft.** The ten that had been written from memory were checked against the web and were correct. Eight closely related papers were added after checking them the same way (Expert Gate, SupSup, AdapterCL, HiDe-Prompt, mixture-of-experts adapters, LoraRetriever, selective classification, certified robustness), and five references for the models used. The other twenty had been looked up for the first paper.

**An independent review of the paper draft, and what it corrected** (2026-11-17, evening). A second agent that had written none of the text read the draft against the result files and the plan and found 22 problems. The ones that change what this log says above:

* The comparison with InfLoRA and GainLoRA says less than "they learn less". Right after learning they are at the level of guessing on the two tasks with two labels (SST-2 55 and 58; Sentiment140 44 and 51, where always answering "negative" gives 63). So in these runs they **had not learned three of the five tasks**, and one step of their code ran with a substitute that was never compared number for number. The comparison says nothing about how well they retain, and nothing about why they learned so little.
* With unedited instructions the scores **do not separate the certified switch from simpler rules** (nearest zone, even the exact key): all reach 79.7. The switch adds the proof and the certificate, not points.
* "The certificate was never wrong" was too wide: true for the 23,640 edited inputs of the fifteen tasks, not for the 875-task collection, where 849 inputs are certified for an older sibling. A certificate promises a stable answer, not the right task.
* The bound on edits is tighter than first stated: `max(l, l') D` instead of `(l + l') D`. Checked on 2,250 edited inputs: it held for all, and the inputs moved at most 41% of it.
* "Radius zero for 8 tasks" was a rounding artefact; computed directly, 14 tasks have exactly the opening of an earlier task. Nothing else changed in the rerun.
* A remark in the paper was mathematically wrong (the exact key is not the small-radius limit of the switch) and three facts from the plan were missing in the paper (how the five tasks were chosen, that the unrelated tasks leave out sentiment tasks, the substitute in the baselines' code). All corrected. Details in `silent_adapters/PLAN.md`.

**The sibling task with trained answers** (`certified_switch/demo_sibling.py`, CPU, t5-small, 26 minutes). After the five tasks a sixth is learned: sst2 sentences under the lightly reworded sst2 definition, with other answer words ("good" / "bad"). Under `certified`: 400 of 400 training inputs blocked, the sibling's test inputs are still answered in the old task's words (0 of 50 right), no old answer changes in any wording. Under `exact_first`: the sibling is learned (31 of 50 right), no answer to an unedited old input changes, and the 50 old sst2 test inputs that carry the sibling's instruction word for word now get the sibling's answer; nothing else changes. One small model, one constructed case.

#### CPU checks made before the GPU run (2026-11-17, first version of the rule)

**Why answers inside a zone cannot change:** the task chosen for such an input is the oldest zone that contains it. New tasks only add newer zones, so the choice never changes, and the adapters used are frozen. This holds for any later tasks and any model. It does not hold outside the zones.

**Where the inputs go** (real benchmark inputs, t5-base word-vectors, radius factor 0.5; `routing_check.py`, about 3 minutes on a CPU). The reworded definitions (light, medium, heavy) were written by machine for this check:

| Sequence, wording | inside own zone (certified) | given to own task | inside a wrong zone | route changed after the task was added |
|---|---|---|---|---|
| five tasks, original | 100% | 100% | 0 | 0 of 500 |
| five tasks, light rewording | 99.8% | 100% | 0 | 0 of 500 |
| five tasks, medium rewording | 20% | 100% | 0 | 0 of 500 |
| five tasks, heavy rewording | 20% | 97.4% | 0 | 1 of 500 |
| fifteen tasks (SuperNI order 1), original | 100% | 100% | 0 | 0 of 1,197 |

What this shows, and what it does not:

1. **On these inputs the task can be recognised without the exact wording.** With the original wording and with light rewording nearly every input is inside its own zone, so its answer is covered by the guarantee. With medium and heavy rewording most inputs are outside the zones and still reach their own task through the nearest centre (100% and 97.4%), with no guarantee.
2. **No input fell inside the zone of another task**, in the five-task and in the fifteen-task sequence. So no task was blocked by an older one here.
3. **The nearest centre alone gave the same routes as the zones on these inputs.** The zones changed no route in this check; what they add is the proof and the flag, and protection in a case not tested here (a later task that is deliberately close to an old one).
4. **Three zone shapes failed before this one** (lesson 32): a ball around the whole-input average, an ellipsoid with the task's own covariance, and a ball around the first 64 tokens. Each let old tasks claim most inputs of new tasks on the fifteen-task sequence (between 2% and 36% of inputs reached their own task, depending on the shape and its settings).
5. **The radius factor was chosen after looking at these routes** (0.3, 0.5 and 0.7 were tried; none put an input in a wrong zone). No answer of the model was used.

**Code check with a tiny model** (t5-small, 3 tasks, 96 training examples, CPU; it shows that the code runs, it is not a result): the official test gave the same scores for `certified` and `locked`; the re-test reproduced the official test on 300 of 300 inputs; with the original wording 0 of 200 answers changed under `certified`; with reworded definitions the exact key matched nothing and behaved like `plain` (143 to 186 of 200 answers changed), while `certified` changed 0, 0 and 3 of 200.

How new this is: recognising the task from frozen pooled word-vectors without a trained router is published (Latent-LoRA). Two searches on 2026-11-17 did not find zones that are never changed with "oldest claim wins", a proof for the inputs inside them, and a flag on each answer. That is a small step on known parts, and the searches may have missed a paper.

#### Wider zones: more ideas tested on CPU with machine paraphrases of 700 real tasks (2026-11-17 and 18)

**Why.** The 875-task test above showed two weak points: with the paper's word-vector zones, a reworded instruction loses its certificate quickly, and a zone wide enough to help collides with sibling tasks. The proof of the guarantee does not depend on the word-vectors. It holds for any frozen feature, and for any zone that is fixed when the task arrives. So four ideas for a wider zone were tried, one after the other. Each rule was written in `silent_adapters/PLAN.md` before the run. All four are routes only, on CPU. No adapter was trained.

**The test.** 875 real tasks of Super-NaturalInstructions. A random 175 are for calibration (choosing settings) and the other 700 are for evaluation (the same 700 for all four ideas). Every task definition was translated by a public translation model into German and back, and into French and back, sentence by sentence (`paper_certified/paraphrase_make.py`). On average 83% of the words are kept. Nobody on the project wrote these rewordings, and some are poor; they were used as they came. The 700 tasks arrive in a fixed order and each gets its zone when it arrives. Four measures:

* **C**: share of tasks that collide, that is, the task's own instruction lies inside an older zone when it arrives.
* **P**: share of paraphrased inputs that are certified for their own task.
* **M**: share of paraphrased inputs that are certified for another task (a stable answer from the wrong adapters). This is the harmful one.
* **R**: share of paraphrased inputs that reach the right task, with or without a certificate.

**What the paper's rule scores on this test** (T5-base word-vectors, 700 tasks, 4,198 paraphrased inputs): **C 16.9%, P 40.5%, M 7.6%, R 75.4%**. This is the first test of the rule with rewordings that the project did not write, at scale. The 100% / 100% / 94% of the five-task runs were measured on tasks that are far apart and on rewordings written by machine. The new test belongs in the limitations of the paper.

**Idea 1: zones in the hidden states of the frozen model** (`paper_certified/feature_study.py`, SmolLM2-135M, numbers in `routing/feature_study.json`). The feature is a pooled hidden state of the frozen base model, instead of the word-vector average. A hidden state should read meaning, so paraphrases might stay together. Layer 0 with the weighted mean is exactly the paper's feature; this was checked and held (the anchor). Settings were chosen on the 175 calibration tasks (last layer, weighted mean, one radius).

* **Rule fixed before the run:** success if at least 20 points more paraphrased inputs are certified for their own task at the same or a lower collision rate, with no more wrong certificates. Against the idea if no setting gains 5 points at a matched collision rate, or if wrong certificates go up.

| SmolLM2-135M, 700 tasks | C | P | M | R |
|---|---|---|---|---|
| word-vectors, rule of the paper | 18.9% | 42.1% | 8.9% | 75.6% |
| last hidden layer, chosen setting | 25.7% | 64.9% | 13.7% | 83.2% |

* **Verdict: against.** P rose by 22.7 points, but C and M rose too. The collision rate chosen on 175 tasks also did not carry over to 700 tasks, which is a flaw of our rule and not a reason to read the result more kindly. Hidden states tolerate paraphrase better and tell sibling tasks apart worse. They move along the trade; they do not leave it.

**Idea 2: a new ruler, in units of what paraphrasing does** (`paper_certified/ruler_study.py`, numbers in `routing/ruler_study.json`). Keep the word-vector feature. Change only how distance is measured: rescale by how a paraphrase moves the feature (estimated on the German paraphrases of the calibration tasks, 10% shrunk towards the identity), so that directions where paraphrases usually move count little.

* **Rule fixed before the run:** read both rulers at the radius where C equals the collision rate of the paper's rule. Success: at least 20 points more P, with no more inputs certified for another task. Against: a gain under 5 points, or more inputs certified for another task. Evaluation on the French-pivot paraphrases, which were not used for the estimate.

| 700 tasks, French pivot, at matched C | Ruler | P | M | R |
|---|---|---|---|---|
| SmolLM2-135M word-vectors | old | 41.6% | 12.3% | 81.8% |
| SmolLM2-135M word-vectors | new | 50.1% | 13.1% | 83.2% |
| T5-base word-vectors | old | 31.8% | 9.5% | 83.0% |
| T5-base word-vectors | new | 42.0% | 10.7% | 84.4% |

* **Verdict: against.** The gain in P is 8.5 and 10.2 points, far below 20, and M went up, which the rule forbade. With the paper's own radius rule the gain on T5-base is zero (39.5% against 39.8%). On the five benchmark tasks with the machine-written heavy rewording the new ruler was worse than the old one (66% to the right task against 94%).
* **What it shows.** The movement of a paraphrase is not confined to a few directions (the ten largest directions hold 35% of it for T5-base and 52% for SmolLM2). So no linear ruler can separate "paraphrase" from "sibling task" in an average of word-vectors.

**Idea 3: a small frozen sentence encoder, and a radius with a stated coverage** (`paper_certified/encoder_study.py`, numbers in `routing/encoder_study.json`). The feature is `all-MiniLM-L6-v2` (22 million weights), made to bring paraphrases together. The radius is a split-conformal quantile of the distance of a paraphrase from its own signature on calibration tasks. If a new task is like the calibration tasks, at least a share 1 - alpha of its rewordings lie inside its zone (alpha = 0.1 primary). Whether an older zone claims them first is not covered and was measured. The variant with the paraphrase ruler was chosen on half of the calibration tasks (ratio 1.73 against 1.50).

* **Rule fixed before the run:** success if P >= 75%, M <= 5% and C <= 19%. Against the idea if P < 55%, or M > 9%, or C > 25%. In between is "partial".

| French pivot, 700 tasks | C | P | M | R | Z (own zone) |
|---|---|---|---|---|---|
| paper's rule, T5-base word-vectors | 16.9% | 40.5% | 7.6% | 75.4% | |
| encoder with ruler (primary) | 16.6% | 71.2% | 11.6% | 86.1% | 82.0% |
| encoder, plain | 22.9% | 73.1% | 17.1% | 82.0% | 89.8% |

* **Verdict: against** (M = 11.6% is above 9%; P = 71.2% is below 75%). A second encoder (bge-small) gave the same picture (P 71.2%, M 12.7%, C 19.9%).
* **What it shows.** The encoder does what it was chosen for: about seven in ten paraphrased inputs are certified for their own task, against four in ten, at the same collision rate. The coverage promise held for the plain metric (Z 89.8% French, 92.0% German, promised 90%) but not for the ruler variant on the other pivot (82.0%). Wrong certificates became more frequent. The colliding tasks are siblings that differ in one decisive word (date format day-month against month-day; closest integers against farthest integers), and to an encoder these are nearly the same text. On the five benchmark tasks, the machine-written medium and heavy rewordings lie mostly outside the zones (38% and 1% inside), but they still reach the right task.

**Idea 4: a zone decided by a frozen judge of "same instruction"** (`paper_certified/judge_study.py`, numbers in `routing/judge_study.json`). The question for a zone is not "how alike" but "is this the same request". A small frozen inference model (`cross-encoder/nli-deberta-v3-xsmall`) says whether each text implies the other. The instruction has to be a separate field of the input. The zone of a task is the set of instructions that pass a wide encoder filter and a judge threshold. Both thresholds come from calibration tasks by split-conformal quantiles. The judge was chosen over a free word-difference judge on calibration tasks (area under the curve 0.969 against 0.677). The run took 91 minutes on four CPU cores.

* **Rule fixed before the run:** the same as for idea 3 (success P >= 75%, M <= 5%, C <= 19%; against P < 55%, or M > 9%, or C > 25%).

| 700 tasks | C | P | M | R | Z |
|---|---|---|---|---|---|
| paper's rule, T5-base word-vectors | 16.9% | 40.5% | 7.6% | 75.4% | |
| judge zone, French pivot (primary) | 36.0% | 58.9% | 34.7% | 64.7% | 90.6% |
| judge zone, German pivot | 36.0% | 56.3% | 41.1% | 58.7% | 95.0% |

* **Verdict: against** (M = 34.7%, C = 36.0%, P = 58.9%).
* **What it shows.** The coverage promise held. The judge is good on single pairs and bad as a zone: a text is compared with up to 20 older tasks, and one wrong "same" is enough. At the chosen threshold 11.5% of the nearest-other-task pairs count as "the same", and that became 35% wrong certificates. A stricter threshold does not rescue it (looked at after the run, nothing chosen from it): at tau = 0.91 M is 10.6% with P 70.4% and C 10.4%; P never reaches 75%, and M falls below 9% only at the last threshold, where almost nothing is covered (P 0.6%). The wrong "same" verdicts are the same sibling tasks as before (remove decimal numbers against remove odd numbers; ATOMIC "reaction" against "intent"; Amazon reviews by category against by polarity). Of the 151 pairs that collide under the paper's rule, the judge calls 60 "the same". One good result: on the five and fifteen benchmark tasks, 14 of the 15 rewordings written by machine lie inside their own zone and none is certified for a wrong task. Cost: 18 judged pairs and 3.5 seconds per text on two CPU threads.

**Common finding.** Four ideas, four different features (hidden states, a new ruler on word-vectors, a sentence encoder, an inference judge), the same end. Sibling tasks differ in one decisive word. A zone wide enough to hold the rewordings of a task also holds the rewordings of its sibling. Better "likeness" raises the correct and the wrong certificates together.

**Decision.** No GPU run for any of the four. No more ideas of this kind. The guarantee stays as proved. The paper reports the price measured at scale (the numbers of the paper's rule above) and the four attempts.

**One confirmation check, and its result.** While reading the table of idea 3, we saw that a setting that was not the primary one (encoder with ruler, alpha = 0.2) beat the paper's rule on all four measures on the French pivot. That comparison was chosen after seeing the results, so it proves nothing by itself. A check was therefore fixed in advance (plan, 2026-11-18) on paraphrases through Spanish (`routing/paraphrases_es.json`), which no study had read: same 700 tasks, nothing fitted again; "confirmed" if the encoder setting gains at least 15 points in P and 8 points in R and has no more wrong certificates. **Result: not confirmed.** On the Spanish pivot the encoder setting has P 63.8%, M 5.3%, R 91.7%; the paper's rule has P 50.4%, M 9.7%, R 77.5%. The gain in P is 13.4 points, and the rule asked for 15. The encoder setting is ahead on each measure, but by less than asked. The paper's rule itself does better on the Spanish paraphrases than on the others (P 39.8% French, 41.2% German, 50.4% Spanish), because they keep more of the original words (86% against 83 to 84%). So the honest summary of the paper's rule is "four to five in ten paraphrased inputs are certified for the right task, depending on the pivot". Over five other random splits of the tasks (Spanish pivot, mean and range): encoder setting P 63.1% (55.5 to 70.0), M 6.0% (2.4 to 9.7), C 12.7%; paper's rule P 49.4% (48.3 to 50.7), M 7.9% (4.2 to 10.4), C 16.0%. On the five benchmark tasks under the encoder setting, none of the medium and heavy rewordings written by machine is inside a zone (they are still sent to the right task). So the encoder zone is reported in the paper as a variant with its costs, not as a shown improvement (`paper_certified/spanish_check.py`, numbers in `routing/spanish_check.json`).

**What was tested and what was not.**

* *Tested:* four ideas on 700 real tasks with machine paraphrases (French and German pivot; routes only, on CPU), against rules written before each run. All four are against. The coverage promise of the split-conformal radius held where it was meant to (plain encoder, judge).
* *Not tested:* rewordings written by people (translation models may move texts differently); a much larger frozen judge, for example the frozen base model itself asked whether two instructions ask for the same thing; tasks outside this collection; any run with trained answers on a GPU.
* *Only expected:* that a larger judge would still meet the sibling-task problem, and that human rewordings behave like the machine ones. Neither was checked.

#### What the zones add at scale: do later tasks re-route reworded inputs? (2026-11-18)

**Why the test was run.** The paper had one hand-built case (the stress test) in which a best-match rule silently re-routes old inputs when a look-alike task arrives. It said of the 875 real tasks that this collection had no edited inputs. Now it has: the machine paraphrases of 700 real tasks. So the central claim could be counted on real tasks nobody constructed. The paper's rule is used as it is. This is not a new idea for wider zones.

**What was fixed before the run.** The same 700 tasks in the same order, T5-base word-vectors, 4,198 paraphrased inputs (French and German pivot), Spanish pivot separately. For each input and rule, the route right after its own task arrived and the route after all 700 tasks. G is the share of paraphrased inputs whose route changed in between. The script had to reproduce the paper's numbers (C 16.9%, P 40.5%, M 7.6%, R 75.4%) and check the theorem. Reading: "the zones add something at scale" if G under the nearest-zone rule is at least 5% and the certified switch has at most half of it; "they add little here" if G under the nearest-zone rule is below 2%. The reading was written before the run. The plan notes that the follow-up rules below were thought of after seeing the first result, and that the first verdict stands.

| 4,198 paraphrased inputs, 700 tasks | Own task at the end | Route changed after arrival | Inside a zone at arrival |
|---|---|---|---|
| plain nearest signature (no radius) | 91.3% | 2.22% (93 inputs, 41 tasks) | no zones |
| nearest zone | 75.4% | 0.31% (13 inputs) | no certificate |
| certified switch (the paper's rule) | 75.4% | 0.31% (the same 13 inputs, 7 tasks) | 48.1% |
| zones first, then plain nearest signature | 85.6% | 1.95% (82 inputs, 37 tasks) | 48.1% |

**First test, verdict: "the zones add little here".** G under the nearest-zone rule is 0.31%, far below 2%, and the certified switch removes none of it. The theorem held: none of the inputs inside a zone at arrival changed its route. All 13 changes are outside every zone. But the nearest-zone rule, which has no guarantee, changed the same 13 inputs and no others. Spanish pivot: 7 of 2,099 changed under every rule. Unedited inputs: no change under any rule, and the nearest-zone rule serves more of them (98.4% against 83.1%). So the hand-built case does not occur here. What the certified switch adds is the proof for the inputs inside a zone, not a measured gain in stability.

**Follow-up: the plain nearest signature (no radius).** Reading fixed before the run: at least 5% is "the radius fixed at arrival keeps routes stable", below 2% is "rare", in between "some". Result: **"some"**. G is 2.22%, and 1.05% of inputs lose their own task. The radius does keep routes more stable (0.31% against 2.22%). The surprise is the other column: the plain rule gives **91.3%** of paraphrased inputs to their own task, against 75.4% for the paper's rule (Spanish: 93.0% against 77.5%; unedited: 98.4% against 83.1%). Of the 93 changed inputs, 34 end at a task whose definition shares at least 80% of its words with their own.

**Last rule: zones first, then the plain nearest signature.** Inside a zone, as in the paper (same certificates, same theorem; checked, no such input changed). Outside every zone, the plain nearest signature. It was designed after seeing the numbers, so it is not an independent test. Result: 85.6% to the own task with the same certificates, G 1.95%, 0.79% lose their own task (Spanish: 84.4%, 2.29%). On the benchmark tasks with the machine-written rewordings the picture reverses:

| 500 inputs, routes only | Certified switch and nearest zone | Plain signature, and zones first then plain signature |
|---|---|---|
| original, light | 100% (0 changed) | 100% (0 changed) |
| medium | 100% (0 changed) | 76.8% (55 changed) on five tasks, 71.6% (74) on fifteen |
| heavy | 93.8% (0 changed) | 60.0% (100 changed) |

So the last rule is **not a repair**: it gains ten points on the large collection and loses thirty-four under heavy rewording on the benchmark. The two fallbacks trade places. No fallback we have is right in both places. Why the plain rule fails on the benchmark was not tested.

**Our guesses.** Wrong: G of 8 to 20% for the nearest-zone rule (found 0.31%); G of 5 to 20% for the plain rule (found 2.22%); that the plain rule on the benchmark would be no better (it is far worse). Right: the last rule at scale (84 to 88% expected, 85.6% found; G 1.4 to 2.0% expected, 1.95% found). Also right: the theorem check.

**What the paper has to say after this.**

1. At scale the certified switch shows no measured stability beyond the nearest-zone rule: both changed the same 13 of 4,198 routes. It adds the proof for the inputs inside a zone, about half of the machine-paraphrased inputs.
2. The plain nearest-signature rule gives more machine-paraphrased inputs to their own task (91.3% against 75.4%) and silently changes about two routes in a hundred.
3. The fallback of the paper's rule was chosen on the benchmark tasks and does not carry over to the large collection. Which fallback is better depends on the collection.
4. The certificates do not depend on the fallback.

**Tested:** routes only, on CPU, with machine paraphrases of 700 real tasks from one collection (Super-NaturalInstructions), against rules written first (the last rule and the follow-up after seeing the first result). **Not tested:** trained answers on a GPU, rewordings written by people, and why the plain rule fails on the benchmark. The paper's rule stays the rule of all GPU runs; a changed fallback would be a variant that has not been run with trained answers.

#### A fallback that is right in both places, and two more tests (2026-11-18)

**Why.** More ideas towards a stronger paper were asked for at 01:45 UTC. The weak part was the fallback of the paper's rule, the rule for inputs outside every zone. It is right on the benchmark tasks (93.8% under heavy rewording) and wrong at scale (75.4%). The plain nearest signature is the other way round (91.3% at scale, 60.0% on the benchmark). All numbers here are routes only, on CPU, T5-base word-vectors. Rules were written in `silent_adapters/PLAN.md` before each run.

**A look at the words (an exploration on the 175 calibration tasks only, not a test).** We looked at why every "likeness" idea failed. Nothing from the 700 evaluation tasks was read.

* A paraphrase drops 13.7% of the words of a definition. It drops rare words more often than common ones: 33% of the words that occur in only one calibration definition, 13% of the words that occur in more than 60.
* Sibling tasks (43 pairs of calibration tasks with different texts that share at least 60% of their words) differ in the same kind of words: 31% of the words in which they differ occur in only one definition, and 55% in at most three.
* A rule "the rare words must all be present" does not work. Only 49% of the paraphrases keep all words that occur in one definition only (37% for words in at most two), and 12% (5%) of the sibling pairs have the same such words.

Reading: rewording and being a sibling task change the same words. So ideas built on key words, or on weights learned from what rewording changes, are dropped without a test. What separates "a synonym was used" from "a different thing is asked" is meaning, sentence by sentence.

**Fallback normalised for the length of the opening.** Idea: the radius might mainly put openings of different lengths on one scale. So divide the distance by sigma(n), the spread of the unrelated inputs at opening length n, and drop the part of the radius that shrinks next to a sibling. No free setting. Rule fixed before the run: success if at least 90% of inputs reach their own task under heavy rewording (five and fifteen tasks), at least 97% under medium, and at least 88% on the Italian pivot (paraphrases through Italian, made new for this test and not read before); against if heavy rewording is below 80% or the Italian pivot below 85%.

* Result: heavy rewording 63.0% (five and fifteen tasks), medium 94.0% and 93.4%, Italian pivot 93.9%. **Verdict: against** (heavy is below 80%).
* Our reading was only partly right. The correlation between sigma(n_k) and the radius is 0.06, so the radius is not mainly a scale for the length of the opening.

**Fallback scaled by the distance to the unrelated tasks (`background`).** The reading that led to it: the radius of the paper is half the distance to the nearest known signature, and "known" includes earlier tasks. On the benchmark the tasks are far apart, so the nearest known signature is an unrelated one and the scale helps. At scale, a sibling's nearest known signature is its older sibling, its radius becomes tiny, and its reworded inputs look far away. So take the scale from the unrelated tasks only: B_k is the distance from the signature of task k to the nearest unrelated signature, fixed when task k arrives. Rule `background`: smallest distance divided by B_k. Rule `certified4`: the zones and certificates of the paper unchanged, and `background` outside the zones. No free setting.

* **Rule fixed before the run (and before any Italian number was seen):** success if at least 90% of inputs reach their own task under heavy rewording on five and on fifteen tasks, at least 97% under medium, and at least 88% on the Italian pivot after all 700 tasks; against if heavy is below 80% in either sequence or the Italian pivot is below 85%; otherwise partial.

| Share of inputs given to their own task | `background` | plain nearest signature | nearest zone (fallback of the paper) |
|---|---|---|---|
| benchmark, heavy rewording (five and fifteen tasks) | 93.8% | 60.0% | 93.8% |
| benchmark, medium rewording | 100% | 76.8% and 71.6% | 100% |
| 700 tasks, Italian pivot (untouched until now; the verdict) | 95.9% | 93.1% | 77.0% |
| 700 tasks, French and German pivot | 94.4% | 91.3% | 75.4% |
| 700 tasks, Spanish pivot | 95.3% | 93.0% | 77.6% |
| routes changed after arrival, French and German (Italian) | 1.60% (1.29%) | 2.22% (2.24%) | 0.31% (0.10%) |
| unedited inputs | 98.4% | 98.4% | 98.4% |

With the zones of the paper in front (`certified4`): 88.6% of the French- and German-pivot inputs to their own task (the paper's rule: 75.4%), 86.6% on the Spanish pivot (77.5%), 91.3% on the Italian pivot (76.8%). Routes changed after arrival: 1.33% (0.31% for the paper's rule). 48.1% of the inputs are inside a zone at arrival and none of them changed. Unedited inputs: 83.1%, as before. On the benchmark tasks the shares are those of the paper's rule in every wording (100%, 100%, 100%, 93.8%; no route changed).

* **Verdict: success.**
* **Our guesses.** Benchmark as expected. On the 700 tasks we expected 86 to 92% for `background` and 82 to 87% for `certified4`. Both came out higher (95.9% and 91.3% on the Italian pivot).
* **Why it works, as far as the numbers show.** B_k has a correlation of 0.52 with the radius. On the benchmark the radius is half of B_k for all five tasks and for 14 of the fifteen, so the two rules agree there. At scale the radius is half of B_k for only 28.6% of the 700 tasks. For the others an earlier task is nearer than every unrelated one, the radius shrinks, and the old fallback turns away the reworded inputs of exactly those tasks. The radius did two things; only the first one is wanted in the fallback.
* **What it costs.** (1) Stability outside the zones: 1.3 to 1.6% of paraphrased inputs change their route after their task has arrived (0.3% with the old fallback). (2) In the stress test a look-alike task takes some uncertified old inputs that the old fallback kept: with the light rewording as the look-alike's definition, 2 of the 100 old inputs with the medium wording and 20 with the heavy wording change (0 and 0 with the old fallback). The 200 certified old inputs do not change under either. (3) It is a variant. Every GPU run and every other route test of the paper used the old fallback. Where the routes are the same the scores are the same, and this was checked input by input (see below). (4) At scale the zones themselves cost about five points of right routes (94.4 to 95.9% without zones against 88.6 to 91.3% with them) and give a proof for about half of the inputs. **Added after the last careful read of the paper (03:37 UTC):** the stress test has two more cases that this entry had left out. With the medium rewording as the look-alike's definition the new fallback lets it take 80 of the 100 old inputs with the heavy wording, where the old fallback lets it take 2; with the heavy rewording as its definition both lose the 100 inputs with that wording. No certified input changes in any case. So the new fallback is clearly less stable for uncertified inputs when a look-alike task arrives. And "the same routes on the benchmark" means: the same route for every test input from the arrival of its own task on, which is everywhere a score is taken.

Other rules on the Italian pivot (own task at the end; routes changed): the paper's rule 76.8% and 0.10% (P 47.0%, M 5.9%, R 76.8%); zones then plain signature 88.5%, 2.24%; length-normalised 93.9%, 1.67%.

**Sixth idea: a zone decided sentence by sentence** (`paper_certified/sentence_judge_study.py`). The exploration says that meaning is what separates a rewording from a sibling. The fourth idea asked a judge about whole instructions of up to 128 tokens (one decisive word is easily lost) and against twenty candidates at once. Here, the instruction is a separate field. A text passes if its distance to the stored instruction in the sentence encoder is at most r, and if every sentence pair (each sentence matched to its most similar sentence in the other text) is scored at least tau by a frozen inference model; identical sentences score 1. Every task within r is judged, so the zone is a fixed set and Theorem 1 applies. Thresholds come from half of the calibration tasks (r at alpha 0.10, tau at alpha 0.05); the judge is chosen among three on that half only. **Rule fixed before the run:** the same as for ideas 3 and 4 (success P >= 75%, M <= 5%, C <= 19%; against P < 55%, M > 9% or C > 25%; otherwise partial), read on the Italian pivot. Our expectation was P 62 to 74%, M 3 to 7%, C 5 to 10%, so most likely "partial". **Result (`routing/sentence_judge_study.json`, about 47 minutes on CPU).** Calibration: r = 0.588; the smallest judge was chosen (area under the curve 0.965, against 0.964 and 0.954). The threshold came out as tau = 0.0017: for about 4% of the true rewordings one sentence pair gets almost no "entailment", so a threshold that keeps 95% of them has to be nearly zero, and at that threshold 24% of the nearest other instructions still pass.

| 700 tasks | C | Z | P | M | R |
|---|---|---|---|---|---|
| **Italian pivot (untouched; the verdict)** | 12.4% | 87.0% | 76.1% | 10.9% | 88.7% |
| French pivot | 12.4% | 85.3% | 75.4% | 10.0% | 89.1% |
| German pivot | 12.4% | 83.1% | 74.3% | 9.0% | 89.7% |
| Spanish pivot | 12.4% | 81.7% | 73.9% | 8.0% | 89.9% |

**Verdict: against** (M = 10.9% on the Italian pivot is above the 9% limit; P = 76.1% and C = 12.4% are on the right side of the success lines). It is the best point so far: three in four rewordings certified for their own task, with about half the wrong certificates and half the collisions of the sentence encoder alone (73.1%, 17.1%, 22.9%). The wrong certificates are the same sibling tasks as always (date formats day-month and month-day; the ATOMIC tasks "intent" and "reaction"; "all elements except the first i" and "all elements from index i to j"). Of the 142 colliding pairs with different texts the judge still passes 50. A small inference model does not know that "reaction" and "intent" ask for different things. Cost: 0.84 seconds per text on three CPU threads. Across the six ideas, the frontier moved (from 40% certified with 8% wrong to 76% certified with 11% wrong) and the wall did not. A large frozen judge (the base model itself, asked directly) is the open experiment.

**What was tested and what was not.**

* *Tested:* routes only, on CPU, with machine paraphrases (French, German, Spanish and Italian pivots) of 700 real tasks from one collection, and the benchmark tasks with the machine-written rewordings, against rules written before each run. The Italian paraphrases were untouched until the verdict of the two fallback tests.
* *Not tested:* trained answers with the new fallback (no GPU run was made); rewordings written by people; the other route tests of the paper (task order, edit test, unrelated inputs, 875-task collisions) with the new fallback.
* *Checked afterwards:* for every benchmark test input (five and fifteen tasks, all four wordings) the new fallback gives the same route as the paper's rule at every arrival from the arrival of the input's own task on: 0 differences (`routing/reroute_study.json`). Differences exist only before an input's own task has arrived (for example 110 inputs on the original wording of the fifteen tasks), and no answer is given from those routes. So every GPU score of the paper is also the score of the variant.

#### A new direction: no instruction at all, and a bound on forgetting (2026-11-18)

**The aim set at 02:51 UTC, as the plan gives it:** Find a big flaw in the field: show with human-written rewordings that most published continual-learning methods only look good because the instruction gives away the task, or another flaw if that is the better one. Fix it: build a method that reaches near-zero forgetting without any instruction and comes with a non-trivial guarantee, and prove it.

**What cannot be done before the deadline, and why.** Rewordings written by people do not exist in this project, so none can be called "human-written". Running the published methods without instructions needs their official code on a GPU for several hours and at least one retry. That does not fit before the deadline. So a claim about "most published methods" cannot be shown tonight, and it is not claimed. What the existing runs show stays as in the paper: a rule without any training recognises the task from the instruction for every test input on these benchmarks, so with unedited instructions they cannot tell retention from recognition. The second half of the request could be attempted: a method that does not use the instruction, with a guarantee that is more than "inside a zone nothing changes".

**The guarantee, in plain words.** When task k arrives, hold out m of its training inputs. Set the size of the zone of task k on these held-out inputs, with a score computed only from frozen parts and the other training inputs: the zone holds every input whose score is at most the quantile q_k of the held-out scores (level alpha). The zone is never changed, and an input goes to the oldest zone that holds it. Then, if a test input of task k is exchangeable with the held-out inputs (for example drawn from the same distribution), for any model, any later tasks and any way of training them, the probability that its answer ever changes after task k has been learned is at most alpha.

*Short proof.* The input lies in the zone of task k with probability at least 1 - alpha (the split-conformal argument, as for the coverage proposition of the paper). If it lies in the zone, the oldest zone that holds it belongs to task k or an older task and can never change, so by the write-once theorem its answer never changes. For an answer score between 0 and 100, the expected forgetting of task k is then at most 100 times alpha points. This holds for all test inputs of the task and needs no instruction.

**What it does not say.** It bounds change. It says nothing about whether the task was learned: inputs of task k that lie inside older zones are answered by older adapters, and their share is reported when the task arrives. The probability is over the held-out and the test input together, and the test inputs must be like the training inputs.

**Pilot A (routes only, CPU, `paper_certified/noinstruction_study.py`, `routing/noinstruction_study.json`).** Official benchmark files, instances only: the definition is never read. 800 training inputs per task for the zones, the official test inputs for the measures. Feature: the weighted word-vector average of the instance. 200 held-out inputs per task, alpha = 0.05. Two scores: the distance to the 10th nearest training input, and (designed after seeing the first numbers, so not an independent test) the ratio of that distance to the distance to a fixed pool of 5,165 unrelated inputs. Lines fixed before: the claim holds if on average at most alpha + two standard errors change and coverage is at least 1 - alpha minus two standard errors; "works without instructions" if at least 90% of test inputs reach their own task and at most 10% are blocked; "does not" if fewer than 75% or more than 25% are blocked. Averages over the tasks of each sequence:

| Sequence | Score | Inside own zone | Route changed after arrival | Blocked by an older zone | Given to own task |
|---|---|---|---|---|---|
| 15 SuperNI tasks | distance to 10th neighbour | 96.1% | 0.7% | 86.5% | 12.7% |
| 15 SuperNI tasks | ratio to unrelated inputs | 94.3% | 0.0% | 65.5% | 33.7% |
| 15 Long Sequence tasks | distance to 10th neighbour | 94.9% | 1.1% | 88.3% | 10.6% |
| 15 Long Sequence tasks | ratio to unrelated inputs | 94.3% | 1.1% | 81.0% | 17.8% |
| all 30 | distance to 10th neighbour | 95.5% | 0.5% | 93.2% | 6.3% |
| all 30 | ratio to unrelated inputs | 94.3% | 1.5% | 78.1% | 20.3% |

The claim holds in every row. The use fails in every row (far more than 25% blocked). The ratio score helps: on the SuperNI tasks a third of the inputs reach their own task instead of an eighth, and five of the fifteen tasks are told apart almost completely (the three summarisation tasks, the dialogue task, the reading-comprehension task). It is not enough: the short-input tasks (the two sentiment tasks, emotion, relation extraction, the question tasks) lie inside older zones.

**Pilot B (trained answers, t5-small, five tasks, no definition in any input, CPU, `certified_switch/demo_no_instruction.py`).** One seed and a small model: a demonstration, not a result about scale.

| Rule | Score right after learning | Score at the end | Answers to earlier tasks changed (of 200) |
|---|---|---|---|
| all adapters on | 64.4 | 9.6 | 200 |
| zones, plain score | only the first task learned (98.1; the others 4.0, 18.0, 0.0, 0.5) | | none |
| zones, ratio score | 36.7 | 33.6 (forgetting 3.8) | 8 (4%) |

With the ratio score none of the 188 inputs inside their own zone changed, but only the first two tasks were learned (blocked test inputs: 0, 0, 23, 48 and 45 of 50). Without its definition the small model learned two of the five tasks only to the level of guessing even with all adapters on (48.0), so the instruction carries part of the task itself, not only its name.

**Honest readings.**

* The bound held in every run: changed shares of 0.0 to 1.5% against 5% promised, and 4% in Pilot B.
* The method does not work yet without instructions, with frozen word-vector features. The zones of early tasks swallow the inputs of later tasks. This is reported when the tasks arrive, so nothing is silent, but then those tasks are not learned.
* The first Pilot B run met badly chosen lines. By the lines as written it "worked", but the result was empty: 377 of 400 training inputs of the second task lay inside the zone of the first, so later tasks were never served by their own adapters and there was nothing to forget. The lines were corrected before the second run was read: "works" now also needs the tasks to be learned (at least 90% of the score with all adapters on, and at least 80% of the inputs given to their own task). By the corrected lines both runs count as "does not work".
* No GPU run was started, because no pilot worked.

**Not shown and not claimed:** that published methods only look good because the instruction gives away the task. That experiment (the published methods without instructions, and rewordings written by people) is the next step.

**Tested / not tested / only expected.**

* *Tested:* the bound on instance features of 30 real tasks (routes only, CPU), and trained answers on one small model, one seed, five tasks.
* *Not tested:* trained answers on a GPU; the published methods without instructions; rewordings written by people; the second score on tasks that have not been used; frozen features other than the word-vector average.
* *Only expected:* that a frozen feature that tells tasks apart without instructions would make the method work. The frozen model's own hidden states and a sentence encoder are the candidates; the sentence encoder was being computed on the same sequences when this was written.

#### The target for this phase, and the signs fixed before any GPU run of our method (2026-11-18, 06:10 UTC)

**The target set at 06:08 UTC:** instruction-free task recognition that works; full learning with no blocked tasks; a non-trivial forgetting guarantee; to beat the state of the art. Experiments and new ideas go on until there are signs of this, and a GPU run is started only when the method is thought to work. In four points:

1. Task recognition that works without any instruction.
2. All tasks are learned and none is blocked.
3. A real guarantee against forgetting.
4. Better than the published methods.

The papers sent at 03:56 UTC stay as the safe version. This is new work. Nothing in these subsections is a result on the standard published setting.

**The "signs" were written down before any GPU run of our method, so that they cannot be moved later:**

1. On routes only, with no instruction, on the 15 SuperNI tasks: at least 90% of the test inputs go to their own task, at most 10% are blocked by an older zone, and at most 5% of the routes change.
2. The same on tasks that no test has used (fresh tasks).
3. With trained answers on the small model: tasks learned (score right after learning at least 90% of the score with all adapters on), at most 20 of 200 answers changed, forgetting at most 3 points.
4. Only then a GPU run of our method.

"Beating the state of the art" can only be measured on a GPU, and only against the published methods run in the same setting without instructions. That baseline run was approved and is described below.

#### Two rules that look at the whole input: `boundary` and `claims` (sentence-encoder feature, development numbers only, 2026-11-18, 07:00 UTC)

**The idea.** Until now a zone meant "everything about as close to the task's inputs as the task's own outliers". That test knows what the task looks like and nothing about what else exists. Two new rules use more:

* `boundary`: a small classifier is trained when task k arrives. It tells the inputs of task k from the pool of unrelated inputs and from all earlier tasks. The zone is the task's side of that boundary, and close to the task's inputs. The oldest zone wins.
* `claims`: the same classifier, but the newest zone wins. A new zone may not hold any stored held-out input of an older task. This is what stops a newcomer from taking old inputs.

**Numbers** (`paper_certified/routing/boundary_study.json`, part 1; sentence-encoder feature, no instruction, averages over tasks; tested on CPU, **development data**: these sequences were used by the earlier scores):

| sequence | rule | given to own task after the last task | route changed after arrival | blocked / coverage at arrival |
|---|---|---|---|---|
| 15 SuperNI | distance to 10th neighbour (oldest wins) | 51.0% | 1.4% | 46.6% blocked |
| 15 SuperNI | ratio to unrelated inputs (oldest wins) | 58.0% | 1.8% | 39.9% blocked |
| 15 SuperNI | `boundary` (oldest wins) | 67.2% | 1.1% | 31.2% blocked |
| 15 SuperNI | `claims` (newest wins) | 87.7% | 1.3% | 81.2% of a newcomer's inputs inside its zone |
| 15 Long Sequence | `boundary` | 49.5% | 1.0% | 48.8% blocked |
| 15 Long Sequence | `claims` | 70.6% | 0.8% | 57.6% inside |
| all 30 | `claims` | 70.9% | 1.0% | 62.5% inside |

**How it was read.**

* `boundary`: the bound holds. The use "does not" work (31% blocked).
* `claims`: the guarantee is met (1.3% changed against a bound of 24.6%; the bound is loose because it counts every input outside a newcomer's zone as lost). The use is "partial" on the SuperNI tasks (87.7%; the line was 90%). On the Long Sequence tasks it does not reach 75%.
* Letting the newest zone win removes the blocking, as intended. Twelve of the fifteen SuperNI tasks have 92 to 100% of their inputs at their own adapters.
* What is left is recognition itself. The classifier alone, without zones, gives 92.4% (SuperNI) and 83.3% (Long Sequence) to the right task.
* The weak tasks share their kind of input with an older task: tweets (sentiment and emotion), sentence pairs (MNLI, CB, RTE), reviews (IMDB and SST-2), passage questions (BoolQ and MultiRC), and Quoref after the news task. For these the newcomer's zone is small (coverage 8 to 49%).

**Not finished when this was written (numbers came later, see the subsections on the zone shapes and the joint recogniser below).** A grid of four zone shapes inside `claims` (one flat boundary against everything, one flat boundary per older task, balls around the task's own inputs, curved boundaries; each with and without 15 simple numbers that describe how a text is written) is still being computed. So is a joint recogniser on other features (`paper_certified/audit_study.py`). The rule for choosing the winner was fixed before the run. Its verdict is read once on fresh tasks that nothing has read.

#### The feature that changed the picture: the frozen model's own hidden states (2026-11-18, 07:23 UTC)

**Feature.** The frozen T5-base encoder, with no adapter. For each instance, the mean of its last-layer states over the tokens (cut at 256). The definition is never read. At most 400 training inputs per task (200 held out), a pool of 2,000 unrelated inputs, level 0.05, the oldest zone wins. 15 SuperNI tasks of order 1.

**Result** (`paper_certified/routing/noinstruction_study.json`, keys `verdict_t5hidden` and `results_reduced`; tested on CPU, one seed, reduced data):

| feature, score | inside own zone | blocked | given to own task at the end | route changed |
|---|---|---|---|---|
| word vectors, ratio | 95.4% | 65.1% | 34.0% | 0.0% |
| sentence encoder, ratio | 96.1% | 40.5% | 58.3% | 0.9% |
| frozen T5-base encoder states, ratio | 94.0% | 10.5% | 87.9% | 1.1% |
| frozen T5-base encoder states, 10th neighbour | 94.9% | 64.7% | 34.2% | 1.0% |

**How it was read.** The rule was written before the run (05:43 UTC). The claim holds: 1.1% changed against a limit of 6.6%, and coverage 94.0% against a limit of 93.1%. The use is "partial": 87.9% is below the line of 90%, and 10.5% blocked is above the line of 10%, both narrowly. So by its own rule this is not "works", and the small trained model was not run on it.

**Our guess was wrong, in the good direction.** We expected 20 to 35% blocked and wrote that "works" would surprise us. The result came within two points of "works".

**Where it fails:** emotion (48% blocked by the other tweet task), evalution (46% by sst2), reddit (22% by quoref), sentiment140 (21%), sciq (13%). All other tasks are at 6% or below.

**Caveats:** one seed, reduced data, a second look at tasks already used, and only the SuperNI sequence was computed.

**What it says.** The feature matters more than the shape of the zone. The sentence encoder was built to ignore how a text is written and to keep what it is about. The model's own states keep both. How a text is written is most of what separates the inputs of one task from another.

#### The two published methods run WITHOUT task definitions (Kaggle T4, five tasks, one seed, 2026-11-18, 07:29 UTC)

**The question.** On these benchmarks the instruction names the task. Do the two published methods that we ran (InfLoRA and GainLoRA, official code) still hold their answers when the instruction is taken away? This covers these two methods and the reduced setting only.

**The run.** Exactly the five-task run of the paper (T5-base, 10 epochs per task, seed 42, same code and patches). One change: the definition of every task is replaced by the empty string in every data file, for training and for testing. The run took 22 minutes. Files: `silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/` (the table was copied from `log.txt` there and from the plan). The re-test counts the answers to the 400 test inputs of the first four tasks that differ at the end from the answer right after the task was learned.

| method | run | learned | score at end (AP) | forgetting (FT) | answers changed |
|---|---|---|---|---|---|
| InfLoRA | with definitions | 56.96 | 58.89 | -1.41 | 98 of 400 |
| InfLoRA | no definition | 51.46 | 37.45 | 18.01 | 217 of 400 |
| GainLoRA | with definitions | 60.85 | 61.31 | -0.32 | 17 of 400 |
| GainLoRA | no definition | 56.08 | 51.94 | 6.51 | 129 of 400 |

On the official test, with one reference answer per example, the no-definition rows are: InfLoRA 51.08 / 37.28 / 17.75 and GainLoRA 55.82 / 51.64 / 6.55 (learned / final / forgetting).

**How it was read** (the rule was written at 06:05 UTC, before the start):

* Was anything learned? Yes for both. InfLoRA 51.5 against 60.0 with instructions (86%); GainLoRA 56.1 against 61.4 (91%). The line was 60%.
* Both **hold their answers less well without instructions**. InfLoRA's forgetting rises by 21 points against its three-seed value with instructions, and 217 answers change against 123 per 400. GainLoRA's forgetting rises by 7.0 points, and 129 answers change against 22 per 400. The lines were: forgetting up by 5 points, or twice as many changed answers and at least 60 more.
* Our guess for GainLoRA was 60 to 150 changed answers. It is 129.

**What must be said with it:**

* One seed, five tasks, two methods, a small model, a tenth of the published budget. It is a first look and no more.
* Without definitions the two published methods also learn two of the five tasks poorly (sst2 and emotion stay near guessing). *Corrected later the same day:* the first version of this line said "every method". That was wrong. Plain adapters in our own run learn all five tasks almost as well without definitions as with them (see the subsection on the GPU run of our method).
* Only InfLoRA and GainLoRA were run. This is not a statement about other published methods.
* Our own method was not part of this run.

**What it means for the target.** In this setting without instructions the number to beat is a final score of 51.64 (official test, GainLoRA), with 6.5 points of forgetting and 129 of 400 answers changed.

#### Frozen model states under the rules that block nothing (development, 15 SuperNI tasks, 2026-11-18, 07:29 UTC)

**Rules** (written at 07:12 and 07:23 UTC, before the run), on the frozen T5-base states of the instance:

* Rule J: one classifier over all tasks learned so far (logistic regression), trained again at every arrival on the stored features.
* Rule L: the same idea as a ridge regression. Two sums per task are added up when the task arrives. No input of any task is stored.
* `claims` with r = 1, 2, 3, as above.

No task can be blocked by J or L: every task is a class.

**Numbers** (`paper_certified/routing/hidden_rules_study.json`, key `dev`; tested on CPU, development data, one split, reduced data):

| rule | own task right after arrival | own task after the last task | route changed | certificate (held-out) | 95% bound |
|---|---|---|---|---|---|
| J, logistic regression, C = 1 | 99.7% | 99.0% | 0.7% | 1.2% | 2.9% |
| L, ridge from added-up sums, 1 | 99.9% | 99.0% | 0.8% | 1.5% | 3.4% |
| L, 10 / 100 | 99.8 / 99.5% | 98.5 / 97.6% | 1.3 / 1.9% | 1.5 / 2.2% | 3.4 / 4.2% |
| `claims`, r = 1 / 2 / 3 | 98.9 / 99.0 / 99.2% | 98.2 / 97.7 / 97.6% | 0.7 / 1.3 / 1.6% | not valid | none |

**Chosen by the rule written before the run: L with ridge 1** (it ties with J at 99.03%, and the rule prefers L). It gives 99.0% of the test inputs to their own task, and 0.8% of routes change. It stores no input. Its weights after all 15 tasks are the same whatever the order of arrival (largest difference with the order reversed: 5e-17). The first sign (at least 90% on the 15 SuperNI tasks) is met. Our guess (95 to 99%) held.

**What the certificate is.** Each task keeps some held-out inputs that the recogniser never trains on. Later, count how many of them now go to a different task than when the task arrived. Because a future test input of that task looks the same to the recogniser as a held-out one, this count gives, on average, a measured limit on how many test inputs changed route.

**Weak spots.** The tweet task (sentiment) falls from 98% to 92% when the emotion task arrives (15 of its 200 held-out inputs change route; its certificate says 8.0%, and 6% of its test inputs did change). The dialogue task falls to 94%. Everything else stays at 99 to 100%. The certificate is above the measured share on average (1.5% against 0.8%).

**Caveats.** Development data, one split, reduced data. The verdict is the check on a fourth batch of fresh tasks, which was still being computed (see below).

#### Rule H and its correction H' (development, 2026-11-18, 07:34 and 07:35 UTC)

**Why.** Rule L blocks nothing, but its guarantee is measured after the fact. A guarantee known before the later tasks arrive is a stronger kind. Rule H puts this on the joint recogniser. A newcomer takes an input only if its margin over the best older task is above a threshold. The threshold is set so that the newcomer takes at most t held-out inputs of each older task. Otherwise the route stays as it was.

**The statement (written before any run).** The chance that a test input of an older task ever changes route is at most (number of later tasks) x (t + 1)/(m + 1), with m the number of held-out inputs.

**Rule H as written** (`routing/hidden_rules_study.json`, key `dev_rule_H`; ridge 1; 15 SuperNI tasks):

| t | own task at arrival | at the end | route changed | bound (average) |
|---|---|---|---|---|
| 0 | 95.1% | 94.7% | 0.5% | 5.8% |
| 1 | 98.7% | 97.7% | 1.3% | 11.6% |
| 2 | 99.7% | 97.3% | 2.4% | 17.4% |
| rule L, for comparison | 99.9% | 99.0% | 0.8% | (certificate 1.5%) |

**The flaw.** With t = 1 or 2 the rule changed more routes than rule L does with no protection at all. When no held-out input of an older task is near the newcomer, the threshold is negative, and then the newcomer takes inputs that even the joint recogniser gives to an older task. We had expected 97 to 99% at the end with t = 0 and got 94.7%. One task pays: quoref takes only 36% of its own inputs, because one held-out input of the news task looks like a quoref passage.

**The fix (H').** The threshold is never below zero. The statement and its proof stay the same (the threshold only became larger).

| t | own task at arrival | at the end | route changed | bound (average) |
|---|---|---|---|---|
| 0 | 95.1% | 95.0% | 0.07% | 5.8% |
| 1 | 98.7% | 98.2% | 0.5% | 11.6% |
| 2 | 99.7% | 99.2% | 0.5% | 17.4% |
| 5 | 99.8% | 99.3% | 0.5% | 34.7% |
| no protection (the cascade alone) | 99.9% | 99.1% | 0.8% | none |

With t = 2, H' gives 99.2% to the own task (rule L: 99.0%) and changes 0.5% of routes (rule L: 0.8%). The tweet task keeps 97% when the emotion task arrives.

**What must be said with it.**

* H' was written after seeing the development numbers of H, on the same 15 tasks. These are development numbers and no more.
* The bound known in advance is loose. For a task with 200 held-out inputs and 14 later tasks it is 3 x 14/201 = 21%. For the three small tasks (50 to 64 held-out inputs) it says nothing. What actually changed is 0.5%.
* A tighter bound in advance needs more held-out inputs. The certificate of rule L stays the tight statement.
* t = 2 was fixed as the smallest t within 0.2 points of the best on the development tasks.

#### A theory note: the price of having no instruction (2026-11-18, 07:20 UTC)

No test is attached to this note. It is a statement with a short proof, written down so that it can be checked before it goes into any paper.

Take two tasks. Where their inputs overlap, a system that does not read an instruction must give one answer to the same text, so it cannot be right for both tasks. The note says that the forgetting of the older task plus what is not learned of the newer task is at least the overlap mass times how much the two best answers conflict. The proof is three lines (at each input, the loss is at least the smaller of the two input probabilities times the conflict). If every input starts with its own task's instruction, no text belongs to two tasks, the bound is zero, and nothing is forced. That is the exact sense in which the instruction gives the task away. A switch with one frozen module per task reaches the bound. How the unavoidable mass is split between the old and the new task is a choice: "oldest zone wins" puts all of it on the newer task, and `claims` puts at most (r + 1)/(m + 1) per later task on the older one. That task recognition limits learning without task labels is already known. Only the form may be new: a bound on forgetting plus not-learning that is zero with instructions and positive without. The plan says this is not claimed as deep.

#### The joint recogniser with the audit, on sentence-encoder features (2026-11-18, 07:39 UTC)

Files: `paper_certified/audit_study.py`, `paper_certified/routing/audit_study.json`. The rule and the lines were fixed at 07:12 UTC. This test does not use the frozen model's states.

**Development** (15 SuperNI and 15 Long Sequence tasks; share of test inputs given to their own task after the last task; tested on CPU, development data):

| recogniser, feature | SuperNI | Long Sequence | changed (SuperNI / Long) | audit (SuperNI / Long) |
|---|---|---|---|---|
| logistic regression, sentence encoder, C = 10 | 96.5% | 90.4% | 2.0 / 5.3% | 3.7 / 6.4% |
| logistic regression, + 16 shape numbers, C = 10 | 99.3% | 95.4% | 0.6 / 2.8% | 2.2 / 2.8% |
| logistic regression, + character n-grams, C = 10 | 98.4% | 95.6% | 1.1 / 2.3% | 2.5 / 2.7% |
| sums of random cosine features, sentence encoder, 0.1 | 94.9% | 86.3% | 3.8 / 5.9% | 6.3 / 6.1% |
| sums of random cosine features, + shape numbers, 0.01 | 98.6% | 92.2% | 1.4 / 4.8% | 3.0 / 5.1% |

Chosen by the rule written before the run: logistic regression on the sentence encoder plus the shape numbers, C = 10. (The plan said 15 shape numbers and listed 16; all 16 were used.) The form without stored inputs gives the same weights for every order of arrival (largest difference with the order reversed: 3e-12).

**Verdict, read once, on the third batch of 30 fresh tasks:** 96.4% of the test inputs go to their own task right after arrival and **93.8%** after the last task. 3.0% of the routes changed. The average audit number is 3.8%. By the lines of 07:12 UTC: use **"works"** (both shares above 85%), audit **"met"**. The plain sentence encoder alone gives 89.5%, with 5.1% changed.

**Where it fails.** Two of the 30 fresh tasks are built on the same texts (answers for a question, and a question for an answer, from one dataset). They get 45% and 49%: each input of the pair goes to one of the two tasks about as often as to the other. Nothing can do better without the instruction. The other 28 tasks are fine. We did not expect a handful of shape numbers to do as well as character n-grams; they do. How an input is written tells the tasks apart.

#### The four zone shapes, and the chosen one on the second fresh batch (2026-11-18, 08:13 UTC)

Files: `paper_certified/routing/boundary_study.json`, keys `dev_grid`, `chosen`, `fresh2`, `verdict_fresh2`. Rule `claims` (newest zone wins; no held-out input of an older task may lie in a new zone). 44 variants on the 15 SuperNI tasks. The Long Sequence part of the grid was cut for CPU time before any grid number was read.

**Development, share given to the own task after the last task** (best penalty; development data):

| shape of the zone | sentence encoder alone | with the 16 shape numbers (best weight) |
|---|---|---|
| one flat boundary against everything | 87.8% | 93.2% |
| one flat boundary per older task | 92.0% | 98.7% |
| balls around the task's own inputs | 87.2% | 77.2% (worse with the numbers) |
| curved boundaries (random cosine waves) | 90.7% | 98.3% |

Chosen by the rule written before the grid: one flat boundary per older task, shape numbers with weight 1, penalty 10, r = 1 (98.7% own task, 0.3% of routes changed, bound known in advance 11.9% on average).

**Verdict, read once, on the second batch of 30 fresh tasks:** 94.1% right after arrival, **93.6%** after the last task, 0.57% of routes changed, bound known in advance 17.7% on average. By the lines fixed before: use **"works"**, guarantee **"met"**. The first form of `claims` (one boundary against everything, no shape numbers) gives 80.2% on the same batch. One weak task (49%), whose misrouted inputs mostly go to one older task.

**What the grid says.** The shape matters (one boundary per older task instead of one against everything: +4 to +6 points). It matters less than the feature (the shape numbers: +5 to +7 points for three of the four shapes). Balls are the worst. The curved boundary is not better than the flat one. No number was written down for the shapes, only the choice rule, so there is no guess to check.

**All fresh-task checks of the day** (each rule chosen on the 15 benchmark tasks, read once on a batch nothing else had used; the batches differ, so the rows are not a comparison of the rules):

| rule | feature | fresh batch | own task after the last task | changed | verdict |
|---|---|---|---|---|---|
| zone, oldest wins (`boundary`) | sentence encoder | 1 | 54.4% | 1.1% | does not |
| zone, newest wins (`claims`) | sentence encoder | 1 | 72.2% | 0.4% | partial |
| zone, newest wins, one boundary per older task | + shape numbers | 2 | 93.6% | 0.6% | works |
| joint recogniser with the audit | + shape numbers | 3 | 93.8% | 3.0% | works |
| joint recogniser (rule L) and rule H' | frozen model's states | 4 | not finished | | |

On batch 1 the classifier of these rules alone, with no zone, gives 83.3% of the test inputs to their own task. So the order of what mattered today stays: first the feature, then one recogniser over all tasks instead of one zone per task, and only then the shape of a zone.

#### Texts that occur in two tasks: a count, not a test (2026-11-18, 07:51 UTC)

Files: `paper_certified/conflict_count.py`, `paper_certified/routing/conflict_count.json`. For the theory note, the part of the conflict that can be counted directly is the share of inputs that occur word for word in two tasks.

| collection | tasks | inputs | inputs that also occur in another task |
|---|---|---|---|
| 15 SuperNI tasks of the benchmark (training and test) | 15 | 10,588 | 0 |
| 15 Long Sequence tasks of the benchmark (training and test) | 15 | 115,381 | 0 |
| fresh batch 1 | 30 | 30,000 | 0 |
| fresh batch 2 | 30 | 30,000 | 0 |
| fresh batch 3 | 30 | 30,000 | 586 (2.0%), all in one pair of tasks |

On the two standard benchmarks no input belongs to two tasks. So nothing forces a method to forget there when the instruction is removed. What the two published methods lost without instructions is not a price that had to be paid. In fresh batch 3 one pair of tasks is built on the same dataset (29% of the inputs of each occur word for word in the other). It is the pair that gets 45% and 49%. The count is a lower limit of the conflict: tasks can also overlap without sharing exact texts. The fourth batch was not opened.

#### An independent check of the three new statements (2026-11-18, 08:02 UTC)

A second, stronger model read the three statements and the code. Nothing was run. The draft is `paper_certified/sec_audit.tex`.

* **The bound known in advance (rule H')** is correct as stated and as coded. One condition was too loose and is now exact: the later tasks and their fit inputs must be independent of the held-out inputs and of the test input (for example fixed in advance).
* **The audit** was stated more weakly than it is. By exchangeability, the share of held-out inputs whose route changed is an unbiased estimate of the share of the task's inputs whose route changed. The number (c + 1)/(m + 1), which the plan called "the certificate", is a slightly larger number with no extra meaning. The real guarantee is the confidence bound (one-sided 95%, in the code as `upper95`). The paper will report the estimate c/m and that bound. The lines fixed in advance stay as they were read. Each bound is for one task and one pair of times; for all tasks and all arrivals at once they do not hold at the same confidence.
* **The price of having no instruction** is true. The wording was tightened. "A method with shared weights pays the same sum" was too strong. It now reads: every method is bound by the same inequality, but does not say how the sum is split.
* **The code matches the text.**
* **A weakness of the fresh-task checks, found and not fixed:** in the fresh batches the test inputs are the last 200 instances of the part of the file that was read, while the held-out inputs are a random part of the first 800. If a file is ordered in some way, the two are not the same kind of draw. Then the statements about change (not the shares given to the own task) rest on an assumption that is not ensured by construction. This applies to the verdicts on batches 1, 3 and 4. They stay as fixed, with this caveat. The GPU feature job gets a second version of the fresh sequences with one random split of all 1,000 instances, so the size of the effect can be seen.

#### An idea for a later GPU run: with instructions, route on the example text alone (run U). NOT yet run (2026-11-18, 07:42 UTC)

**The idea.** The zones of the paper read the instruction, and that is where they are weak: a reworded instruction can leave the zone. The recogniser never reads the instruction. So in the normal setting, with instructions, it can set the route from the example text alone, while the adapters still read the whole input. Then no rewording of the instruction can change a route. What is given up: exactly zero changed answers on unedited inputs, which only the zones give. It assumes that the system can find where the example begins in the input (the benchmark gives instruction and example as two fields).

**The run (run U).** Five tasks, seed 42, T5-base, 10 epochs, with the definitions, plain adapters, the same route table as the run without instructions. Re-test at four wordings of the definition, rules "ridge" and "handicap". The numbers of the same seed from the run of 2026-11-17 (tested):

| seed 42, with definitions | original | light | medium | heavy |
|---|---|---|---|---|
| certified switch of the paper: final score | 79.33 | 79.15 | 77.61 | 72.12 |
| certified switch: answers changed of 400 / right task | 0 / 100% | 0 / 100% | 0 / 100% | 0 / 93.8% |
| GainLoRA: final score / answers changed of 400 | 61.31 / 17 | 59.49 / 15 | 58.61 / 32 | 52.01 / 49 |
| InfLoRA: final score / answers changed of 400 | 58.89 / 98 | 59.87 / 85 | 60.45 / 91 | 56.37 / 151 |
| every input answered by its own task's adapters (ceiling) | 79.33 | 79.15 | 77.61 | 76.72 |

**Reading, fixed before any run.** "Repairs the weak spot" if, for rule "ridge": at the heavy wording the final score is at least 74.1, at the original wording at least 77.3, and at every wording at most 8 of the 400 compared answers change. "Does not" if the heavy score is below 72.12 or the original score below 75.3. Anything else: mixed. What we expect (not a result): original 78.5 to 79.3 with about 6 changed answers, heavy 75.5 to 76.7. The three rewordings were written by machine, not by people. The notebook (`silent_adapters/five_tasks_instance_routes_with_definitions.ipynb`) was prepared here. *Update:* it was run later the same day; the result is in the subsection "Run U" below.

#### THE MAIN RESULT OF THE DAY: the GPU run of our method without instructions (Kaggle T4, five tasks, one seed, 2026-11-18, 08:15 UTC)

**The run.** 20 minutes of training and re-tests on a Kaggle T4. Five tasks, seed 42, T5-base, 10 epochs. Every definition replaced by the empty string. Plain adapters, nothing blocked. Routes from the table written and pushed before the run (`silent_adapters/noinstruction_routes.json`). Files: `silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks/`. Re-test scores, with every reference answer; the same scoring for all rows.

| no definition anywhere | learned | final score (AP) | forgetting (FT) | answers changed of 400 | right task |
|---|---|---|---|---|---|
| ours: the recogniser (rule L) | 77.70 | 77.50 | 0.50 | 6 | 98.4% |
| ours: the recogniser with the handicap (rule H') | 77.70 | 77.90 | 0.00 | 1 | 99.4% |
| every input to its own task (the ceiling for any route) | 78.10 | 78.10 | 0.00 | 0 | 100% |
| rule J (logistic regression) / zones, newest wins | 77.70 / 77.10 | 77.50 / 76.70 | 0.50 / 0.50 | 6 / 5 | 98.6 / 97.6% |
| nearest centre of word vectors (rule "centre") | 76.30 | 74.74 | 1.94 | 11 | 92.8% |
| plain adapters, all on | 78.10 | 14.18 | 79.90 | 400 | 20% |
| GainLoRA (published), same setting | 56.08 | 51.94 | 6.51 | 129 | |
| InfLoRA (published), same setting | 51.46 | 37.45 | 18.01 | 217 | |
| with definitions (run of 2026-11-17): certified switch / GainLoRA / InfLoRA | 79.33 / 60.85 / 56.96 | 79.33 / 61.31 / 58.89 | 0.00 / -0.32 / -1.41 | 0 / 17 / 98 | |

**Verdict by the rule of 07:29 UTC: "beats the published methods in this setting".** For rule L all three lines hold: the final score is 25.6 points above the better published method without definitions (the line was 2 points); forgetting is 0.50 (line: below 2); 6 of 400 answers changed (line: at most 32). Rule H', read by the same lines: 26.0 points above, forgetting 0.00, 1 answer changed.

**Caveats, as the plan states them:**

* **A wording slip in the rule.** The rule said "AP, official test". But the official test of this run scores the plain adapters with all adapters on (14.17); the routed rules exist only in the re-test. So the comparison is re-test against re-test (GainLoRA 51.94). The official-test and re-test numbers of the published methods differ by 0.3 points (51.64 against 51.94), which changes nothing here.
* **Our guess for the learned score was far too low.** We expected 54 to 58; it is 77.7. We had assumed that without definitions our adapters would learn as poorly as the published methods did. They do not: plain adapters learn the five tasks almost as well without definitions as with them (78.1 against 79.3), sst2 at 85 and emotion at 70, where the two published methods reach 43 to 48 and 12 to 28 in this budget. Our guesses for changed answers (5 to 15; got 6) and forgetting (below 1; got 0.50) held.
* *Update, 12:24 UTC:* the next line belongs to the 10-epoch comparison. At the published 100 epochs the gap is forgetting (see the subsection "Run D" below).
* **Most of the 25 points is learning, not retention.** The published methods restrict what a new adapter may change in order to protect old tasks, and with a tenth of their training budget they learn less. The part that is about forgetting is the last two columns (0.5 against 6.5 and 18.0 points; 6 against 129 and 217 changed answers). That part does not depend on how much was learned.
* **The published methods might close the learning gap with their full number of epochs.**
* One seed, five tasks, a small model, 10 epochs. This is not the published setting.
* *Update, 09:02 UTC:* seeds 43 and 44 were run later for ours and for the two published methods. The verdict stands for three seeds (see the subsection "Three seeds without instructions" below).
* A sentence of the earlier subsection was wrong as written ("every method learns two of the five tasks poorly"). It is the two published methods that do. It is corrected there.

**The audit next to what happened** (rule L, 200 held-out inputs per task): the estimate says 0%, 0%, 0% and 4.0% of the routes of the first four tasks changed (95% bounds 1.5%, 1.5%, 1.5% and 7.1%). On the test inputs 0, 1, 0 and 5 of 100 routes changed. The 6 changed routes are the 6 changed answers.

**What this is.** It is the first GPU evidence for all four points of the target at once, in this small setting: the task is recognised without an instruction (98.4% and 99.4% of test inputs), no task is blocked, the guarantees were met, and the two published methods are beaten in this setting, also when they are given their definitions (61.3 and 58.9) and ours is not.

#### Why the order of the last steps was changed (2026-11-18, 07:45 UTC)

At 06:10 and 07:29 UTC we wrote that the GPU run of our method would start only after the check on fresh tasks with the frozen model's states. We changed this, and wrote it down before doing it. That check needed an hour more on this machine, and the run costs about half an hour of the roughly 15 GPU hours left this week. The gate was there so as not to spend GPU hours on a recogniser that only works on the tasks it was chosen on; since then the joint recogniser had passed on fresh tasks with a weaker feature (93.8%) and gave the same 99.0% for every order of the 15 tasks (development, `routing/hidden_rules_study.json`, key `dev_other_orders`). All readings were already fixed and do not depend on each other, so the order of reading cannot bend any of them, and all would be reported whatever they said.

#### Three seeds without instructions (Kaggle T4, five tasks, 10 epochs, 2026-11-18, 08:40 and 09:02 UTC)

Files: `silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_seeds43_44/` and `..._ours_five_tasks_seeds43_44/`, with the two runs of seed 42. Re-test, no definition anywhere. Each cell is: learned / final score / forgetting / answers changed of 400.

| method | seed 42 | seed 43 | seed 44 | mean of three; changed of 1,200 |
|---|---|---|---|---|
| InfLoRA | 51.46 / 37.45 / 18.01 / 217 | 48.89 / 36.07 / 16.27 / 278 | 54.34 / 43.23 / 14.63 / 251 | 51.56 / 38.92 / 16.31 / 746 |
| GainLoRA | 56.08 / 51.94 / 6.51 / 129 | 50.88 / 48.67 / 4.27 / 147 | 56.53 / 52.45 / 6.10 / 138 | 54.50 / 51.02 / 5.63 / 414 |
| ours, rule L | 77.70 / 77.50 / 0.50 / 6 | 77.43 / 76.83 / 0.75 / 6 | 77.73 / 77.33 / 0.50 / 6 | 77.62 / 77.22 / 0.58 / 18 |
| ours, rule H' | 77.70 / 77.90 / 0.00 / 1 | 77.43 / 77.43 / 0.00 / 1 | 77.73 / 77.73 / 0.00 / 1 | 77.62 / 77.68 / 0.00 / 3 |
| nearest centre of word vectors | 76.30 / 74.74 / 1.94 / 11 | 75.63 / 73.88 / 2.18 / 11 | 76.53 / 74.99 / 1.93 / 11 | 76.15 / 74.54 / 2.02 / 33 |
| with definitions, three seeds (paper): certified switch / GainLoRA / InfLoRA | | | | final 79.7 / 62.0 / 62.9; changed 0 / 65 / 370 |

Other columns for ours: plain adapters with all on end at 14.18, 15.35 and 14.78 (mean 14.77); every input to its own task ends at 78.10, 77.83 and 78.13 (mean 78.02).

**Reading (the rule was fixed at 08:16 UTC).**

1. Ours against the published methods: the three lines of 07:29 UTC hold for the means. The final score is 26.2 points above the better published method. Forgetting is 0.58. 18 answers change, against a quarter of GainLoRA's 414, which is 103. Rule L is above the better published method in each of the three seeds. So **"beats the published methods in this setting" stands for three seeds**, for rule L and for rule H'.
2. The published methods against themselves with definitions: InfLoRA's forgetting rises from -3.1 to 16.3 points and its changed answers from 370 to 746 of 1,200. GainLoRA's forgetting rises from -0.5 to 5.6 points and its changed answers from 65 to 414. **Both hold their answers less well without instructions, in every seed.**
3. Our guesses of 08:16 UTC (76 to 79 per seed; 4 to 9 changed answers for rule L; 0 to 3 for rule H') held.

**The same things must be said as for one seed:** most of the gap in the final score is learning, not retention; the published methods had a tenth of their published epochs; two methods only; one sequence of five tasks; the five tasks are among the 15 on which the rule was chosen.

**Two more observations.** The three seeds of ours differ very little. That is expected: the routes are the same in every seed, so only how well the adapters were trained varies. The seeds test the adapters and not the rule. And the simple rule "nearest centre of word vectors" already ends at 74.5 with 33 changed answers. Any rule that recognises the task goes most of the way in this setting. Rules L and H' add the last three points, fewer changed answers, and the statements about them.

#### The recogniser on the fourth fresh batch, and all fresh-batch checks (2026-11-18, 08:46 UTC)

Read once (`paper_certified/routing/hidden_rules_study.json`, key `verdict_fresh4`; lines of 07:23 and 07:34 UTC). Rule L on the frozen T5-base states, 30 fresh tasks.

| rule | own task at arrival | own task after the last task | route changed (ever) | audit c/m, (c+1)/(m+1), 95% bound | verdict |
|---|---|---|---|---|---|
| rule L | 97.5% | 94.6% | 3.4% (3.8%) | 3.3%, 3.7%, 5.3% | use "works"; audit "met" |
| rule H' (t = 2), next to it | 93.9% | 93.6% | 0.5% (0.5%) | bound in advance 21.6% | "keeps the recognition"; "bound met" |
| rule J (logistic regression), for comparison | 97.4% | 94.5% | 3.2% | | |

**Read against our guess.** We expected 80 to 90% "because random task collections contain pairs of tasks built on the same texts". The share is higher than we guessed. The pairs are there as guessed. Two tasks are built on the same reading passages (an answer for a question, and a question for the passage). The older one falls from 100% to 53% when the newer one arrives, and the newer one gets 52%. 47% of the older one's routes change, and its audit says so (102 of its 200 held-out inputs changed). Two tasks on the same kind of product reviews end at 68% and 84%. The other 26 tasks are at 95 to 100%. Under rule H' the newer task of the first pair is kept out instead (it takes few inputs) and the older one keeps its routes: 0.5% changed on average against 3.4%. The price moves from the older task to the newer one, as the theory note says it must. The caveat of the independent check applies: the test inputs are the last 200 instances of a file, the held-out inputs a random part of the first 800.

**All fresh-batch checks, now all read** (each rule chosen on the 15 benchmark tasks, read once on a batch nothing else had used; the batches differ, so the rows are not a comparison of the rules):

| rule | feature | fresh batch | own task after the last task | changed | verdict |
|---|---|---|---|---|---|
| zone, oldest wins (`boundary`) | sentence encoder | 1 | 54.4% | 1.1% | does not |
| zone, newest wins (`claims`) | sentence encoder | 1 | 72.2% | 0.4% | partial |
| zone, newest wins, one boundary per older task | + shape numbers | 2 | 93.6% | 0.6% | works |
| joint recogniser with the audit | + shape numbers | 3 | 93.8% | 3.0% | works |
| joint recogniser (rule L) and rule H' | frozen model's states | 4 | 94.6% (H': 93.6%) | 3.4% (H': 0.5%) | works |

#### The small trained model on the CPU, with the routes of the recogniser (2026-11-18, 08:46 UTC)

File: `certified_switch/demo_results/no_instruction_t5-small_seed0_ridge.json`. t5-small, five tasks, no definition anywhere, 400 training and 50 test inputs per task, 5 epochs, one seed. Routes by rule L on the frozen t5-small states. The lines are the corrected lines of Pilot B (03:13 UTC).

| | learned | at the end | forgetting | answers changed of 200 | own task at the end |
|---|---|---|---|---|---|
| all adapters on | 64.35 | 9.60 | 68.44 | 200 | |
| routes of rule L | 63.95 | 62.75 | 1.50 | 6 | 97.2% |

**Verdict "works", both parts.** (a) 6 of 200 answers changed (line: at most 20); forgetting 1.5 points (at most 3); 200 changed with all adapters on (more than 100). (b) The learned score with the routes is 99% of that with all adapters on (line: 90%), and 97.2% of the test inputs go to their own task (line: 80%). Two of the five tasks are learned poorly by this small model with either rule (48.0 on both), so they add nothing to the comparison. The audit per task: 0, 1, 0 and 6 of 200 held-out inputs changed; on the 50 test inputs per task 0, 2, 0 and 4 routes changed, and each changed route changed its answer. By the order fixed at 06:10 UTC this check was to come before the GPU run. It came after it and agrees with it. (Pilot B above, with zones, did not work; this is a different rule.)

#### The feature job on the GPU: the second benchmark and up to 120 tasks (Kaggle T4, about 25 minutes, no training; 2026-11-18, 09:04 UTC)

Files: `silent_adapters/results/2026-11-18_kaggle_hidden_states_scale/`. Frozen T5-base states of the instance text; the rules are exactly as on the CPU (lines of 07:48 and 08:01 UTC). Checks inside the job: all 90 downloaded fresh tasks have the same texts as the CPU files (fingerprints), and the 15 SuperNI tasks reproduce the CPU numbers (rule L 99.03% and 0.84% changed; rule H' 99.23% and 0.50%; rule J 99.03%).

| sequence | tasks | rule L: own task at arrival / at the end | changed | audit (c+1)/(m+1); 95% bound | rule H' (t = 2): at the end / changed | rule J at the end |
|---|---|---|---|---|---|---|
| SuperNI order 1 | 15 | 99.9 / 99.0% | 0.8% | 1.5%; 3.4% | 99.2% / 0.5% | 99.0% |
| SuperNI order 2 | 15 | 99.7 / 99.0% | 0.9% | 1.5%; 3.3% | 99.4% / 0.1% | 99.0% |
| Long Sequence order 3 (all official test inputs) | 15 | 99.3 / 97.8% | 1.6% | 2.7%; 4.4% | 96.7% / 0.4% | 98.0% |
| Long Sequence order 4 | 15 | 99.0 / 97.8% | 1.6% | 2.6%; 4.3% | 96.3% / 0.3% | 98.0% |
| fresh batch 1 | 30 | 99.2 / 98.2% | 1.4% | 1.8%; 3.2% | 97.5% / 0.5% | 98.8% |
| fresh batches 1+2 | 60 | 97.5 / 93.7% | 4.6% | 5.2%; 7.1% | 91.8% / 0.7% | |
| fresh batches 1+2+3 | 90 | 96.4 / 91.7% | 5.4% | 6.2%; 8.3% | 89.3% / 0.9% | |
| everything (15 + 15 + 90) | 120 | 95.6 / 90.5% | 6.0% | 6.7%; 9.0% | 88.2% / 1.1% | |

**Read against the lines.**

* On the second benchmark (15 Long Sequence tasks), rule L **"works"**: 97.8% (the line was 90%), audit **"met"**, for both official orders. The share at the end is the same for both orders, as it must be for this rule.
* **The weak tasks are the ones we named.** Of the three entailment tasks, mnli falls from 100% to 84% when rte arrives (its audit says so: 38 of its 200 held-out inputs changed route) and rte gets 90%. cb is not weak (100%).
* **What the handicap does to them:** with rule H' the older task is protected and the newer one pays. mnli keeps 96.5%, rte takes only 59% of its own inputs.
* **Many tasks** (a description, as said in advance): the share given to the own task falls slowly with the number of tasks: 99.0% with 15, 98.2% with 30, 93.7% with 60, 91.7% with 90, 90.5% with 120 (rule L). The changed routes rise from 0.8% to 6.0% and the audit follows them (6.7%). Rule H' changes about 1% of the routes at every size and gives 2 to 3 points fewer inputs to their own task. Its bound known in advance says nothing at this size (above 100% on average for 120 tasks with 200 held-out inputs).
* **Our guesses held** (Long Sequence 94 to 98%; 120 tasks 90 to 95%, falling slowly).

**The random-split check.** The earlier checks of fresh tasks used "the last 200 instances as test inputs", which does not ensure that test and held-out inputs are the same kind of draw. With one random split of all 1,000 instances instead, the numbers move by 0.4 points or less (own task at the end: at most 0.25 points for rule L and 0.41 for rule H'; changed routes -0.2 to +0.1 points). So the split mattered little on these tasks. (Two slips were corrected in the plan at 09:16 UTC: this line first said "0.2 points or less", and the 120-task value is 90.55%.)

**A second look at an old result.** On the first fresh batch the two zone rules on the sentence encoder gave 54.4% and 72.2%. On the same 30 tasks, rule L on the frozen model's states gives 98.2%. That batch was used before, so this is a description, not a verdict. The verdict for this rule is the fourth batch (94.6%).

#### Run U: with instructions, routes from the example alone (Kaggle T4, five tasks, one seed, 2026-11-18, 09:52 UTC)

Files: `silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions/`. 36 minutes of training and re-tests; seed 42, T5-base, 10 epochs, with the definitions; plain adapters; the same route table as in the runs without instructions. Re-test at the four wordings of the definition. Each cell is: final score / answers changed of 400 / share given to the own task.

| rule | original | light | medium | heavy |
|---|---|---|---|---|
| certified switch (zones on the instruction), this run | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 77.61 / 0 / 100% | 72.12 / 0 / 93.8% |
| recogniser, routes from the example alone (rule L) | 78.73 / 3 / 98.4% | 78.35 / 3 / 98.4% | 76.81 / 3 / 98.4% | 76.12 / 3 / 98.4% |
| the same with the handicap (rule H') | 79.13 / 1 / 99.4% | 78.75 / 1 / 99.4% | 77.21 / 1 / 99.4% | 76.52 / 1 / 99.4% |
| every input to its own task (ceiling) | 79.33 | 79.15 | 77.61 | 76.72 |
| zone first, else rule L | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 76.81 / 3 / 98.4% | 76.12 / 3 / 98.4% |
| zone first, else rule H' | 79.33 / 0 / 100% | 79.15 / 0 / 100% | 77.21 / 1 / 99.4% | 76.52 / 1 / 99.4% |
| nearest centre of word vectors of the whole input | 79.33 / 0 | 79.15 / 0 | 77.21 / 0 | 58.91 / 58 / 72.2% |
| GainLoRA, same seed (run of 2026-11-17) | 61.31 / 17 | 59.49 / 15 | 58.61 / 32 | 52.01 / 49 |
| InfLoRA, same seed | 58.89 / 98 | 59.87 / 85 | 60.45 / 91 | 56.37 / 151 |

The certified switch's numbers are the same as in the run of 2026-11-17 to the last digit, so plain training without blocking gave the same adapters here, as expected.

**The line fixed before the run (07:42 UTC):** "repairs the weak spot" if, for rule L, the final score at the heavy wording is at least 74.1, at the original wording at least 77.3, and at most 8 of the 400 compared answers change at every wording. "Does not" if the heavy score is below 72.12 or the original below 75.3.

**Verdict: "repairs the weak spot".** Rule L: 76.12 at the heavy wording, 78.73 at the original, 3 changed answers at every wording. Rule H': 76.52, 79.13 and 1.

**Guesses.** They held (original 78.5 to 79.3; heavy 75.5 to 76.7; a gain of about 4 points at the heavy wording and a loss of up to 1 point at the original), except one, in the good direction: we expected about 6 changed answers for rule L and there are 3. Six routes change, as in the run without definitions. But with the definition in front of it, the model gives the same answer for three of those inputs whichever of the two tasks' adapters answer. **So changed routes and changed answers are not the same count; both must be counted.**

**What it shows.** The routes of the recogniser are the same at every wording, because it never reads the instruction. The score stays close to the ceiling at every wording (rule L 0.6 to 0.8 points below it, rule H' 0.2 to 0.4 points; corrected at 10:00 UTC, the first wording was "within 0.2 to 0.6 points", which was wrong for rule L). What is lost under rewording is only what the adapters themselves lose when their instruction is reworded (79.3 to 76.7), not misrouting. The zones are exact where the instruction is unedited (0 changed answers, where the recogniser changes 1 to 3 of 400), and they lose 4.6 points at the heavy wording through misrouted inputs (76.72 against 72.12). The mixed rule (the zone where an input is certified, the recogniser elsewhere) has the best of both at the original, light and heavy wordings. At the medium wording it is 0.4 to 0.8 points below the certified switch, whose own fallback happens to be right for all inputs there.

**Caveats.** One seed, five tasks, three rewordings written by machine and not by people: a first look. It repairs on the benchmark what five attempts to widen the zones did not. A system that receives one string must find where the example begins; that is assumed here (the benchmark gives instruction and example as two fields).

#### Run D: the two published methods without definitions at their published 100 epochs (Kaggle T4 x2, 128 minutes, 2026-11-18, read at 12:24 UTC)

**The run.** Both methods at the same time, five tasks, seed 42, T5-base, no definitions, 100 epochs per task. That is the published number and ten times what all earlier runs here had. Files: `silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_e100/`. Re-test, every reference answer; the official test (one random reference) in brackets.

| five tasks, no definitions, seed 42 | score right after learning | final score | forgetting | answers changed of 400 |
|---|---|---|---|---|
| InfLoRA, 100 epochs | 78.37 (77.72) | 25.37 (25.13) | 66.25 (65.74) | 382 |
| GainLoRA, 100 epochs | 76.29 (75.66) | 56.41 (55.88) | 25.35 (25.22) | 178 |
| InfLoRA, 10 epochs (earlier run) | 51.46 | 37.45 | 18.01 | 217 |
| GainLoRA, 10 epochs | 56.08 | 51.94 | 6.51 | 129 |
| ours, 10 epochs: plain adapters behind the recogniser (rule L) | 77.70 | 77.50 | 0.50 | 6 |
| ours, 10 epochs: with the handicap (rule H') | 77.70 | 77.90 | 0.00 | 1 |

**Per task** (official test, right after learning, then at the end). InfLoRA: 99.2 then 11.0; 83.0 then 3.0; 62.4 then 36.6; 76.0 then 7.0; 68.0 then 68.0. GainLoRA: 99.7 then 65.8; 72.0 then 31.0; 62.6 then 60.6; 74.0 then 52.0; 70.0 then 70.0.

**Read against the rule of 08:21 UTC.**

1. **Learning: "they close the learning gap".** Both reach at least 73 right after learning (78.37 and 76.29; ours has 77.70 with 10 epochs). With their full number of epochs they learn sst2 (83 and 72) and emotion (68 and 70), the two tasks they had learned poorly with 10 epochs.
2. **Retention: "hold their answers less well than ours".** Forgetting is 66.25 and 25.35 points against 0.50 (the line was 5 points above ours). 382 and 178 of 400 answers change against 6 (the line was 60 more).
3. **The headline number:** ours with 10 epochs ends at 77.50. They end at 56.41 (GainLoRA) and 25.37 (InfLoRA) with 100 epochs.

**Our guesses.**

* Learning 68 to 78: right for GainLoRA; InfLoRA is just above it (78.37).
* Changed answers 60 to 200: right for GainLoRA (178), far too low for InfLoRA (382).
* Forgetting 4 to 15 points: wrong for both. It is 25 and 66.
* Our doubt ("a method that has learned more may also separate the tasks better") did not come true. They learned more and lost more.

All our guesses were wrong in the direction of too little forgetting.

**What this changes.** The sentence "most of the gap in the final score is learning, not retention" belongs to the 10-epoch comparison. With their published number of epochs the two methods learn the five tasks as well as plain adapters do, and the gap in the final score (77.5 against 56.4 and 25.4) is forgetting. Forgetting grew with what was learned: from 10 to 100 epochs it went from 18.0 to 66.3 points (InfLoRA) and from 6.5 to 25.4 (GainLoRA), and the changed answers from 217 to 382 and from 129 to 178.

**What it does not show.**

* One seed, five tasks, T5-base (the published setting uses a larger model).
* Our adapters had a tenth of the epochs. That is against us, not for us.
* **The control is missing:** the same two methods at 100 epochs WITH their definitions, in this setting. Their papers report little forgetting with definitions, on a larger model. Here, with definitions and 10 epochs, their forgetting was near zero, but then they had learned little. Without the control the loss at 100 epochs cannot be put on the missing instruction alone: it could also come from 100 epochs on this smaller model. Run E (below) is that control.

#### The next GPU runs, how each will be read, and their state (fixed 08:21 UTC; state at about 10:00 UTC)

The remaining GPU hours of the week are about 13 by our count (the exact number is only visible in the Kaggle editor). Kaggle allows two GPU sessions at a time. If the hours run out, the later runs are dropped and that is said. **None of these has a result yet.**

| run | what | how it will be read (fixed before) | state now |
|---|---|---|---|
| D | the two published methods without definitions at their published 100 epochs | as fixed at 08:21 UTC | **done**: see the subsection "Run D" above |
| E (the control for D) | the two published methods WITH their definitions at 100 epochs, same five tasks and seed 42, official code, both at the same time; re-test at the four wordings (`five_tasks_published_e100_with_definitions.ipynb`; about 2.5 hours) | for each method: "the instruction was what kept the answers" if, with definitions, forgetting is at most half of its value without definitions AND at most half as many answers change (GainLoRA: at most 12.7 and 89 of 400; InfLoRA: at most 33.1 and 191); "it was not the instruction" if forgetting with definitions is at least 80% of that without (GainLoRA at least 20.3; InfLoRA at least 53.0); otherwise mixed. Also reported as it is: their final score at the original wording next to ours (79.33 and 78.73) and at the heavy wording (ours 76.12 for the recogniser, 72.12 for the certified switch) | started on Kaggle 2026-11-18 at 12:41 UTC (notebook tested with a dry run first); NOT a result yet |
| U | with definitions, routes from the example text alone (the idea above) | as fixed at 07:42 UTC | done: see the subsection "Run U" above |
| feature job | frozen T5-base states on the second benchmark and up to 120 tasks, with a second random split of the fresh tasks; no training | as fixed at 07:48 and 08:01 UTC | done: see the subsection on the feature job above |
| C | the 15 SuperNI tasks without definitions, order 1, seed 42, 10 epochs: InfLoRA and GainLoRA, and ours with the table `noinstruction_routes_15.json` | "holds on fifteen tasks" if for rule L the final score is at least 2 points above the better published method, forgetting is below 2, and changed answers are at most a quarter of the better published method's count; "does not" if below it or at least as many answers change; otherwise mixed | ours: still running at about 12:30 UTC; the published methods' 15-task run: built, after run E if hours remain |
| second feature job | the feature job with other frozen models (`hidden_states_models.ipynb`) | not described in the plan entries read | built |

**Run D was read at 12:24 UTC (above); our expectation written for it is replaced by that reading.**

**What we expect for E (not a result):** they learn 79 to 84 right after learning. GainLoRA: forgetting 0 to 6 points, 15 to 80 changed answers, final score 75 to 83. InfLoRA: forgetting 1 to 12 points, 60 to 200 changed answers, final 68 to 82. So "the instruction was what kept the answers" for both, less sure for InfLoRA. More likely than not, GainLoRA ends above our 10-epoch score at the original wording, and both lose several points under the heavy rewording. The plan counts about 19.5 of the week's 30 GPU hours as used when run D ended; the running 15-task run of ours needs about 3 more and run E about 2.5, which leaves about 5 hours for the 15-task run of the published methods (the exact number is only visible in the Kaggle editor).

**Not planned for lack of hours:** our method at 100 epochs; trained answers on the second benchmark; a larger model. They stay open and the paper will say so.

#### The paper: restructured on 2026-11-18, best method first (state at about 12:32 UTC)

On 2026-11-18 (between 09:30 and 09:50 UTC) the paper was to be rewritten with all tests in it, the method found last put first, and the others integrated. Decided at 09:50 UTC: the introduction first, then related work, method, results, discussion, limitations, conclusion, appendix. No test is dropped.

**Long version** (`paper_certified/main.tex`, now 80 pages):

1. Abstract, 1 Introduction, 2 Related work, 3 Method (3.1 the setting and the lemma "an answer changes only if its route changes"; 3.2 without instructions: recogniser, audit, handicap; then word-vector features, protected zones, the certified switch, what the zones guarantee, the variant for sibling tasks, zones from the inputs of a task, the compared methods, the protocol).
2. 4 Results: 4.1 published methods without instructions; 4.2 recognition without instructions; 4.3 trained answers without instructions (with the 100-epoch run); 4.4 five tasks with instructions; 4.5 reworded instructions; 4.6 routes from the example alone; 4.7 fifteen tasks; 4.8 where the gap comes from; then stress test, 875 tasks, edits, orders, unrelated inputs, other models, rewording at scale, zones from inputs.
3. 5 Discussion, 6 Limitations, 7 Conclusion, appendices.

**Short version** (`paper_certified/short.tex`, now 78 pages, main text 9 pages; the Conclusion ends on page 9): the same order. 3.2 recogniser, audit and handicap; 3.3 zones and the certified switch; 3.4 zones from inputs; 4.2 without instructions; 4.3 with instructions. The word-vector feature details and the full statements of four propositions are in the appendix. Title and authors unchanged.

**Corrections.** Both versions were corrected after two independent readings: about 45 points in the long version and about 35 in the short one (leftovers of the old order, caveats for the newest results, exact wording for tasks that share texts, a symbol clash). Both now contain run D. The Word files and the split PDFs were rebuilt (the split PDFs are now small files, made with pypdf). The files were sent at about 12:32 UTC. The PDFs sent at 03:56 UTC are older than these results. (The page counts were not checked.)

#### The state of the day and what is running or prepared right now (about 12:30 UTC). The items below are NOT results yet.

* **The pause.** From about 10:22 to 12:20 UTC nothing was done. The corrections of the paper were half applied at that moment and were finished afterwards.
* **Done and in the subsections above:** three seeds without instructions, the fourth fresh batch, the small trained model, the feature job, run U, run D.
* **Running on Kaggle:** the 15-task run of our method without definitions (still running at about 12:30 UTC; about 3 hours in all).
* **Next:** run E, the control for run D (published methods WITH definitions at 100 epochs; started on Kaggle at 12:41 UTC; about 2.5 hours; NOT a result yet). Then, if hours remain, the 15-task run of the published methods without definitions, and the job with frozen states of six other models.
* **Hours.** The free GPU hours of this week are nearly used up. If the hours end while a run is going, Kaggle stops it and its results may be lost.
* **Not planned:** our method at 100 epochs, trained answers on the second benchmark, a larger model.
