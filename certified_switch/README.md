# The certified switch as stand-alone code

`certified_switch.py` is the method of `paper_certified/` written for any Hugging Face model that
has an embedding table and linear attention layers. It uses no code of another paper.

| Part | What it does |
|---|---|
| `Zones` | builds a protected zone for each task from the frozen word-vectors of its instruction, and gives every input a route: the oldest zone that contains it, or else the nearest zone |
| `RoutedLoRA`, `add_adapters` | one low-rank adapter per task on every attention projection; older adapters are frozen; for each input only the adapters of tasks 1 to its route are on |
| `ContinualModel` | learns tasks one after another (`learn`) and answers inputs (`answer`), for encoder-decoder and decoder-only models |

Why an answer inside a zone can never change: its route is the oldest zone that contains it; later
tasks only add newer zones; the adapters the route switches on are frozen.

What it needs: every task's inputs start with the same instruction text. A few "unrelated" tasks
(inputs that will never be learned) are needed once, to set the scale of distances.

## Two rules

`ContinualModel(model, tokenizer, unrelated_tasks, rule=...)`

| `rule` | What happens when a new task looks like an old one | What is promised |
|---|---|---|
| `"certified"` (default, the rule of the paper) | the new task's inputs lie inside the old zone and stay with the old task; `learn` returns how many | the answer to every input inside a zone never changes |
| `"exact_first"` (variant, added 2026-11-17) | an input that carries a task's instruction token for token goes to that task (the oldest, if two tasks have the same instruction), so the new task is learned | the answer to an input that carries a learned instruction exactly never changes; the answer to another input inside a zone never changes *unless* a task is learned later whose instruction that input carries exactly |

On tasks whose instructions do not collide the two rules give the same routes. In a test with 875
real task instructions (routes only, see the `RESULTS_LOG.md`, section 4L) the first rule refused 17% of
the tasks because they were siblings of an older task; the second served all but 14, which have an
instruction identical to an older one. `demo_sibling.py` shows the difference with a small model.

## Demo

```bash
python demo.py --model t5-small --data /path/to/CL_Benchmark --train 400 --test 50 --epochs 5 --lr 5e-3
```

`CL_Benchmark` is the data folder of the GainLoRA repository (SuperNI and Long_Sequence). The demo
teaches the model five tasks one after another and then answers every test question again, with the
original and with reworded instructions (`rewordings.json`), under three rules: `certified`,
`nearest` (nearest zone only) and `all` (every adapter on: plain adapters with no protection).
Results are written to `demo_results/`.

## What has been run (CPU only, small models, few examples)

| Model | Kind | Score right after learning | At the end, certified switch | Answers changed | At the end, all adapters on | Answers changed |
|---|---|---|---|---|---|---|
| `t5-small` (61M) | encoder-decoder | 65.8 | 65.8 | 0 of 200 | 12.0 | 200 of 200 |
| `gpt2` (124M) | decoder-only | 67.9 | 67.9 | 0 of 120 | 15.8 | 119 of 120 |
| `SmolLM2-135M` | decoder-only | 74.3 | 74.3 | 0 of 120 | 27.7 | 100 of 120 |

(400 training examples and 5 epochs per task; 50 test inputs per task for `t5-small`, 30 for the
two decoder-only models; original instructions. With heavily reworded instructions the certified
switch ends at 56.2, 54.8 and 65.1 with no answer changed; 88.8%, 81.3% and 80.7% of those inputs
reach their own task.)

These runs show that the code works on different kinds of models and that answers are kept. They
are not benchmark results: the models are small and barely trained.

## A small test (no model needed)

```bash
python test_zero_radius.py
```

It checks one edge case of the rule: a later task that has the instruction of an older task, word
for word. Its zone has radius zero, all of its inputs are reported as blocked when it arrives, and
under every rule the older task keeps them (ties go to the older task). Before 2026-11-18 the code
divided zero by zero in this case and the nearest-zone rule could pick the newer task.

## Without instructions: the recogniser, the audit and the handicap

`recogniser.py` (numpy only) routes inputs when there is no instruction text. It keeps one ridge
recogniser over the features of any frozen encoder, from running sums: no stored training inputs, and
with the plain rule the same result in any order of arrival. Each task brings fit features (used for the
sums) and held-out features (stored, never used for fitting). With `nu=None` the route is the argmax of
the newest recogniser; with `nu=0, 1, 2, ...` the newcomer takes an input only if its margin beats a
threshold set by the held-out inputs of older tasks (the handicap).

```python
from recogniser import JointRecogniser, standardiser
st = standardiser(pool_features)            # features of unrelated inputs, from any frozen encoder
rec = JointRecogniser(dim=768, ridge=1.0, nu=None)      # or nu=2 for the handicap
for fit_x, held_x in tasks:                 # each (n x dim) and (m x dim), already standardised by st
    rec.add_task(fit_x, held_x)
    print(rec.newcomer_report())            # own share and changed held-out inputs, before any training
routes = rec.route(st(new_features))        # task index for every input
print(rec.audit(0))                         # m, changed, changed_ever, estimate, upper, valid
print(rec.bound_in_advance(0))              # handicap only: later tasks * (nu + 1) / (m + 1)
```

What is guaranteed: with the plain rule, `audit` gives a one-sided 95% upper bound on the share of
test inputs of a task whose route has changed (valid if the held-out and test inputs are exchangeable).
With the handicap, the share of inputs of an older task that ever change is at most
(later tasks)(nu + 1)/(m + 1) in expectation, known before anything is trained. What is not:
(i) these bound change, not quality: an input the recogniser gives to the wrong task from the start
does not change and is answered by the wrong adapters; (ii) the price of the handicap is paid by the
newcomer, as the share of its own held-out inputs it does not take (with `nu=0` one look-alike
held-out input can keep most of the newcomer out); (iii) the bound in advance grows with the number
of later tasks and falls with m, and is far above what is measured. Under the handicap the audit is
not valid (`valid=False`), because the same held-out inputs set the thresholds.

```bash
python test_recogniser.py     # equal to paper_certified/hidden_rules_study.py, order, handicap, theorem; prints OK
```
