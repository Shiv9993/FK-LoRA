# Silent adapters

The plan, with the success and failure rules fixed in advance, is in `PLAN.md`. Read that first.

## Stage 0: run the published methods (`stage0_colab.ipynb`)

Runs the **official GainLoRA code** (InfLoRA and GainLoRA) on SuperNI order 1, 15 tasks one after
another, in a reduced setting (t5-base instead of t5-large, 10 epochs instead of 100).

1. Open `stage0_colab.ipynb` in Colab from GitHub (repo `Shiv9993/FK-LoRA`).
2. Runtime → Change runtime type → **T4 GPU**. Runtime → Run all. Allow Google Drive.
3. If Colab disconnects, open it again and press Run all. It continues from the last finished task.

Everything is saved to Google Drive in `MyDrive/silent_adapters/<run_tag>/`:

| On Drive | What it is |
|---|---|
| `<run>/outputs/<n>-<task>/` | results of each finished task, and the weights the next task needs |
| `<run>/logs/<n>-<task>.log` | the full training log of each task |
| `<run>/progress.json` | minutes each task took |
| `results.json` | the final scores (AP and FT) of every method |

A task counts as finished only when its folder on Drive contains a file called `DONE`. If the run is
cut off in the middle of a task, only that task is repeated.

Settings are in the first code cell. To start a separate run, change `run_tag`. If the GPU runs out
of memory, set `train_batch` to 8.

## Files

| File | What it is |
|---|---|
| `PLAN.md` | the plan and the pass/fail rules, written before any run |
| `stage0_colab.ipynb` | the notebook to run |
| `quick_test_colab.ipynb` (source `quick_test.py`) | **short tests on five tasks**, one seed. Methods: `plain`, `locked` (exact switch), `certified2` (certified switch, final rule), `certified` (its first version: shown if on Drive, not trained again), `inflora`, `gainlora_inflora` (official code, unchanged), and the older `silent`, `silent2`. Finished methods are skipped. Every run made with it also tests the reworded task definitions. See the main README, sections 4J to 4L |
| `certified15_colab.ipynb` | the same code set to the 15 tasks of SuperNI order 1, `certified2` only (about 3 to 4 hours on a T4). Built from `quick_test.py` by `make_notebook.py` |
| `five_tasks_seeds_colab.ipynb` | the same code set to seeds 42, 43 and 44 on the five tasks (`certified2`, InfLoRA, GainLoRA). Built from `quick_test.py` by `make_notebook.py` |
| `results/` | raw output of GPU runs: logs, and for Kaggle runs `small_files.zip` (scores, every answer of the re-test, training logs) |
| `kaggle_fetch.py` | `python3 kaggle_fetch.py <kaggle user> <notebook slug> [name]`: says whether a Kaggle run is still going and, when it has finished, saves its log and small result files under `results/<name>/`. The login comes from a secret of the cloud session (or from `KAGGLE_API_TOKEN`); it is never written to a file here |
| `silent_run.py` | our code: starts the official training script unchanged and adds the silence loss, the exact switch (`SILENT_LOCK=1`) or the certified switch (`SILENT_CLAIM=1`). With `SILENT_BASELINE=1` it changes nothing in an official method and only repeats its test with reworded definitions. The notebooks carry a copy of it |
| `routing_check.py` | CPU check of the certified switch: where every test input goes, for five and fifteen tasks, original and reworded definitions. No training, no GPU, about 3 minutes |
| `check_progress_colab.ipynb` (source `check_progress.py`) | read-only: prints the scores and timings of every run saved on Drive so far. Needs no GPU and can be run while the main notebook is training |
| `stage0.py` | the same code as a script (cells marked `# %%`); edit this, then run `python3 make_notebook.py` |

## What has and has not been tested (2026-11-16)

Tested on a CPU with a tiny model (t5-small, 3 tasks, 8 training examples, a stand-in for the GPU
library): both official methods run from start to finish through this notebook's code; a run killed
in the middle of a task, with the local machine wiped, continued from the saved copy and gave the
same final score; a finished run is skipped entirely.

First Colab try (2026-11-16) stopped in the set-up check, and the notebook did not show why. Fixed the
same day: every set-up error is now printed; the GPU is checked first; the GPU libraries that come
with PyTorch are put on the library path so CuPy can find them; and if CuPy still does not work, the
one step that uses it (an SVD) is done with PyTorch instead. `results.json` records which was used
(`svd_backend`). The PyTorch stand-in was self-tested on CPU; it has not been compared number for
number with real CuPy.

First real run (Colab T4, 2026-11-16/17): InfLoRA finished all 15 tasks in 3.0 hours with batch 16.
GainLoRA ran out of GPU memory on task 3 (the first task with long inputs) at batch 16. The notebook
now handles this itself: when a task runs out of memory it halves the batch and tries that task again,
keeps the smaller batch for the rest of that method, and records the batch used for every task in
`progress.json`. The effective batch size (batch times accumulation steps) stays at 32, and the
official code reads all of a task's data when it builds its matrices, so only memory use changes.

**Still not checked:** the PyTorch stand-in for CuPy against real CuPy, number for number.

## Short first test (`quick_test_colab.ipynb`)

Self-tested on CPU with a tiny model (2026-11-17): all four methods run through the notebook on three
tasks; the built-in self-check confirms that "new adapter switched off" gives exactly the same outputs
as "new adapter's weights set to zero" (difference 0.00) after real weight updates, while the trained
adapter does change those outputs. **Not yet run on a GPU.** Expected time on a T4: about 3 to 3.5
hours for all four methods (an estimate).

## The exact switch (`locked`), self-tested on CPU 2026-11-17

With a tiny model and a deliberately large learning rate, `plain` fell from 60.3 to 15.0 on the first
task and from 49.0 to 30.4 on the second. `locked` scored exactly the same at the end as right after
learning on every task (60.3, 49.0, 29.9), all 616 test inputs were matched to a task, and its scores
right after learning equal `plain`'s. **Not yet run on a GPU.** `silent2` also runs end to end on CPU;
its tiny-test numbers mean nothing.

## The certified switch (2026-11-17)

Rule, proof, the pass/fail rules and the result read against them are in `PLAN.md` (last sections).
In short: each task has a protected zone made from the frozen word-vectors of its fixed opening; an
input inside a zone belongs to the oldest task that claims it, and its answer can never change again;
an input outside every zone goes to the nearest zone, with no guarantee.

**First GPU run (Colab T4, five tasks, one seed, first version of the rule):** every rule held;
numbers in the `RESULTS_LOG.md`, section 4L. A CPU check on the 15-task sequence then showed that the
first version's rule for inputs outside the zones fails under rewording when long-text tasks are
present. The final rule is called `certified2` (nearest zone, common words count less) and is frozen.

**Second GPU run (Kaggle T4, five tasks, one seed):** `certified2`, InfLoRA and GainLoRA, 36 minutes
each; the rules held again (`RESULTS_LOG.md`, section 4L).

**Next GPU runs:**

1. `certified15_colab.ipynb`: `certified2`, InfLoRA and GainLoRA on the 15 tasks of SuperNI order 1.
   On Kaggle with two GPUs about 7 hours in one go; on one GPU about 10 hours.
2. `five_tasks_seeds_colab.ipynb`: the five-task comparison with seeds 42, 43 and 44, nothing to
   edit (nine runs; about 3 hours on two GPUs, about 5.5 hours on one). Seed 42 is repeated on
   purpose: it checks that a run can be reproduced, and all three seeds then come from the same
   code. On Kaggle two "Save & Run All" runs can go at the same time, so this one can be started
   while the 15-task run is still going.

At the end every notebook prints all tables (one block per seed, then all seeds together), and
writes the small result files (scores, every answer, logs) into one archive,
`<run_tag>_small_files.zip`, next to the results folder. A run that fails does not stop the
others; it is listed at the end.

`check_progress_colab.ipynb` prints every table for the tasks finished so far and can be run while
training goes on.

**Running on Kaggle instead of Colab.** `quick_test_colab.ipynb` and `certified15_colab.ipynb` find
out by themselves that they are on Kaggle and then write their results to the notebook's output
(`/kaggle/working/silent_adapters/<run_tag>/`) and everything else to a temporary folder.

1. On kaggle.com create a notebook and bring the `.ipynb` in (File → Import Notebook; upload the
   file downloaded from GitHub, or paste its link).
2. In the right-hand panel (Session options): *Accelerator* **GPU T4 x2**, *Internet* **on**. Both
   need a phone-verified account (kaggle.com → Settings → Phone verification); until then the
   Internet switch is not shown. With the internet off the notebook stops at `git clone` and now says
   so. Google Drive cannot be connected on Kaggle and is not needed there.
3. **Save Version → Save & Run All.** The run continues in the background with the browser closed;
   a session can last up to 12 hours, which is more than either notebook needs.
4. When it has finished, open the saved version: everything printed is there (copy it into the
   chat), and the result files are under *Output*.

**Two GPUs.** One method cannot be made faster by a second GPU, because its tasks are learned one
after another. With two GPUs (Kaggle's "T4 x2") the notebook trains two *methods* at the same time,
one per GPU, and marks every printed line with the GPU and the method (`use_all_gpus` in the first
cell switches this off). Checked on CPU with a tiny model: the same three methods run two at a time
and one after another gave exactly the same answers and scores. Not yet run on real GPUs this way.
The 15-task notebook has one method, so it uses one GPU.

`plain`, `locked` and the first `certified` are on Google Drive, not on Kaggle. They are not trained
again there (`only_if_finished` in the first cell): the re-test of `certified2` already contains
plain adapters (`all`) and the exact switch (`key`). Checked locally: the first cell picks the
Kaggle folders. **Not yet run on Kaggle itself.**

**CPU check of the routes** (no training), inside the folder of the official code and with its
Python environment:

```bash
cp /path/to/silent_adapters/silent_run.py /path/to/silent_adapters/routing_check.py .
```

```bash
python routing_check.py --model t5-base
```

Tested on CPU with a tiny model (t5-small, 3 tasks; this shows that the code runs, nothing more):
`certified2` gives the same official scores as `locked`; its re-test reproduces the official test on
every input; with `SILENT_BASELINE=1` the re-test of InfLoRA and of GainLoRA also reproduces their
official test on every input (300 of 300 each); the 15-task setting runs on its first three tasks,
two of which have no reworded definition; the older `silent` method still runs.
