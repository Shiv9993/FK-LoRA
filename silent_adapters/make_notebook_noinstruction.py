"""Builds five_tasks_noinstruction_colab.ipynb: the five-task run (seed 42; certified2, InfLoRA, GainLoRA) with
the task definition removed from every data file of the five tasks before anything is trained.

Usage: python3 make_notebook_noinstruction.py [three|published|ours|all]    (run it in this folder; it only writes notebooks)
  three      five_tasks_noinstruction_colab.ipynb (default): certified2, InfLoRA, GainLoRA
  published  five_tasks_noinstruction_published.ipynb: InfLoRA and GainLoRA only, one per GPU

The notebook is made from quick_test.py (the code of the earlier five-task runs) and silent_run.py, with
these changes and no others:
  * the first text cell, the run tag, the list of methods (certified2, inflora, gainlora_inflora);
  * a new code cell that sets "Definition" to [""] in train/dev/test of the five tasks (in the copy of the
    official repository that the notebook downloads), checks it, and shows the exact input of each code path;
  * in the embedded copy of silent_run.py the re-test wordings are cut to "original": the three reworded
    definitions of the earlier runs would put a task description back into the input;
  * the summary printed at the end also compares with the earlier run WITH definitions.
It does not import make_notebook.py, because that file writes all its notebooks when it is imported."""
import json, re, sys

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
VARIANT_ARGS = sys.argv[1:] or ["three"]

# The earlier run WITH definitions: seed 42, five tasks, original wording. The numbers are in
# noinstruction_earlier_numbers.json, read from silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip
# with the notebook's own functions scores_of and retest_table, and are embedded in the notebook as constants.
EARLIER = {k: v for k, v in json.load(open(HERE + "/noinstruction_earlier_numbers.json")).items() if not k.startswith("_")}


def edit(text, old, new):
    assert text.count(old) == 1, f"expected exactly one place to change: {old[:70]!r} (found {text.count(old)})"
    return text.replace(old, new)


FIRST_THREE = '''# %% [markdown]
# # Five tasks without the task instruction: certified switch and the two published methods
#
# **This notebook repeats the earlier five-task run (seed 42, t5-base, 10 epochs per task) with one change: the
# task definition is removed from every input.** Before anything is trained, the "Definition" of every data
# file of the five tasks (train, dev and test) is replaced by the empty string. The input of every task is then
# the same fixed text followed by the example itself (`Definition: ` + blank line + `Now complete the following
# example -` + `Input: ...` + `Output: `), so the input no longer tells which task it belongs to. The question is
# how the three methods behave when that is so: InfLoRA and GainLoRA (official code) and `certified2` (our
# method, kept because its rule `all` gives plain adapters and its rule `centre` uses the whole input).
# The rules of `certified2` that need the instruction (`certified`, `nearest`, `key`) are expected to fail; they
# are reported as they come out, not hidden.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** and *Internet* to **on**, then
# **Save Version -> Save & Run All**. With two GPUs two jobs run at the same time and the third follows: about
# **1.5 hours** (the earlier run took 36 minutes per job; set-up about 10 minutes). With one GPU about 2 hours.
# **The result is the small-files zip** `/kaggle/working/silent_adapters/__TAG___small_files.zip`
# (scores, every answer of the re-test, training logs). The tables at the end of the printed output are also
# in the saved version.
#
# Differences from the earlier run, and only these: (1) the definitions are removed (cell "Remove the task
# definitions"); (2) only seed 42 and these three jobs; (3) the re-test uses the original wording only: the three
# reworded definitions of the earlier run would put a task description back into the input; (4) the summary
# compares with the numbers of the earlier run WITH definitions (embedded in the last cell, source named there).
#
# After each task, every test input is answered again (the re-test, as in the earlier run) and the answers that
# changed since the task was learned are counted. For `certified2` this is done under five rules with the same
# trained adapters: `certified`, `nearest`, `centre`, `key` and `all` (every adapter on = plain adapters).
# The published methods are tested as they are.
'''

REMOVE_CELL = '''# %% [markdown]
# ## Remove the task definitions
# The only change against the earlier run. In the downloaded copy of the official code, the "Definition" of
# train, dev and test of every one of the five tasks becomes `[""]`. The official input builder then gives
# `Definition: ` + empty text, and the same is true for the input builder of `silent_run.py`. Nothing else in the
# data is touched (the examples, the answers and the unused fields stay). The cell checks every file afterwards.

# %%
NOINST_TASKS = list(CFG["tasks"])
NOINST_EMPTY_INPUT = "Definition: \\n\\nNow complete the following example -\\nInput: {0}\\nOutput: "   # the input text, same for every task


def remove_definitions(repo, tasks):
    """Set "Definition" to [""] in train/dev/test of `tasks` everywhere under `repo` (outside the git folder and the
    run outputs). Returns {path: length of the definition that was removed}."""
    changed = {}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in (".git", "logs_and_outputs", "__pycache__")]
        if os.path.basename(root) in tasks:
            for f in files:
                if f in ("train.json", "dev.json", "test.json"):
                    p = os.path.join(root, f)
                    data = json.load(open(p, encoding="utf-8"))
                    if not isinstance(data, dict) or "Definition" not in data:
                        continue
                    old = data["Definition"]
                    changed[p] = len(old[0] if isinstance(old, list) and old else (old if isinstance(old, str) else ""))
                    data["Definition"] = [""]
                    tmp = p + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, ensure_ascii=False)
                    os.replace(tmp, p)
    return changed


def check_definitions_removed(repo, tasks):
    """Every train/dev/test file of every task must now have Definition == [""] and still have its examples."""
    n = 0
    for t in tasks:
        for split in ("train", "dev", "test"):
            p = os.path.join(repo, "CL_Benchmark", "SuperNI", t, split + ".json")
            data = json.load(open(p, encoding="utf-8"))
            assert data["Definition"] == [""], f"definition not removed in {p}: {str(data['Definition'])[:80]}"
            assert len(data["Instances"]) > 0, f"no examples in {p}"
            n += 1
    return n


NOINST_CHANGED = remove_definitions(REPO, NOINST_TASKS)
assert len(NOINST_CHANGED) >= 3 * len(NOINST_TASKS), f"expected train, dev and test of every task; changed only {len(NOINST_CHANGED)} files"
print(f"definitions removed in {len(NOINST_CHANGED)} files; every one of the {check_definitions_removed(REPO, NOINST_TASKS)} "
      f"train/dev/test files of the {len(NOINST_TASKS)} tasks now has Definition == [\\"\\"]")
for p in NOINST_CHANGED:
    if "CL_Benchmark" + os.sep + "SuperNI" not in p:
        print("   also changed (another copy of a task):", p)
shutil.rmtree(os.path.join(CFG["work_dir"], "hf_cache", "datasets"), ignore_errors=True)   # no cached copy of the old data
json.dump({"task_definitions_removed": True, "tasks": NOINST_TASKS, "files": {os.path.relpath(p, REPO): n for p, n in NOINST_CHANGED.items()},
           "note": "values: length (characters) of the definition that was removed"}, open(os.path.join(SAVE_DIR, "noinstruction.json"), "w"), indent=1)

# The input that each code path builds for the first example of every task and split. Hard check: no task words.
code = ("import sys, json, types\\nsys.path.insert(0, 'src')\\nfrom cl_dataset import CLInstructions\\n"
        "me = types.SimpleNamespace(_load_dataset=lambda p: CLInstructions._load_dataset(None, p))\\n"
        "out = {}\\nfor t, split in " + repr([(t, s) for t in NOINST_TASKS for s in ("train", "dev", "test")]) + ":\\n"
        "    g = CLInstructions.load_SuperNI_dataset(me, 'CL_Benchmark/SuperNI/%s/%s.json' % (t, split), None, t, 'full', None, split)\\n"
        "    out[t + '/' + split] = next(g)['Instance']['instruction']\\nprint('OFFICIAL_INPUTS' + json.dumps(out))\\n")
r = subprocess.run([ENV_PY, "-c", code], capture_output=True, text=True, env=RUN_ENV, cwd=REPO)
line = [l for l in r.stdout.splitlines() if l.startswith("OFFICIAL_INPUTS")]
if r.returncode == 0 and line:
    official_inputs = json.loads(line[0][len("OFFICIAL_INPUTS"):])
    assert all(v == NOINST_EMPTY_INPUT for v in official_inputs.values()), "an official input still contains something task-specific"
    print("official input builder: the text before the example is identical for all", len(official_inputs), "task/split pairs")
else:                                                  # only a check; the data themselves were checked above
    print("WARNING: could not run the official input builder as a check:\\n" + (r.stdout + r.stderr)[-1500:])
first = json.load(open(os.path.join(REPO, "CL_Benchmark", "SuperNI", NOINST_TASKS[0], "test.json"), encoding="utf-8"))["Instances"][0]["input"]
print("input of the first test example of the first task, as the official code builds it:\\n" + "-" * 30 + "\\n"
      + NOINST_EMPTY_INPUT.replace("{0}", first) + "\\n" + "-" * 30)
'''

# The comparison with the earlier run, printed at the end.
COMPARE_CELL = '''
# ---- comparison with the earlier run WITH definitions ----
# Source of the embedded numbers: silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip (run tag
# superni_short5_t5base_e10, seed 42, Kaggle T4, the same five tasks, t5-base, 10 epochs). "official": scores of the
# official test, read with this notebook's own function scores_of. Rules: the re-test with the original wording, every
# reference answer, read with this notebook's own function retest_table: learned = average score right after learning,
# AP = average score at the end, FT = forgetting, changed/compared = answers that changed between right after learning
# and the end.
EARLIER_WITH_DEFINITIONS = __EARLIER__
EARLIER_SOURCE = "silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip (seed 42, with definitions)"


def compare_with_earlier():
    print("\\n" + "=" * 30 + " this run (NO task definition) against the earlier run WITH definitions " + "=" * 30)
    print("earlier numbers:", EARLIER_SOURCE)
    print("learned: average score right after learning each task | AP: average score at the end | FT: forgetting (lower is better)")
    print("changed: answers to the compared test inputs that differ between right after learning and the end")
    here, result = BY_SEED.get("42", {"summary": {}, "retest": {}}), {}
    print(f"\\n{'method / rule':<34}{'run':<18}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}")
    for m in CFG["methods"]:
        rules = ["certified", "nearest", "centre", "key", "all"] if m == "certified2" else ["official"]
        for rule in rules:
            now = (here["retest"].get(m) or {}).get("table", {}).get("original|" + rule)
            old = EARLIER_WITH_DEFINITIONS.get(m, {}).get("retest", {}).get(rule)
            result[f"{m}|{rule}"] = {"no_definition": now and {k: now[k] for k in ("learned", "AP", "FT", "changed", "compared")}, "with_definition": old}
            for label, v in (("no definition", now), ("with definitions", old)):
                if v is None:
                    print(f"{(m + ' / ' + rule) if label == 'no definition' else '':<34}{label:<18}{'(not available)':>43}")
                else:
                    print(f"{(m + ' / ' + rule) if label == 'no definition' else '':<34}{label:<18}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}"
                          f"{str(v['changed']) + ' of ' + str(v['compared']):>18}")
    print("\\nofficial test (one random reference answer per example, as in the main table above):")
    print(f"{'method':<34}{'run':<18}{'learned':>9}{'AP':>8}{'FT':>8}")
    for m in CFG["methods"]:
        now, old = here["summary"].get(m), EARLIER_WITH_DEFINITIONS.get(m, {}).get("official")
        result[m + "|official_test"] = {"no_definition": now and {k: now[k] for k in ("learned", "AP", "FT", "tasks_done")}, "with_definition": old}
        for label, v in (("no definition", now), ("with definitions", old)):
            if v is None:
                print(f"{m if label == 'no definition' else '':<34}{label:<18}{'(not available)':>25}")
            else:
                print(f"{m if label == 'no definition' else '':<34}{label:<18}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}")
    return result


COMPARISON = compare_with_earlier()
if "certified2" in BY_SEED.get("42", {}).get("summary", {}):
    print("\\ncertified2 without definitions: every task has the same fixed opening, so every task's signature is the same. Zones of the "
          "tasks learned later are therefore empty or claimed by task 1; the zone lines above show it. Expect the rules certified, nearest and "
          "key to send every input to the first task, and `all` and `centre` to be the only rules that can use the later adapters.")

'''


FIRST_PUBLISHED = '''# %% [markdown]
# # Five tasks without the task instruction: the two published methods
#
# **This notebook repeats the earlier five-task run of InfLoRA and GainLoRA (official code, seed 42, t5-base, 10
# epochs per task) with one change: the task definition is removed from every input.** Before anything is trained,
# the "Definition" of every data file of the five tasks (train, dev and test) is replaced by the empty string. The
# input of every task is then the same fixed text followed by the example itself (`Definition: ` + blank line +
# `Now complete the following example -` + `Input: ...` + `Output: `), so the input no longer tells which task it
# belongs to. The question is how the two published methods behave when that is so.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** and *Internet* to **on**, then
# **Save Version -> Save & Run All**. The two methods run at the same time, one on each GPU, and the session ends
# when both are done: about **45 to 50 minutes** (the earlier run took 36 minutes per method; set-up about 10
# minutes). With one GPU they run one after the other, about 85 minutes.
# **The result is the small-files zip** `/kaggle/working/silent_adapters/__TAG___small_files.zip`
# (scores, every answer of the re-test, training logs). The tables at the end of the printed output are also
# in the saved version.
#
# Differences from the earlier run, and only these: (1) the definitions are removed (cell "Remove the task
# definitions"); (2) only seed 42 and these two methods; (3) the re-test uses the original wording only: the three
# reworded definitions of the earlier run would put a task description back into the input; (4) the summary
# compares with the numbers of the earlier run WITH definitions (embedded in the last cell, source named there).
#
# After each task, every test input is answered again (the re-test, as in the earlier run) and the answers that
# changed since the task was learned are counted. The published methods are tested as they are.
'''

FIRST_OURS = '''# %% [markdown]
# # Five tasks without the task instruction: our own method, plain training and a route table
#
# **This notebook runs the project's own job (`certified2`, seed 42, t5-base, 10 epochs per task) on the five tasks
# with the task definition removed from every input, and with training that blocks nothing.** Before anything is
# trained, the "Definition" of every data file of the five tasks (train, dev and test) is replaced by the empty
# string, so the input of every task is the same fixed text followed by the example itself and no longer tells
# which task it belongs to.
#
# **Why training is changed (`SA_NO_BLOCK=1`).** In the earlier setting the certified switch gave each training
# input to the oldest task whose protected zone contains it, and the new adapter is then switched off for that
# input and learns nothing. Without definitions all five tasks have the same opening, so every training input of
# tasks 2 to 5 falls inside the zone of task 1 and the new adapter would learn nothing. With `SA_NO_BLOCK=1`, while
# task t is learned, adapters 1..t are on for every training input (plain sequential training: one adapter per
# task, older ones frozen and active). The official test after each task is then also made with all adapters on.
#
# **What is evaluated.** After each task every test input is answered again (original wording only) under these
# rules, all from the same trained adapters: `all` (every adapter on = plain adapters), `centre` (the nearest
# centre of the whole input chooses how many adapters are on) and every `table:<name>` rule, where the route of a
# test input is read from a table (the position of the task whose adapters 1..route answer it). The
# tables are `table:ridge` and `table:handicap` (the two rules of our method, which decide from the input alone which
# task it belongs to), `table:joint`, `table:claims` and `table:own` (every input goes to its own task: perfect
# recognition, for comparison only). They were written before the run and are embedded in this notebook (file
# `noinstruction_routes.json` of the repository). If a table lacks an entry the job stops before training; a table
# rule that fails is left out and `all` and `centre` are kept.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** (one GPU is used) and *Internet* to
# **on**, then **Save Version -> Save & Run All**. One job: about **45 to 50 minutes** (36 minutes of training and
# testing, set-up about 10 minutes). **The result is the small-files zip**
# `/kaggle/working/silent_adapters/__TAG___small_files.zip` (scores, every answer of the re-test, training logs and
# the route table that was used). The tables at the end of the printed output are also in the saved version, next
# to the numbers of the earlier run WITH definitions.
'''

TABLE_CELL = '''# %% [markdown]
# ## Route tables
# The route of a test input under a `table:<name>` rule is read from a table:
# `routes[name][task folder][index of the example in test.json][number of tasks learned so far] = position (from 1) of
# the task whose adapters 1..route answer the input`. The table `oracle` is built here from the test files (every input
# goes to its own task). Tables from `noinstruction_routes.json` are embedded below. The job checks the table before
# it trains anything and stops with a message if an entry is missing.

# %%
EMBEDDED_TASK_ORDER = __ORDER__
EMBEDDED_ROUTES = __ROUTES__          # from noinstruction_routes.json ({} if that file did not exist when the notebook was made)
NOINST_N = len(CFG["tasks"])
ROUTES = dict(EMBEDDED_ROUTES)
if "own" not in ROUTES:                                           # "own" (every input to its own task) is in the file; else build it here
    ROUTES["oracle"] = {}
    for j, t in enumerate(CFG["tasks"], start=1):
        n = len(json.load(open(os.path.join(REPO, "CL_Benchmark", "SuperNI", t, "test.json"), encoding="utf-8"))["Instances"])
        ROUTES["oracle"][t] = [{str(k): j for k in range(j, NOINST_N + 1)} for _ in range(n)]      # example of task j: route j after j.. tasks

if TEST:                                                          # CPU self-test only: also a table that always uses adapter 1 alone
    ROUTES["first_only"] = {t: [{str(k): 1 for k in range(j, NOINST_N + 1)} for _ in next(iter(ROUTES.values()))[t]] for j, t in enumerate(CFG["tasks"], start=1)}
assert list(CFG["tasks"]) == EMBEDDED_TASK_ORDER[:len(CFG["tasks"])] or not EMBEDDED_ROUTES, "the route tables were made for another task order"


def check_routes(routes, tasks, repo):
    for name, by_task in routes.items():
        for t in range(1, len(tasks) + 1):
            for j in range(t):
                assert tasks[j] in by_task, f"table {name}: no task {tasks[j]}"
                n = len(json.load(open(os.path.join(repo, "CL_Benchmark", "SuperNI", tasks[j], "test.json"), encoding="utf-8"))["Instances"])
                assert len(by_task[tasks[j]]) == n, f"table {name}: task {tasks[j]} has {len(by_task[tasks[j]])} entries, test.json has {n}"
                for i, row in enumerate(by_task[tasks[j]]):
                    r = row.get(str(t))
                    assert isinstance(r, int) and 1 <= r <= t, f"table {name}: task {tasks[j]}, example {i}, after {t} tasks: route {r!r}"


check_routes(ROUTES, CFG["tasks"], REPO)
ROUTE_FILE = os.path.join(CFG["work_dir"], "route_table.json")
json.dump({"routes": ROUTES}, open(ROUTE_FILE, "w"))
shutil.copy(ROUTE_FILE, os.path.join(SAVE_DIR, "route_table.json"))
CFG["certified2"]["SA_ROUTE_TABLE"] = ROUTE_FILE
print("route tables written and checked:", ", ".join("table:" + n for n in ROUTES), "->", ROUTE_FILE)
'''

PUBLISHED = {k: v for k, v in json.load(open(HERE + "/noinstruction_published_numbers.json")).items() if not k.startswith("_")}

OURS_COMPARE = '''
# ---- our job against the published methods WITHOUT definitions (the run of 2026-11-18) and against our earlier run WITH definitions ----
# Source of the published numbers: silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/small_files.zip
# (seed 42, the same five tasks, no task definition, t5-base, 10 epochs), read with this notebook's own functions scores_of and
# retest_table. Source of "with definitions": silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip.
PUBLISHED_WITHOUT_DEFINITIONS = __PUBLISHED__
EARLIER_WITH_DEFINITIONS = __EARLIER__
EARLIER_SOURCE = "silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip (seed 42, with definitions)"
PUBLISHED_SOURCE = "silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/small_files.zip (seed 42, no definitions)"


def compare_with_earlier():
    here = BY_SEED.get("42", {"summary": {}, "retest": {}})
    table = ((here["retest"].get("certified2") or {}).get("table", {}))
    rules = [k.split("|")[1] for k in table if k.startswith("original|")]
    first = [r for r in ("table:ridge", "table:handicap") if r in rules]          # these two are our method
    rules = first + [r for r in rules if r not in first]
    print("\\n" + "=" * 24 + " all rules, and the published methods without definitions " + "=" * 24)
    print("learned: average score right after learning each task (route at the task's own arrival) | AP: average score after all tasks")
    print("FT: forgetting (lower is better) | answers changed: answers to the inputs of the first four tasks (400) that differ between")
    print("right after learning the task and the end. Re-test numbers use every reference answer; the official-test row uses one.")
    print(f"\\n{'run':<44}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}")
    result = {"rules": {}, "published_without_definitions": PUBLISHED_WITHOUT_DEFINITIONS, "published_source": PUBLISHED_SOURCE}
    for rule in rules:
        v = table["original|" + rule]
        result["rules"][rule] = {k: v[k] for k in ("learned", "AP", "FT", "changed", "compared")}
        print(f"{'ours, ' + rule:<44}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}{str(v['changed']) + ' of ' + str(v['compared']):>18}")
    s = here["summary"].get("certified2")
    if s:
        result["official_test"] = {k: s[k] for k in ("learned", "AP", "FT", "tasks_done")}
        print(f"{'ours, official test (= all)':<44}{s['learned']:>9.2f}{s['AP']:>8.2f}{s['FT']:>8.2f}")
    print("-" * 87)
    for m, name in (("inflora", "InfLoRA"), ("gainlora_inflora", "GainLoRA")):
        v = PUBLISHED_WITHOUT_DEFINITIONS[m]["retest"]["official"]
        print(f"{name + ', no definitions, re-test':<44}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}{str(v['changed']) + ' of ' + str(v['compared']):>18}")
        o = PUBLISHED_WITHOUT_DEFINITIONS[m]["official"]
        print(f"{name + ', no definitions, official test':<44}{o['learned']:>9.2f}{o['AP']:>8.2f}{o['FT']:>8.2f}")
    print("-" * 87)
    for rule in ("all", "centre"):
        v = EARLIER_WITH_DEFINITIONS["certified2"]["retest"].get(rule)
        if v:
            print(f"{'ours WITH definitions (earlier), ' + rule:<44}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}{str(v['changed']) + ' of ' + str(v['compared']):>18}")
    failed = {}
    for i, t in enumerate(CFG["tasks"]):
        f = os.path.join(SAVE_DIR, f"{CFG['run_tag']}_certified2_seed42", "outputs", f"{i + 1}-{t}", "extra_eval.json")
        if os.path.exists(f):
            for r, why in (json.load(open(f)).get("failed_rules") or {}).items():
                failed[f"{r} after task {i + 1}"] = why
    if failed:
        print("\\nRULES THAT FAILED (left out of the table):", failed)
    result["failed_rules"] = failed
    return result


COMPARISON = compare_with_earlier()

'''

VARIANTS = {
    "three": dict(tag="superni_noinstruction5_t5base_e10", methods=["certified2", "inflora", "gainlora_inflora"], first=FIRST_THREE,
                  out="five_tasks_noinstruction_colab.ipynb", what="the three jobs of this run"),
    "published": dict(tag="superni_noinstruction5_published_t5base_e10", methods=["inflora", "gainlora_inflora"], first=FIRST_PUBLISHED,
                      out="five_tasks_noinstruction_published.ipynb", what="the two published methods, one per GPU"),
    "ours": dict(tag="superni_noinstruction5_ours_t5base_e10", methods=["certified2"], first=FIRST_OURS,
                 out="five_tasks_noinstruction_ours.ipynb", what="the project's own job only"),
    "published_seeds": dict(tag="superni_noinstruction5_published_seeds43_44_t5base_e10", methods=["inflora", "gainlora_inflora"],
                            first=FIRST_PUBLISHED, seeds=[43, 44], silent_run="silent_run_frozen_published.py", out="five_tasks_noinstruction_published_seeds43_44.ipynb",
                            what="the two published methods, seeds 43 and 44"),
    "ours_seeds": dict(tag="superni_noinstruction5_ours_seeds43_44_t5base_e10", methods=["certified2"], first=FIRST_OURS,
                       seeds=[43, 44], silent_run="silent_run_frozen_ours.py", out="five_tasks_noinstruction_ours_seeds43_44.ipynb", what="the project's own job, seeds 43 and 44"),
}
BASE_OF = {"fifteen_ours": "ours", "published_seeds": "published", "ours_seeds": "ours"}


def build(variant):
    V = VARIANTS[variant]
    RUN_TAG, OUT = V["tag"], (__import__("os").environ.get("NOINST_OUT_DIR") or HERE) + "/" + V["out"]
    FIRST_CELL = V["first"].replace("__TAG__", RUN_TAG)
    if V.get("seeds"):
        two_gpu, one_gpu = (("about 85 minutes", "about 2.5 hours") if len(V["methods"]) == 2 else ("about 45 to 50 minutes", "about 85 minutes"))
        note = ("# **SEEDS 43 AND 44.** This is the same notebook as its seed-42 twin, with only the seeds and the run tag changed: every job is run for\n"
                "# seed 43 and for seed 44 (" + str(len(V["methods"]) * 2) + " runs, each with its own folder), and the route table does not depend on the seed. With two GPUs\n"
                "# two runs go at the same time, with one GPU one after the other. **Time for this notebook: " + two_gpu + " on T4 x2, " + one_gpu +
                " on one GPU.**\n# The time estimate further down is for one seed. At the end the printed output has one block per seed and then the mean over the two seeds.\n#\n")
        i = FIRST_CELL.index("\n#\n") + len("\n#\n")
        FIRST_CELL = FIRST_CELL[:i] + note + FIRST_CELL[i:].replace("seed 42", "seed 43 and seed 44")
    src = open(HERE + "/quick_test.py").read()
    a, b = src.index("# %% [markdown]"), src.index("# %%\nimport os, re, sys")
    src = src[:a] + FIRST_CELL + "\n" + src[b:]
    src = edit(src, 'run_tag="superni_short5_t5base_e10",', f'run_tag="{RUN_TAG}",')
    src = edit(src, 'methods=["plain", "locked", "certified", "certified2", "inflora", "gainlora_inflora"],   # finished ones are skipped',
               "methods=" + json.dumps(V["methods"]) + ",   # " + V["what"])
    src = edit(src, 'only_if_finished=["plain", "locked", "certified"],   # shown when a finished copy is in the results folder, never trained',
               'only_if_finished=[],                   # nothing is only shown here')
    assert "seeds=[42]," in src
    if V.get("epochs"):
        src = edit(src, "    epochs=10,\n", f"    epochs={V['epochs']},\n")
    if V.get("progressive_zip"):                         # the small-files zip is refreshed after every finished task, not only at the end
        src = edit(src, "FAILED = {}                                            # runs that stopped with an error; the others carry on\n",
                   ZIP_FUNCTION + "\nFAILED = {}                                            # runs that stopped with an error; the others carry on\n")
        src = edit(src, '        say(f"      {name} done in {(time.time() - t0) / 60:.1f} min, saved")\n',
                   '        say(f"      {name} done in {(time.time() - t0) / 60:.1f} min, saved")\n'
                   '        try:\n            make_small_zip()\n        except Exception as err:                         # never stops a run\n'
                   '            say(f"      (the small-files zip could not be refreshed: {err})")\n')
        z0 = src.index("import zipfile                                         # the small result files")
        z1 = src.index('print("\\nsaved:", os.path.join(SAVE_DIR, "results.json")')
        src = src[:z0] + "ZIP = make_small_zip()\n" + src[z1:]
    if V.get("preset") == "fifteen":
        src = edit(src, 'PRESET = "five"', 'PRESET = "fifteen"')
        src = edit(src, 'CFG.update(run_tag="superni_o1_t5base_e10_ours", methods=["certified2", "inflora", "gainlora_inflora"], tasks=[',
                   f'CFG.update(run_tag="{RUN_TAG}", methods={json.dumps(V["methods"])}, tasks=[')
    # silent_run.py goes inside the notebook, with the re-test cut to the original wording
    code = open(__import__("os").environ.get("NOINST_SILENT_RUN") or HERE + "/" + V.get("silent_run", "silent_run.py")).read()   # the file that is embedded
    assert "'''" not in code
    code = edit(code, 'LEVELS = ["original", "light", "medium", "heavy"]',
                'LEVELS = ["original"]   # no-instruction run: the reworded definitions of the earlier run would put a task description back')
    src = edit(src, "SILENT_RUN_SOURCE = None  # filled in by make_notebook.py", "SILENT_RUN_SOURCE = r'''" + code + "'''")
    # the new cell goes right after the cell that writes silent_run.py, before "The runs"
    src = edit(src, 'print("silent_run.py written:", len(SILENT_RUN_SOURCE.splitlines()), "lines")\n\n',
               'print("silent_run.py written:", len(SILENT_RUN_SOURCE.splitlines()), "lines")\n\n' + REMOVE_CELL + "\n")
    # the comparison goes before results.json is written (not for 15 tasks: the earlier numbers are for the five-task run)
    if V.get("preset") == "fifteen":
        src = edit(src, 'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED},',
                   'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED, "no_task_definition": True},')
    else:
        compare_text = (E100_COMPARE.replace("__PUB__", repr(PUBLISHED)).replace("__EARLIER__", repr(EARLIER))
                        if V.get("compare") == "e100" else COMPARE_CELL.replace("__EARLIER__", repr(EARLIER)))
        src = edit(src, 'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED},',
                   compare_text.lstrip("\n") +
                   'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED, "no_task_definition": True,\n'
                   '           "comparison_with_definitions": COMPARISON, "earlier_source": EARLIER_SOURCE},')
    if BASE_OF.get(variant, variant) == "ours":
        if V.get("preset") != "fifteen":
            c0 = src.index('if "certified2" in BY_SEED.get("42"'); c1 = src.index('json.dump({"cfg": CFG, "svd_backend"')
            src = src[:c0] + src[c1:]                      # the remark about blocked zones is for the three-job notebook
            a0, b0 = src.index("# ---- comparison with the earlier run WITH definitions ----"), src.index("COMPARISON = compare_with_earlier()\n") + len("COMPARISON = compare_with_earlier()\n")
            src = src[:a0] + OURS_COMPARE.replace("__EARLIER__", repr(EARLIER)).replace("__PUBLISHED__", repr(PUBLISHED)).lstrip("\n") + src[b0:]
        routes_file = HERE + "/" + V.get("routes", "noinstruction_routes.json")
        embedded = json.load(open(routes_file)) if __import__("os").path.exists(routes_file) else {}
        embedded = embedded.get("routes", embedded)
        src = edit(src, 'SILENT_LAMBDA="0", SILENT_CLAIM="1", SILENT_KAPPA="0.5", SILENT_SIF="0.001", SILENT_EVAL="1"),',
                   'SILENT_LAMBDA="0", SILENT_CLAIM="1", SILENT_KAPPA="0.5", SILENT_SIF="0.001", SILENT_EVAL="1",\n'
                   '                      SA_NO_BLOCK="1"),    # plain training: adapters 1..t on for every training input')
        order = (json.load(open(routes_file)).get("task_order") if __import__("os").path.exists(routes_file) else None) or []
        cell = TABLE_CELL.replace("__ROUTES__", repr(embedded)).replace("__ORDER__", repr(order))
        src = edit(src, "# %% [markdown]\n# ## The runs", cell + "\n# %% [markdown]\n# ## The runs")
        # a rule that failed after one task is left out of the table, the other rules are kept
        src = edit(src, 'rules = steps[-1].get("rules") or ["certified", "nearest", "key", "all"]',
                   'rules = [r for r in (steps[-1].get("rules") or ["certified", "nearest", "key", "all"]) if all(r in (x.get("rules") or [r]) for x in steps)]')
        # the re-test table of a run must not stop the summary of the other runs
        src = edit(src, "        retest = retest_table(run_id)\n        if not retest:",
                   "        try:\n            retest = retest_table(run_id)\n        except Exception as err:\n"
                   "            print(f\"\\n{method}: the re-test table could not be read: {type(err).__name__}: {err}\")\n            continue\n        if not retest:")
    if V.get("seeds"):                                    # same notebook, other seeds: one run after the other, a block per seed and the mean
        src = edit(src, 'seeds=[42],                            # one full run per seed; for more seeds write for example [43, 44]',
                   "seeds=" + json.dumps(V["seeds"]) + ",                        # one full run per seed")
        src = src.replace('BY_SEED.get("42", {"summary": {}, "retest": {}})', 'BY_SEED.get(SEED_KEY, {"summary": {}, "retest": {}})')
        assert 'BY_SEED.get(SEED_KEY' in src
        src = edit(src, "COMPARISON = compare_with_earlier()\n",
                   'COMPARISON = {}\nfor SEED_KEY in [str(x) for x in CFG["seeds"]]:\n'
                   '    print("\\n" + "#" * 12 + f" seed {SEED_KEY}: against the numbers of the earlier runs (those are seed 42) " + "#" * 12)\n'
                   '    COMPARISON[SEED_KEY] = compare_with_earlier()\n')
    assert src.count("S0_TEST") >= 1

    cells, kind, buf = [], None, []


    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        elif text:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


    for line in src.split("\n"):
        if line.startswith("# %%"):
            flush(); kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
    json.dump(nb, open(OUT, "w"), indent=1)
    print(len(cells), "cells written to", OUT)



# =================================================================================================================
# Run U: the task definitions are KEPT, training blocks nothing (SA_NO_BLOCK=1), the re-test uses every rule at four wordings
# =================================================================================================================
ZIP_FUNCTION = '''
import zipfile, threading
ZIP_LOCK = threading.Lock()


def make_small_zip():
    """The small result files (scores, every answer of the re-test, logs) in one archive; written after every finished task."""
    zip_path = os.path.join(CFG["drive_dir"], CFG["run_tag"] + "_small_files.zip")
    with ZIP_LOCK:
        tmp = zip_path + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(SAVE_DIR):
                if ".partial" in root:
                    continue
                for f in files:
                    if f.endswith((".json", ".jsonl", ".log", ".txt")) and not f.startswith(("tokenizer", "special_tokens", "spiece")):
                        try:
                            z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), CFG["drive_dir"]))
                        except OSError:
                            pass
        os.replace(tmp, zip_path)
    return zip_path

'''

E100_COMPARE = '''
# ---- the 100-epoch run against the 10-epoch runs of the same seed (42) ----
# Sources: no definitions, 10 epochs: silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/small_files.zip;
# with definitions, 10 epochs: silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip (both read with this
# notebook's own functions scores_of and retest_table).
PUBLISHED_10_EPOCHS = __PUB__
EARLIER_WITH_DEFINITIONS = __EARLIER__
EARLIER_SOURCE = "10 epochs: silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks (no definitions) and 2026-11-17_kaggle_five_tasks_seed42 (with definitions)"


def compare_with_earlier():
    here = BY_SEED.get("42", {"summary": {}, "retest": {}})
    print("\\n" + "=" * 20 + " 100 epochs against 10 epochs, no task definition, seed 42 " + "=" * 20)
    print("learned: average score right after learning each task | AP: average score at the end | FT: forgetting (lower is better)")
    print("answers changed: answers to the inputs of the first four tasks (400) that differ between right after learning and the end")
    result = {}
    print(f"\\n{'method':<20}{'run':<34}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}")
    for m in CFG["methods"]:
        now = (here["retest"].get(m) or {}).get("table", {}).get("original|official")
        old = PUBLISHED_10_EPOCHS.get(m, {}).get("retest", {}).get("official")
        wd = EARLIER_WITH_DEFINITIONS.get(m, {}).get("retest", {}).get("official")
        result[m] = {"retest_100_epochs": now and {k: now[k] for k in ("learned", "AP", "FT", "changed", "compared")}, "retest_10_epochs_no_definition": old,
                     "retest_10_epochs_with_definition": wd}
        for label, v in (("100 epochs, no definitions", now), ("10 epochs, no definitions", old), ("10 epochs, with definitions", wd)):
            if v is None:
                print(f"{m if label.startswith('100') else '':<20}{label:<34}{'(not available)':>43}")
            else:
                print(f"{m if label.startswith('100') else '':<20}{label:<34}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}{str(v['changed']) + ' of ' + str(v['compared']):>18}")
    print("\\nofficial test (one random reference answer per example):")
    print(f"{'method':<20}{'run':<34}{'learned':>9}{'AP':>8}{'FT':>8}")
    for m in CFG["methods"]:
        now, old = here["summary"].get(m), PUBLISHED_10_EPOCHS.get(m, {}).get("official")
        result[m + "|official_test"] = {"100_epochs": now and {k: now[k] for k in ("learned", "AP", "FT", "tasks_done")}, "10_epochs_no_definition": old}
        for label, v in (("100 epochs, no definitions", now), ("10 epochs, no definitions", old)):
            if v is None:
                print(f"{m if label.startswith('100') else '':<20}{label:<34}{'(not available)':>25}")
            else:
                print(f"{m if label.startswith('100') else '':<20}{label:<34}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}")
    return result


COMPARISON = compare_with_earlier()

'''

FIRST_E100 = '''# %% [markdown]
# # Five tasks without the task instruction: the two published methods, 100 epochs
#
# **This notebook repeats the earlier no-instruction run of InfLoRA and GainLoRA (official code, seed 42, t5-base) with the PUBLISHED
# number of epochs: 100 per task instead of 10.** Everything else is as in the 10-epoch run: the task definition is removed from every
# input (the "Definition" of every data file of the five tasks becomes the empty string before anything is trained), the same
# five tasks, the same learning rate, batch size (effective 32), maximum lengths and thresholds. The official script ties the
# development-set check and the checkpoint to the number of epochs (one check and one saved checkpoint every 5 epochs, the best
# one by development ROUGE-L is kept), so with 100 epochs the best of 20 checkpoints is kept per task, with 10 epochs the best of 2.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** and *Internet* to **on**, then **Save Version -> Save & Run All**.
# The two methods run at the same time, one on each GPU. The 10-epoch run took 36 minutes per method; with ten times the training this
# takes about **4.5 to 5 hours** (training scales with the epochs; set-up, test and re-test do not). Nothing in the notebook has a time
# limit of its own. After every finished task the task's results are copied to the output folder and the small-files zip
# `/kaggle/working/silent_adapters/__TAG___small_files.zip` is refreshed, so an interrupted run keeps its finished tasks.
# After each task every test input is answered again (original wording only) and the answers that changed since the task was learned
# are counted, as before. The printed summary at the end shows these runs next to the 10-epoch no-definition numbers of the same seed.
'''

FIRST_15_PUBLISHED = '''# %% [markdown]
# # Fifteen tasks without the task instruction: the two published methods
#
# **This notebook runs InfLoRA and GainLoRA (official code, seed 42, t5-base, 10 epochs per task) on the 15 tasks of SuperNI order 1,
# one after another, with the task definition removed from every input.** Before anything is trained, the "Definition" of every
# data file of the 15 tasks (train, dev and test) is replaced by the empty string, so the input of every task is the same fixed
# text followed by the example itself (`Definition: ` + blank line + `Now complete the following example -` + `Input: ...` +
# `Output: `) and no longer tells which task it belongs to. The two methods run at the same time, one on each GPU.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** and *Internet* to **on**, then **Save Version -> Save & Run All**.
# Time, estimated from the earlier 15-task run with definitions (InfLoRA 188 minutes, GainLoRA 198 minutes of training and testing,
# the two in parallel): about **3.3 to 3.7 hours** including about 10 minutes of set-up. (Shorter inputs without definitions may make it a
# little faster; this is not counted on.) With one GPU the two methods run one after the other: about 6.5 hours. **The result is the
# small-files zip** `/kaggle/working/silent_adapters/__TAG___small_files.zip` (scores, every answer of the re-test, training logs).
# After each task every test input is answered again (original wording only) and the answers that changed since the task was
# learned are counted, as in the earlier runs. The tables are printed at the end.
'''

FIRST_15_OURS = '''# %% [markdown]
# # Fifteen tasks without the task instruction: our own method, plain training and route tables
#
# **This notebook runs the project's own job (`certified2` code with plain training, seed 42, t5-base, 10 epochs per task) on the 15
# tasks of SuperNI order 1, with the task definition removed from every input and with training that blocks nothing
# (`SA_NO_BLOCK=1`).** Before anything is trained, the "Definition" of every data file of the 15 tasks is replaced by the empty string.
# While task t is learned, adapters 1..t are on for every training input (plain sequential training: one adapter per task, older ones
# frozen and active).
#
# **What is evaluated.** After each task every test input is answered again (original wording only) under these rules, all from the
# same trained adapters: `all` (every adapter on = plain adapters), `centre`, and the route tables `table:ridge`, `table:joint`,
# `table:handicap` and `table:own` (every input to its own task: perfect recognition, for comparison). The route of a test input
# under a table rule is read from a table written before the run and embedded here (file `noinstruction_routes_15.json` of the
# repository). If a table lacks an entry the job stops before training; a table rule that fails is left out and the other rules are kept.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** (one GPU is used) and *Internet* to **on**, then
# **Save Version -> Save & Run All**. One job. Time, estimated from the earlier 15-task run (the certified job took 210 minutes with four
# wordings and five rules): about **3 to 3.5 hours** including about 10 minutes of set-up. **The result is the small-files zip**
# `/kaggle/working/silent_adapters/__TAG___small_files.zip` (scores, every answer of the re-test, training logs and the route table
# used). The tables are printed at the end.
'''

FIRST_WITHDEF = '''# %% [markdown]
# # Five tasks WITH task definitions: plain training, every rule, route tables
#
# **This notebook runs the project's own job (`certified2`, seed 42, t5-base, 10 epochs per task) on the same five tasks as the
# run of 2026-11-17, with the task definitions KEPT (nothing is removed from the data), and with training that blocks nothing
# (`SA_NO_BLOCK=1`).** While task t is learned, adapters 1..t are on for every training input (plain sequential training: one
# adapter per task, older ones frozen and active). With definitions no training input of these five tasks lies inside the zone of
# an older task, so this is the same training as in the certified run; the protected zones are still recorded while training, so
# the rules that need them can be evaluated.
#
# **What is evaluated.** After each task every test input is answered again at four wordings of the task definition (original,
# light, medium and heavy rewording; the same rewordings as in the earlier run), under these rules, all from the same trained
# adapters: `certified`, `nearest`, `centre`, `key`, `all` (as before), the route tables `table:ridge`, `table:handicap`
# (the two rules of our method that decide from the example text alone which task it belongs to), `table:joint`, `table:claims`,
# `table:own` (every input to its own task: perfect recognition, for comparison) and two mixed rules `zone+table:ridge` and
# `zone+table:handicap`: the certified route where the input is inside a zone, otherwise the table route. The tables depend only
# on the example text, so the same table is used at every wording. They were written before the run and are embedded in this
# notebook (file `noinstruction_routes.json` of the repository). If a table lacks an entry the job stops before training; a table
# rule that fails is left out and the other rules are kept.
#
# **Nothing has to be edited.** On Kaggle set *Accelerator* to **GPU T4 x2** (one GPU is used) and *Internet* to **on**, then
# **Save Version -> Save & Run All**. One job: about **55 to 65 minutes** (the earlier certified job took 36 minutes with four
# wordings and five rules; this one has twelve rules; set-up about 10 minutes). **The result is the small-files zip**
# `/kaggle/working/silent_adapters/__TAG___small_files.zip`. At the end the printed output has, for each wording and each rule:
# AP, answers changed of the 400 compared, and the share given to the right task, next to the numbers of the run of 2026-11-17.
'''

WITHDEF_COMPARE = '''
# ---- the final table, next to the run of 2026-11-17 (with definitions, seed 42) ----
# Source of the earlier numbers: silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip, read with this
# notebook's own functions scores_of and retest_table (AP, answers changed of the 400 compared, share given to the right task).
EARLIER_LEVELS = __LEVELS__
EARLIER_SOURCE = "silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip (seed 42, with definitions)"
WORDINGS = ["original", "light", "medium", "heavy"]
RULE_ORDER = ["all", "certified", "nearest", "centre", "key", "table:ridge", "table:handicap", "table:joint", "table:claims", "table:own",
              "zone+table:ridge", "zone+table:handicap"]


def compare_with_earlier():
    here = BY_SEED.get("42", {"summary": {}, "retest": {}})
    table = ((here["retest"].get("certified2") or {}).get("table", {}))
    have = {k.split("|")[1] for k in table}
    rules = [r for r in RULE_ORDER if r in have] + sorted(have - set(RULE_ORDER))
    print("\\n" + "=" * 20 + " every rule at every wording, and the run of 2026-11-17 (with definitions) " + "=" * 20)
    print("AP: average score after all tasks | changed: answers to the inputs of the first four tasks (400) that differ between right after")
    print("learning the task and the end | right task: share of test inputs given to their own task (not meaningful for `all`).")
    result = {"rules": {}, "earlier_source": EARLIER_SOURCE, "earlier": EARLIER_LEVELS}
    for level in WORDINGS:
        print(f"\\nwording: {level}\\n{'rule':<24}{'AP':>8}{'answers changed':>18}{'right task':>12}")
        for rule in rules:
            v = table.get(level + "|" + rule)
            if v is None:
                print(f"{rule:<24}{'(not available)':>38}")
                continue
            result["rules"][level + "|" + rule] = {k: v[k] for k in ("learned", "AP", "FT", "changed", "compared", "right_task")}
            right = f"{100 * v['right_task']:.1f}%" if (v["right_task"] is not None and rule != "all") else "-"
            print(f"{rule:<24}{v['AP']:>8.2f}{str(v['changed']) + ' of ' + str(v['compared']):>18}{right:>12}")
        print(f"{'earlier run 2026-11-17:':<24}")
        for m, rule, name in (("certified2", "certified", "certified switch"), ("gainlora_inflora", "official", "GainLoRA"), ("inflora", "official", "InfLoRA")):
            e = EARLIER_LEVELS[m][rule][level]
            right = f"{100 * e['right_task']:.1f}%" if e["right_task"] is not None else "-"
            print(f"  {name:<22}{e['AP']:>8.2f}{str(e['changed']) + ' of ' + str(e['compared']):>18}{right:>12}")
    s = here["summary"].get("certified2")
    if s:
        result["official_test"] = {k: s[k] for k in ("learned", "AP", "FT", "tasks_done")}
        print(f"\\nofficial test of this run (= rule all, original wording, one random reference answer): learned {s['learned']:.2f}, AP {s['AP']:.2f}, FT {s['FT']:.2f}")
    failed = {}
    for i, t in enumerate(CFG["tasks"]):
        f = os.path.join(SAVE_DIR, f"{CFG['run_tag']}_certified2_seed42", "outputs", f"{i + 1}-{t}", "extra_eval.json")
        if os.path.exists(f):
            for r, why in (json.load(open(f)).get("failed_rules") or {}).items():
                failed[f"{r} after task {i + 1}"] = why
    if failed:
        print("\\nRULES THAT FAILED (left out of the tables):", failed)
    result["failed_rules"] = failed
    return result


COMPARISON = compare_with_earlier()

'''

WITHDEF = dict(tag="superni_instance_routes_defs5_t5base_e10", out="five_tasks_instance_routes_with_definitions.ipynb")


def build_withdef():
    import os
    RUN_TAG, OUT = WITHDEF["tag"], (os.environ.get("NOINST_OUT_DIR") or HERE) + "/" + WITHDEF["out"]
    src = open(HERE + "/quick_test.py").read()
    a, b = src.index("# %% [markdown]"), src.index("# %%\nimport os, re, sys")
    src = src[:a] + FIRST_WITHDEF.replace("__TAG__", RUN_TAG) + "\n" + src[b:]
    src = edit(src, 'run_tag="superni_short5_t5base_e10",', f'run_tag="{RUN_TAG}",')
    src = edit(src, 'methods=["plain", "locked", "certified", "certified2", "inflora", "gainlora_inflora"],   # finished ones are skipped',
               'methods=["certified2"],   # the project\'s own job only')
    src = edit(src, 'only_if_finished=["plain", "locked", "certified"],   # shown when a finished copy is in the results folder, never trained',
               'only_if_finished=[],                   # nothing is only shown here')
    code = open(HERE + "/silent_run.py").read()
    assert "'''" not in code
    src = edit(src, "SILENT_RUN_SOURCE = None  # filled in by make_notebook.py", "SILENT_RUN_SOURCE = r'''" + code + "'''")
    routes_file = HERE + "/noinstruction_routes.json"
    data = json.load(open(routes_file))
    cell = TABLE_CELL.replace("__ROUTES__", repr(data["routes"])).replace("__ORDER__", repr(data["task_order"]))
    t0 = cell.index("if TEST:"); t1 = cell.index("def check_routes")
    cell = cell[:t0] + "\n" + cell[t1:]                             # no CPU-test-only table here
    src = edit(src, 'print("silent_run.py written:", len(SILENT_RUN_SOURCE.splitlines()), "lines")\n\n',
               'print("silent_run.py written:", len(SILENT_RUN_SOURCE.splitlines()), "lines")\n\n' + cell + "\n")
    src = edit(src, 'SILENT_LAMBDA="0", SILENT_CLAIM="1", SILENT_KAPPA="0.5", SILENT_SIF="0.001", SILENT_EVAL="1"),',
               'SILENT_LAMBDA="0", SILENT_CLAIM="1", SILENT_KAPPA="0.5", SILENT_SIF="0.001", SILENT_EVAL="1",\n'
               '                      SA_NO_BLOCK="1", SA_KEEP_RULES="1"),    # plain training; every rule is kept in the re-test')
    src = edit(src, 'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED},',
               WITHDEF_COMPARE.replace("__LEVELS__", repr({k: v for k, v in json.load(open(HERE + "/noinstruction_earlier_levels.json")).items() if not k.startswith("_")})).lstrip("\n") +
               'json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED, "task_definitions_kept": True,\n'
               '           "comparison": COMPARISON, "earlier_source": EARLIER_SOURCE},')
    src = edit(src, 'rules = steps[-1].get("rules") or ["certified", "nearest", "key", "all"]',
               'rules = [r for r in (steps[-1].get("rules") or ["certified", "nearest", "key", "all"]) if all(r in (x.get("rules") or [r]) for x in steps)]')
    src = edit(src, "        retest = retest_table(run_id)\n        if not retest:",
               "        try:\n            retest = retest_table(run_id)\n        except Exception as err:\n"
               "            print(f\"\\n{method}: the re-test table could not be read: {type(err).__name__}: {err}\")\n            continue\n        if not retest:")
    write_notebook(src, OUT)


def write_notebook(src, out):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        elif text:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})

    for line in src.split("\n"):
        if line.startswith("# %%"):
            flush(); kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
    json.dump(nb, open(out, "w"), indent=1)
    print(len(cells), "cells written to", out)


VARIANTS["fifteen_published"] = dict(tag="superni_noinstruction15_published_t5base_e10", methods=["inflora", "gainlora_inflora"],
                                     first=FIRST_15_PUBLISHED, preset="fifteen", silent_run="silent_run_frozen_published.py", out="fifteen_tasks_noinstruction_published.ipynb",
                                     what="the two published methods, 15 tasks, one per GPU")
VARIANTS["fifteen_ours"] = dict(tag="superni_noinstruction15_ours_t5base_e10", methods=["certified2"], first=FIRST_15_OURS,
                                preset="fifteen", silent_run="silent_run_frozen_ours.py", routes="noinstruction_routes_15.json", out="fifteen_tasks_noinstruction_ours.ipynb",
                                what="the project's own job, 15 tasks")

VARIANTS["published_e100"] = dict(tag="superni_noinstruction5_published_t5base_e100", methods=["inflora", "gainlora_inflora"], first=FIRST_E100,
                                  epochs=100, progressive_zip=True, compare="e100", silent_run="silent_run_frozen_published.py",
                                  out="five_tasks_noinstruction_published_e100.ipynb", what="the two published methods, 100 epochs, one per GPU")

for v in (list(VARIANTS) if VARIANT_ARGS == ["all"] else VARIANT_ARGS):
    build_withdef() if v == "withdef" else build(v)
