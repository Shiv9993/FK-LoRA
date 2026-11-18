# %% [markdown]
# # Silent adapters — short tests on five tasks
#
# Five short SuperNI tasks learned one after another (t5-base, 10 epochs, one seed). Exploratory: it
# shows whether an idea does anything on a real model before many GPU-hours are spent.
#
# | Method | What it is |
# |---|---|
# | `plain` | one new adapter per task, older ones frozen, all added together. No protection |
# | `locked` | **exact switch:** `plain`, and each input is answered with the adapters added after its task switched off. The task is recognised from the exact task definition at the start of the input |
# | `certified2` | **certified switch (final rule):** like `locked`, but the task is recognised from the frozen word-vectors of the input, not from the exact wording. Each task has a protected zone; an input inside a zone belongs to the oldest task that claims it, for ever. Inputs outside every zone go to the nearest zone. Common words count less |
# | `certified` | the first version of the certified switch (inputs outside every zone went to the nearest whole-input centre). Shown if it is on Drive; it is not trained again |
# | `inflora`, `gainlora_inflora` | published methods, official code, unchanged |
# | `silent`, `silent2` | `plain`, plus the new adapter is trained to change nothing on stand-in inputs (earlier idea) |
#
# **What this run does.** Methods that are already finished in the results folder are skipped.
# `plain`, `locked` and the first `certified` are only shown if a finished copy is there. `certified2`
# and the two published methods are trained on the same five tasks: 36 minutes each on a T4, so
# about 1.8 hours on one GPU and about 1.3 hours on two. For more seeds, change `seeds` in the first
# code cell (for example `seeds=[43, 44]`); every seed is a full run of every method.
#
# After each task, every test input is answered again with **reworded task definitions** (light,
# medium, heavy rewording). For `certified2` this is done under five rules with the same trained
# adapters: `certified`, `nearest` (nearest zone only, older zones have no priority), `centre` (nearest
# whole-input centre, the fallback of the first version), `key` (the exact switch of `locked`) and
# `all` (every adapter on, which is `plain`). The published methods are tested as they are.
#
# **How to use it on Colab:** Runtime → Change runtime type → **T4 GPU** → Run all. Allow Google Drive.
# After a disconnect, open it again and Run all: it continues from the last finished task.
# Results are saved in `MyDrive/silent_adapters/<run_tag>/`.
#
# **How to use it on Kaggle** (found out by the notebook itself): in the right-hand panel set
# *Accelerator* to **GPU T4 x2** and *Internet* to **on**, then **Save Version → Save & Run All**. It
# then runs in the background with the browser closed. Results are written to the notebook's output,
# `/kaggle/working/silent_adapters/<run_tag>/`, and everything printed is in the saved version.
# With two GPUs, two methods are trained at the same time, one on each GPU (one method alone is not
# faster: its tasks have to be learned one after another).

# %%
import os, re, sys, json, time, shlex, shutil, subprocess

PRESET = "five"       # "five": the five short tasks.  "fifteen": the whole SuperNI order 1 (15 tasks), `certified` only.

CFG = dict(
    run_tag="superni_short5_t5base_e10",   # folder name on Drive. Change it to start a new, separate run.
    model="t5-base",
    epochs=10,
    seeds=[42],                            # one full run per seed; for more seeds write for example [43, 44]
    tasks=["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
           "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
           "task875_emotion_classification"],
    methods=["plain", "locked", "certified", "certified2", "inflora", "gainlora_inflora"],   # finished ones are skipped
    only_if_finished=["plain", "locked", "certified"],   # shown when a finished copy is in the results folder, never trained
    # here: the re-test of `certified2` already contains plain adapters (`all`) and the exact switch (`key`)
    train_batch=16,                        # halved automatically if the GPU runs out of memory
    eval_batch=16,
    silent=dict(SILENT_LAMBDA="1.0", SILENT_POOL="400", SILENT_NEG_BATCH="8", SILENT_MAX_LEN="256", SILENT_KINDS="a,b"),
    silent2=dict(SILENT_LAMBDA="1.0", SILENT_POOL="800", SILENT_NEG_BATCH="8", SILENT_MAX_LEN="256", SILENT_KINDS="a,b,c,c"),
    locked=dict(SILENT_LAMBDA="0", SILENT_LOCK="1"),
    certified2=dict(SILENT_LAMBDA="0", SILENT_CLAIM="1", SILENT_KAPPA="0.5", SILENT_SIF="0.001", SILENT_EVAL="1"),
    certified=None,                        # first version (code of commit 7892ef6): only shown if finished, never trained again
    retest_baselines=True,                 # also test the published methods with reworded task definitions
    inflora=dict(SILENT_BASELINE="1"), gainlora_inflora=dict(SILENT_BASELINE="1"),
    keep_all_files=False,
    use_all_gpus=True,                     # with two GPUs (Kaggle "T4 x2") two methods are trained at the same time
    drive_dir="/content/drive/MyDrive/silent_adapters",
    work_dir="/content/silent_adapters_work",
    repo_url="https://github.com/liangyanshuo/gainlora",
    repo_commit="e9d0d700538c8d12892f7fd52b216291f7e42acc",   # the exact version of the official code
)

# --- used only for the CPU self-test of this notebook; has no effect in Colab ---
TEST = os.environ.get("S0_TEST") == "1"
if TEST and os.environ.get("S0_PRESET"):
    PRESET = os.environ["S0_PRESET"]

if PRESET == "fifteen":                    # the 15 tasks of SuperNI order 1, in the official order
    CFG.update(run_tag="superni_o1_t5base_e10_ours", methods=["certified2", "inflora", "gainlora_inflora"], tasks=[
        "task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
        "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
        "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
        "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
        "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
        "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"])

if TEST:
    CFG.update(run_tag="superni_quick_selftest" + ("15" if PRESET == "fifteen" else ""), model=os.environ["S0_MODEL"],
               epochs=1, tasks=CFG["tasks"][:3],
               drive_dir=os.environ["S0_DRIVE"], work_dir=os.environ["S0_WORK"])
    CFG["silent"].update(SILENT_POOL="16", SILENT_NEG_BATCH="4", SILENT_SELFCHECK="9")
    CFG["silent2"].update(SILENT_POOL="16", SILENT_NEG_BATCH="4")
    CFG["certified2"].update(SILENT_BG="1200")
    CFG["only_if_finished"] = ["certified"]
    CFG["train_batch"] = 8
    if os.environ.get("S0_METHODS"):
        CFG["methods"] = os.environ["S0_METHODS"].split(",")
    if os.environ.get("S0_SEEDS"):
        CFG["seeds"] = [int(x) for x in os.environ["S0_SEEDS"].split(",")]

IN_COLAB = "google.colab" in sys.modules
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/working")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
elif ON_KAGGLE and not TEST:                # Kaggle: results go to the notebook's output folder, the rest to a temporary one
    CFG.update(drive_dir="/kaggle/working/silent_adapters",
               work_dir=("/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp") + "/silent_adapters_work")
SAVE_DIR = os.path.join(CFG["drive_dir"], CFG["run_tag"])
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CFG["work_dir"], exist_ok=True)
print("results are saved in:", SAVE_DIR)

# %% [markdown]
# ## Set-up
# The official code needs old library versions (Python 3.10, PyTorch 2.1.0, Transformers 4.30.2).
# This cell builds a separate Python environment with exactly those versions, so Colab's own
# libraries are left alone. It takes a few minutes and is repeated after every disconnect.

# %%
def sh(cmd):
    """Run a shell command. Its output is shown only if it fails."""
    print("$", cmd if len(cmd) < 160 else cmd[:157] + "...", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-4000:])
        raise RuntimeError("this command failed (its output is printed above): " + cmd[:120])


def env_python(code):
    """Run a short piece of Python in the separate environment. Returns (worked, what it printed)."""
    r = subprocess.run([ENV_PY, "-c", code], capture_output=True, text=True, env=RUN_ENV)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


if not TEST:                                          # check the GPU first, before spending minutes on set-up
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    assert gpu, ("No GPU. On Colab: Runtime -> Change runtime type -> T4 GPU. On Kaggle: panel on the right -> "
                 "Session options -> Accelerator -> GPU T4 x2. Then run all again.")
    print("GPU:", gpu.replace("\n", " | "))
N_GPUS = int(os.environ.get("S0_GPUS") or (1 if TEST else len(gpu.splitlines())))

REPO = os.environ.get("S0_REPO") or os.path.join(CFG["work_dir"], "gainlora")
ENV_PY = os.environ.get("S0_ENV_PY") or os.path.join(CFG["work_dir"], "env", "bin", "python")
ENV_DIR = os.path.dirname(os.path.dirname(ENV_PY))

if not os.path.isdir(os.path.join(REPO, "src")):
    shutil.rmtree(REPO, ignore_errors=True)               # anything left from a download that failed half-way
    try:
        sh(f"git clone {CFG['repo_url']} {REPO}")
    except RuntimeError:
        import socket
        try:
            socket.create_connection(("github.com", 443), timeout=10).close()
        except OSError:
            raise RuntimeError(
                "NO INTERNET: the notebook cannot download the code. On Kaggle the internet is off until you switch it on: "
                "panel on the right -> Session options (or Notebook options) -> Internet -> on. The switch appears only "
                "after your phone number is verified (kaggle.com -> Settings -> Phone verification). Then run all again. "
                "Google Drive is not needed on Kaggle.") from None
        raise
    sh(f"git -C {REPO} checkout {CFG['repo_commit']}")
head = subprocess.run(f"git -C {REPO} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
assert head == CFG["repo_commit"], f"unexpected version of the official code: {head}"

if not os.path.exists(ENV_PY):
    sh(f"{sys.executable} -m pip install -q uv")
    sh(f"uv venv --python 3.10.12 {ENV_DIR}")
    sh(f"uv pip install -q --python {ENV_PY} torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
    sh(f"uv pip install -q --python {ENV_PY} transformers==4.30.2 datasets==2.14.6 accelerate==0.24.1 numpy==1.22.4 "
       "fsspec==2023.10.0 pyarrow==14.0.2 loralib nltk ipdb sentencepiece tabulate absl-py six rouge-score 'protobuf<4'")
    try:                                              # CuPy is optional: see below
        sh(f"uv pip install -q --python {ENV_PY} cupy-cuda12x==12.1.0 numpy==1.22.4")
    except RuntimeError:
        print("CuPy could not be installed; PyTorch will be used for that step instead.")

# The GPU libraries that came with PyTorch, so that CuPy can find them whatever Colab has installed.
import glob
nvidia_libs = sorted(glob.glob(os.path.join(ENV_DIR, "lib", "python3.10", "site-packages", "nvidia", "*", "lib")))
RUN_ENV = dict(os.environ, TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true", CUDA_VISIBLE_DEVICES="0",
               HF_HOME=os.path.join(CFG["work_dir"], "hf_cache"),
               LD_LIBRARY_PATH=":".join(nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":"))

ok, out = env_python("import torch, transformers, datasets, numpy; "
                     "print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', "
                     "datasets.__version__, '| numpy', numpy.__version__, '| GPU seen by PyTorch:', "
                     "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')")
print(out)
assert ok, "The Python environment is broken; the error is printed above."
assert TEST or "NONE" not in out, "PyTorch in the separate environment cannot see the GPU; the details are printed above."

# The official code uses the CuPy library for one step (finding the main directions of a matrix, an
# SVD, on the GPU). If CuPy does not work on this machine, the same step is done with PyTorch.
FALLBACK = os.path.join(CFG["work_dir"], "cupy_fallback")
FALLBACK_CODE = """\
# Stand-in for CuPy that does the same maths with PyTorch. Arrays are ordinary PyTorch tensors.
import torch
from torch.utils.dlpack import to_dlpack as _to, from_dlpack as _from
torch.Tensor.toDlpack = lambda self: _to(self)
def fromDlpack(capsule): return _from(capsule)
def dot(a, b): return a @ b
def hstack(parts): return torch.hstack(tuple(parts))
def ones(shape): return torch.ones(shape, device="cuda" if torch.cuda.is_available() else "cpu")
class linalg:
    @staticmethod
    def svd(a, full_matrices=True): return torch.linalg.svd(a, full_matrices=full_matrices)
"""
cupy_test = "import cupy as cp; u, s, v = cp.linalg.svd(cp.ones((4, 4)), full_matrices=False); print('ok', float((s ** 2).sum()))"
ok, out = (False, "forced for the self-test") if os.environ.get("S0_FORCE_FALLBACK") else env_python(cupy_test)
if ok and "cupy_fallback" not in out:
    SVD_BACKEND = "cupy"
else:
    print("CuPy does not work here. Reason:\n   ", "\n    ".join(out.splitlines()[-6:]))
    os.makedirs(os.path.join(FALLBACK, "cupy"), exist_ok=True)
    open(os.path.join(FALLBACK, "cupy", "__init__.py"), "w").write(FALLBACK_CODE)
    RUN_ENV["PYTHONPATH"] = ":".join([FALLBACK, RUN_ENV.get("PYTHONPATH", "")]).strip(":")
    ok, out = env_python(cupy_test)
    assert ok, "The PyTorch stand-in for CuPy failed too:\n" + out
    SVD_BACKEND = "pytorch (stand-in for cupy)"
print("SVD step done with:", SVD_BACKEND)

# %% [markdown]
# ## Our method's code
# `silent_run.py` starts the official training script unchanged and adds one extra loss: on stand-in
# inputs, the model with the new adapter must answer exactly as it did without it. The stand-in inputs
# are the task's own inputs under an unrelated task definition, and text from unrelated tasks. No
# other task of this benchmark is used, and nothing about old tasks is stored.

# %%
SILENT_RUN_SOURCE = None  # filled in by make_notebook.py
if SILENT_RUN_SOURCE is None:
    SILENT_RUN_SOURCE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "silent_run.py")).read()
open(os.path.join(REPO, "silent_run.py"), "w").write(SILENT_RUN_SOURCE)
print("silent_run.py written:", len(SILENT_RUN_SOURCE.splitlines()), "lines")

# %% [markdown]
# ## The runs
# Every setting that does not depend on the task order is read from the official scripts. The task
# order, the folders and the per-task files are built here for our five tasks.

# %%
def get(args, name):
    return args[args.index(name) + 1]


def put(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args += [name, str(value)]


def copy_dir(src, dst):
    """Copy a folder so that `dst` only appears when the copy is complete."""
    tmp = dst + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)


def latest_bar(log_path):
    """The newest progress bar in the log, e.g. ' (latest progress bar: 310/500)'. Training, checking and
    final testing each have their own bar, so the numbers restart a few times within one task."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 4000))
            bars = re.findall(r"(\d+)/(\d+) \[", f.read().decode(errors="replace"))
        return f" (latest progress bar: {bars[-1][0]}/{bars[-1][1]})" if bars else ""
    except OSError:
        return ""


ORDER = CFG["run_tag"]
for i, t in enumerate(CFG["tasks"]):                    # the small files that tell the official code which tasks to load
    d = os.path.join(REPO, "configs", f"{ORDER}_configs", t)
    os.makedirs(d, exist_ok=True)
    one = {"SuperNI": [{"sampling strategy": "full", "dataset name": t}]}
    json.dump(one, open(os.path.join(d, "train_tasks.json"), "w"), indent=2)
    json.dump(one, open(os.path.join(d, "dev_tasks.json"), "w"), indent=2)
    json.dump({"SuperNI": [{"sampling strategy": "full", "dataset name": x} for x in CFG["tasks"][:i + 1]]},
              open(os.path.join(d, "test_tasks.json"), "w"), indent=2)


def official_first_command(method):
    """The first command of the official script for this method: the source of all fixed settings."""
    text = open(os.path.join(REPO, f"gen_script_superni_order1_t5_{method}.sh")).read().replace("\\\n", " ")
    line = next(l for l in text.split("\n") if "python src/run_t5.py" in l)
    return shlex.split(line.split("python src/run_t5.py", 1)[1].replace("$2", "MODEL_PATH"))


def official_effective_batch():
    """Batch size times accumulation steps for each task, from the official script (32 for every task)."""
    text = open(os.path.join(REPO, "gen_script_superni_order1_t5_inflora.sh")).read().replace("\\\n", " ")
    out = {}
    for line in text.split("\n"):
        if "python src/run_t5.py" in line:
            a = shlex.split(line.split("python src/run_t5.py", 1)[1])
            out[os.path.basename(get(a, "--task_config_dir"))] = int(get(a, "--per_device_train_batch_size")) * \
                int(get(a, "--gradient_accumulation_steps"))
    return out


EFFECTIVE = official_effective_batch()


def build_command(method, i, run_id, batch, seed):
    tasks, t = CFG["tasks"], CFG["tasks"][i]
    out = f"logs_and_outputs/{run_id}/outputs"
    args = official_first_command("gainlora_inflora" if method == "gainlora_inflora" else "inflora")
    if method in OURS or method == "plain":             # the official plain-adapter mode, with its two penalties set to zero
        k = args.index("--threshold"); del args[k:k + 2]
        put(args, "--model_name", "olora"); put(args, "--lambda1", 0); put(args, "--lambda2", 0)
    put(args, "--model_name_or_path", CFG["model"])
    put(args, "--task_order", ",".join(tasks))
    put(args, "--task_config_dir", f"configs/{ORDER}_configs/{t}")
    put(args, "--output_dir", f"{out}/{i + 1}-{t}")
    put(args, "--metric_for_best_model", f"eval_rougeL_for_{t}")
    put(args, "--num_train_epochs", CFG["epochs"])
    put(args, "--per_device_train_batch_size", batch)
    put(args, "--gradient_accumulation_steps", max(1, EFFECTIVE[t] // batch))   # effective batch stays the same
    put(args, "--per_device_eval_batch_size", CFG["eval_batch"])
    put(args, "--run_name", run_id)
    put(args, "--seed", seed)
    if i > 0:                                           # the same extra arguments the official scripts give to later tasks
        prev = f"{out}/{i}-{tasks[i - 1]}/saved_weights"
        put(args, "--load_checkpoint_from", f"{prev}/trans_input.pt")
        put(args, "--previous_lora_path", ",".join(f"{out}/{j + 1}-{tasks[j]}/saved_weights" for j in range(i)))
        put(args, "--previous_prompt_key_path", f"{prev}/prompts_keys_till_now.pt")
        put(args, "--gen_data_dir", "generated_data/lora_gen_superni_t5")
        args.append("--add_instruction_replay")
    if TEST:                                            # tiny data for the CPU self-test
        put(args, "--max_train_samples", 96); put(args, "--max_eval_samples", 8)
        put(args, "--learning_rate", 0.01); put(args, "--num_train_epochs", 2)      # big changes, so forgetting shows up
    return args


OURS = ("silent", "silent2", "locked", "certified", "certified2")   # methods that go through silent_run.py
BASELINES = ("inflora", "gainlora_inflora")            # official methods; silent_run.py only adds the re-test to them


def run_method(method, seed, gpu=0, prefix=""):
    def say(text):
        print(prefix + text if prefix else text, flush=True)

    tasks = CFG["tasks"]
    run_id = f"{CFG['run_tag']}_{method}_seed{seed}"
    local_out = os.path.join(REPO, "logs_and_outputs", run_id, "outputs")
    saved_out = os.path.join(SAVE_DIR, run_id, "outputs")
    log_dir = os.path.join(SAVE_DIR, run_id, "logs")
    if (CFG.get(method, 0) is None or method in CFG["only_if_finished"]) and \
            not all(os.path.exists(os.path.join(saved_out, f"{i + 1}-{t}", "DONE")) for i, t in enumerate(tasks)):
        say(f"=== {method}: no finished copy in the results folder; it is only shown, never trained here (`only_if_finished`), skipped ===")
        return None
    os.makedirs(local_out, exist_ok=True); os.makedirs(saved_out, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    prog_path = os.path.join(SAVE_DIR, run_id, "progress.json")
    progress = json.load(open(prog_path)) if os.path.exists(prog_path) else {"tasks": {}}
    ours = method in OURS or (method in BASELINES and CFG["retest_baselines"])   # started through silent_run.py
    env = dict(RUN_ENV, **(CFG[method] or {})) if ours else dict(RUN_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    script = "silent_run.py" if ours else "src/run_t5.py"
    say(f"=== {method}{', seed ' + str(seed) if len(CFG['seeds']) > 1 else ''}: {len(tasks)} tasks ===")
    for i, t in enumerate(tasks):
        name = f"{i + 1}-{t}"
        saved_task, local_task = os.path.join(saved_out, name), os.path.join(local_out, name)
        if os.path.exists(os.path.join(saved_task, "DONE")):
            if not os.path.exists(os.path.join(local_task, "DONE")):   # after a disconnect: bring it back from Drive
                shutil.rmtree(local_task, ignore_errors=True); shutil.copytree(saved_task, local_task)
            say(f"[{i + 1}/{len(tasks)}] {name}: already done, skipped")
            continue
        t0, log_path = time.time(), os.path.join(log_dir, name + ".log")
        say(f"[{i + 1}/{len(tasks)}] {name}: running ...")
        while True:
            batch = progress.get("train_batch", CFG["train_batch"])    # smaller if this method ran out of memory before
            args = build_command(method, i, run_id, batch, seed)
            shutil.rmtree(local_task, ignore_errors=True)              # remove anything left from an interrupted try
            with open(log_path, "w") as log:
                proc = subprocess.Popen([ENV_PY, script] + args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    time.sleep(5 if TEST else 120)
                    if proc.poll() is None:
                        say(f"      ... {(time.time() - t0) / 60:.0f} min{latest_bar(log_path)}")
            if proc.returncode == 0 and os.path.exists(os.path.join(local_task, "all_results.json")):
                break
            log_text = open(log_path, errors="replace").read()
            if "out of memory" in log_text.lower() and batch > 2:      # GPU memory: halve the batch and try this task again
                progress["train_batch"] = batch // 2
                json.dump(progress, open(prog_path, "w"), indent=1)
                say(f"      the GPU ran out of memory with batch {batch}; trying again with batch {batch // 2} "
                    f"(the effective batch size does not change)")
                continue
            say(f"{name} failed. The end of its log:\n" + "".join(l[:300] + ("\n" if len(l) > 300 else "") for l in log_text.splitlines(True)[-40:]))
            raise RuntimeError(f"{name} failed; the full log is in {log_path}")
        for d in os.listdir(local_task):                               # as the official script does
            if d.startswith("checkpoint"):
                shutil.rmtree(os.path.join(local_task, d), ignore_errors=True)
        open(os.path.join(local_task, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        copy_dir(local_task, saved_task)                               # now it is safe on Drive
        if i > 0 and not CFG["keep_all_files"]:                        # the previous task's helper matrices are no longer needed
            for base in (os.path.join(saved_out, f"{i}-{tasks[i - 1]}"), os.path.join(local_out, f"{i}-{tasks[i - 1]}")):
                for f in os.listdir(base):
                    if f.startswith("reg_") and f.endswith(".pt"):
                        os.remove(os.path.join(base, f))
                shutil.rmtree(os.path.join(base, "trans_input"), ignore_errors=True)
        log_text = open(log_path, errors="replace").read()
        silence = re.findall(r"average silence loss over the last 50 steps ([0-9.e+-]+)", log_text)
        lock = re.findall(r"inputs seen while testing: (\d+); not matched to any task: (\d+)", log_text)
        zone = re.findall(r"\[certified\] task \d+: .*", log_text)
        same = re.findall(r"same answers as the official test on (\d+) of (\d+)", log_text)
        progress["tasks"][name] = {"minutes": (time.time() - t0) / 60, "train_batch": batch,
                                   "silence_loss_first_last": [float(silence[0]), float(silence[-1])] if silence else None,
                                   "test_inputs_seen_unmatched": [int(lock[-1][0]), int(lock[-1][1])] if lock else None,
                                   "zone": zone[-1] if zone else None,
                                   "retest_same_as_official": [int(same[-1][0]), int(same[-1][1])] if same else None}
        json.dump(progress, open(prog_path, "w"), indent=1)
        say(f"      {name} done in {(time.time() - t0) / 60:.1f} min, saved")
    return run_id


FAILED = {}                                            # runs that stopped with an error; the others carry on


def run_all(jobs):
    """Run every (method, seed): one after another, or two at a time when there are two GPUs (one run per GPU)."""
    many = len(CFG["seeds"]) > 1
    gpus = N_GPUS if CFG["use_all_gpus"] else 1
    done = {}

    def one(job, gpu, prefix):
        try:
            done[job] = run_method(job[0], job[1], gpu=gpu, prefix=prefix)
        except Exception as e:                                           # reported at the end; nothing else is stopped
            FAILED[f"{job[0]}, seed {job[1]}"] = str(e)
            print(f"{prefix}FAILED: {e}", flush=True)

    if gpus < 2 or len(jobs) < 2:
        for job in jobs:
            print()
            one(job, 0, "")
    else:
        import queue, threading
        print(f"{gpus} GPUs: {gpus} runs are trained at the same time, one per GPU. Lines are marked with the GPU and the run.")
        cost = {"gainlora_inflora": 3, "inflora": 2}                   # the slowest methods are started first
        todo = queue.Queue()
        for job in sorted(jobs, key=lambda job: -cost.get(job[0], 1)):
            todo.put(job)

        def worker(gpu):
            while True:
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                one(job, gpu, f"[GPU {gpu}, {job[0]}{', seed ' + str(job[1]) if many else ''}] ")

        threads = [threading.Thread(target=worker, args=(g,)) for g in range(gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return {job: run_id for job, run_id in done.items() if run_id}


RUNS = run_all([(m, s) for s in CFG["seeds"] for m in CFG["methods"]])

# %% [markdown]
# ## Scores
# **AP**: average score over the tasks at the very end (higher is better). **FT**: forgetting, how far
# earlier tasks fell from their best score (lower is better). **learned**: average score of each task
# right after it was learned. Scores are Rouge-L on about 100 test examples per task, so differences of
# a few points on one task are noise.

# %%
def scores_of(run_id):
    out = os.path.join(SAVE_DIR, run_id, "outputs")
    rows = []
    for i, t in enumerate(CFG["tasks"]):
        f = os.path.join(out, f"{i + 1}-{t}", "all_results.json")
        if not os.path.exists(os.path.join(out, f"{i + 1}-{t}", "DONE")):
            break
        res = json.load(open(f))
        rows.append([res[f"predict_eval_rougeL_for_{CFG['tasks'][j]}"] for j in range(i + 1)])
    return rows


def retest_table(run_id):
    """The re-test stored by a run: every rule and every wording, from the same trained weights."""
    out, tasks = os.path.join(SAVE_DIR, run_id, "outputs"), CFG["tasks"]
    steps = []
    for i, t in enumerate(tasks):
        f = os.path.join(out, f"{i + 1}-{t}", "extra_eval.json")
        if not (os.path.exists(f) and os.path.exists(os.path.join(out, f"{i + 1}-{t}", "DONE"))):
            break
        steps.append(json.load(open(f)))
    if not steps:
        return None
    T, table = len(steps), {}
    rules = steps[-1].get("rules") or ["certified", "nearest", "key", "all"]
    for level, last in steps[-1]["levels"].items():
        have = [j for j in range(T) if any(r["dataset"] == tasks[j] for r in last)]   # tasks that have this wording
        for rule in rules:
            score, by_id = {}, []
            for i in range(T):                                   # after task i+1: score of every task learned so far
                data = steps[i]["levels"].get(level, [])
                by_id.append({(r["dataset"], r["id"]): r for r in data})
                for j in have:
                    if j <= i:
                        s = [r[rule]["score"] for r in data if r["dataset"] == tasks[j]]
                        score[i, j] = sum(s) / len(s)
            earlier = [j for j in have if j < T - 1]
            changed = total = 0
            for j in earlier:                                    # answers right after learning task j+1 against the end
                for key, r in by_id[j].items():
                    if key[0] == tasks[j]:
                        total += 1
                        changed += r[rule]["pred"] != by_id[-1][key][rule]["pred"]
            routed = [r for r in last if r[rule]["route"] is not None]
            table[level, rule] = {
                "tasks": len(have), "learned": sum(score[j, j] for j in have) / len(have),
                "AP": sum(score[T - 1, j] for j in have) / len(have),
                "FT": sum(max(score[i, j] for i in range(j, T - 1)) - score[T - 1, j] for j in earlier) / len(earlier) if earlier else 0.0,
                "changed": changed, "compared": total,
                "right_task": sum(r[rule]["route"] == tasks.index(r["dataset"]) + 1 for r in routed) / len(routed) if routed else None,
                "certified": sum(r["in_zone"] for r in last), "inputs": len(last),
                "final_scores": {tasks[j]: score[T - 1, j] for j in have}, "learned_scores": {tasks[j]: score[j, j] for j in have}}
    return {"tasks_done": T, "table": {f"{a}|{b}": v for (a, b), v in table.items()},
            "same_as_official": [s.get("same_as_official") for s in steps]}


NAMES = {"certified": "certified (ours)", "nearest": "nearest only", "centre": "nearest centre", "key": "exact key (= locked)",
         "all": "all on (= plain)", "official": "as published"}


def report(run_ids):
    """Print the scores of the runs of one seed and return them as numbers."""
    summary = {}
    for method, run_id in run_ids.items():
        rows = scores_of(run_id)
        if not rows:
            continue
        T = len(rows)
        prog = json.load(open(os.path.join(SAVE_DIR, run_id, "progress.json")))
        summary[method] = {
            "tasks_done": T, "AP": sum(rows[-1]) / T,
            "FT": sum(max(r[j] for r in rows[j:-1]) - rows[-1][j] for j in range(T - 1)) / (T - 1) if T > 1 else 0.0,
            "learned": sum(rows[i][i] for i in range(T)) / T,
            "minutes": sum(x["minutes"] for x in prog["tasks"].values()), "rows": rows,
            "silence_loss_first_last": [x.get("silence_loss_first_last") for x in prog["tasks"].values()],
            "unmatched": [x.get("test_inputs_seen_unmatched") for x in prog["tasks"].values()],
        }

    print(f"{'method':<20}{'tasks':>6}{'learned':>10}{'AP':>8}{'FT':>8}{'minutes':>10}")
    for m, s in summary.items():
        print(f"{m:<20}{s['tasks_done']:>6}{s['learned']:>10.2f}{s['AP']:>8.2f}{s['FT']:>8.2f}{s['minutes']:>10.0f}")
    print("\neach task: score right after it was learned -> score at the end")
    print(f"{'task':<40}" + "".join(f"{m:>22}" for m in summary))
    for j, t in enumerate(CFG["tasks"]):
        cells = [f"{s['rows'][j][j]:6.1f} -> {s['rows'][-1][j]:5.1f}" if len(s["rows"]) > j else "-" for s in summary.values()]
        print(f"{t[:38]:<40}" + "".join(f"{c:>22}" for c in cells))
    for m in ("silent", "silent2"):
        if m in summary:
            print(f"\n{m}: silence loss per task (first 50 steps, last 50 steps):", summary[m]["silence_loss_first_last"])
    if "locked" in summary:
        print("\nlocked: test inputs (seen, not matched to a task) after each task:", summary["locked"]["unmatched"])
        same = all(abs(summary["locked"]["rows"][j][j] - summary["locked"]["rows"][-1][j]) < 1e-9 for j in range(len(summary["locked"]["rows"])))
        print("locked: every task scores exactly the same at the end as right after it was learned:", same)
    all_retests = {}
    for method, run_id in run_ids.items():
        retest = retest_table(run_id)
        if not retest:
            continue
        all_retests[method] = retest
        if method.startswith("certified"):
            prog = json.load(open(os.path.join(SAVE_DIR, run_id, "progress.json")))
            print(f"\n{method}: the protected zone of each task")
            for name, x in prog["tasks"].items():
                print("  ", x.get("zone"))
        print(f"\n{method} run, re-test after {retest['tasks_done']} tasks. Scores here use every reference answer (the official score")
        print("picks one at random), so they can differ a little from the table above. `tasks`: how many tasks have this wording.")
        print(f"{'wording':<10}{'rule':<22}{'tasks':>6}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}{'right task':>12}{'certified':>14}")
        for key, v in retest["table"].items():
            level, rule = key.split("|")
            right = f"{v['right_task']:.3f}" if (v["right_task"] is not None and rule != "all") else ""
            cert = f"{v['certified']} of {v['inputs']}" if rule == "certified" else ""
            print(f"{level:<10}{NAMES.get(rule, rule):<22}{v['tasks']:>6}{v['learned']:>9.2f}{v['AP']:>8.2f}{v['FT']:>8.2f}"
                  f"{str(v['changed']) + ' of ' + str(v['compared']):>18}{right:>12}{cert:>14}")
        print("re-test equals the official test (same answers, of inputs), per task:", retest["same_as_official"])
    if all_retests:
        print("\nanswers changed: answers to a task's test inputs that differ between right after learning it and the end.")
        print("right task: share of test inputs at the end that were given to their own task. certified: inputs inside a protected zone.")
    return {"summary": summary, "retest": all_retests}


def spread(values):
    return f"{sum(values) / len(values):.2f} ({min(values):.2f} to {max(values):.2f})"


BY_SEED = {}
for seed in CFG["seeds"]:
    if len(CFG["seeds"]) > 1:
        print("\n" + "=" * 40 + f" seed {seed} " + "=" * 40)
    BY_SEED[str(seed)] = report({m: RUNS[m, seed] for m in CFG["methods"] if (m, seed) in RUNS})

if len(CFG["seeds"]) > 1:
    full = len(CFG["tasks"])
    print("\n" + "=" * 40 + " all seeds together " + "=" * 40)
    print("Mean over the seeds, with the lowest and the highest seed in brackets. Only seeds in which the method finished every task.")
    print(f"{'method':<20}{'seeds':>6}{'learned':>26}{'AP':>26}{'FT':>26}")
    for m in CFG["methods"]:
        done = [b["summary"][m] for b in BY_SEED.values() if m in b["summary"] and b["summary"][m]["tasks_done"] == full]
        if done:
            print(f"{m:<20}{len(done):>6}" + "".join(f"{spread([d[k] for d in done]):>26}" for k in ("learned", "AP", "FT")))
    print("\nre-test (every reference answer): mean over the seeds; answers changed are added up")
    print(f"{'method':<20}{'wording':<10}{'rule':<22}{'seeds':>6}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>20}")
    for m in CFG["methods"]:
        tables = [b["retest"][m]["table"] for b in BY_SEED.values() if m in b["retest"] and b["retest"][m]["tasks_done"] == full]
        for key in (tables[0] if tables else []):
            level, rule = key.split("|")
            vals = [t[key] for t in tables if key in t]
            print(f"{m:<20}{level:<10}{NAMES.get(rule, rule):<22}{len(vals):>6}"
                  + "".join(f"{sum(v[k] for v in vals) / len(vals):>{w}.2f}" for k, w in (("learned", 9), ("AP", 8), ("FT", 8)))
                  + f"{str(sum(v['changed'] for v in vals)) + ' of ' + str(sum(v['compared'] for v in vals)):>20}")

json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "by_seed": BY_SEED, "failed": FAILED},
          open(os.path.join(SAVE_DIR, "results.json"), "w"), indent=1)

import zipfile                                         # the small result files in one archive, easy to download and keep
ZIP = os.path.join(CFG["drive_dir"], CFG["run_tag"] + "_small_files.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(SAVE_DIR):
        for f in files:
            if f.endswith((".json", ".jsonl", ".log", ".txt")) and not f.startswith(("tokenizer", "special_tokens", "spiece")):
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), CFG["drive_dir"]))
print("\nsaved:", os.path.join(SAVE_DIR, "results.json"), "| scores, answers and logs in one file:", ZIP)
if FAILED:
    print("\nRUNS THAT FAILED (they are missing or incomplete in the tables above; run all again to continue them):")
    for name, why in FAILED.items():
        print("  ", name, "-", why[:300])
print(f"{len(CFG['seeds'])} seed(s), {len(CFG['tasks'])} tasks, reduced setting: an early look, not a result. Copy everything above into the chat.")
