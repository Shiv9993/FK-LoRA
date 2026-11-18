# %% [markdown]
# # Silent adapters — stage 0: run the published methods (GainLoRA, InfLoRA)
#
# This notebook runs the **official GainLoRA code** on the SuperNI benchmark (15 tasks, learned one
# after another) in a reduced setting that fits a free Colab T4. It is step one of
# `silent_adapters/PLAN.md`: before testing our own idea, check that the published methods run and
# see how long they take.
#
# **How to use it**
# 1. Runtime → Change runtime type → **T4 GPU**.
# 2. Runtime → Run all. Allow Google Drive access when asked.
# 3. If Colab disconnects, open the notebook again and press Run all again. **It continues from the
#    last finished task.** Nothing has to be changed.
#
# **What is saved, and where**
# Everything goes to Google Drive, in `MyDrive/silent_adapters/<run_tag>/`:
# after each task, its results and the weights the next task needs are copied there. If the run is
# cut off in the middle of a task, only that one task is repeated.
#
# To start again from nothing, change `run_tag` in the settings cell (or delete the folder on Drive).

# %%
import os, re, sys, json, time, shlex, shutil, subprocess

CFG = dict(
    run_tag="superni_o1_t5base_e10",     # folder name on Drive. Change it to start a new, separate run.
    model="t5-base",                     # the paper uses t5-large; t5-base fits a T4
    epochs=10,                           # the paper uses 100
    order="superni_order1",              # which official task order to run
    methods=["inflora", "gainlora_inflora"],
    seed=42,                             # the official scripts use the default seed, 42
    train_batch=16,                      # lower to 8 if the GPU runs out of memory (effective batch stays 32)
    eval_batch=16,
    max_tasks=15,
    keep_all_files=False,                # False: large helper files of older tasks are deleted from Drive once
                                         # they are no longer needed (saves about 2 GB). Scores are always kept.
    drive_dir="/content/drive/MyDrive/silent_adapters",
    work_dir="/content/silent_adapters_work",
    repo_url="https://github.com/liangyanshuo/gainlora",
    repo_commit="e9d0d700538c8d12892f7fd52b216291f7e42acc",   # the exact version of the official code
)

# --- used only for the CPU self-test of this notebook; has no effect in Colab ---
TEST = os.environ.get("S0_TEST") == "1"
if TEST:
    CFG.update(run_tag="superni_selftest", model=os.environ["S0_MODEL"], epochs=1, max_tasks=3,
               drive_dir=os.environ["S0_DRIVE"], work_dir=os.environ["S0_WORK"])
    if os.environ.get("S0_METHODS"):
        CFG["methods"] = os.environ["S0_METHODS"].split(",")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
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
    print("GPU:", gpu)

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
# ## The runs
# The commands are read from the official scripts, one per task, so every setting is the authors'
# own. Only these are changed: the model, the number of epochs, the batch size (the effective batch
# of 32 is kept), the seed, and the folder names.

# %%
def official_commands(method):
    """Parse the official script into one argument list per task."""
    script = os.path.join(REPO, f"gen_script_{CFG['order']}_t5_{method}.sh")
    text = open(script).read().replace("\\\n", " ")
    official_run = f"gen_script_{CFG['order']}_t5_{method}"
    cmds = []
    for line in text.split("\n"):
        if "python src/run_t5.py" in line:
            args = shlex.split(line.split("python src/run_t5.py", 1)[1].replace("$2", "MODEL_PATH"))
            cmds.append(args)
    return cmds, official_run


def get(args, name):
    return args[args.index(name) + 1]


def put(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args += [name, str(value)]


def our_command(args, official_run, run_id, train_batch=None):
    train_batch = train_batch or CFG["train_batch"]
    args = [a.replace(f"logs_and_outputs/{official_run}/", f"logs_and_outputs/{run_id}/") for a in args]
    effective = int(get(args, "--per_device_train_batch_size")) * int(get(args, "--gradient_accumulation_steps"))
    put(args, "--model_name_or_path", CFG["model"])
    put(args, "--num_train_epochs", CFG["epochs"])
    put(args, "--per_device_train_batch_size", train_batch)
    put(args, "--gradient_accumulation_steps", max(1, effective // train_batch))   # effective batch stays the same
    put(args, "--per_device_eval_batch_size", CFG["eval_batch"])
    put(args, "--run_name", run_id)
    put(args, "--seed", CFG["seed"])
    if TEST:                                           # tiny data for the CPU self-test
        put(args, "--max_train_samples", 8); put(args, "--max_eval_samples", 8)
    return args


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


def run_method(method):
    cmds, official_run = official_commands(method)
    cmds = cmds[:CFG["max_tasks"]]
    run_id = f"{CFG['run_tag']}_{method}_seed{CFG['seed']}"
    local_out = os.path.join(REPO, "logs_and_outputs", run_id, "outputs")
    saved_out = os.path.join(SAVE_DIR, run_id, "outputs")
    log_dir = os.path.join(SAVE_DIR, run_id, "logs")
    os.makedirs(local_out, exist_ok=True); os.makedirs(saved_out, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    prog_path = os.path.join(SAVE_DIR, run_id, "progress.json")
    progress = json.load(open(prog_path)) if os.path.exists(prog_path) else {"tasks": {}}
    print(f"\n=== {method}: {len(cmds)} tasks ===")
    for i, raw in enumerate(cmds):
        args = our_command(list(raw), official_run, run_id)
        name = os.path.basename(get(args, "--output_dir"))             # e.g. "3-task1290_xsum_summarization"
        saved_task, local_task = os.path.join(saved_out, name), os.path.join(local_out, name)
        if os.path.exists(os.path.join(saved_task, "DONE")):
            if not os.path.exists(os.path.join(local_task, "DONE")):   # after a disconnect: bring it back from Drive
                shutil.rmtree(local_task, ignore_errors=True); shutil.copytree(saved_task, local_task)
            print(f"[{i + 1}/{len(cmds)}] {name}: already done, skipped")
            continue
        t0, log_path = time.time(), os.path.join(log_dir, name + ".log")
        print(f"[{i + 1}/{len(cmds)}] {name}: running ...", flush=True)
        while True:
            batch = progress.get("train_batch", CFG["train_batch"])    # smaller if this method ran out of memory before
            args = our_command(list(raw), official_run, run_id, batch)
            shutil.rmtree(local_task, ignore_errors=True)              # remove anything left from an interrupted try
            with open(log_path, "w") as log:
                fake_oom = TEST and os.environ.get("S0_FAKE_OOM") and i == 1 and batch == CFG["train_batch"]
                cmd = [ENV_PY, "-c", "print('CUDA out of memory (pretend, self-test)'); raise SystemExit(1)"] if fake_oom \
                    else [ENV_PY, "src/run_t5.py"] + args
                proc = subprocess.Popen(cmd, cwd=REPO, env=RUN_ENV, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    time.sleep(5 if TEST else 120)
                    if proc.poll() is None:
                        print(f"      ... {(time.time() - t0) / 60:.0f} min{latest_bar(log_path)}", flush=True)
            if proc.returncode == 0 and os.path.exists(os.path.join(local_task, "all_results.json")):
                break
            log_text = open(log_path, errors="replace").read()
            if "out of memory" in log_text.lower() and batch > 2:      # GPU memory: halve the batch and try this task again
                progress["train_batch"] = batch // 2
                json.dump(progress, open(prog_path, "w"), indent=1)
                print(f"      the GPU ran out of memory with batch {batch}; trying again with batch {batch // 2} "
                      f"(the effective batch size does not change)", flush=True)
                continue
            print("".join(l[:300] + ("\n" if len(l) > 300 else "") for l in log_text.splitlines(True)[-40:]))
            raise RuntimeError(f"{name} failed; the full log is in {log_path}")
        for d in os.listdir(local_task):                               # as the official script does
            if d.startswith("checkpoint"):
                shutil.rmtree(os.path.join(local_task, d), ignore_errors=True)
        open(os.path.join(local_task, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        copy_dir(local_task, saved_task)                               # now it is safe on Drive
        order_file = os.path.join(local_out, "task_order.txt")
        if os.path.exists(order_file):
            shutil.copy(order_file, os.path.join(saved_out, "task_order.txt"))
        if i > 0 and not CFG["keep_all_files"]:                        # task i-1's helper matrices are only read by task i
            older = os.path.basename(get(our_command(list(cmds[i - 1]), official_run, run_id), "--output_dir"))
            for base in (os.path.join(saved_out, older), os.path.join(local_out, older)):
                for f in os.listdir(base):
                    if f.startswith("reg_") and f.endswith(".pt"):
                        os.remove(os.path.join(base, f))
                shutil.rmtree(os.path.join(base, "trans_input"), ignore_errors=True)
        progress["tasks"][name] = {"minutes": (time.time() - t0) / 60, "train_batch": batch}
        json.dump(progress, open(prog_path, "w"), indent=1)
        print(f"      done in {(time.time() - t0) / 60:.1f} min, saved to Drive", flush=True)
    return run_id


RUN_IDS = {m: run_method(m) for m in CFG["methods"]}

# %% [markdown]
# ## Scores
# Same definitions as the official `score.py`. **AP** is the average score over all tasks at the very
# end (higher is better). **FT** is forgetting: how far each earlier task fell from its best score
# (lower is better). The score of a task is Rouge-L, as in the papers.

# %%
def scores_of(run_id):
    out = os.path.join(SAVE_DIR, run_id, "outputs")
    names = sorted((d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "DONE"))),
                   key=lambda d: int(d.split("-")[0]))
    tasks = [n.split("-", 1)[1] for n in names]
    rows = []
    for i, n in enumerate(names):
        res = json.load(open(os.path.join(out, n, "all_results.json")))
        rows.append([res[f"predict_eval_rougeL_for_{tasks[j]}"] for j in range(i + 1)])
    return tasks, rows


def ap_ft(rows):
    T = len(rows)
    ap = sum(rows[-1]) / T
    ft = (sum(max(r[j] for r in rows[j:-1]) - rows[-1][j] for j in range(T - 1)) / (T - 1)) if T > 1 else 0.0
    return ap, ft


summary = {}
for method, run_id in RUN_IDS.items():
    tasks, rows = scores_of(run_id)
    if not rows:
        continue
    ap, ft = ap_ft(rows)
    prog = json.load(open(os.path.join(SAVE_DIR, run_id, "progress.json")))
    minutes = sum(t["minutes"] for t in prog["tasks"].values())
    summary[method] = {"AP": ap, "FT": ft, "tasks_done": len(rows), "minutes": minutes, "final_scores": dict(zip(tasks, rows[-1])),
                       "right_after_learning": {t: rows[i][i] for i, t in enumerate(tasks)}}
    print(f"{method:>18}: {len(rows)} of {CFG['max_tasks']} tasks | AP {ap:6.2f} | FT {ft:6.2f} | {minutes / 60:.1f} GPU-hours")
json.dump({"cfg": CFG, "svd_backend": SVD_BACKEND, "summary": summary}, open(os.path.join(SAVE_DIR, "results.json"), "w"), indent=1)

if summary:
    tasks = list(next(iter(summary.values()))["final_scores"])
    print("\nscore of each task at the end (and right after it was learned)")
    print(f"{'task':<48}" + "".join(f"{m:>26}" for m in summary))
    for t in tasks:
        print(f"{t[:46]:<48}" + "".join(
            f"{s['final_scores'].get(t, float('nan')):>14.1f} ({s['right_after_learning'].get(t, float('nan')):5.1f})   "
            for s in summary.values()))
print("\nsaved:", os.path.join(SAVE_DIR, "results.json"))
print("This is a reduced setting (smaller model, fewer epochs). Do not compare these numbers with the papers' tables.")
