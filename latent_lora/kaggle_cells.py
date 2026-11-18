# %% [markdown]
# # Latent-LoRA against the certified switch: five tasks, three seeds
#
# **A comparison run, nothing to edit.** Five SuperNI tasks are learned one after another (t5-base, 10 epochs, 800 training
# examples per task, seeds 42, 43, 44), once with **our method** (`ContinualModel` of `certified_switch.py`, rule "certified";
# from the same adapters also "nearest" and "all" = plain adapters) and once with **Latent-LoRA** (Azghan et al., arXiv:2607.23837,
# *re-implemented from the paper's text*, because no official code exists; rank 32, alpha 16, lambda 0.05, Gaussian-mixture router
# with K=5). Both go through the same data loading, training loop (AdamW, constant learning rate 3e-4, effective batch 32), greedy
# decoding and Rouge-L scoring (`run_five.py`). After every task all test inputs of all tasks learned so far are answered again, with the
# original definition and with three reworded definitions, and every answer is saved.
#
# **How to use it on Kaggle:** right-hand panel, *Session options*: Accelerator **GPU T4 x2** (one GPU also works), **Internet on**
# (needed for the data repository and for the t5-base weights; if it is off, the notebook stops at once and says so). Then
# **Save Version -> Save & Run All**. With two GPUs, two (method, seed) jobs run at the same time, one per GPU, each as its own
# process with its own log file.
#
# **What it writes** (all under `/kaggle/working/latent_lora_five/`): `results/<method>_seed<seed>.json` (every answer, after every task),
# `logs/*.log`, `summary.json`, and at the end `/kaggle/working/latent_lora_five_small_files.zip` with all `.json` and `.log` files (no model weights).
# Every `run_five.py` process first runs a self-test on the real device (two tiny tasks per method); if it fails, nothing else is started.

# %%
import os, sys, re, json, time, shutil, subprocess, threading, zipfile

NB_TEST = os.environ.get("LL_NB_TEST") == "1"          # only used to test this notebook on a CPU; has no effect on Kaggle
ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = os.environ.get("LL_NB_OUT") or "/kaggle/working/latent_lora_five"
ZIP = OUT.rstrip("/") + "_small_files.zip"
CODE = os.path.join(OUT, "code")
LOGS = os.path.join(OUT, "logs")
WORK = os.environ.get("LL_NB_WORK") or ("/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp") + "/latent_lora_work"
CFG = dict(
    model="t5-base",
    seeds=[42, 43, 44],
    methods=["ours", "latent_lora"],
    repo_url="https://github.com/liangyanshuo/gainlora",
    repo_commit="e9d0d700538c8d12892f7fd52b216291f7e42acc",     # the version of the data repository used for the earlier runs
    minutes_per_job=45,                                          # rough guess for one (method, seed) on a T4, see the estimate below
    progress_every_s=180,
)
if NB_TEST:
    CFG.update(seeds=[42], progress_every_s=20)
os.makedirs(CODE, exist_ok=True); os.makedirs(LOGS, exist_ok=True); os.makedirs(WORK, exist_ok=True)
print("results are written to:", OUT)

# %% [markdown]
# ## The code
# These cells hold the Python files; the next cell writes them to disk. `certified_switch.py` and `rewordings.json` are copies of the
# project's files (not edited); `latent_lora.py` and `run_five.py` are the new files.

# %%
FILES = {}

# %%
FILES["certified_switch.py"] = None  # filled in by make_notebook.py

# %%
FILES["latent_lora.py"] = None  # filled in by make_notebook.py

# %%
FILES["run_five.py"] = None  # filled in by make_notebook.py

# %%
FILES["rewordings.json"] = None  # filled in by make_notebook.py

# %%
for name, text in FILES.items():
    with open(os.path.join(CODE, name), "w") as f:
        f.write(text)
print("written to", CODE, ":", ", ".join(f"{n} ({len(t.splitlines())} lines)" for n, t in FILES.items()))

# %% [markdown]
# ## Machine, data, model

# %%
NO_INTERNET = ("NO INTERNET: the notebook cannot download what it needs ({what}). On Kaggle the internet is off until you switch it on: panel on "
               "the right -> Session options (or Notebook options) -> Internet -> on. The switch appears only after your phone number is "
               "verified (kaggle.com -> Settings -> Phone verification). Then run all again.")

gpu_lines = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
if not NB_TEST:
    assert gpu_lines, ("No GPU. On Kaggle: panel on the right -> Session options -> Accelerator -> GPU T4 x2 (or P100). Then run all again.")
N_GPUS = len(gpu_lines)
WORKERS = int(os.environ.get("LL_NB_WORKERS") or (N_GPUS if N_GPUS else 1))    # one job per GPU
print(f"GPUs seen: {N_GPUS}" + ("" if not gpu_lines else " -> " + " | ".join(gpu_lines)))
print(f"{WORKERS} job(s) at a time" + (", one per GPU" if not NB_TEST else " (test mode, CPU)"))

DATA = os.environ.get("LL_NB_DATA")
if not DATA:
    repo = os.path.join(WORK, "gainlora")
    if not os.path.isdir(os.path.join(repo, "CL_Benchmark", "SuperNI")):
        shutil.rmtree(repo, ignore_errors=True)
        r = subprocess.run(["git", "clone", CFG["repo_url"], repo], capture_output=True, text=True)
        if r.returncode != 0:
            import socket
            try:
                socket.create_connection(("github.com", 443), timeout=10).close()
            except OSError:
                raise RuntimeError(NO_INTERNET.format(what="git clone " + CFG["repo_url"])) from None
            raise RuntimeError("git clone failed although github.com can be reached:\n" + (r.stdout + r.stderr)[-2000:])
        subprocess.run(["git", "-C", repo, "checkout", CFG["repo_commit"]], check=True, capture_output=True, text=True)
    DATA = os.path.join(repo, "CL_Benchmark")
TASKS = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification", "task748_glucose_reverse_cause_event_detection",
         "task1687_sentiment140_classification", "task875_emotion_classification"]
for t in TASKS:
    for split in ("train", "test"):
        assert os.path.exists(os.path.join(DATA, "SuperNI", t, split + ".json")), f"missing data file {t}/{split}.json in {DATA}"
for t in "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(","):
    assert os.path.exists(os.path.join(DATA, "Long_Sequence", t, "train.json")), f"missing Long_Sequence/{t}/train.json in {DATA}"
print("data:", DATA)

# the model and tokenizer, downloaded once here (so that the parallel jobs do not download at the same time)
check = ("import torch, transformers; from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; "
         f"AutoTokenizer.from_pretrained('{CFG['model']}'); AutoModelForSeq2SeqLM.from_pretrained('{CFG['model']}'); "
         "print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())")
r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
if r.returncode != 0 and "sentencepiece" in (r.stdout + r.stderr).lower():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"], capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
if r.returncode != 0:
    text = r.stdout + r.stderr
    if any(w in text for w in ("ConnectionError", "Max retries", "Temporary failure", "Name or service", "offline", "LocalEntryNotFound", "couldn't connect", "Can't load")):
        raise RuntimeError(NO_INTERNET.format(what=f"the {CFG['model']} weights from huggingface.co") + "\n\nDetails:\n" + text[-1500:])
    raise RuntimeError("loading the model failed:\n" + text[-3000:])
print(r.stdout.strip())

# %% [markdown]
# ## Time estimate
# Rough guess, not measured on a GPU: about 45 minutes for one (method, seed) job on a T4 (10 epochs x 800 examples x 5 tasks of training, plus
# re-answering about 6000 test inputs per rule after the tasks; the earlier notebook needed 36 minutes per method for a similar setting).

# %%
jobs = [(m, s) for s in CFG["seeds"] for m in CFG["methods"]]
if not NB_TEST:
    rounds = -(-len(jobs) // WORKERS)
    print(f"{len(jobs)} jobs ({len(CFG['methods'])} methods x {len(CFG['seeds'])} seeds), {WORKERS} at a time, about {CFG['minutes_per_job']} min each: "
          f"estimated total about {rounds * CFG['minutes_per_job'] / 60:.1f} hours (estimate, not measured).")
else:
    print(f"{len(jobs)} jobs in tiny test mode")

# %% [markdown]
# ## The runs
# One process per job, one log file per job in `logs/`. A progress line per running job is printed every few minutes.

# %%
FAILED, RUNNING, ABORT = {}, {}, []
lock = threading.Lock()
queue_ = list(jobs)


def tail(path, n=1):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 6000))
            lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def run_job(job, gpu):
    method, seed = job
    log_path = os.path.join(LOGS, f"{method}_seed{seed}.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    if NB_TEST:
        env["LL_TEST"] = "1"
    cmd = [sys.executable, os.path.join(CODE, "run_five.py"), "--data", DATA, "--out", OUT, "--methods", method, "--seeds", str(seed)]
    t0 = time.time()
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=CODE)
        with lock:
            RUNNING[job] = (proc, log_path, t0, gpu)
        code = proc.wait()
    with lock:
        RUNNING.pop(job, None)
        if code != 0:
            FAILED[job] = code
            if code == 2:                                            # the self-test failed: the same would happen to every job
                ABORT.append(job)
    print(f"[GPU {gpu}] {method} seed {seed}: {'finished' if code == 0 else 'FAILED (exit code %d)' % code} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if code != 0:
        print("   the end of its log:\n   " + "\n   ".join(tail(log_path, 40)), flush=True)


def worker(gpu):
    while True:
        with lock:
            if not queue_ or ABORT:
                return
            job = queue_.pop(0)
        run_job(job, gpu)


def reporter(stop):
    while not stop.wait(CFG["progress_every_s"]):
        with lock:
            now = list(RUNNING.items())
        for (m, s), (proc, path, t0, gpu) in now:
            last = tail(path)
            print(f"  ... GPU {gpu}, {m} seed {s}, {(time.time() - t0) / 60:.0f} min: {last[0][:200] if last else '(no output yet)'}", flush=True)


print(f"starting {len(jobs)} jobs on {WORKERS} worker(s); the first thing every job does is a self-test on its GPU", flush=True)
stop = threading.Event()
threading.Thread(target=reporter, args=(stop,), daemon=True).start()
threads = [threading.Thread(target=worker, args=(g,)) for g in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
stop.set()
if ABORT:
    print("\n" + "=" * 90 + "\nTHE SELF-TEST FAILED, so no further job was started. Its message is printed above (the end of the log). "
          "Copy it into the chat.\n" + "=" * 90)

# %% [markdown]
# ## Results
# **Learned**: mean score of each task right after it was learned. **AP**: mean score after the last task. **FT**: forgetting.
# **Answers changed**: test inputs of tasks 1 to 4 whose answer at the end differs from the answer right after their task was learned
# (for "ours" with rule "certified" this is 0 by construction). Scores are Rouge-L, mean over the seeds.

# %%
r = subprocess.run([sys.executable, os.path.join(CODE, "run_five.py"), "--summary", "--out", OUT], capture_output=True, text=True, cwd=CODE,
                   env=dict(os.environ, LL_TEST="1") if NB_TEST else None)
print(r.stdout + r.stderr[-1500:])
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:       # the small result files only; there are no model weights anywhere
    for root, _, files in os.walk(OUT):
        for f in files:
            if f.endswith((".json", ".log")):
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), os.path.dirname(OUT.rstrip("/"))))
print("zip with all .json and .log files:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")
if FAILED:
    print("\nJOBS THAT FAILED (their finished tasks are in results/; run all again to repeat the unfinished jobs):")
    for (m, s), c in FAILED.items():
        print(f"   {m} seed {s}: exit code {c}, log {os.path.join(LOGS, f'{m}_seed{s}.log')}")
else:
    print("\nall jobs finished")
