"""Writes no_instruction_kaggle.ipynb (run in any folder: python3 make_noinstruction_notebook.py). The project files are embedded as they are on disk now."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED = ["certified_switch.py", "demo_no_instruction.py"]   # demo_no_instruction.py imports only certified_switch (checked)
CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md("""
# Continual learning without task instructions: "all adapters" against "zones"

**Nothing has to be edited.** Five SuperNI tasks are learned one after another with t5-base, with the task definition removed from every
input. Each task gets its own low-rank adapter (800 training examples, 10 epochs, learning rate 3e-4). The same adapters are then used in two ways:
**all** (all adapters switched on) and **zones** (each task claims a zone built from the inputs of its training examples; an input is answered
with the adapters up to the oldest task whose zone holds it). After the last task the test inputs (100 per task) are answered again and compared
with the answers right after learning. Two runs, seeds 42 and 43, one per GPU, in parallel.

**How to use it on Kaggle:** right-hand panel, *Session options*: Accelerator **GPU T4 x2**, **Internet on** (needed for the data and the
t5-base weights; if it is off the notebook stops at once and says so). Then **Save Version -> Save & Run All**.

**How long:** the earlier notebook needed about 45 minutes per job on a T4; the two jobs here run at the same time, so expect about
**45 to 60 minutes** in total (an estimate, not measured on a GPU for this script).

**The result** is the file `/kaggle/working/no_instruction_small_files.zip` in the notebook's output (two JSON result files and two logs, no model weights).
A small table is also printed at the end.
""")
code('''
import os, sys, json, time, shutil, subprocess, threading, zipfile

NB_TEST = os.environ.get("NI_NB_TEST") == "1"          # only to test this notebook on a CPU; has no effect on Kaggle
OUT = os.environ.get("NI_NB_OUT") or "/kaggle/working/no_instruction"
ZIP = "/kaggle/working/no_instruction_small_files.zip" if not NB_TEST else OUT.rstrip("/") + "_small_files.zip"
CODE = os.path.join(OUT, "code")
LOGS = os.path.join(OUT, "logs")
RESULTS = os.path.join(OUT, "results")
WORK = os.environ.get("NI_NB_WORK") or ("/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp") + "/no_instruction_work"
CFG = dict(
    model="t5-base", seeds=[42, 43], train=800, test=100, epochs=10, lr="3e-4",
    repo_url="https://github.com/liangyanshuo/gainlora",
    repo_commit="e9d0d700538c8d12892f7fd52b216291f7e42acc",     # the version of the data repository used for the earlier runs
    progress_every_s=60,
)
if NB_TEST:
    CFG.update(model="t5-small", seeds=[42], train=16, test=4, epochs=1, lr="3e-4", progress_every_s=20)
for d in (CODE, LOGS, RESULTS, WORK):
    os.makedirs(d, exist_ok=True)
print("results are written to:", OUT)
''')
md("""
## The code
The next cell holds the project's two Python files (copies, not edited); the cell after it writes them to disk.
""")
for name in EMBED:
    text = open(os.path.join(HERE, name)).read()
    assert "'''" not in text and not text.endswith("\\"), name
    code(f'FILES = globals().get("FILES", {{}})\nFILES["{name}"] = r\'\'\'' + text + "'''")
code('''
for name, text in FILES.items():
    with open(os.path.join(CODE, name), "w") as f:
        f.write(text)
print("written to", CODE, ":", ", ".join(f"{n} ({len(t.splitlines())} lines)" for n, t in FILES.items()))
if "cuda" not in FILES["demo_no_instruction.py"] and "device" not in FILES["demo_no_instruction.py"]:
    print("NOTE: demo_no_instruction.py never mentions cuda/device, so each job runs on the CPU of the machine unless the script is changed.")
''')
md("## Machine, data, model")
code('''
NO_INTERNET = ("NO INTERNET: the notebook cannot download what it needs ({what}). On Kaggle the internet is off until you switch it on: panel on "
               "the right -> Session options (or Notebook options) -> Internet -> on. The switch appears only after your phone number is "
               "verified (kaggle.com -> Settings -> Phone verification). Then run all again.")

gpu_lines = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
if not NB_TEST:
    assert gpu_lines, "No GPU. On Kaggle: panel on the right -> Session options -> Accelerator -> GPU T4 x2. Then run all again."
N_GPUS = len(gpu_lines)
print(f"GPUs seen: {N_GPUS}" + ("" if not gpu_lines else " -> " + " | ".join(gpu_lines)))
if not NB_TEST and N_GPUS < len(CFG["seeds"]):
    print(f"only {N_GPUS} GPU(s): the {len(CFG['seeds'])} jobs will share them (the second one starts when the first has finished if there is 1 GPU)")
WORKERS = max(1, min(N_GPUS, len(CFG["seeds"]))) if not NB_TEST else 1

DATA = os.environ.get("NI_NB_DATA")
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
            raise RuntimeError("git clone failed although github.com can be reached:\\n" + (r.stdout + r.stderr)[-2000:])
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
        raise RuntimeError(NO_INTERNET.format(what=f"the {CFG['model']} weights from huggingface.co") + "\\n\\nDetails:\\n" + text[-1500:])
    raise RuntimeError("loading the model failed:\\n" + text[-3000:])
print(r.stdout.strip())
''')
md("""
## The runs
One process per seed, one GPU each (`CUDA_VISIBLE_DEVICES`), one log file each in `logs/`. The last line of each log is printed every minute.
""")
code('''
FAILED, RUNNING = {}, {}
lock = threading.Lock()
queue_ = list(CFG["seeds"])


def tail(path, n=1):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 6000))
            lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def run_job(seed, gpu):
    log_path = os.path.join(LOGS, f"seed{seed}.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    cmd = [sys.executable, os.path.join(CODE, "demo_no_instruction.py"), "--model", CFG["model"], "--data", DATA, "--train", str(CFG["train"]),
           "--test", str(CFG["test"]), "--epochs", str(CFG["epochs"]), "--lr", CFG["lr"], "--seed", str(seed), "--out", RESULTS, "--tag", "gpu"]
    t0 = time.time()
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=CODE)
        with lock:
            RUNNING[seed] = (proc, log_path, t0, gpu)
        code = proc.wait()
    with lock:
        RUNNING.pop(seed, None)
        if code != 0:
            FAILED[seed] = code
    print(f"[GPU {gpu}] seed {seed}: {'finished' if code == 0 else 'FAILED (exit code %d)' % code} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if code != 0:
        print("   the end of its log:\\n   " + "\\n   ".join(tail(log_path, 40)), flush=True)


def worker(gpu):
    while True:
        with lock:
            if not queue_:
                return
            seed = queue_.pop(0)
        run_job(seed, gpu)


def reporter(stop):
    while not stop.wait(CFG["progress_every_s"]):
        with lock:
            now = list(RUNNING.items())
        for seed, (proc, path, t0, gpu) in now:
            last = tail(path, 2)
            print(f"  ... GPU {gpu}, seed {seed}, {(time.time() - t0) / 60:.0f} min: " + (" | ".join(l[:160] for l in last) if last else "(no output yet)"), flush=True)


print(f"starting {len(CFG['seeds'])} job(s) on {WORKERS} worker(s)", flush=True)
stop = threading.Event()
threading.Thread(target=reporter, args=(stop,), daemon=True).start()
threads = [threading.Thread(target=worker, args=(g,)) for g in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
stop.set()
''')
md("""
## Results
Per rule: **learned** = mean score of each task right after it was learned; **at the end** = mean score after the last task; **forgetting** = mean drop
of tasks 1 to 4; **answers changed** = test inputs of tasks 1 to 4 whose answer at the end differs from the one right after learning. Scores are Rouge-L (0 to 100).
""")
code('''
rows = []
for seed in CFG["seeds"]:
    p = os.path.join(RESULTS, f"no_instruction_{CFG['model'].replace('/', '_')}_seed{seed}gpu.json")
    if not os.path.exists(p):
        print(f"seed {seed}: no result file ({p})")
        continue
    d = json.load(open(p))
    for rule in ("all", "zones"):
        x = d["rules"][rule]
        rows.append((seed, rule, x["learned"], x["AP"], x["FT"], f"{x['changed']} of {x['compared']}", d["verdict"] if rule == "zones" else ""))
print(f"{'seed':<6}{'rule':<8}{'learned':>9}{'at the end':>12}{'forgetting':>12}{'answers changed':>18}  verdict")
for s, rule, a, b, c, ch, v in rows:
    print(f"{s:<6}{rule:<8}{a:9.2f}{b:12.2f}{c:12.2f}{ch:>18}  {v}")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:       # small result files only; no model weights anywhere
    for root, _, files in os.walk(OUT):
        for f in files:
            if f.endswith((".json", ".log")):
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), os.path.dirname(OUT.rstrip("/"))))
print("zip with all .json and .log files:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.2f} MB)")
if FAILED:
    print("\\nJOBS THAT FAILED:", {s: c for s, c in FAILED.items()}, "- logs in", LOGS)
else:
    print("\\nall jobs finished")
''')
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": CELLS,
      "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open(os.path.join(HERE, "no_instruction_kaggle.ipynb"), "w"), indent=1)
print(len(CELLS), "cells written to no_instruction_kaggle.ipynb")
