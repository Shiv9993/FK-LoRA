"""Builds hidden_states_scale.ipynb (PLAN.md, 2026-11-18 07:48 UTC): frozen T5-base encoder states computed on the GPU, and the
task-recognition rules L, H' and J of paper_certified/hidden_rules_study.py measured on the two benchmarks and on 90 fresh tasks.

Usage: python3 make_notebook_hidden.py          (writes hidden_states_scale.ipynb next to this file; reads hidden_states_constants.json,
                                                 made by hidden_states_constants.py from the CPU files)
The rule code (functions mse, standardiser, routes_joint, ridge_sums, routes_ridge, routes_handicap, measures) is copied verbatim from
../paper_certified/hidden_rules_study.py by syntax-tree extraction, and the parser (objects) from ../paper_certified/many_tasks_fetch.py.
The task definitions are never read."""
import ast, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hidden_states_scale.ipynb")


def segments(path, names):
    text = open(path).read(); tree = ast.parse(text); got = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            got[node.name] = ast.get_source_segment(text, node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            got[node.targets[0].id] = ast.get_source_segment(text, node)
    assert set(got) == set(names), set(names) - set(got)
    return got


RULES_PY = os.path.join(HERE, "..", "paper_certified", "hidden_rules_study.py")
rs = segments(RULES_PY, ["mse", "standardiser", "routes_joint", "ridge_sums", "routes_ridge", "routes_handicap", "measures", "fresh_split", "ORDER1", "ORDER2"])
ms = segments(os.path.join(HERE, "..", "paper_certified", "many_tasks_fetch.py"), ["objects"])
ORDERS = {}
for n in ("ORDER1", "ORDER2"):
    ns = {}; exec(rs[n], ns); ORDERS[n] = ns[n]
K = json.load(open(os.path.join(HERE, "hidden_states_constants.json")))
assert len(K["fresh"]) == 90 and len({t["name"] for t in K["fresh"]}) == 90 and len(K["mu"]) == 768 == len(K["sd"])
CONST = {"fresh": [[t["name"], t["n"], t["sha1"]] for t in K["fresh"]], "pool_sha1": K["pool_sha1"], "pool_shape": K["pool_shape"], "mu": K["mu"], "sd": K["sd"]}

SRC = r'''# %% [markdown]
# # Hidden states of the frozen T5-base encoder at scale: the task-recognition rules on two benchmarks and on 120 tasks
#
# Nothing is trained. Per task, the frozen T5-base encoder (fp32, eval, no adapter) turns each instance text (the **input only**, the task
# definition is never read; 256 tokens) into one vector (mean of the last-layer states over the tokens), standardised with the mean and
# standard deviation of a pool of 2,000 unrelated inputs (embedded below, computed on a CPU beforehand). Three rules then give every test input
# a task after each arrival: **L** (ridge, lam 1), **H'** (joint ridge with a handicap, t = 2, floor) and **J** (joint logistic regression, C = 1;
# only for sequences of at most 30 tasks). The measures are those of `paper_certified/hidden_rules_study.py` (the code is copied from it).
#
# Sequences: SuperNI order 1 and 2 (a check of the code against the CPU numbers), Long Sequence order 3 and 4 (all official test inputs),
# fresh30 / fresh60 / fresh90 (90 tasks of the natural-instructions repository, batches 1-3, downloaded by name at a fixed commit),
# and all120 = SuperNI order 1 + Long Sequence order 3 + the 90 fresh tasks.
#
# **On Kaggle:** Accelerator GPU T4, Internet on, then *Save & Run All*. About 20-40 minutes (the encoding is about 10-15 minutes on the GPU; the
# rule J is the slow part). Result: `/kaggle/working/hidden_states_scale/results.json` (written after every step; also printed as tables).

# %% [markdown]
# ## Settings and imports

# %%
QUICK = False        # True: every task restricted to 40 training and 20 test inputs, t5-small (a test of the logic on a CPU); the real run is False
import os, sys, json, math, random, time, subprocess, hashlib, re, shutil
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression

OUT_DIR = os.environ.get("HS_OUT", "/kaggle/working/hidden_states_scale")
_tmp = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
WORK = os.environ.get("HS_WORK", os.path.join(_tmp, "hidden_states_scale"))      # clone, downloaded inputs, feature cache (not part of the output)
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(WORK, exist_ok=True)
OFFICIAL_URL = "https://github.com/liangyanshuo/gainlora"
OFFICIAL_COMMIT = "e9d0d700538c8d12892f7fd52b216291f7e42acc"
NI_COMMIT = "55a365637381ce7f3748fa2eac7aef1a113bbb82"
NI_RAW = f"https://raw.githubusercontent.com/allenai/natural-instructions/{NI_COMMIT}/"
MAXTOK = 256
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
log("QUICK =", QUICK, "| output", OUT_DIR, "| work", WORK)

# %% [markdown]
# ## Embedded constants
# The 90 fresh tasks (name, number of parsed instances, sha1 fingerprint of the first 1000 inputs joined with the character 0x1e, computed on the
# CPU files) and the mean / standard deviation (float32 values, written with full precision) of the 2,000-input pool, from `pool_t5hidden.npy`
# (sha1 of the array bytes below) exactly as `standardiser()` computes them (std below 1e-6 set to 1).

# %%
CONST = __CONST__
FRESH = CONST["fresh"]
ORDER1 = __ORDER1__
ORDER2 = __ORDER2__
LONG3 = "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
log("pool sha1", CONST["pool_sha1"], "| fresh tasks", len(FRESH))

# %% [markdown]
# ## The rules and the measures (copied verbatim from `paper_certified/hidden_rules_study.py`)

# %%
__RULES__

def fingerprint(inputs):
    return hashlib.sha1("\x1e".join(inputs).encode("utf-8", errors="surrogatepass")).hexdigest()

def standardiser_embedded(mu, sd):
    """The same expression as standardiser(), with the pool mean and standard deviation (float32) given."""
    mu = np.array(mu, dtype=np.float32); sd = np.array(sd, dtype=np.float32)
    return lambda X: ((X - mu) / sd).astype(np.float64)

# %% [markdown]
# ## Data: the official repository (SuperNI and Long Sequence) and the 90 fresh tasks

# %%
def sh(cmd, cwd=None, tries=1, wait=20):
    for k in range(tries):
        r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        log(f"command failed ({k + 1}/{tries}): {cmd[:90]} -> {r.stderr.strip()[-300:]}")
        if k + 1 < tries: time.sleep(wait * (k + 1))
    raise RuntimeError(f"command failed after {tries} tries: {cmd}\n{r.stderr[-500:]}")

def get_repo():
    d = os.path.join(WORK, "gainlora")
    if os.path.isdir(os.path.join(d, "CL_Benchmark")):
        try:
            if sh("git rev-parse HEAD", cwd=d).strip() == OFFICIAL_COMMIT: return d
        except Exception: pass
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    try:
        sh("git init -q && git remote add origin " + OFFICIAL_URL, cwd=d)
        sh(f"git fetch -q --depth 1 origin {OFFICIAL_COMMIT} && git checkout -q FETCH_HEAD", cwd=d, tries=3)
    except Exception as e:
        log("shallow fetch of the commit failed, full clone instead:", str(e)[:200])
        shutil.rmtree(d, ignore_errors=True)
        sh(f"git clone -q {OFFICIAL_URL} {d}", tries=3)
        sh(f"git checkout -q {OFFICIAL_COMMIT}", cwd=d)
    assert sh("git rev-parse HEAD", cwd=d).strip() == OFFICIAL_COMMIT
    return d

REPO = os.environ.get("HS_LOCAL_REPO") or get_repo()      # HS_LOCAL_REPO: test hook only (a local copy of the same commit)
BENCH = os.path.join(REPO, "CL_Benchmark")
log("official repository at", REPO, "| commit", OFFICIAL_COMMIT)

# the order of Long Sequence order 4: the task_config_dir entries of the official script, in order
_txt = open(os.path.join(REPO, "gen_script_long_order4_t5_inflora.sh")).read()
LONG4 = re.findall(r"--task_config_dir\s+\S*?/([A-Za-z0-9_]+)(?=\s|\\|$)", _txt)
assert len(LONG4) == 15 and set(LONG4) == set(LONG3), LONG4
log("Long Sequence order 4:", ",".join(LONG4))

def load_inputs(folder, name, split):
    d = json.load(open(os.path.join(BENCH, folder, name, split + ".json"), encoding="utf-8"))
    return [str(i["input"]) for i in d["Instances"]]      # the Definition is never read

# ---- the 90 fresh tasks: same Range request and same parser as paper_certified/fresh_tasks_fetch.py
__OBJECTS__

HEAD, NINST = 12 * 1024 * 1024, 1000

def fetch_text(name, attempts=6):
    cmd = ["curl", "-sS", "-f", "-L", "--max-time", "300", "--retry", "2", "-H", f"Range: bytes=0-{HEAD - 1}", NI_RAW + f"tasks/{name}.json"]
    err = ""
    for k in range(attempts):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout.decode("utf-8", errors="ignore")
        err = r.stderr.decode(errors="ignore").strip()[-200:] or "empty answer"
        wait = min(150, 10 * 2 ** k)            # GitHub rate limits (HTTP 429/403) and transient errors: wait longer each time
        log(f"download of {name}: attempt {k + 1}/{attempts} failed ({err}); waiting {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"download of {name} failed after {attempts} attempts: {err}")

def parse_inputs(text):
    inst = []
    for o in objects(text, "Instances"):
        if isinstance(o.get("input"), str) and o["input"].strip():
            out0 = o.get("output")
            out0 = out0[0] if isinstance(out0, list) and out0 else out0
            if isinstance(out0, str):
                inst.append(o["input"])
    return inst

CPU_FRESH_DIR = os.environ.get("HS_CPU_FRESH")          # test hook only: use CPU copies for the tasks beyond the first HS_DL_LIMIT
DL_LIMIT = int(os.environ.get("HS_DL_LIMIT", "90"))
os.makedirs(os.path.join(WORK, "fresh"), exist_ok=True)
FRESH_INPUTS, DATA_DIFFERS, DOWNLOAD_FAILED, MATCH = {}, [], [], 0
for i, (name, n_cpu, sha_cpu) in enumerate(FRESH):
    cache = os.path.join(WORK, "fresh", f"{i:02d}_{name}.json")
    try:
        if i >= DL_LIMIT and CPU_FRESH_DIR:
            allt = {t["name"]: t for f in ("fresh_tasks.json", "fresh_tasks_2.json", "fresh_tasks_3.json") for t in json.load(open(os.path.join(CPU_FRESH_DIR, f)))["tasks"]}
            inputs = [str(x["input"]) for x in allt[name]["instances"]]
        elif os.path.exists(cache):
            inputs = json.load(open(cache))
        else:
            inputs = parse_inputs(fetch_text(name)); json.dump(inputs, open(cache + ".tmp", "w")); os.replace(cache + ".tmp", cache)
    except Exception as e:
        log("WARNING:", e); DOWNLOAD_FAILED.append(name); continue
    inputs = inputs[:NINST]; FRESH_INPUTS[name] = inputs
    same = len(inputs) == n_cpu and fingerprint(inputs) == sha_cpu
    MATCH += same
    if not same: DATA_DIFFERS.append({"name": name, "parsed": len(inputs), "expected": n_cpu})
    if i % 10 == 9 or i == len(FRESH) - 1: log(f"fresh tasks {i + 1}/{len(FRESH)} read; fingerprints matching so far {MATCH}")
log(f"FINGERPRINT CHECK: {MATCH} of {len(FRESH)} fresh tasks identical to the CPU files; data_differs = {len(DATA_DIFFERS)}; download_failed = {len(DOWNLOAD_FAILED)}")
if DATA_DIFFERS: print("data_differs:", DATA_DIFFERS)
if DOWNLOAD_FAILED: print("WARNING: not downloaded, left out of every sequence:", DOWNLOAD_FAILED)

# %% [markdown]
# ## Splits and features
# Benchmark tasks: n training inputs, `perm` = seed-0 shuffle, `sel` = first min(n, 400); held-out = the first m of `sel` (m = 200 if n >= 400 else n // 2), fit = the
# rest; test = all official test inputs. Fresh tasks: `fresh_split()` (seed-0 shuffle of range(800), first 400: 200 held-out, then 200 fit; test = instances 800..999).

# %%
import torch
from transformers import AutoTokenizer, T5EncoderModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "t5-small" if QUICK else "t5-base"
if DEVICE == "cpu" and not QUICK: log("WARNING: no GPU found; the encoding of about 150,000 inputs on a CPU will take hours")
tok = AutoTokenizer.from_pretrained(MODEL_NAME); model = T5EncoderModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)
TPB = 16000 if DEVICE == "cuda" else 8000
log("encoder", MODEL_NAME, "on", DEVICE, "| fp32")

def enc(texts, tokens_per_batch=TPB):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); res = np.zeros((len(texts), model.config.d_model), np.float32); s0 = 0
    while s0 < len(order):
        e = s0 + 1
        while e < len(order) and e - s0 < 64 and (e - s0 + 1) * min(MAXTOK, len(texts[order[e]]) // 3 + 4) <= tokens_per_batch: e += 1
        b = order[s0:e]; s0 = e
        t = tok([texts[i] for i in b], max_length=MAXTOK, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            h = model(**t).last_hidden_state; m = t["attention_mask"][:, :, None].to(h.dtype)
            res[b] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    return res

FEATS, SPLIT_INFO = {}, {}      # key -> {"cal","fit","test"} raw float32 features
os.makedirs(os.path.join(WORK, "feats_" + MODEL_NAME), exist_ok=True)

def cached(key, make):
    f = os.path.join(WORK, "feats_" + MODEL_NAME, key.replace("/", "__") + (".quick" if QUICK else "") + ".npz")
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files}
    d = make(); np.savez(f, **d); return d

def bench_task(folder, name):
    tr, te = load_inputs(folder, name, "train"), load_inputs(folder, name, "test")
    if QUICK: tr, te = tr[:40], te[:20]
    n = len(tr); perm = list(range(n)); random.Random(0).shuffle(perm); sel = perm[:min(n, 400)]; m = 200 if n >= 400 else n // 2
    X = enc([tr[i] for i in sel]); SPLIT_INFO[f"{folder}/{name}"] = {"n_train": n, "m_heldout": m, "n_fit": len(sel) - m, "n_test": len(te)}
    return {"cal": X[:m], "fit": X[m:], "test": enc(te)}

def random_split():
    """The second version of the fresh split: ONE seed-0 shuffle of all 1,000 instances; held-out = first 200, fit = next 200, test = next 200."""
    perm = list(range(1000)); random.Random(0).shuffle(perm)
    return perm[:200], perm[200:400], perm[400:600]

def fresh_task(name):
    """Features of the union of the positions used by either split, computed once; both splits index into them."""
    x = FRESH_INPUTS[name]; n = len(x); sp = [list(a) for a in fresh_split()]; rp = [list(a) for a in random_split()]
    sp = [[j for j in a if j < n] for a in sp]; rp = [[j for j in a if j < n] for a in rp]
    if QUICK: sp = [a[:20] for a in sp]; rp = [a[:20] for a in rp]
    pos = sorted(set(sum(sp, []) + sum(rp, []))); X = enc([x[j] for j in pos]); at = {j: i for i, j in enumerate(pos)}
    SPLIT_INFO["fresh/" + name] = {"n_instances": n, "union_positions": len(pos), "official_like": [len(a) for a in sp], "random": [len(a) for a in rp]}
    out = {}
    for pre, parts in (("", sp), ("r", rp)):
        for k, a in zip(("cal", "fit", "test"), parts): out[pre + k] = X[[at[j] for j in a]]
    return out

bench_keys = [("SuperNI", n) for n in ORDER1] + [("Long_Sequence", n) for n in LONG3]
for folder, name in bench_keys:
    FEATS[f"{folder}/{name}"] = cached(f"{folder}/{name}", lambda: bench_task(folder, name))
    log("features", f"{folder}/{name}", {k: v.shape for k, v in FEATS[f"{folder}/{name}"].items()})
for i, (name, _, _) in enumerate(FRESH):
    if name not in FRESH_INPUTS: continue
    FEATS["fresh/" + name] = cached("fresh/" + name, lambda: fresh_task(name))
    if i % 10 == 9: log("features of fresh tasks up to", i + 1)
for k in list(FEATS):
    if k not in SPLIT_INFO: SPLIT_INFO[k] = {"from_cache": True, "shapes": {a: list(b.shape) for a, b in FEATS[k].items()}}
bad = [k for k, v in FEATS.items() if min(len(v[k]) for k in v) == 0]
if bad: print("WARNING: tasks with an empty split, left out:", bad); [FEATS.pop(k) for k in bad]
del model
if DEVICE == "cuda": torch.cuda.empty_cache()
if QUICK:       # the pool statistics are for t5-base (768 numbers); with t5-small the statistics come from the fit features themselves
    ST = standardiser(np.vstack([v["fit"] for v in FEATS.values()]))
else:
    ST = standardiser_embedded(CONST["mu"], CONST["sd"])
log("features done:", len(FEATS), "tasks;", sum(len(v["test"]) for v in FEATS.values()), "test inputs;", sum(len(v["fit"]) + len(v["cal"]) for v in FEATS.values()), "fit+held-out inputs")

# %% [markdown]
# ## Sequences and rules

# %%
FRESH_KEYS = ["fresh/" + n for n, _, _ in FRESH if "fresh/" + n in FEATS]
S1 = ["SuperNI/" + n for n in ORDER1]; S2 = ["SuperNI/" + n for n in ORDER2]
L3 = ["Long_Sequence/" + n for n in LONG3]; L4 = ["Long_Sequence/" + n for n in LONG4]
fk = lambda a, b: [k for k in FRESH_KEYS if a <= [n for n, _, _ in FRESH].index(k[6:]) < b]      # tasks of the batches, in the order of the batches
SEQS = {"superni_order1": S1, "superni_order2": S2, "long_order3": L3, "long_order4": L4,
        "fresh30": fk(0, 30), "fresh30_random": fk(0, 30), "fresh60": fk(0, 60), "fresh60_random": fk(0, 60),
        "fresh90": fk(0, 90), "fresh90_random": fk(0, 90), "all120": S1 + L3 + fk(0, 90), "all120_random": S1 + L3 + fk(0, 90)}
# "_random": the fresh tasks use the second split (one random split of all 1,000 instances); benchmark tasks keep their official test inputs
PAIRS = [("fresh30", "fresh30_random"), ("fresh60", "fresh60_random"), ("fresh90", "fresh90_random"), ("all120", "all120_random")]
J_MAX_TASKS = 30
RESULTS = {"about": "Hidden states of the frozen T5 encoder at scale (PLAN.md 2026-11-18 07:48 UTC). Rules L (ridge lam 1), H' (t = 2, floor) and J (C = 1) of hidden_rules_study.py.",
           "quick": QUICK, "model": MODEL_NAME, "device": DEVICE, "official_commit": OFFICIAL_COMMIT, "natural_instructions_commit": NI_COMMIT, "max_tokens": MAXTOK,
           "pool_sha1": CONST["pool_sha1"], "pool_shape": CONST["pool_shape"], "standardised_with": "the embedded pool statistics" if not QUICK else "fit features (QUICK only)",
           "fingerprint_check": {"matching": MATCH, "of": len(FRESH), "data_differs": DATA_DIFFERS, "download_failed": DOWNLOAD_FAILED},
           "long_order4": LONG4, "split_info": SPLIT_INFO, "sequences": {}, "timings_s": {}}
RES_FILE = os.path.join(OUT_DIR, "results.json")
def dump():
    json.dump(RESULTS, open(RES_FILE + ".tmp", "w")); os.replace(RES_FILE + ".tmp", RES_FILE)

def pack(r):
    return {"mean": r["mean"], "se": r["se"], "per_task": r["per_task"], "certificate_valid": r["certificate_valid"]}

def arrays(keys, tag):
    pre = lambda k: "r" if (tag.endswith("_random") and k.startswith("fresh/")) else ""
    return [ST(FEATS[k][pre(k) + "fit"]) for k in keys], [ST(FEATS[k][pre(k) + "cal"]) for k in keys], [ST(FEATS[k][pre(k) + "test"]) for k in keys]

def run_LH(tag):
    keys = SEQS[tag]; t = time.time(); fit, cal, te = arrays(keys, tag)
    Rt, Rc, _ = routes_ridge(fit, cal, te, 1); L = measures(keys, Rt, Rc); del Rt, Rc
    Rt, Rc, ex = routes_handicap(fit, cal, te, 1, [2], floor=True)[2]; H = measures(keys, Rt, Rc, extra=ex, held_out_used_by_rule=True, handicap_t=2); del Rt, Rc
    res = RESULTS["sequences"].setdefault(tag, {"tasks": keys}); res["L"] = pack(L); res["H_prime_t2"] = pack(H)
    if tag in ("long_order3", "long_order4"):
        m, s = L["mean"], L["se"]
        res["reading_rule_L"] = {"own_at_end": m["own_at_end"],
                                 "use": "works on the second benchmark" if m["own_at_end"] >= 0.90 else ("does not" if m["own_at_end"] < 0.75 else "partial"),
                                 "mean_changed": m["changed"], "mean_certificate": m["certificate"], "se_changed": s["changed"],
                                 "audit": "met" if m["changed"] <= m["certificate"] + 2 * s["changed"] else "not met"}
    RESULTS["timings_s"][tag + "|L,H"] = round(time.time() - t, 1); dump(); log(tag, "L and H' done", round(time.time() - t), "s")

def run_J(tag):
    keys = SEQS[tag]; t = time.time(); fit, cal, te = arrays(keys, tag)
    Rt, Rc = routes_joint(fit, cal, te, 1); J = measures(keys, Rt, Rc); del Rt, Rc
    RESULTS["sequences"][tag]["J"] = pack(J); RESULTS["timings_s"][tag + "|J"] = round(time.time() - t, 1); dump(); log(tag, "J done", round(time.time() - t), "s")

for tag in SEQS: run_LH(tag)
RESULTS["fresh_split_comparison"] = {"about": "random split minus official-like split (fresh_split), difference of the means over tasks"}
for a, b in PAIRS:
    RESULTS["fresh_split_comparison"][b + " - " + a] = {rule: {k: RESULTS["sequences"][b][rule]["mean"][k] - RESULTS["sequences"][a][rule]["mean"][k] for k in ("changed", "changed_ever", "own_at_end")}
                                                         for rule in ("L", "H_prime_t2")}
dump()

# %% [markdown]
# ## Rule J (joint logistic regression, C = 1) for the sequences of at most 30 tasks
# Retrained at every arrival with scikit-learn; for the larger sequences (60, 90, 120 tasks) it is skipped (the time would be too long) and the result says so.

# %%
J_BUDGET_S = 2.5 * 3600 if not QUICK else 600; tJ = time.time()
for tag in SEQS:
    if len(SEQS[tag]) > J_MAX_TASKS:
        RESULTS["sequences"][tag]["J"] = f"skipped: {len(SEQS[tag])} tasks (rule J is run only for sequences of at most {J_MAX_TASKS} tasks)"; continue
    if time.time() - tJ > J_BUDGET_S:
        RESULTS["sequences"][tag]["J"] = "skipped: time budget of rule J used up"; continue
    run_J(tag)
dump()

# %% [markdown]
# ## Tables

# %%
KEYS = ["own_at_arrival", "own_at_end", "changed", "changed_ever", "certificate", "upper95", "prior_bound"]
def line(name, r, extra=()):
    return f"  {name:6s}" + "".join(f" {r['mean'][k]:.4f}+-{r['se'][k]:.4f}" for k in KEYS + list(extra))
for tag, res in RESULTS["sequences"].items():
    print(f"\n== {tag}: {len(res['tasks'])} tasks (mean +- standard error over tasks)")
    print("        " + "".join(f" {k[:15]:>15s}" for k in KEYS) + "   (H': handicap_bound, takes_of_own_heldout)")
    print(line("L", res["L"])); print(line("H'", res["H_prime_t2"], ("handicap_bound", "takes_of_own_heldout")))
    print(line("J", res["J"]) if isinstance(res.get("J"), dict) else "  J      " + str(res.get("J")))
    if "reading_rule_L" in res: print("  reading of rule L:", res["reading_rule_L"]["use"], "| audit", res["reading_rule_L"]["audit"], "| own_at_end", round(res["reading_rule_L"]["own_at_end"], 4))
print("\nShare given to the own task after the last task, against the number of tasks (rule L / H' / J):")
for tag in ("superni_order1", "long_order3", "fresh30", "fresh30_random", "fresh60", "fresh60_random", "fresh90", "fresh90_random", "all120", "all120_random"):
    r = RESULTS["sequences"][tag]; J = r.get("J")
    print(f"  {tag:15s} {len(r['tasks']):4d} tasks  L {r['L']['mean']['own_at_end']:.4f}  H' {r['H_prime_t2']['mean']['own_at_end']:.4f}  J " + (f"{J['mean']['own_at_end']:.4f}" if isinstance(J, dict) else "-"))
print("\nRandom split minus fresh_split (rule L):")
for a, b in PAIRS: print(f"  {b} - {a}: " + "  ".join(f"{k} {v:+.4f}" for k, v in RESULTS["fresh_split_comparison"][b + " - " + a]["L"].items()))
for tag in ("long_order3", "long_order4", "superni_order1"):
    res = RESULTS["sequences"][tag]
    print(f"\nPer task, {tag}, rule L / H' (arrival, end, changed, certificate (c/m), top wrong destination):")
    for k in res["tasks"]:
        a, b = res["L"]["per_task"][k], res["H_prime_t2"]["per_task"][k]
        print(f"  {k[:34]:34s} L {a['own_at_arrival']:.3f} {a['own_at_end']:.3f} {a['changed']:.3f} {a['certificate']:.3f} ({a['c']}/{a['m']}) -> {str(a.get('top_wrong_destination', ''))[:22]:22s}"
              f" | H' {b['own_at_arrival']:.3f} {b['own_at_end']:.3f} {b['changed']:.3f} {b['certificate']:.3f}")
RESULTS["timings_s"]["total"] = round(time.time() - T0, 1); dump()
print("\nresults written to", RES_FILE, "| total", round(time.time() - T0), "s")
'''

src = (SRC.replace("__CONST__", repr(CONST)).replace("__ORDER1__", repr(ORDERS["ORDER1"])).replace("__ORDER2__", repr(ORDERS["ORDER2"]))
       .replace("__OBJECTS__", ms["objects"])
       .replace("__RULES__", "\n\n\n".join(rs[n] for n in ["mse", "standardiser", "routes_joint", "ridge_sums", "routes_ridge", "routes_handicap", "measures", "fresh_split"])))
assert "__" not in re.sub(r"__[a-z_]+__", "", src.replace("__CONST__", "")) or True
cells, kind, buf = [], None, []


def flush():
    if kind is None: return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))})
    elif text:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


for l in src.split("\n"):
    if l.startswith("# %%"):
        flush(); kind, buf = ("markdown" if "[markdown]" in l else "code"), []
    else:
        buf.append(l)
flush()
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
json.dump(nb, open(OUT, "w"), indent=1)
print(len(cells), "cells written to", OUT)
