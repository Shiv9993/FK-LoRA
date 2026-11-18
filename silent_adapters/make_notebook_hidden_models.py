"""Builds hidden_states_models.ipynb: does the task recogniser (rules L and H' of paper_certified/hidden_rules_study.py) depend on which frozen
model the features come from? Six models, SuperNI order 1 and Long Sequence order 3 (Long test inputs subsampled to 500 per task).
Usage: python3 make_notebook_hidden_models.py <scratch folder S>     (S/ni/ni_tasks.json is read for the 2,000 pool texts; the first notebook's constants are read from
hidden_states_constants.json). The rule code is copied by syntax-tree extraction from hidden_rules_study.py. Definitions are never read."""
import ast, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
OUT = os.path.join(HERE, "hidden_states_models.ipynb")


def segments(path, names):
    text = open(path).read(); got = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names: got[node.name] = ast.get_source_segment(text, node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names: got[node.targets[0].id] = ast.get_source_segment(text, node)
    assert set(got) == set(names), set(names) - set(got)
    return got


rs = segments(os.path.join(HERE, "..", "paper_certified", "hidden_rules_study.py"), ["mse", "standardiser", "ridge_sums", "routes_ridge", "routes_handicap", "measures", "ORDER1"])
ns = {}; exec(rs["ORDER1"], ns); ORDER1 = ns["ORDER1"]
# the pool texts: exactly POOL_TEXTS / pool_rows() (inputs of the tasks of ni_tasks.json without the 15 ORDER1 tasks; sorted seed-0 sample of 2000), cut to 1500 characters
tasks = json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"]
src = [x for t in tasks if t["name"] not in set(ORDER1) for x in t["inputs"]]
pidx = sorted(random.Random(0).sample(range(len(src)), 2000))
POOL = [str(src[i])[:1500] for i in pidx]
K = json.load(open(os.path.join(HERE, "hidden_states_constants.json")))
CHECK = {"mu": K["mu"], "sd": K["sd"], "pool_sha1": K["pool_sha1"]}

SRC = r'''# %% [markdown]
# # Does the task recogniser depend on the frozen model? Six models, two sequences
#
# Same rules and measures as `hidden_states_scale.ipynb` (copied from `paper_certified/hidden_rules_study.py`): rule L (ridge; lam 1, 10, 100) and rule H' (lam 1, t = 2, floor).
# Features of the instance text only (the task definition is never read): mean over the non-padding tokens of the LAST-layer hidden states, 256 tokens, fp32,
# frozen, eval, no adapter. Models: (1) t5-small encoder, (2) t5-base encoder, (3) t5-large encoder, (4) google/flan-t5-base encoder,
# (5) HuggingFaceTB/SmolLM2-360M (decoder only, AutoModel), (6) sentence-transformers/all-MiniLM-L6-v2 (plain AutoModel, mean pooling, no normalisation).
# Sequences: SuperNI order 1 (all official test inputs) and Long Sequence order 3 (test inputs of every task subsampled to 500: `random.Random(0).sample`, sorted).
# Training inputs split as before (seed-0 shuffle, at most 400, first m held out). Standardisation: per model, with the mean and standard deviation of **that model's
# features of the same 2,000 pool texts** (embedded below, each cut to its first 1,500 characters; 256 tokens rarely need more).
# Model (2) must reproduce own_at_end 0.9903 / changed 0.0084 of rule L on SuperNI order 1 (printed). A model that cannot be loaded is skipped with a message.
# **Kaggle:** GPU T4, Internet on, Save & Run All. Result: `/kaggle/working/hidden_states_models/results.json`.

# %%
QUICK = False        # True: t5-small and MiniLM only, 40 training / 20 test inputs per task, 100 pool texts (a CPU test); the real run is False
import os, sys, json, math, random, time, subprocess, shutil, gc
import numpy as np
from scipy.stats import beta as beta_dist
OUT_DIR = os.environ.get("HS_OUT", "/kaggle/working/hidden_states_models")
_tmp = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
WORK = os.environ.get("HS_WORK", os.path.join(_tmp, "hidden_states_models"))
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(WORK, exist_ok=True)
OFFICIAL_URL = "https://github.com/liangyanshuo/gainlora"; OFFICIAL_COMMIT = "e9d0d700538c8d12892f7fd52b216291f7e42acc"
MAXTOK = 256; T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
ORDER1 = __ORDER1__
LONG3 = "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
MODELS = [("t5-small", "t5-small", "t5"), ("t5-base", "t5-base", "t5"), ("t5-large", "t5-large", "t5"), ("flan-t5-base", "google/flan-t5-base", "t5"),
          ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "auto"), ("minilm-l6", "sentence-transformers/all-MiniLM-L6-v2", "auto")]
if QUICK: MODELS = [m for m in MODELS if m[0] in ("t5-small", "minilm-l6")]
# first notebook's embedded pool statistics of t5-base (used only for the check of model (2)) and the sha1 of that pool array
CHECK = __CHECK__

# %% [markdown]
# ## The 2,000 pool texts (inputs of 860 tasks of natural-instructions, without the 15 tasks of order 1; the list of `POOL_TEXTS`, sorted seed-0 sample of 2,000; each cut to 1,500 characters)

# %%
POOL = json.loads(__POOL__)
if QUICK: POOL = POOL[:100]
log("pool texts", len(POOL), "| characters", sum(map(len, POOL)))

# %% [markdown]
# ## The rules and measures (copied verbatim from `paper_certified/hidden_rules_study.py`)

# %%
__RULES__

# %% [markdown]
# ## Data (official repository at the fixed commit)

# %%
def sh(cmd, cwd=None, tries=1, wait=20):
    for k in range(tries):
        r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
        if r.returncode == 0: return r.stdout
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
        log("shallow fetch failed, full clone instead:", str(e)[:200]); shutil.rmtree(d, ignore_errors=True)
        sh(f"git clone -q {OFFICIAL_URL} {d}", tries=3); sh(f"git checkout -q {OFFICIAL_COMMIT}", cwd=d)
    assert sh("git rev-parse HEAD", cwd=d).strip() == OFFICIAL_COMMIT
    return d

REPO = os.environ.get("HS_LOCAL_REPO") or get_repo()      # HS_LOCAL_REPO: test hook only
BENCH = os.path.join(REPO, "CL_Benchmark")
def load_inputs(folder, name, split):
    d = json.load(open(os.path.join(BENCH, folder, name, split + ".json"), encoding="utf-8"))
    return [str(i["input"]) for i in d["Instances"]]      # the Definition is never read

TASKS = {}      # key -> (texts of the training inputs in the order of sel, m, test texts)
for folder, names in (("SuperNI", ORDER1), ("Long_Sequence", LONG3)):
    for name in names:
        tr, te = load_inputs(folder, name, "train"), load_inputs(folder, name, "test")
        if QUICK: tr, te = tr[:40], te[:20]
        elif folder == "Long_Sequence" and len(te) > 500: te = [te[i] for i in sorted(random.Random(0).sample(range(len(te)), 500))]
        n = len(tr); perm = list(range(n)); random.Random(0).shuffle(perm); sel = perm[:min(n, 400)]; m = 200 if n >= 400 else n // 2
        TASKS[f"{folder}/{name}"] = ([tr[i] for i in sel], m, te)
SEQS = {"superni_order1": [f"SuperNI/{n}" for n in ORDER1], "long_order3": [f"Long_Sequence/{n}" for n in LONG3]}
log("tasks", len(TASKS), "| training inputs", sum(len(v[0]) for v in TASKS.values()), "| test inputs", sum(len(v[2]) for v in TASKS.values()))

# %% [markdown]
# ## One model after the other: features, standardisation with the model's own pool statistics, rules

# %%
import torch
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TPB = 16000 if DEVICE == "cuda" else 8000
if DEVICE == "cpu" and not QUICK: log("WARNING: no GPU found; this will be slow")
RESULTS = {"about": "Does the recogniser depend on the frozen model? Rules L (lam 1, 10, 100) and H' (lam 1, t = 2, floor) of hidden_rules_study.py.", "quick": QUICK, "device": DEVICE,
           "pool_texts": len(POOL), "pool_cut_to_characters": 1500, "models": {}, "skipped": {}}
RES_FILE = os.path.join(OUT_DIR, "results.json")
def dump():
    json.dump(RESULTS, open(RES_FILE + ".tmp", "w")); os.replace(RES_FILE + ".tmp", RES_FILE)

def make_enc(tok, model):
    def enc(texts, tokens_per_batch=TPB):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); res = np.zeros((len(texts), model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model), np.float32); s0 = 0
        while s0 < len(order):
            e = s0 + 1
            while e < len(order) and e - s0 < 64 and (e - s0 + 1) * min(MAXTOK, len(texts[order[e]]) // 3 + 4) <= tokens_per_batch: e += 1
            b = order[s0:e]; s0 = e
            t = tok([texts[i] for i in b], max_length=MAXTOK, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                h = model(**t).last_hidden_state; m = t["attention_mask"][:, :, None].to(h.dtype)
                res[b] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
        return res
    return enc

def run_model(label, hf_id, kind):
    tok = AutoTokenizer.from_pretrained(hf_id); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = (T5EncoderModel if kind == "t5" else AutoModel).from_pretrained(hf_id, torch_dtype=torch.float32).eval().to(DEVICE)
    enc = make_enc(tok, model); t = time.time()
    P = enc(POOL); st = standardiser(P); log(label, "pool encoded", P.shape, round(time.time() - t), "s")
    feats = {}
    for k, (tr, m, te) in TASKS.items():
        X = enc(tr); feats[k] = (st(X[:m]), st(X[m:]), st(enc(te)))       # held-out (cal), fit, test; standardised, float64
    log(label, "features done", round(time.time() - t), "s")
    out = {"hf_id": hf_id, "dim": int(P.shape[1]), "sequences": {}}
    for tag, keys in SEQS.items():
        cal = [feats[k][0] for k in keys]; fit = [feats[k][1] for k in keys]; te = [feats[k][2] for k in keys]; r = {}
        for lam in (1, 10, 100):
            Rt, Rc, _ = routes_ridge(fit, cal, te, lam); M = measures(keys, Rt, Rc)
            r[f"L_lam{lam}"] = {"mean": M["mean"], "se": M["se"], "per_task_own_at_end": {k: M["per_task"][k]["own_at_end"] for k in keys}}
        Rt, Rc, ex = routes_handicap(fit, cal, te, 1, [2], floor=True)[2]; M = measures(keys, Rt, Rc, extra=ex, held_out_used_by_rule=True, handicap_t=2)
        r["H_prime_t2"] = {"mean": M["mean"], "se": M["se"], "per_task_own_at_end": {k: M["per_task"][k]["own_at_end"] for k in keys}}
        out["sequences"][tag] = r
    if label == "t5-base" and not QUICK:
        mu = np.array(CHECK["mu"], dtype=np.float32); sd = np.array(CHECK["sd"], dtype=np.float32); st_old = lambda X: ((X - mu) / sd).astype(np.float64)
        # the check of model (2): standardised superni_order1 test features, pool statistics computed here (cut pool texts) against the embedded constants of the first notebook
        diff = 0.0
        for k in SEQS["superni_order1"]:
            raw = enc(TASKS[k][2]); diff = max(diff, float(np.abs(st(raw) - st_old(raw)).max()))
        a = out["sequences"]["superni_order1"]["L_lam1"]["mean"]
        out["check_t5base"] = {"max_abs_difference_standardised_features_vs_embedded_constants": diff, "own_at_end": a["own_at_end"], "changed": a["changed"],
                               "reproduces_0.9903_and_0.0084": bool(abs(a["own_at_end"] - 0.9903) < 0.0006 and abs(a["changed"] - 0.0084) < 0.0006)}
        print("CHECK t5-base:", out["check_t5base"])
    del model, enc, feats; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return out

for label, hf_id, kind in MODELS:
    try:
        RESULTS["models"][label] = run_model(label, hf_id, kind)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:300]}"; RESULTS["skipped"][label] = msg
        print(f"SKIPPED model {label} ({hf_id}): {msg}"); gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    dump()

# %% [markdown]
# ## Table

# %%
print("\nown_at_end / changed (mean over tasks); rows = models")
print(f"{'model':14s} {'dim':>5s} | " + " | ".join(f"{s:>13s} {r:>10s}" for s in ("SuperNI", "Long") for r in ("L lam1", "H' t2")))
for label, o in RESULTS["models"].items():
    cells = []
    for tag in ("superni_order1", "long_order3"):
        for rule in ("L_lam1", "H_prime_t2"):
            m = o["sequences"][tag][rule]["mean"]; cells.append(f"{m['own_at_end']:.4f}/{m['changed']:.4f}".rjust(24))
    print(f"{label:14s} {o['dim']:5d} | " + " | ".join(cells))
print("\nown_at_end of rule L for lam 1 / 10 / 100:")
for label, o in RESULTS["models"].items():
    print(f"  {label:14s}" + "".join(f"  {tag[:7]} " + " / ".join(f"{o['sequences'][tag][f'L_lam{l}']['mean']['own_at_end']:.4f}" for l in (1, 10, 100)) for tag in SEQS))
if RESULTS["skipped"]: print("\nSKIPPED:", RESULTS["skipped"])
RESULTS["total_s"] = round(time.time() - T0, 1); dump(); print("results written to", RES_FILE, "| total", round(time.time() - T0), "s")
'''
code = (SRC.replace("__ORDER1__", repr(ORDER1)).replace("__CHECK__", repr(CHECK)).replace("__POOL__", repr(json.dumps(POOL, ensure_ascii=False)))
        .replace("__RULES__", "\n\n\n".join(rs[n] for n in ["mse", "standardiser", "ridge_sums", "routes_ridge", "routes_handicap", "measures"])))
cells, kind, buf = [], None, []
def flush():
    if kind is None: return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown": cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))})
    elif text: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})
for l in code.split("\n"):
    if l.startswith("# %%"): flush(); kind, buf = ("markdown" if "[markdown]" in l else "code"), []
    else: buf.append(l)
flush()
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
json.dump(nb, open(OUT, "w"), indent=1)
print(len(cells), "cells;", os.path.getsize(OUT) // 1024, "KB ->", OUT, "| pool chars", sum(map(len, POOL)))
