"""Zones in the frozen model's own hidden states (plan: last section of silent_adapters/PLAN.md). CPU only.

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    cd $S/ext_gainlora/repo && OMP_NUM_THREADS=4 python /path/to/paper_certified/feature_study.py
Stages are cached in $S/feature_study/ (base.npz, chunks); a rerun resumes. Writes routing/feature_study.json.
"""
import json, os, random, sys, time, types, glob
import numpy as np

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{S}/feature_study"
MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_LEN = 512
LAYERS = [0, 2, 4, 8, 16, 30]
POOLS = ["mean", "last"]
CFGS = [(l, p) for l in LAYERS for p in POOLS]
KAPPAS = [round(0.1 * i, 1) for i in range(1, 11)]
QS = [0.5, 0.7, 0.8, 0.9, 0.95]
SETTINGS = [("R1", k) for k in KAPPAS] + [("R2", q) for q in QS]
SNAP = 1e-3            # distances below this are float noise between two passes over the same tokens: set to 0
ONLY_ANCHOR = "--anchor" in sys.argv

import torch
from transformers import AutoModel, AutoTokenizer
torch.set_num_threads(4)

class _Official:
    def __init__(self, *a, **k): pass
    def forward(self, *a, **k): pass
    def compute_loss(self, *a, **k): pass
    def predict(self, *a, **k): pass
for _name, _attrs in (("t5_inflora", ("LoRALayer", "T5Attention", "T5Stack")), ("cl_trainer_olora", ("OLoRATrainer",)), ("compute_metrics", ())):
    _m = types.ModuleType(_name)
    for _a in _attrs: setattr(_m, _a, type(_a, (_Official,), {}))
    sys.modules[_name] = _m
sys.modules["cl_trainer_olora"].skip_instructions = None
sys.modules["compute_metrics"].metric_max_over_ground_truths = sys.modules["compute_metrics"].rougeL_score = None
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
import silent_run as sr

FIVE = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
        "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
        "task875_emotion_classification"]
ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
EMB = model.get_input_embeddings()
D = model.config.hidden_size
t_start = time.time()
def log(*a): print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)

def template(definition):
    return "Definition: " + definition + "\n\nNow complete the following example -\nInput: {0}\nOutput: "
def ids_of(text): return tok(text, max_length=MAX_LEN, truncation=True)["input_ids"]

# ---------------------------------------------------------------- feature extraction
def extract(ids_list, nvals_list, cfgs, tokens_per_batch=3000):
    """Yield (i, {cfg: array (len(nvals_i), D) float64}) of the RAW pooled feature at each n in nvals_i (n clipped to the length).
    mean: weighted mean of layer-L states over positions 1..n; last: layer-L state at position n.  Forward only on the first
    min(length, max n) tokens (causal model: later tokens cannot change earlier states)."""
    need_fwd = any(l > 0 for l, _ in cfgs)
    T = [min(len(x), int(max(nv))) for x, nv in zip(ids_list, nvals_list)]
    order = sorted(range(len(ids_list)), key=lambda i: T[i])
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and (e - s + 1) * T[order[e]] <= tokens_per_batch and e - s < 32: e += 1
        b = order[s:e]; s = e
        L = max(T[i] for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, :T[i]] = torch.tensor(ids_list[i][:T[i]]); att[r, :T[i]] = 1
        with torch.no_grad():
            if need_fwd: hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
            else: hs = (EMB(ids),)
        for r, i in enumerate(b):
            w = WEIGHT[torch.tensor(ids_list[i][:T[i]])]
            pos = torch.from_numpy(np.minimum(np.asarray(nvals_list[i]), len(ids_list[i])) - 1)
            res = {}
            for (l, p) in cfgs:
                h = hs[l][r, :T[i]].double()
                if p == "mean":
                    f = (torch.cumsum(h * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None])[pos]
                else:
                    f = h[pos]
                res[(l, p)] = f
            yield i, res

# ---------------------------------------------------------------- the fixed scale (as _claim_base)
WEIGHT = None
def unrelated_base():
    global WEIGHT
    rng = random.Random(0)
    defs, texts = [], []
    for name in sr.POOL_TASKS:
        data = json.load(open(os.path.join("CL_Benchmark", "Long_Sequence", name, "train.json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
        defs.append(d); texts.append([inst["input"] for inst in data["Instances"]])
    picks = [rng.randrange(len(defs)) for _ in range(sr.BG_N)]
    sources = [sr._source(defs[k], rng.choice(texts[k])) for k in picks]
    rows = tok(sources, max_length=MAX_LEN, truncation=True)["input_ids"]
    V = EMB.weight.shape[0]
    counts = torch.zeros(V, dtype=torch.double)
    for r in rows:
        r = torch.tensor(r); r = r[r != tok.pad_token_id]
        counts += torch.bincount(r, minlength=V).double()
    WEIGHT = sr.SIF_A / (sr.SIF_A + counts / counts.sum())
    WEIGHT[tok.pad_token_id] = 0.0
    return defs, texts, rows

def build_base():
    path = f"{OUT}/base.npz"
    if os.path.exists(path):
        z = np.load(path); unrelated_base()
        return {k: z[k] for k in z.files}
    defs, texts, rows = unrelated_base()
    log("unrelated inputs", len(rows), "forward on first 64 tokens")
    feats = {c: [None] * len(rows) for c in CFGS}
    for i, res in extract([r[:64] for r in rows], [[64]] * len(rows), CFGS):
        for c in CFGS: feats[c][i] = res[c][0]
    base = {}
    for c in CFGS:
        x = torch.stack(feats[c]); mu, W = sr._whitener(x)
        base[f"mu|{c}"] = mu.numpy(); base[f"W|{c}"] = W.numpy()
    sig_rows = []
    for d, xs in zip(defs, texts):
        r = tok([sr._source(d, x) for x in xs[:20]], max_length=MAX_LEN, truncation=True)["input_ids"]
        n = sr._prefix_len(r); assert n >= 5
        sig_rows.append((r[0][:n], n))
    for i, res in extract([x for x, _ in sig_rows], [[n] for _, n in sig_rows], CFGS):
        for c in CFGS:
            base.setdefault(f"bg|{c}", [None] * len(sig_rows))[i] = res[c][0].numpy()
    for c in CFGS:
        base[f"bg|{c}"] = (np.stack(base[f"bg|{c}"]) - base[f"mu|{c}"]) @ base[f"W|{c}"]
    np.savez(path, **base)
    return base

def whiten(base, c, f): return (f - base[f"mu|{c}"]) @ base[f"W|{c}"]

def opening(definition):
    a = ids_of(template(definition).replace("{0}", "A B")); b = ids_of(template(definition).replace("{0}", "zzq xkj"))
    n = min(sr._prefix_len([a, b]), MAX_LEN)
    return a[:n], n

def signatures(base, defs, cfgs):
    ops = [opening(d) for d in defs]
    sig = {c: [None] * len(defs) for c in cfgs}
    for i, res in extract([o for o, _ in ops], [[n] for _, n in ops], cfgs):
        for c in cfgs: sig[c][i] = whiten(base, c, res[c].numpy())[0]
    return np.array([n for _, n in ops]), {c: np.stack(sig[c]) for c in cfgs}

def distances(base, texts, tasks_n, tasks_sig, cfgs, store=None):
    """float32 matrices (n_texts, n_tasks): distance of the rescaled feature at n_k from signature k, per config."""
    ids = [ids_of(t) for t in texts]
    uniq = np.unique(tasks_n); where = np.searchsorted(uniq, tasks_n)
    out = {c: np.zeros((len(texts), len(tasks_n)), dtype=np.float32) for c in cfgs}
    for i, res in extract(ids, [uniq] * len(ids), cfgs):
        for c in cfgs:
            z = whiten(base, c, res[c].numpy())[where]
            out[c][i] = np.linalg.norm(z - tasks_sig[c], axis=1)
    return out

# ---------------------------------------------------------------- routes and measures
def snap(x): x = np.asarray(x, dtype=np.float64).copy(); x[x < SNAP] = 0.0; return x

def radii_r1(d_bg, d_sig, kappa):
    """tasks in arrival order (rows/cols of d_sig already in that order)."""
    K = len(d_bg); r = np.zeros(K)
    for j in range(K):
        near = d_bg[j] if j == 0 else min(d_bg[j], d_sig[j, :j].min())
        r[j] = kappa * near
    return r

def route(dist, radius):
    """columns in arrival order (oldest first). Oldest zone that contains the input, else the smallest distance/radius (ties: older)."""
    inside = dist <= radius[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = dist / radius[None]
    rel = np.where(np.isnan(rel), 0.0, rel)
    nearest = rel.argmin(1)
    anyin = inside.any(1)
    return np.where(anyin, inside.argmax(1), nearest), anyin

def measures(Du, ou, Dp, op, radius):
    K = len(radius); res = {}
    d = snap(Du); older = np.arange(K)[None, :] < ou[:, None]
    blocked = ((d <= radius[None]) & older).any(1)
    has = np.bincount(ou, minlength=K) > 0
    coll = np.bincount(ou, weights=blocked.astype(float), minlength=K) > 0
    ru, _ = route(d, radius)
    res["C"] = float(coll[has].sum() / has.sum()); res["U"] = float((ru == ou).mean())
    p = snap(Dp); rp, cert = route(p, radius)
    own = rp == op
    res["P"] = float((cert & own).mean()); res["M"] = float((cert & ~own).mean()); res["R"] = float(own.mean())
    res["tasks_with_unedited"] = int(has.sum()); res["paraphrased_inputs"] = int(len(op)); res["radius_zero"] = int((radius == 0).sum())
    return res

class SetData:
    """Distances of one set of tasks (own arrival order), for one config."""
    def __init__(self, base, c, task_idx, Du, ou, Dp, op, sig, owners_of_bg_sig_dist=None):
        self.idx = np.array(task_idx); colmap = {t: j for j, t in enumerate(task_idx)}
        mu = np.isin(ou, self.idx); mp = np.isin(op, self.idx)
        self.Du = Du[np.ix_(mu, self.idx)]; self.ou = np.array([colmap[t] for t in ou[mu]])
        self.Dp = Dp[np.ix_(mp, self.idx)]; self.op = np.array([colmap[t] for t in op[mp]])
        s = sig[self.idx]
        self.d_sig = snap(np.linalg.norm(s[:, None, :] - s[None, :, :], axis=2))
        bg = base[f"bg|{c}"]
        self.d_bg = snap(np.linalg.norm(s[:, None, :] - bg[None, :, :], axis=2).min(1))
    def radius(self, setting, r2=None):
        kind, v = setting
        return radii_r1(self.d_bg, self.d_sig, v) if kind == "R1" else np.full(len(self.idx), r2[v])
    def run(self, setting, r2=None):
        r = self.radius(setting, r2)
        return measures(self.Du, self.ou, self.Dp, self.op, r)

# ---------------------------------------------------------------- the standard check (anchor and secondary)
def load_ni(name):
    out = {}
    for split in ("train", "test"):
        data = json.load(open(os.path.join("CL_Benchmark", "SuperNI", name, split + ".json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) else data["Definition"]
        out[split] = [i["input"] for i in data["Instances"]]
    out["definition"] = d.strip()
    return out

def standard_check(base, cfgs, radius_of, tag):
    """radius_of(cfg, d_bg, d_sig) -> radii in arrival order. Returns {cfg: {"five": {...}, "fifteen": {...}}}."""
    names = sorted(set(FIVE) | set(ORDER1))
    data = {n: load_ni(n) for n in names}
    defs = {n: [data[n]["definition"]] + sr.REWORDED.get(n.split("_")[0], []) for n in names}
    n_k, sig = signatures(base, [defs[n][0] for n in names], cfgs)
    tn = {n: i for i, n in enumerate(names)}
    texts, meta = [], []
    for n in names:
        for li, d in enumerate(defs[n]):
            for x in data[n]["test"]:
                texts.append(template(d).replace("{0}", x)); meta.append((n, li))
    log(tag, "standard check:", len(names), "tasks,", len(texts), "test inputs")
    dist = distances(base, texts, n_k, sig, cfgs)
    res = {}
    for c in cfgs:
        res[c] = {}
        for label, order in (("five", FIVE), ("fifteen", ORDER1)):
            cols = [tn[n] for n in order]
            s = sig[c][cols]
            d_sig = snap(np.linalg.norm(s[:, None] - s[None], axis=2)); d_bg = snap(np.linalg.norm(s[:, None] - base[f"bg|{c}"][None], axis=2).min(1))
            rad = radius_of(c, d_bg, d_sig)
            out = {}
            for li, lev in enumerate(sr.LEVELS):
                rows = [i for i, (n, l) in enumerate(meta) if l == li and n in order]
                own = np.array([order.index(meta[i][0]) for i in rows])
                r, cert = route(snap(dist[c][rows][:, cols]), rad)
                out[lev] = {"zone": float((cert & (r == own)).mean()), "certified": float((r == own).mean()), "inputs": len(rows),
                            "tasks": len(set(own.tolist())), "wrong": int((cert & (r != own)).sum())}
            res[c][label] = out
    return res, dist, meta, names

def fmt_std(res):
    f = lambda d, k: "/".join(f"{d[l][k] * 100:.0f}" for l in sr.LEVELS)
    return f"five: zone {f(res['five'], 'zone')} certified {f(res['five'], 'certified')} | fifteen: zone {f(res['fifteen'], 'zone')} certified {f(res['fifteen'], 'certified')}"

# ================================================================ main
base = build_base()
log("base ready; weights: tokens with weight", int((WEIGHT > 0).sum()))
anchor, _, _, _ = standard_check(base, [(0, "mean")], lambda c, d_bg, d_sig: radii_r1(d_bg, d_sig, 0.5), "anchor")
anchor = anchor[(0, "mean")]
pub = [r for r in json.load(open(os.path.join(HERE, "routing", "models.json")))["runs"] if r["model"] == MODEL][0]
log("ANCHOR  ", fmt_std(anchor))
log("PUBLISHED", fmt_std({"five": pub["five"], "fifteen": pub["fifteen"]}))
json.dump({"anchor": anchor, "published": {"five": pub["five"], "fifteen": pub["fifteen"]}}, open(f"{OUT}/anchor.json", "w"), indent=1)
if ONLY_ANCHOR: sys.exit()

# ---- the 875 tasks and their inputs
tasks = json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"]
para = json.load(open(os.path.join(HERE, "routing", "paraphrases.json")))
PIV = [p for p in ("de", "fr") if all(p in v for v in para["tasks"].values())]
by_name = {t["name"]: t for t in tasks}
names = sorted(by_name)
rr = random.Random(0); shuffled = names[:]; rr.shuffle(shuffled)
cal_names, ev_names = shuffled[:175], shuffled[175:]
tidx = {n: i for i, n in enumerate(names)}           # column index = position in the sorted list
cal = [tidx[n] for n in cal_names]; ev = [tidx[n] for n in ev_names]
n_k, sig = signatures(base, [by_name[n]["definition"] for n in names], CFGS)
log("opening lengths: min/median/max", int(n_k.min()), int(np.median(n_k)), int(n_k.max()), "| distinct", len(np.unique(n_k)))
items = []   # (task column, kind, text); kind 0 unedited, 1.. paraphrased
for n in names:
    t = by_name[n]
    for x in t["inputs"][:6]: items.append((tidx[n], 0, template(t["definition"]).replace("{0}", x)))
    for k, p in enumerate(PIV):
        for x in t["inputs"][:3]: items.append((tidx[n], 1 + k, template(para["tasks"][n][p]).replace("{0}", x)))
owner = np.array([i[0] for i in items]); kind = np.array([i[1] for i in items])
log("items", len(items), "unedited", int((kind == 0).sum()), "paraphrased", int((kind > 0).sum()), "pivots", PIV)
lens = [len(ids_of(i[2])) for i in items]
log("input lengths: median", int(np.median(lens)), "max", max(lens), "tokens")
CH = 300
order = sorted(range(len(items)), key=lambda i: lens[i])
chunks = [order[s:s + CH] for s in range(0, len(order), CH)]
for ci, ch in enumerate(chunks):
    path = f"{OUT}/chunk_{ci:03d}.npz"
    if os.path.exists(path): continue
    t0 = time.time()
    dist = distances(base, [items[i][2] for i in ch], n_k, sig, CFGS)
    np.savez(path, idx=np.array(ch), **{f"{c[0]}|{c[1]}": dist[c] for c in CFGS})
    log(f"chunk {ci + 1}/{len(chunks)} done in {time.time() - t0:.0f}s")
DIST = {c: np.zeros((len(items), len(names)), dtype=np.float32) for c in CFGS}
for ci in range(len(chunks)):
    z = np.load(f"{OUT}/chunk_{ci:03d}.npz")
    for c in CFGS: DIST[c][z["idx"]] = z[f"{c[0]}|{c[1]}"]
log("all distance matrices loaded")

# ---- calibration
mu_, mp_ = kind == 0, kind > 0
def setdata(c, idx): return SetData(base, c, idx, DIST[c][mu_], owner[mu_], DIST[c][mp_], owner[mp_], sig[c])
cal_data = {c: setdata(c, cal) for c in CFGS}
R2 = {}
for c in CFGS:
    dd = DIST[c][mp_]; oo = owner[mp_]; m = np.isin(oo, cal)
    own_d = snap(dd[m, oo[m]])
    R2[c] = {q: float(np.quantile(own_d, q)) for q in QS}
grid = {c: {s: cal_data[c].run(s, R2[c]) for s in SETTINGS} for c in CFGS}
ref_cfg = (0, "mean"); ref_set = ("R1", 0.5)
C_ref = grid[ref_cfg][ref_set]["C"]
log("C_ref (calibration, layer 0 mean, R1 0.5) =", C_ref, " P_ref =", grid[ref_cfg][ref_set]["P"])
best = {}
for c in CFGS:
    ok = [(s, m) for s, m in grid[c].items() if m["C"] <= C_ref + 1e-12]
    if not ok: best[c] = None; continue
    s, m = max(ok, key=lambda sm: (sm[1]["P"], -sm[1]["C"], -sm[1]["M"]))
    best[c] = s
    log(f"cal best {c}: setting {s} C={m['C']:.3f} P={m['P']:.3f} M={m['M']:.3f} R={m['R']:.3f} U={m['U']:.3f}")
cand = [c for c in CFGS if best[c] is not None]
chosen = max(cand, key=lambda c: grid[c][best[c]]["P"])
log("chosen", chosen, best[chosen])

# ---- evaluation on the 700
ev_data = {c: setdata(c, ev) for c in CFGS}
ev_ref = ev_data[ref_cfg].run(ref_set, R2[ref_cfg])
ev_chosen = ev_data[chosen].run(best[chosen], R2[chosen])
ev_grid = {c: (ev_data[c].run(best[c], R2[c]) if best[c] else None) for c in CFGS}
log("EVAL reference", ev_ref); log("EVAL chosen", chosen, best[chosen], ev_chosen)

# ---- secondary: the standard fifteen-task check with the reference and the chosen configuration
rad_ref = lambda c, d_bg, d_sig: radii_r1(d_bg, d_sig, ref_set[1])
def rad_chosen(c, d_bg, d_sig):
    s = best[chosen]
    return radii_r1(d_bg, d_sig, s[1]) if s[0] == "R1" else np.full(len(d_bg), R2[chosen][s[1]])
secondary = {"reference": standard_check(base, [ref_cfg], rad_ref, "secondary-reference")[0][ref_cfg],
             "chosen": standard_check(base, [chosen], rad_chosen, "secondary-chosen")[0][chosen]}
log("SECONDARY reference", fmt_std(secondary["reference"])); log("SECONDARY chosen   ", fmt_std(secondary["chosen"]))

sk = lambda c: f"L{c[0]}-{c[1]}"
st = lambda s: f"{s[0]}({s[1]})"
json.dump({
    "model": MODEL, "pivots": PIV, "paraphrase_stats": para.get("stats"),
    "split": {"calibration": cal_names, "evaluation": ev_names},
    "items": {"unedited": int(mu_.sum()), "paraphrased": int(mp_.sum()), "tasks_without_inputs": int(sum(len(t["inputs"]) == 0 for t in tasks))},
    "opening_tokens": {"min": int(n_k.min()), "median": float(np.median(n_k)), "max": int(n_k.max())},
    "anchor": anchor, "published": {"five": pub["five"], "fifteen": pub["fifteen"]},
    "C_ref_calibration": C_ref,
    "R2_radii": {sk(c): {str(q): v for q, v in R2[c].items()} for c in CFGS},
    "calibration_grid": {sk(c): {st(s): m for s, m in grid[c].items()} for c in CFGS},
    "calibration_best": {sk(c): ({"setting": st(best[c]), **grid[c][best[c]]} if best[c] else None) for c in CFGS},
    "chosen": {"config": sk(chosen), "setting": st(best[chosen]), "calibration": grid[chosen][best[chosen]]},
    "evaluation": {"reference": {"config": sk(ref_cfg), "setting": st(ref_set), **ev_ref},
                   "chosen": {"config": sk(chosen), "setting": st(best[chosen]), **ev_chosen}},
    "evaluation_full_grid_NOT_USED_FOR_CONCLUSION": {sk(c): ({"setting": st(best[c]), **ev_grid[c]} if ev_grid[c] else None) for c in CFGS},
    "secondary_fifteen": secondary,
    "notes": {"snap": SNAP, "layer30": "hidden_states[30] of the HF model (after the final norm)"},
}, open(os.path.join(HERE, "routing", "feature_study.json"), "w"), indent=1)

print("\nconfig   best setting (cal)  | cal: C     P     M   | eval(700): C     P     M     R     U")
for c in CFGS:
    if best[c] is None: print(f"{sk(c):<9} none"); continue
    a, b = grid[c][best[c]], ev_grid[c]
    print(f"{sk(c):<9} {st(best[c]):<14} | {a['C']:.3f} {a['P']:.3f} {a['M']:.3f} | {b['C']:.3f} {b['P']:.3f} {b['M']:.3f} {b['R']:.3f} {b['U']:.3f}")
print("reference", ev_ref); print("chosen", sk(chosen), st(best[chosen]), ev_chosen)
print("done")
