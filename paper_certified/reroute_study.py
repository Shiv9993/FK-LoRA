"""Third pre-registered analysis (last section of silent_adapters/PLAN.md, "what the zones add, at scale"): do later tasks
re-route reworded inputs? CPU, T5-base word-vectors, the paper's rule (`certified`, rule B, kappa 0.5, old ruler, snap 1e-3),
the nearest-zone rule (`nearest`) and the variant (`exactfirst`) as in many_tasks_analysis.py / exact_first_analysis.py,
on the 700 evaluation tasks in arrival order (random.Random(0) split, 175 calibration, 700 evaluation).

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=3 \
        $S/mm_env/bin/python reroute_study.py
Caches in $S/reroute_study/. Writes routing/reroute_study.json. Step 0 reproduces the reference numbers and stops if they differ.
Data building (items, distances, radii) is a copy of spanish_check.PaperRule / ruler_study.run_table.
"""
import collections, json, os, random, re, sys, time, types
import numpy as np

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", f"{S}/hf_cache")
OUT = f"{S}/reroute_study"; os.makedirs(OUT, exist_ok=True)
MAX_LEN = 512; SNAP = 1e-3; KAPPA = 0.5
import torch
from transformers import AutoTokenizer, T5EncoderModel
torch.set_num_threads(3)

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

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
def template(definition): return "Definition: " + definition + "\n\nNow complete the following example -\nInput: {0}\nOutput: "
def snap(x): x = np.asarray(x, dtype=np.float64).copy(); x[x < SNAP] = 0.0; return x
def inv_sqrt(M):
    w, v = np.linalg.eigh(M); return (v * w ** -0.5) @ v.T

def words(text): return collections.Counter(re.findall(r"\w+", text.lower()))
def alike(a, b):                       # as many_tasks_analysis.py
    wa, wb = words(a), words(b)
    return sum((wa & wb).values()) / max(sum(wa.values()), sum(wb.values()), 1)

# ------------------------------------------------------------------ data (as ruler_study / spanish_check)
tasks = json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"]
para = json.load(open(os.path.join(HERE, "routing", "paraphrases.json")))
es = json.load(open(os.path.join(HERE, "routing", "paraphrases_es.json")))
by_name = {t["name"]: t for t in tasks}; names = sorted(by_name)
rr = random.Random(0); shuffled = names[:]; rr.shuffle(shuffled)
ev_names = shuffled[175:]; K = len(ev_names); assert K == 700
ITALIAN = "--italian" in sys.argv                    # the Italian file is read only with this flag (after everything else is written)
KINDS = {0: "unedited", 1: "German", 2: "French", 3: "Spanish"}
items = []                              # (task position, kind, text)
for c, n in enumerate(ev_names):
    t = by_name[n]
    for x in t["inputs"][:6]: items.append((c, 0, template(t["definition"]).replace("{0}", x)))
    for kd, src in ((1, para["tasks"][n]["de"]), (2, para["tasks"][n]["fr"]), (3, es["tasks"][n]["es"])):
        for x in t["inputs"][:3]: items.append((c, kd, template(src).replace("{0}", x)))
if ITALIAN:
    it = json.load(open(os.path.join(HERE, "routing", "paraphrases_it.json")))
    assert len(it["tasks"]) == 875 and all("it" in v for v in it["tasks"].values())
    KINDS[4] = "Italian"
    for c, n in enumerate(ev_names):
        for x in by_name[n]["inputs"][:3]: items.append((c, 4, template(it["tasks"][n]["it"]).replace("{0}", x)))
owner = np.array([i[0] for i in items]); kind = np.array([i[1] for i in items]); N = len(items)
log("items", N, {KINDS[k]: int((kind == k).sum()) for k in KINDS})

tok = AutoTokenizer.from_pretrained("t5-base")
ids_of = lambda text: tok(text, max_length=MAX_LEN, truncation=True)["input_ids"]
model = T5EncoderModel.from_pretrained("t5-base", torch_dtype=torch.float32).eval()
Emb = model.get_input_embeddings().weight.detach(); V, Dm = Emb.shape
rng = random.Random(0); defs, texts = [], []
for name in sr.POOL_TASKS:
    data = json.load(open(os.path.join("CL_Benchmark", "Long_Sequence", name, "train.json")))
    d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
    defs.append(d); texts.append([inst["input"] for inst in data["Instances"]])
picks = [rng.randrange(len(defs)) for _ in range(sr.BG_N)]
sources = [sr._source(defs[k], rng.choice(texts[k])) for k in picks]
rows = tok(sources, max_length=MAX_LEN, truncation=True)["input_ids"]
bg_rows = rows
counts = torch.zeros(V, dtype=torch.double)
for r in rows:
    r = torch.tensor(r); r = r[r != tok.pad_token_id]; counts += torch.bincount(r, minlength=V).double()
WEIGHT = sr.SIF_A / (sr.SIF_A + counts / counts.sum()); WEIGHT[tok.pad_token_id] = 0.0
def cum(ids, nmax):
    T = min(len(ids), int(nmax)); t = torch.tensor(ids[:T]); w = WEIGHT[t]
    return (torch.cumsum(Emb[t].double() * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None]).numpy()
def phi(ids, ns):
    ns = np.asarray(ns); c = cum(ids, ns.max()); return c[np.minimum(ns, len(ids)) - 1]
x64 = np.stack([phi(r[:64], [64])[0] for r in rows]); mu = x64.mean(0); xc = x64 - mu
cov = xc.T @ xc / (len(x64) - 1); cov = 0.5 * cov + 0.5 * np.eye(Dm) * np.trace(cov) / Dm
W = inv_sqrt(cov)
bg_raw = []
for d, xs in zip(defs, texts):
    r = tok([sr._source(d, x) for x in xs[:20]], max_length=MAX_LEN, truncation=True)["input_ids"]
    n = sr._prefix_len(r); assert n >= 5
    bg_raw.append(phi(r[0][:n], [n])[0])
bg_raw = np.stack(bg_raw)
SIGMA = {}
def get_sigma(lengths):
    """sigma(n): root-mean-square distance of the unrelated inputs (bg_rows) from their own mean in psi_n (same weights, mu, W)."""
    todo = sorted({int(n) for n in lengths} - set(SIGMA))
    if not todo: return
    ns = np.array(todo); sm = np.zeros((len(ns), Dm)); sq = np.zeros(len(ns))
    for r in bg_rows:
        c = cum(r, 10 ** 9)
        P = (c[np.minimum(ns, len(r)) - 1] - mu) @ W
        sm += P; sq += (P ** 2).sum(1)
    cnt = len(bg_rows)
    for n, a, b in zip(todo, sm, sq): SIGMA[n] = float(np.sqrt(max(b / cnt - float((a / cnt) @ (a / cnt)), 0.0)))
nk_ev = np.array([min(sr._prefix_len([ids_of(template(by_name[n]["definition"]).replace("{0}", "A B")), ids_of(template(by_name[n]["definition"]).replace("{0}", "zzq xkj"))]), MAX_LEN) for n in ev_names])
get_sigma(nk_ev)
SIG_EV = np.array([SIGMA[int(n)] for n in nk_ev])
log("sigma over", len(set(nk_ev.tolist())), "opening lengths", min(SIGMA.values()), max(SIGMA.values()))
cache = f"{OUT}/dist_exact_it.npz" if ITALIAN else f"{OUT}/dist_exact.npz"
if os.path.exists(cache):
    z = np.load(cache); D, EX, sigz, bgz = z["D"], z["EX"], z["sigz"], z["bgz"]
    assert D.shape == (N, K) and EX.shape == (N, K)
    log("loaded cache")
else:
    ops = {}
    for n in ev_names:
        a = ids_of(template(by_name[n]["definition"]).replace("{0}", "A B")); b = ids_of(template(by_name[n]["definition"]).replace("{0}", "zzq xkj"))
        nk = min(sr._prefix_len([a, b]), MAX_LEN); ops[n] = (a[:nk], nk)
    nk = np.array([ops[n][1] for n in ev_names]); uniq = np.unique(nk); where = np.searchsorted(uniq, nk)
    sig_raw = np.stack([phi(ops[n][0], [ops[n][1]])[0] for n in ev_names])
    sigz = (sig_raw - mu) @ W; bgz = (bg_raw - mu) @ W
    openings = collections.defaultdict(list)      # exact claims as in exact_first_analysis.py
    for k, n in enumerate(ev_names): openings[tuple(ops[n][0][:min(ops[n][1], MAX_LEN - 1)])].append(k)
    lengths = sorted({len(o) for o in openings})
    D = np.zeros((N, K), np.float32); EX = np.zeros((N, K), bool)
    for i, (_, _, text) in enumerate(items):
        ids = ids_of(text)
        f = phi(ids, uniq) - mu
        D[i] = np.linalg.norm((f @ W)[where] - sigz, axis=1)
        for m in lengths:
            if m <= len(ids):
                for j in openings.get(tuple(ids[:m]), ()): EX[i, j] = True
        if i % 2000 == 0: log("item", i)
    np.savez(cache, D=D, EX=EX, sigz=sigz, bgz=bgz)

# ------------------------------------------------------------------ radii in arrival order (rule B, kappa 0.5)
d_sig = snap(np.linalg.norm(sigz[:, None] - sigz[None], axis=2))
d_bg = snap(np.linalg.norm(sigz[:, None] - bgz[None], axis=2).min(1))
radius = np.zeros(K)
for j in range(K):
    near = d_bg[j] if j == 0 else min(d_bg[j], d_sig[j, :j].min())
    radius[j] = KAPPA * near
Ds = snap(D)                                          # (N, K) snapped distances

def route(rows, upto, rule):
    """route of items `rows` when only tasks 0..upto exist (arrival order); ties go to the older task. -> (route, inside any, first zone)"""
    d = Ds[rows][:, :upto + 1]; r = radius[:upto + 1]
    inside = d <= r[None]
    with np.errstate(divide="ignore", invalid="ignore"): rel = d / r[None]
    rel = np.where(np.isnan(rel), 0.0, rel)
    near = rel.argmin(1); anyin = inside.any(1)
    if rule in ("background", "certified4"):                   # distance / B_k (B_k: distance to the nearest UNRELATED signature); certified4: zones first
        with np.errstate(divide="ignore", invalid="ignore"): q = d / d_bg[:upto + 1][None]
        bgr = np.where(np.isnan(q), 0.0, q).argmin(1)
        return (bgr if rule == "background" else np.where(anyin, inside.argmax(1), bgr)), anyin
    if rule in ("normalised", "certified3"):                   # distance / sigma(n_k); certified3: zones first
        nrm = (d / SIG_EV[:upto + 1][None]).argmin(1)
        return (nrm if rule == "normalised" else np.where(anyin, inside.argmax(1), nrm)), anyin
    if rule == "signature": return d.argmin(1), anyin          # plain distance, no radius; ties to the older task
    if rule in ("hybrid", "exactfirst_hybrid"):                # zones first (oldest holding zone), else plain nearest signature
        hyb = np.where(anyin, inside.argmax(1), d.argmin(1))
        if rule == "hybrid": return hyb, anyin
        ex = EX[rows][:, :upto + 1]
        return np.where(ex.any(1), ex.argmax(1), hyb), anyin
    if rule == "nearest": return near, anyin
    cert = np.where(anyin, inside.argmax(1), near)
    if rule == "certified": return cert, anyin
    ex = EX[rows][:, :upto + 1]
    return np.where(ex.any(1), ex.argmax(1), cert), anyin

def routes_end(rule):
    return route(np.arange(N), K - 1, rule)

def routes_arrival(rule):
    out = np.zeros(N, int); anyin = np.zeros(N, bool)
    for j in range(K):
        rows = np.nonzero(owner == j)[0]
        out[rows], anyin[rows] = route(rows, j, rule)
    return out, anyin

# ------------------------------------------------------------------ step 0
mu_k = kind == 0; both = (kind == 1) | (kind == 2)
Du, ou = Ds[mu_k], owner[mu_k]
older = np.arange(K)[None, :] < ou[:, None]
blocked = ((Du <= radius[None]) & older).any(1)
has = np.bincount(ou, minlength=K) > 0
coll = np.bincount(ou, weights=blocked.astype(float), minlength=K) > 0
rt_u, _ = route(np.nonzero(mu_k)[0], K - 1, "certified")
rp, cp = route(np.nonzero(both)[0], K - 1, "certified"); own = rp == owner[both]
ref = {"C": 0.1686, "U": 0.8314, "P": 0.4052, "M": 0.0755, "R": 0.7542}
got = {"C": float(coll[has].sum() / has.sum()), "U": float((rt_u == ou).mean()), "P": float((cp & own).mean()),
       "M": float((cp & ~own).mean()), "R": float(own.mean())}
ok = all(abs(got[k] - ref[k]) < 0.00051 for k in ref) and int(both.sum()) == 4198
reproduced = {**{k: got[k] for k in got}, "n_paraphrased": int(both.sum()), "reference": ref, "ok": bool(ok)}
log("reproduction", {k: round(v, 4) for k, v in got.items()}, "n", int(both.sum()), "ok", ok)
if not ok:
    json.dump({"reproduced": reproduced}, open(os.path.join(HERE, "routing", "reroute_study.json"), "w"), indent=1)
    log("REPRODUCTION FAILED, stopping"); sys.exit(2)

# ------------------------------------------------------------------ step 1 and 2
RULES = ("certified", "nearest", "exactfirst", "signature", "hybrid", "exactfirst_hybrid", "normalised", "certified3", "background", "certified4")
R = {}
for rule in RULES:
    a, anyin = routes_arrival(rule); e, _ = routes_end(rule)
    R[rule] = {"arr": a, "end": e, "cert": anyin}
# theorem: an input inside a zone at arrival never changes its route under certified
c = R["certified"]
assert not ((c["arr"] != c["end"]) & c["cert"]).any(), "a certified route changed: the guarantee is broken"
# sanity: route "at arrival" of the last task equals the end route
last = np.nonzero(owner == K - 1)[0]
assert (c["arr"][last] == c["end"][last]).all()
# and, by theorem for exactfirst? (not asserted; reported below)

SETS = {"unedited": kind == 0, "French": kind == 2, "German": kind == 1, "French+German": both, "Spanish": kind == 3}
if ITALIAN: SETS["Italian"] = kind == 4
def share(m, sel): return float(m[sel].sum() / sel.sum())
rules_out = {}
for rule in RULES:
    a, e, cert = R[rule]["arr"], R[rule]["end"], R[rule]["cert"]; rules_out[rule] = {}
    for sname, sel in SETS.items():
        ch = (a != e)
        blk = {"n": int(sel.sum()), "own_at_arrival": share(a == owner, sel), "own_at_end": share(e == owner, sel),
               "lost": share((a == owner) & (e != owner), sel), "changed": share(ch, sel),
               "n_own_at_arrival": int(((a == owner) & sel).sum()), "n_own_at_end": int(((e == owner) & sel).sum()),
               "n_lost": int(((a == owner) & (e != owner) & sel).sum()), "n_changed": int((ch & sel).sum()),
               "tasks_with_changed": int(len(set(owner[ch & sel].tolist())))}
        if rule in ("certified", "hybrid", "exactfirst_hybrid", "certified3", "certified4"):
            blk["certified_at_arrival"] = share(cert, sel)
            blk["changed_among_certified"] = int((ch & cert & sel).sum())
            if rule != "exactfirst_hybrid": assert blk["changed_among_certified"] == 0, "a zone-certified route changed" 
            blk["changed_among_uncertified"] = share(ch & ~cert, sel)
        rules_out[rule][sname] = blk

# ------------------------------------------------------------------ step 3: protected
prot = {}
for sname in ("French+German", "Spanish"):
    sel = SETS[sname]; ca, na, ne = R["certified"]["cert"], R["nearest"]["arr"], R["nearest"]["end"]
    certown = sel & ca & (R["certified"]["arr"] == owner)
    lost = certown & (na == owner) & (ne != owner)
    prot[sname] = {"n_certified_own": int(certown.sum()), "lost_by_nearest": int(lost.sum()),
                   "share_lost_by_nearest": float(lost.sum() / certown.sum()),
                   "share_other_task_at_end_under_nearest": float((certown & (ne != owner)).sum() / certown.sum()),
                   "n_other_task_at_end_under_nearest": int((certown & (ne != owner)).sum())}

# ------------------------------------------------------------------ step 4: verdict
Gn = rules_out["nearest"]["French+German"]["changed"]; Gc = rules_out["certified"]["French+German"]["changed"]
Ge = rules_out["exactfirst"]["French+German"]["changed"]
if Gn >= 0.05 and Gc <= 0.5 * Gn: outcome = "adds something at scale"
elif Gn < 0.02 or Gc > 0.75 * Gn: outcome = "adds little here"
else: outcome = "something, but less than half of the changes are removed"
verdict = {"G_nearest": Gn, "G_certified": Gc, "G_exactfirst": Ge, "ratio_certified_to_nearest": (Gc / Gn if Gn > 0 else None), "outcome": outcome}
log("verdict", verdict)

# ------------------------------------------------------------------ step 5
na, ne = R["nearest"]["arr"], R["nearest"]["end"]
ch_n = na != ne
cand = np.nonzero(both & ch_n & R["certified"]["cert"] & (R["certified"]["arr"] == owner))[0]
pick = cand[np.unique(np.linspace(0, len(cand) - 1, 15).round().astype(int))] if len(cand) else cand
examples = [{"task": ev_names[owner[i]], "pivot": KINDS[kind[i]], "nearest_at_end_task": ev_names[ne[i]],
             "own_position": int(owner[i]) + 1, "other_position": int(ne[i]) + 1,
             "tasks_arrived_in_between": int(ne[i] - owner[i] - 1), "tasks_arrived_until_holder": int(ne[i] - owner[i]),
             "nearest_at_arrival_task": ev_names[na[i]]} for i in pick]
groups = np.minimum(owner * 5 // K, 4)
by_group = {}
for g in range(5):
    gs = (groups == g)
    row = {"arrival_positions": f"{g * 140 + 1}-{g * 140 + 140}"}
    for sname in ("French+German", "Spanish"):
        sel = SETS[sname] & gs
        row[sname] = {"n": int(sel.sum()), "changed_nearest": share(ch_n, sel), "n_changed_nearest": int((ch_n & sel).sum()),
                      "changed_certified": share(R["certified"]["arr"] != R["certified"]["end"], sel),
                      "changed_exactfirst": share(R["exactfirst"]["arr"] != R["exactfirst"]["end"], sel)}
    by_group[f"group{g + 1}"] = row
defn = [by_name[n]["definition"] for n in ev_names]
sib = {"measure": "word overlap of the two task definitions, as alike() in many_tasks_analysis.py; threshold 0.8"}
for sname in ("French+German", "Spanish", "unedited"):
    sel = SETS[sname] & ch_n
    idx = np.nonzero(sel)[0]
    hits = sum(alike(defn[owner[i]], defn[ne[i]]) >= 0.8 for i in idx)
    sib[sname] = {"n_changed_nearest": int(len(idx)), "n_to_sibling": int(hits), "share_of_changed": (hits / len(idx) if len(idx) else None)}
notes = {"rule": "certified/nearest/exactfirst as in many_tasks_analysis.py and exact_first_analysis.py; radii in arrival order, kappa 0.5, snap 1e-3 (as ruler_study.py); old ruler (whitening) with T5-base word-vectors",
         "arrival": "route at arrival of task j uses only tasks 1..j and their radii (radii depend only on earlier tasks); end = all 700",
         "items": "unedited inputs[:6], paraphrased inputs[:3] per pivot, as ruler_study.py / spanish_check.py; paraphrases identical to the original are kept",
         "primary": "French+German together",
         "ties": "older task first (argmin / argmax over arrival order)",
         "own_at_arrival_nearest_note": "under nearest the own task at arrival may be chosen by distance; under certified the oldest containing zone wins",
         "examples": "15 inputs, French/German, changed under nearest and certified for their own task under certified (certified at arrival, routed to own), spaced evenly along the list ordered by task arrival",
         "tasks_arrived_in_between": "tasks that arrived after the own task and before the task that takes the input at the end (exclusive of both)",
         "by_arrival_group": "five equal groups of 140 arrival positions"}
# ------------------------------------------------------------------ follow-up: rule `signature` (plain nearest signature)
sg_a, sg_e = R["signature"]["arr"], R["signature"]["end"]; ch_s = sg_a != sg_e
Gs = rules_out["signature"]["French+German"]["changed"]
verdict_followup = {"G_signature": Gs, "outcome": ("the radius fixed at arrival is what keeps routes stable" if Gs >= 0.05 else
                    "re-routing is rare in this collection under any best-match rule" if Gs < 0.02 else "some")}
prot_s = {}
for sname in ("French+German", "Spanish"):
    sel = SETS[sname]; certown = sel & R["certified"]["cert"] & (R["certified"]["arr"] == owner)
    lost = certown & (sg_a == owner) & (sg_e != owner); oth = certown & (sg_e != owner)
    prot_s[sname] = {"n_certified_own": int(certown.sum()), "lost_by_signature": int(lost.sum()), "share_lost_by_signature": float(lost.sum() / certown.sum()),
                     "n_other_task_at_end": int(oth.sum()), "share_other_task_at_end": float(oth.sum() / certown.sum())}
cs = np.nonzero(both & ch_s)[0]
pk = cs[np.unique(np.linspace(0, len(cs) - 1, 15).round().astype(int))] if len(cs) else cs
examples_signature = [{"task": ev_names[owner[i]], "pivot": KINDS[kind[i]], "task_at_end": ev_names[sg_e[i]], "tasks_arrived_in_between": int(sg_e[i] - owner[i] - 1) if sg_e[i] > owner[i] else None,
                       "own_position": int(owner[i]) + 1, "end_position": int(sg_e[i]) + 1, "word_share_of_definitions": alike(defn[owner[i]], defn[sg_e[i]])} for i in pk]
sib_s = {"measure": sib["measure"]}
for sname in ("French+German", "Spanish", "unedited"):
    idx = np.nonzero(SETS[sname] & ch_s)[0]; hits = sum(alike(defn[owner[i]], defn[sg_e[i]]) >= 0.8 for i in idx)
    sib_s[sname] = {"n_changed_signature": int(len(idx)), "n_to_sibling": int(hits), "share_of_changed": (hits / len(idx) if len(idx) else None)}
bag_s = {}
for g in range(5):
    row = {"arrival_positions": f"{g * 140 + 1}-{g * 140 + 140}"}
    for sname in ("French+German", "Spanish"):
        sel = SETS[sname] & (groups == g)
        row[sname] = {"n": int(sel.sum()), "changed_signature": share(ch_s, sel), "n_changed_signature": int((ch_s & sel).sum())}
    bag_s[f"group{g + 1}"] = row
# ------------------------------------------------------------------ summary table
summary = {}
ALLR = ("signature", "nearest", "certified", "exactfirst", "hybrid", "exactfirst_hybrid", "normalised", "certified3", "background", "certified4")
for sname in ("French+German", "Spanish") + (("Italian",) if ITALIAN else ()):
    summary[sname] = {}
    for rule in ALLR:
        b = rules_out[rule][sname]
        summary[sname][rule] = {"own_at_end": b["own_at_end"], "changed": b["changed"], "lost": b["lost"],
                                "certified_at_arrival": b.get("certified_at_arrival", 0.0) if rule in ("certified", "hybrid", "exactfirst_hybrid", "certified3", "certified4") else 0.0}
summary["unedited"] = {rule: {"own_at_end": rules_out[rule]["unedited"]["own_at_end"]}
                       for rule in ALLR}
summary["note"] = "certified_at_arrival is 0 for signature and nearest (no zone certificate); exactfirst has no certificate of its own and is also given 0"

# ------------------------------------------------------------------ benchmark view (five / fifteen tasks, four wordings), routes only
FIVE = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
        "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification", "task875_emotion_classification"]
ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]
names15 = sorted(set(FIVE) | set(ORDER1)); bdata = {}
for n in names15:
    dd_ = {}
    for split in ("train", "test"):
        dj = json.load(open(os.path.join("CL_Benchmark", "SuperNI", n, split + ".json")))
        if split == "train": dd_["definition"] = (dj["Definition"][0] if isinstance(dj["Definition"], list) else dj["Definition"]).strip()
        else: dd_["test"] = [i["input"] for i in dj["Instances"]]
    bdata[n] = dd_
bdefs = {n: [bdata[n]["definition"]] + sr.REWORDED.get(n.split("_")[0], []) for n in names15}
bops = {}
for n in names15:
    a = ids_of(template(bdefs[n][0]).replace("{0}", "A B")); b = ids_of(template(bdefs[n][0]).replace("{0}", "zzq xkj"))
    nkb = min(sr._prefix_len([a, b]), MAX_LEN); bops[n] = (a[:nkb], nkb)
bnk = np.array([bops[n][1] for n in names15]); buniq = np.unique(bnk); bwhere = np.searchsorted(buniq, bnk)
bsig = (np.stack([phi(bops[n][0], [bops[n][1]])[0] for n in names15]) - mu) @ W
bbg = (bg_raw - mu) @ W
btexts, bmeta = [], []
for n in names15:
    for li, dfn in enumerate(bdefs[n]):
        for x in bdata[n]["test"]: btexts.append(template(dfn).replace("{0}", x)); bmeta.append((n, li))
BD = np.zeros((len(btexts), len(names15)))
for i, tx in enumerate(btexts):
    BD[i] = np.linalg.norm(((phi(ids_of(tx), buniq) - mu) @ W)[bwhere] - bsig, axis=1)
log("benchmark inputs", len(btexts), {n.split("_")[0]: len(bdefs[n]) for n in names15 if len(bdefs[n]) != 4})
bcol = {n: i for i, n in enumerate(names15)}
get_sigma(bnk); BSIG = np.array([SIGMA[int(n)] for n in bnk])
def bench_route(d, r, upto, rule, sg, bk=None):
    d = d[:, :upto + 1]; r = r[:upto + 1]; inside = d <= r[None]
    with np.errstate(divide="ignore", invalid="ignore"): rel = d / r[None]
    rel = np.where(np.isnan(rel), 0.0, rel); anyin = inside.any(1); near = rel.argmin(1)
    if rule == "nearest": return near, anyin
    if rule == "signature": return d.argmin(1), anyin
    if rule in ("background", "certified4"):
        with np.errstate(divide="ignore", invalid="ignore"): q = d / bk[:upto + 1][None]
        bgr = np.where(np.isnan(q), 0.0, q).argmin(1)
        return (bgr if rule == "background" else np.where(anyin, inside.argmax(1), bgr)), anyin
    nrm = (d / sg[:upto + 1][None]).argmin(1)
    if rule == "normalised": return nrm, anyin
    if rule == "certified3": return np.where(anyin, inside.argmax(1), nrm), anyin
    first = inside.argmax(1)
    if rule == "certified": return np.where(anyin, first, near), anyin
    return np.where(anyin, first, d.argmin(1)), anyin
bench = {}; bstat = {}; bident = {}
stored = json.load(open(os.path.join(HERE, "routing", "ruler_study.json")))["tables"]["t5-base"]["secondary"]["OLD"]
for label, order in (("five", FIVE), ("fifteen", ORDER1)):
    cols = [bcol[n] for n in order]; s_ = bsig[cols]; Kb = len(order); sgb = BSIG[cols]; bkb = dbg_b = None
    dsg = snap(np.linalg.norm(s_[:, None] - s_[None], axis=2)); dbg = snap(np.linalg.norm(s_[:, None] - bbg[None], axis=2).min(1))
    rad = np.zeros(Kb); bkb = dbg
    bstat[label] = {"B_min": float(dbg.min()), "B_median": float(np.median(dbg)), "B_max": float(dbg.max()),
                    "share_radius_equals_half_B": float(np.mean(np.array([KAPPA * (dbg[j] if j == 0 else min(dbg[j], dsg[j, :j].min())) for j in range(Kb)]) == KAPPA * dbg))}
    for j in range(Kb):
        rad[j] = KAPPA * (dbg[j] if j == 0 else min(dbg[j], dsg[j, :j].min()))
    bench[label] = {}
    if label == "fifteen":
        bident["fifteen_radius_not_half_B"] = []
        for j in range(Kb):
            if rad[j] != KAPPA * dbg[j]:
                e_ = int(dsg[j, :j].argmin())
                bident["fifteen_radius_not_half_B"].append({"task": order[j], "position": j + 1, "radius": float(rad[j]), "half_B": float(KAPPA * dbg[j]),
                    "nearest_earlier_task": order[e_], "distance_to_it": float(dsg[j, e_]), "B": float(dbg[j]), "half_distance_to_it": float(KAPPA * dsg[j, e_])})
    for li, lev in enumerate(sr.LEVELS):
        sel = [i for i, (n, l) in enumerate(bmeta) if l == li and n in order]
        own_ = np.array([order.index(bmeta[i][0]) for i in sel]); dsel = snap(BD[sel][:, cols])
        bench[label][lev] = {}
        for rule in ("certified", "nearest", "signature", "hybrid", "normalised", "certified3", "background", "certified4"):
            end, in_end = bench_route(dsel, rad, Kb - 1, rule, sgb, bkb)
            arr = np.zeros(len(sel), int); in_arr = np.zeros(len(sel), bool)
            for j in range(Kb):
                m = own_ == j
                if m.any(): arr[m], in_arr[m] = bench_route(dsel[m], rad, j, rule, sgb, bkb)
            blk = {"n": int(len(sel)), "own_at_end": float((end == own_).mean()), "own_at_arrival": float((arr == own_).mean()),
                   "changed": int((arr != end).sum()), "inside_zone_at_end": float(in_end.mean()), "inside_zone_at_arrival": float(in_arr.mean())}
            if rule in ("certified", "hybrid", "certified3", "certified4"): assert not ((arr != end) & in_arr).any(), "a zone-certified benchmark route changed"
            bench[label][lev][rule] = blk
        def allt(rule):
            return np.stack([bench_route(dsel, rad, u, rule, sgb, bkb)[0] for u in range(Kb)])      # (Kb, n): route after each arrival u
        ident = {}
        for a_, b_ in (("certified", "certified4"), ("nearest", "background")):
            A_, B_ = allt(a_), allt(b_); diff = A_ != B_
            exist = np.arange(Kb)[:, None] >= own_[None, :]                                          # the input's own task has arrived
            ident[f"{a_}_vs_{b_}"] = {"n": int(len(sel)), "differ_after_last_task": int(diff[Kb - 1].sum()),
                                      "differ_at_any_arrival_from_own_task_on": int((diff & exist).any(0).sum()),
                                      "differ_at_any_arrival_incl_before_own_task": int(diff.any(0).sum())}
        bident.setdefault(label, {})[lev] = ident
        st = stored[label][lev]                                   # reproduction of certified against ruler_study.json
        ok_b = bench[label][lev]["certified"]["n"] == st["inputs"] and abs(bench[label][lev]["certified"]["own_at_end"] - st["certified"]) < 1e-9
        log("benchmark", label, lev, "certified own", round(bench[label][lev]["certified"]["own_at_end"], 4), "stored", round(st["certified"], 4), "ok", ok_b)
        if not ok_b:
            json.dump({"benchmark_reproduction_failed": [label, lev]}, open(os.path.join(OUT, "bench_failed.json"), "w")); sys.exit(3)

# ------------------------------------------------------------------ sigma summary
sv = np.array(list(SIGMA.values())); lens_ev = sorted(set(nk_ev.tolist()))
sig_ev_vals = np.array([SIGMA[n] for n in lens_ev])
sigma_out = {"by_length": {str(n): SIGMA[n] for n in sorted(SIGMA)}, "evaluation_lengths": {"n_distinct": len(lens_ev), "min": float(sig_ev_vals.min()),
             "median": float(np.median(sig_ev_vals)), "max": float(sig_ev_vals.max())},
             "corr_sigma_radius_700_tasks": float(np.corrcoef(SIG_EV, radius)[0, 1]),
             "corr_sigma_radius_700_tasks_spearman": float(np.corrcoef(np.argsort(np.argsort(SIG_EV)), np.argsort(np.argsort(radius)))[0, 1]),
             "definition": "RMS distance of the 4000 unrelated inputs (the paper's token-weight inputs) from their own mean in psi_n = (token-weighted average of the first n tokens - mu) W; shorter inputs use all tokens"}

# ------------------------------------------------------------------ stress test (construction of exact_first_analysis.py: five tasks, then a sixth with SST-2 inputs under a reworded definition)
stress = {}
old = {"definition": bdata[FIVE[1]]["definition"], "test": bdata[FIVE[1]]["test"]}
old_train = [i["input"] for i in json.load(open(os.path.join("CL_Benchmark", "SuperNI", FIVE[1], "train.json")))["Instances"]]
sdefs = [old["definition"]] + sr.REWORDED["task363"]
def opening_of(defn):
    a = ids_of(template(defn).replace("{0}", "A B")); b = ids_of(template(defn).replace("{0}", "zzq xkj"))
    n_ = min(sr._prefix_len([a, b]), MAX_LEN); return a[:n_], n_
def dist_to(texts, sigs, nks):
    un = np.unique(nks); wh = np.searchsorted(un, nks)
    return np.stack([np.linalg.norm(((phi(ids_of(tx), un) - mu) @ W)[wh] - sigs, axis=1) for tx in texts])
five_idx = [bcol[n] for n in FIVE]
for li_new in (1, 2, 3):
    op6, n6 = opening_of(sdefs[li_new]); get_sigma([n6])
    sig6 = (phi(op6, [n6])[0] - mu) @ W
    sigs = np.vstack([bsig[five_idx], sig6[None]]); nks = np.concatenate([bnk[five_idx], [n6]]); sg = np.concatenate([BSIG[five_idx], [SIGMA[n6]]])
    dsg6 = snap(np.linalg.norm(sigs[:, None] - sigs[None], axis=2)); dbg6 = snap(np.linalg.norm(sigs[:, None] - bbg[None], axis=2).min(1))
    rad6 = np.zeros(6)
    for j in range(6): rad6[j] = KAPPA * (dbg6[j] if j == 0 else min(dbg6[j], dsg6[j, :j].min()))
    texts_old = [template(sdefs[l]).replace("{0}", x) for l in range(4) for x in old["test"]]
    lev_old = np.repeat(np.arange(4), len(old["test"]))
    d_old = snap(dist_to(texts_old, sigs, nks))
    d_new = snap(dist_to([template(sdefs[li_new]).replace("{0}", x) for x in old_train], sigs, nks))
    row = {}
    for rule in ("certified3", "normalised", "certified4", "background", "certified", "nearest"):
        before, _ = bench_route(d_old, rad6, 4, rule, sg, dbg6); after, in6 = bench_route(d_old, rad6, 5, rule, sg, dbg6)
        own6, _ = bench_route(d_new, rad6, 5, rule, sg, dbg6)
        row[rule] = {"changed_per_wording": {sr.LEVELS[l]: int(((before != after) & (lev_old == l)).sum()) for l in range(4)},
                     "n_old_inputs_per_wording": int(len(old["test"])), "new_task_own": int((own6 == 5).sum()), "new_task_inputs": int(len(old_train))}
        if rule in ("certified", "certified3", "certified4"): assert not ((before != after) & (lev_old >= 0) & bench_route(d_old, rad6, 4, "certified", sg, dbg6)[1]).any(), "stress: a certified route changed"
    stress[sr.LEVELS[li_new]] = row
    log("stress", sr.LEVELS[li_new], {r: v["changed_per_wording"] for r, v in row.items()})

extra = {}
B_out = {"definition": "B_k = distance from the signature of task k to the nearest signature of the unrelated tasks only (d_bg of the radius rule)",
         "700_tasks": {"min": float(d_bg.min()), "median": float(np.median(d_bg)), "max": float(d_bg.max()),
                       "corr_B_radius": float(np.corrcoef(d_bg, radius)[0, 1]),
                       "share_radius_equals_half_B": float(np.mean(radius == KAPPA * d_bg)), "n_B_zero": int((d_bg == 0).sum())},
         "benchmark": bstat}
extra["B"] = B_out
if ITALIAN:
    selI = kind == 4
    rI, certI = route(np.nonzero(selI)[0], K - 1, "certified"); ownI = rI == owner[selI]
    extra["italian_paper_measures_certified"] = {"P": float((certI & ownI).mean()), "M": float((certI & ~ownI).mean()), "R": float(ownI.mean()), "n": int(selI.sum())}
    hf, hF = bench["five"]["heavy"]["normalised"]["own_at_end"], bench["fifteen"]["heavy"]["normalised"]["own_at_end"]
    mf, mF = bench["five"]["medium"]["normalised"]["own_at_end"], bench["fifteen"]["medium"]["normalised"]["own_at_end"]
    iw = rules_out["normalised"]["Italian"]["own_at_end"]
    out_n = ("success" if (hf >= 0.90 and hF >= 0.90 and mf >= 0.97 and mF >= 0.97 and iw >= 0.88) else
             "against" if (hf < 0.80 or hF < 0.80 or iw < 0.85) else "partial")
    bh = lambda a, l, r: bench[a][l][r]["own_at_end"]; ib = rules_out["background"]["Italian"]["own_at_end"]
    hf2, hF2, mf2, mF2 = bh("five", "heavy", "background"), bh("fifteen", "heavy", "background"), bh("five", "medium", "background"), bh("fifteen", "medium", "background")
    out_b = ("success" if (hf2 >= 0.90 and hF2 >= 0.90 and mf2 >= 0.97 and mF2 >= 0.97 and ib >= 0.88) else
             "against" if (hf2 < 0.80 or hF2 < 0.80 or ib < 0.85) else "partial")
    extra["verdict_background"] = {"heavy_five": hf2, "heavy_fifteen": hF2, "medium_five": mf2, "medium_fifteen": mF2, "italian_own_at_end": ib, "outcome": out_b}
    extra["verdict_normalised"] = {"heavy_five": hf, "heavy_fifteen": hF, "medium_five": mf, "medium_fifteen": mF, "italian_own_at_end": iw, "outcome": out_n}
res = {"notes": notes, "reproduced": reproduced, "rules": rules_out, "protected": prot, "verdict": verdict,
       "examples": examples, "n_examples_candidates": int(len(cand)), "by_arrival_group": by_group, "siblings": sib,
       "verdict_followup": verdict_followup, "protected_signature": prot_s, "examples_signature": examples_signature,
       "siblings_signature": sib_s, "by_arrival_group_signature": bag_s,
       "summary_table": summary, "benchmark": bench,
       "sigma": sigma_out, "stress": stress, "benchmark_identity": bident, **extra}
json.dump(res, open(os.path.join(HERE, "routing", "reroute_study.json"), "w"), indent=1)
log("written")
