"""Third pre-registered experiment (last section of silent_adapters/PLAN.md): a small frozen sentence encoder as the feature
(all-MiniLM-L6-v2 primary; bge-small-en-v1.5 reported), prefix of the opening at a grid length, plain or with the paraphrase
ruler, one radius for all tasks from a split-conformal rule (half B of the 175 calibration tasks). CPU only, 2 threads.
Steps 1-5 of the brief for the primary encoder, then second views a (curve), e (collisions), d (benchmark check), f (cost),
b (instruction as a separate field: MiniLM, bge, all-mpnet-base-v2), c (steps 1-5 for bge).

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2 \
        $S/mm_env/bin/python encoder_study.py
Caches (feature arrays) in $S/encoder_study/; writes routing/encoder_study.json next to this script (after every stage).
Conventions reused from feature_study.py / ruler_study.py: split of 875 tasks (Random(0) on sorted names, 175 calibration /
700 evaluation in shuffled order = arrival order), template, opening = common token prefix of the template with two different
inputs, route rule (oldest zone containing the input, else nearest; ties older), measures C, P, M, R, U, benchmark check.
"""
import json, os, random, sys, time, types, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", f"{S}/hf_cache")
OUT = f"{S}/encoder_study"; os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(HERE, "routing", "encoder_study.json")
GRID = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 254]
MAXTOK = 254
ALPHAS = [0.05, 0.1, 0.2]
SNAP = 1e-4
QGRID = [round(0.0125 * i, 4) for i in range(1, 81)]
C_REF = 0.189
ENCODERS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean"), "bge": ("BAAI/bge-small-en-v1.5", "cls"),
            "mpnet": ("sentence-transformers/all-mpnet-base-v2", "mean")}

import torch
from transformers import AutoModel, AutoTokenizer
torch.set_num_threads(2)

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
t_start = time.time()
def log(*a): print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)
def template(definition): return "Definition: " + definition + "\n\nNow complete the following example -\nInput: {0}\nOutput: "
def grid_len(n): return n if n < GRID[0] else max(g for g in GRID if g <= n)

RESULT = {"notes": {}}
def dump(): json.dump(RESULT, open(JSON_PATH, "w"), indent=1)

# ---------------------------------------------------------------- the encoder
class Enc:
    def __init__(self, key):
        self.key = key; name, self.pool = ENCODERS[key]; self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()
        self.dim = self.model.config.hidden_size; self.cache = {}
    def ids(self, text, maxlen=MAXTOK):
        return tuple(self.tok(text, add_special_tokens=False, max_length=maxlen, truncation=True)["input_ids"])
    def encode(self, seqs, use_cache=True, tokens_per_batch=6000):
        """seqs: list of tuples of token ids WITHOUT special tokens (specials are added here). -> (len, dim) float32, unit length."""
        keys = list(dict.fromkeys(seqs))
        todo = [k for k in keys if (not use_cache) or k not in self.cache]
        todo.sort(key=len)
        s = 0; local = {}
        while s < len(todo):
            e = s + 1
            while e < len(todo) and (e - s + 1) * (len(todo[e]) + 2) <= tokens_per_batch and e - s < 256: e += 1
            b = todo[s:e]; s = e
            full = [[self.tok.cls_token_id] + list(k) + [self.tok.sep_token_id] for k in b]
            L = max(len(f) for f in full)
            ids = torch.full((len(b), L), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
            for r, f in enumerate(full): ids[r, :len(f)] = torch.tensor(f); att[r, :len(f)] = 1
            with torch.inference_mode():
                h = self.model(input_ids=ids, attention_mask=att).last_hidden_state
                if self.pool == "mean":
                    m = att[:, :, None].to(h.dtype); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                else: v = h[:, 0]
                v = torch.nn.functional.normalize(v, dim=-1).numpy()
            for k, vec in zip(b, v):
                if use_cache: self.cache[k] = vec
                else: local[k] = vec
        src = self.cache if use_cache else local
        return np.stack([src[k] for k in seqs]) if len(seqs) else np.zeros((0, self.dim), np.float32)

# ---------------------------------------------------------------- data
tasks = json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"]
para = json.load(open(os.path.join(HERE, "routing", "paraphrases.json")))
PIV = [p for p in ("de", "fr") if all(p in v for v in para["tasks"].values())]
assert PIV == ["de", "fr"]
by_name = {t["name"]: t for t in tasks}
names = sorted(by_name)
rr = random.Random(0); shuffled = names[:]; rr.shuffle(shuffled)
cal_names, ev_names = shuffled[:175], shuffled[175:]
cal_sorted = sorted(cal_names)
half_A = cal_sorted[0::2]; half_B = cal_sorted[1::2]
ev_col = {n: i for i, n in enumerate(ev_names)}
K = len(ev_names)

def inputs_text(defn, x): return template(defn).replace("{0}", x)

def prepare_openings(enc):
    """n_k, L_k, signature (unit vector) for all 875 tasks."""
    nk, Lk, ops = {}, {}, {}
    for n in names:
        d = by_name[n]["definition"]
        a = enc.ids(inputs_text(d, "A B")); b = enc.ids(inputs_text(d, "zzq xkj"))
        k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
        nk[n] = k; Lk[n] = grid_len(k); ops[n] = a[:Lk[n]]
    sig = dict(zip(names, enc.encode([ops[n] for n in names])))
    return nk, Lk, ops, sig

def dist_matrix(F, tasksL_idx, sigs, T=None):
    """F (n, nU, dim); tasksL_idx (T,) index of the prefix length of each task in the nU axis; sigs (T, dim).
    -> (n, T) float64 Euclidean distance (after linear map T, if given) between F[:, tasksL_idx[t]] and sigs[t]."""
    n = F.shape[0]; out = np.zeros((n, len(sigs)))
    for li in np.unique(tasksL_idx):
        cols = np.where(tasksL_idx == li)[0]
        S_ = torch.from_numpy(sigs[cols].astype(np.float64))
        if T is not None: S_ = S_ @ torch.from_numpy(T)
        for s in range(0, n, 2000):
            A = torch.from_numpy(F[s:s + 2000, li].astype(np.float64))
            if T is not None: A = A @ torch.from_numpy(T)
            out[s:s + 2000][:, cols] = torch.cdist(A, S_, compute_mode="donot_use_mm_for_euclid_dist").numpy()
    return out

# ---------------------------------------------------------------- measures
def snap(x): x = np.asarray(x, dtype=np.float64).copy(); x[x < SNAP] = 0.0; return x
def route(dist, r):
    inside = dist <= r
    return np.where(inside.any(1), inside.argmax(1), dist.argmin(1)), inside.any(1)
def measures(Du, ou, Dp, op, r):
    """Du: unedited distances (n_u, K), ou their owners; Dp paraphrased (n_p, K), op owners; one radius r for all."""
    d = snap(Du); Kc = d.shape[1]
    older = np.arange(Kc)[None, :] < ou[:, None]
    blocked = ((d <= r) & older).any(1)
    has = np.bincount(ou, minlength=Kc) > 0
    coll = np.bincount(ou, weights=blocked.astype(float), minlength=Kc) > 0
    ru, _ = route(d, r)
    p = snap(Dp); rp, cert = route(p, r); own = rp == op
    return {"C": float(coll[has].sum() / has.sum()), "U": float((ru == ou).mean()), "P": float((cert & own).mean()),
            "M": float((cert & ~own).mean()), "R": float(own.mean()),
            "Z": float((p[np.arange(len(op)), op] <= r).mean()), "tasks_with_unedited": int(has.sum()),
            "unedited_inputs": int(len(ou)), "paraphrased_inputs": int(len(op))}

def conformal_radius(scores, alpha):
    m = len(scores); idx = math.ceil(round((1 - alpha) * (m + 1), 9)); s = np.sort(scores)
    if idx > m: return float(s[-1]), True
    return float(s[idx - 1]), False

def inv_sqrt(M):
    w, v = np.linalg.eigh(M); return (v * w ** -0.5) @ v.T

def match(xs, curve, Cref):
    C = np.array([m["C"] for m in curve]); ok = np.where(C <= Cref + 1e-12)[0]
    if len(ok) == 0: return {"x": float(xs[0]), **{k: curve[0][k] for k in "PMRUZ"}, "flag": "C above C_ref at the smallest radius"}
    i = int(ok.max())
    if i == len(C) - 1: return {"x": float(xs[i]), **{k: curve[i][k] for k in "PMRUZ"}, "flag": "C never exceeds C_ref (end of curve)"}
    t = (Cref - C[i]) / (C[i + 1] - C[i])
    return {"x": float(xs[i] + t * (xs[i + 1] - xs[i])), "t": float(t), "grid_index": i,
            **{k: float(curve[i][k] + t * (curve[i + 1][k] - curve[i][k])) for k in "PMRUZ"}, "flag": None}

# ---------------------------------------------------------------- steps 1-5 for one encoder
class Run: pass

def run_encoder(key):
    enc = Enc(key); R = Run(); R.enc = enc
    log(key, "encoder loaded", enc.name, "dim", enc.dim)
    nk, Lk, ops, sig = prepare_openings(enc); R.nk, R.Lk, R.ops, R.sig = nk, Lk, ops, sig
    nks = np.array([nk[n] for n in names]); Ls = np.array([Lk[n] for n in names])
    log(key, "n_k min/median/max", int(nks.min()), float(np.median(nks)), int(nks.max()), "| L_k distinct", sorted(set(Ls.tolist())),
        "| n_k<8:", int((nks < 8).sum()), "| n_k>=254:", int((nks >= 254).sum()))
    out = {"encoder": enc.name, "pooling": enc.pool,
           "opening_tokens": {"min": int(nks.min()), "median": float(np.median(nks)), "max_capped_254": int(nks.max()), "below_8": int((nks < 8).sum()),
                              "at_cap_254": int((nks >= 254).sum()), "L_k_counts": {str(g): int((Ls == g).sum()) for g in sorted(set(Ls.tolist()))}}}

    # ---- calibration: ruler displacements (half A), radius scores (half B)
    def feat_at(name, defn, x, L): return enc.ids(inputs_text(defn, x))[:L]
    seqs, meta = [], []
    for n in half_A:
        t = by_name[n]
        for x in t["inputs"][:3]:
            seqs += [feat_at(n, t["definition"], x, Lk[n]), feat_at(n, para["tasks"][n]["de"], x, Lk[n])]; meta.append(n)
    FA = enc.encode(seqs).astype(np.float64)
    disp = FA[1::2] - FA[0::2]
    Sig_p = disp.T @ disp / len(disp)
    Sigma = 0.9 * Sig_p + 0.1 * (np.trace(Sig_p) / enc.dim) * np.eye(enc.dim)
    Rul = inv_sqrt(Sigma); R.Rul = Rul
    out["ruler"] = {"n_displacements": int(len(disp)), "half_A_tasks": len(half_A), "trace_Sigma_p": float(np.trace(Sig_p)),
                    "rank_Sigma_p": int((np.linalg.eigvalsh(Sig_p) > 1e-12 * np.linalg.eigvalsh(Sig_p).max()).sum()),
                    "ruler_eigs_min_max": [float(np.linalg.eigvalsh(Rul).min()), float(np.linalg.eigvalsh(Rul).max())]}
    seqs, bn = [], []
    for n in half_B:
        t = by_name[n]
        if not t["inputs"]: continue
        x = t["inputs"][0]; seqs.append(feat_at(n, para["tasks"][n]["de"], x, Lk[n])); bn.append(n)
    FB = enc.encode(seqs).astype(np.float64)
    SB = np.stack([sig[n] for n in bn]).astype(np.float64)
    metrics = {"plain": np.eye(enc.dim), "ruler": Rul}
    sigB = np.stack([sig[n] for n in half_B]).astype(np.float64)
    out["halves"] = {"A": len(half_A), "B": len(half_B), "B_with_inputs_m": len(bn)}
    out["radius"] = {}; out["ratio"] = {}; R.radius = {}
    for mname, T in metrics.items():
        scores = snap(np.linalg.norm((FB - SB) @ T, axis=1))
        R.radius[mname] = {}; out["radius"][mname] = {}
        for a in ALPHAS:
            r, over = conformal_radius(scores, a); R.radius[mname][a] = r
            out["radius"][mname][str(a)] = {"r": r, "index_exceeds_m_used_largest": over}
        Z_ = sigB @ T; dd = np.linalg.norm(Z_[:, None] - Z_[None], axis=2); np.fill_diagonal(dd, np.inf)
        med = float(np.median(dd.min(1)))
        out["ratio"][mname] = {"median_nearest_other_halfB_signature": med, "ratio_to_r_0.1": med / R.radius[mname][0.1],
                               "scores_min_median_max": [float(scores.min()), float(np.median(scores)), float(scores.max())]}
    R.primary = max(("plain", "ruler"), key=lambda m: out["ratio"][m]["ratio_to_r_0.1"])
    out["primary_variant"] = R.primary
    log(key, "radii", {m: {a: round(v, 4) for a, v in R.radius[m].items()} for m in R.radius}, "ratios",
        {m: round(out["ratio"][m]["ratio_to_r_0.1"], 3) for m in metrics}, "primary", R.primary)

    # ---- evaluation items (arrival order)
    items = []   # (col, kind 0 unedited / 1 de / 2 fr, text)
    for n in ev_names:
        t = by_name[n]
        for x in t["inputs"][:6]: items.append((ev_col[n], 0, inputs_text(t["definition"], x)))
        for k, p in enumerate(PIV):
            for x in t["inputs"][:3]: items.append((ev_col[n], 1 + k, inputs_text(para["tasks"][n][p], x)))
    R.items = items
    owner = np.array([i[0] for i in items]); kind = np.array([i[1] for i in items])
    ev_L = np.array([Lk[n] for n in ev_names]); U = np.unique(ev_L); where = np.searchsorted(U, ev_L)
    ev_sig = np.stack([sig[n] for n in ev_names])
    cache_f = f"{OUT}/F_{key}.npy"
    if os.path.exists(cache_f): F = np.load(cache_f)
    else:
        ids = [enc.ids(i[2]) for i in items]
        F = np.zeros((len(items), len(U), enc.dim), dtype=np.float32)
        for s in range(0, len(items), 1000):
            chunk = ids[s:s + 1000]
            flat = [x[:int(L)] for x in chunk for L in U]
            F[s:s + len(chunk)] = enc.encode(flat).reshape(len(chunk), len(U), enc.dim)
            log(key, f"encoded items {min(s + 1000, len(items))}/{len(items)}  cache {len(enc.cache)}")
        np.save(cache_f, F)
    R.F, R.U, R.where, R.owner, R.kind, R.ev_sig, R.ev_L = F, U, where, owner, kind, ev_sig, ev_L
    log(key, "items", len(items), "unedited", int((kind == 0).sum()), "de", int((kind == 1).sum()), "fr", int((kind == 2).sum()), "| distinct L in eval", U.tolist())
    R.DM = {m: dist_matrix(F, where, ev_sig, T if m == "ruler" else None) for m, T in metrics.items()}
    # sanity: unedited inputs at own signature
    chk = {}
    for m in metrics:
        own_u = R.DM[m][kind == 0][np.arange((kind == 0).sum()), owner[kind == 0]]
        chk[m] = {"n_unedited": int(len(own_u)), "not_below_1e-4": int((own_u >= SNAP).sum()), "max": float(own_u.max()), "exactly_0": int((own_u == 0).sum())}
    out["unedited_own_distance_check"] = chk; log(key, "unedited own-signature distance check", chk)

    # ---- measures
    sets = {"French": kind == 2, "German": kind == 1, "both": kind > 0}
    mu_k = kind == 0
    out["table"] = {}
    for m in metrics:
        out["table"][m] = {}
        for a in ALPHAS:
            out["table"][m][str(a)] = {s: measures(R.DM[m][mu_k], owner[mu_k], R.DM[m][sel], owner[sel], R.radius[m][a]) for s, sel in sets.items()}
    # verdict
    pm = R.primary; v = out["table"][pm]["0.1"]["French"]
    out["verdict"] = {"variant": pm, "alpha": 0.1, "pivot": "French", "P": v["P"], "M": v["M"], "C": v["C"],
                      "outcome": "success" if (v["P"] >= 0.75 and v["M"] <= 0.05 and v["C"] <= 0.19)
                      else ("against" if (v["P"] < 0.55 or v["M"] > 0.09 or v["C"] > 0.25) else "partial")}
    log(key, "verdict", out["verdict"])
    R.out = out
    return R

# ---------------------------------------------------------------- second views
def view_curve(R):
    kind, owner = R.kind, R.owner; res = {}
    sets = {"French": kind == 2}
    for m in ("plain", "ruler"):
        own_d = snap(R.DM[m][kind > 0][np.arange((kind > 0).sum()), owner[kind > 0]])
        radii = [float(np.quantile(own_d, q)) for q in QGRID]
        mu_k = kind == 0
        curve = [measures(R.DM[m][mu_k], owner[mu_k], R.DM[m][kind == 2], owner[kind == 2], r) for r in radii]
        res[m] = {"quantiles": QGRID, "radii": radii, "curve_French": curve, "at_C_0.189": match(radii, curve, C_REF),
                  "C_monotone": bool(np.all(np.diff([c["C"] for c in curve]) >= -1e-12))}
    return res

def view_collisions(R):
    m = R.primary; r = R.radius[m][0.1]; kind, owner = R.kind, R.owner
    mu_k = np.where(kind == 0)[0]; d = snap(R.DM[m][mu_k]); ou = owner[mu_k]
    older = np.arange(K)[None, :] < ou[:, None]; inside = (d <= r) & older
    blocked = inside.any(1)
    tasks_coll = sorted(set(ou[blocked].tolist()))
    has = len(set(ou.tolist()))
    first = []
    for c in tasks_coll[:15]:
        rows = np.where((ou == c) & blocked)[0]
        claimers = np.where(inside[rows[0]])[0]
        first.append({"task": ev_names[c], "arrival_position": int(c), "blocked_inputs": int(len(rows)), "claimed_by_oldest": ev_names[int(claimers[0])],
                      "older_zones_containing_first_blocked_input": int(len(claimers))})
    return {"variant": m, "alpha": 0.1, "tasks_with_unedited": has, "tasks_colliding": len(tasks_coll), "share": len(tasks_coll) / has, "first_15": first}

def view_benchmark(R):
    enc = R.enc; m = R.primary; T = R.Rul if m == "ruler" else None; r = R.radius[m][0.1]
    names15 = sorted(set(FIVE) | set(ORDER1)); data = {}
    for n in names15:
        dd = {}
        for split in ("train", "test"):
            dj = json.load(open(os.path.join("CL_Benchmark", "SuperNI", n, split + ".json")))
            if split == "train": dd["definition"] = (dj["Definition"][0] if isinstance(dj["Definition"], list) else dj["Definition"]).strip()
            else: dd["test"] = [i["input"] for i in dj["Instances"]]
        data[n] = dd
    defs = {n: [data[n]["definition"]] + sr.REWORDED.get(n.split("_")[0], []) for n in names15}
    tk = {};
    for n in names15:
        a = enc.ids(inputs_text(defs[n][0], "A B")); b = enc.ids(inputs_text(defs[n][0], "zzq xkj")); k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
        tk[n] = (grid_len(k), a[:grid_len(k)], k)
    sg = {n: enc.encode([tk[n][1]])[0] for n in names15}
    Ls = np.array([tk[n][0] for n in names15]); U = np.unique(Ls); wh = np.searchsorted(U, Ls)
    texts, meta = [], []
    for n in names15:
        for li, d in enumerate(defs[n]):
            for x in data[n]["test"]: texts.append(enc.ids(inputs_text(d, x))); meta.append((n, li))
    flat = [x[:int(L)] for x in texts for L in U]
    F = enc.encode(flat).reshape(len(texts), len(U), enc.dim)
    D = dist_matrix(F, wh, np.stack([sg[n] for n in names15]), T)
    tn = {n: i for i, n in enumerate(names15)}
    res = {"variant": m, "alpha": 0.1, "radius": r, "reworded_tasks": sorted(sr.REWORDED), "L_k": {n: tk[n][0] for n in names15}, "n_k": {n: tk[n][2] for n in names15}}
    for label, order in (("five", FIVE), ("fifteen", ORDER1)):
        cols = [tn[n] for n in order]; res[label] = {}
        for li, lev in enumerate(sr.LEVELS):
            rows = [i for i, (n, l) in enumerate(meta) if l == li and n in order]
            own = np.array([order.index(meta[i][0]) for i in rows]); Dm = snap(D[rows][:, cols])
            rt, cert = route(Dm, r)
            res[label][lev] = {"inputs": len(rows), "tasks": len(set(own.tolist())),
                               "in_own_zone": float((Dm[np.arange(len(rows)), own] <= r).mean()),
                               "certified_own": float((cert & (rt == own)).mean()), "routed_own": float((rt == own).mean()),
                               "certified_wrong": int((cert & (rt != own)).sum())}
    res["note"] = "light/medium/heavy exist only for the five tasks with rewordings (sr.REWORDED); in 'fifteen' they cover those five only"
    return res

def view_cost(R):
    enc = R.enc; m = R.primary; T = R.Rul if m == "ruler" else None; r = R.radius[m][0.1]
    sel = list(range(0, len(R.items), 42))[:200]
    sg = R.ev_sig.astype(np.float64); U, wh = R.U, R.where
    if T is not None: sgT = sg @ T
    for i in sel[:3]: enc.encode([enc.ids(R.items[i][2])[:int(L)] for L in U], use_cache=False)   # warm-up
    times = []
    for i in sel:
        t0 = time.perf_counter()
        ids = enc.ids(R.items[i][2])
        f = enc.encode([ids[:int(L)] for L in U], use_cache=False).astype(np.float64)   # (nU, dim)
        fz = f @ T if T is not None else f
        ref = sgT if T is not None else sg
        d = np.linalg.norm(fz[wh] - ref, axis=1)
        inside = d <= r; _ = inside.argmax() if inside.any() else d.argmin()
        times.append((time.perf_counter() - t0) * 1000)
    return {"variant": m, "inputs": len(sel), "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)), "p90_ms": float(np.quantile(times, 0.9)),
            "distinct_prefix_lengths": int(len(U)), "threads": 2, "note": "sequential, one input at a time, no cache; includes tokenisation, all prefix encodings, distances to 700 signatures, route"}

def view_instruction_field(key, R=None):
    enc = R.enc if R is not None else Enc(key)
    def e(texts): return enc.encode([enc.ids(t, MAXTOK) for t in texts])
    orig = e([by_name[n]["definition"] for n in names]); idx = {n: i for i, n in enumerate(names)}
    pv = {p: e([para["tasks"][n][p] for n in names]) for p in PIV}
    def dist(A, B): return torch.cdist(torch.from_numpy(A.astype(np.float64)), torch.from_numpy(B.astype(np.float64)), compute_mode="donot_use_mm_for_euclid_dist").numpy()
    scores = snap(np.array([np.linalg.norm(pv["de"][idx[n]] - orig[idx[n]]) for n in half_B]))
    ev_i = [idx[n] for n in ev_names]
    Du = dist(orig[ev_i], orig[ev_i]); Du[np.arange(K), np.arange(K)] = 0.0
    out = {"encoder": enc.name, "m": len(scores), "radius": {}, "table": {}}
    Dp = {p: dist(pv[p][ev_i], orig[ev_i]) for p in PIV}
    ou = np.arange(K)
    for a in ALPHAS:
        r, over = conformal_radius(scores, a); out["radius"][str(a)] = {"r": r, "index_exceeds_m_used_largest": over}
        out["table"][str(a)] = {}
        for sname, ps in (("French", ["fr"]), ("German", ["de"]), ("both", ["de", "fr"])):
            D_ = np.concatenate([Dp[p] for p in ps]); o_ = np.concatenate([ou] * len(ps))
            out["table"][str(a)][sname] = measures(Du, ou, D_, o_, r)
    return out

# ================================================================ main
if __name__ == "__main__":
    RESULT["notes"] = {
        "split": {"calibration": len(cal_names), "evaluation": len(ev_names), "half_A": len(half_A), "half_B": len(half_B)},
        "snap": SNAP, "grid": GRID, "token_cap": MAXTOK,
        "dedupe": "identical token prefixes are encoded once and share the vector, so an unedited input starting with the opening is at distance exactly 0 from its signature",
        "opening": "common token prefix (no special tokens, this tokenizer, capped at 254 tokens) of the template with inputs 'A B' and 'zzq xkj'",
        "Z": "share of paraphrased inputs whose distance to their own signature is <= radius",
        "benchmark_zone_certified_routed": "in_own_zone = dist to own signature <= r; certified_own = some zone contains the input and the route is the own task; routed_own = route is the own task (certified or nearest)",
        "paraphrase_stats": para.get("stats")}
    P = run_encoder("minilm"); RESULT["minilm"] = P.out; dump(); log("stage 1-5 minilm written")
    RESULT["view_a_curve"] = view_curve(P); dump(); log("curve", {m: RESULT["view_a_curve"][m]["at_C_0.189"] for m in RESULT["view_a_curve"]})
    RESULT["view_e_collisions"] = view_collisions(P); dump(); log("collisions", RESULT["view_e_collisions"]["tasks_colliding"])
    RESULT["view_d_benchmark"] = view_benchmark(P); dump(); log("benchmark done")
    RESULT["view_f_cost"] = view_cost(P); dump(); log("cost", RESULT["view_f_cost"]["mean_ms"])
    RESULT["view_b_instruction_field"] = {"minilm": view_instruction_field("minilm", P)}; dump()
    B = run_encoder("bge"); RESULT["bge"] = B.out; dump(); log("stage c bge written")
    RESULT["view_b_instruction_field"]["bge"] = view_instruction_field("bge", B); dump()
    RESULT["view_b_instruction_field"]["mpnet"] = view_instruction_field("mpnet"); dump()
    log("done")
