"""Fourth pre-registered experiment (last section of silent_adapters/PLAN.md): a zone decided by a frozen judge of
"same instruction". Everything works on instruction texts (one text per task and wording). Zone of task j = all texts u with
(1) MiniLM distance d(u, o_j) <= r_w and (2) judge(u, o_j) >= tau. Judge A = cross-encoder/nli-deberta-v3-xsmall (min of the two
entailment probabilities, each text cut at 128 judge tokens, 1 for identical strings); judge B = 1 - max T5-token weight of the
tokens that are in exactly one of the two texts (paper's weights a/(a+p), a=0.001, p from the 4000 unrelated inputs).
r_w and tau from half B of the 175 calibration tasks (German pivot); judge chosen by AUC on half B; 700 evaluation tasks in
arrival order, French pivot primary; at most 20 nearest tasks inside r_w are judged per text. CPU only, 2 threads.

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2 \
        $S/mm_env/bin/python judge_study.py
Caches (embeddings, every judge-A pair score) in $S/judge_study/ (a rerun resumes); writes routing/judge_study.json next to this
script after every stage (originals + French first, then German, then the extra views).
Reuses encoder_study.py (data, split, MiniLM encoder, benchmark rewordings) and ruler_study.py (word weights).
"""
import json, os, sys, time, math, hashlib, pickle, random
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_study as es            # data, split, Enc (MiniLM), FIVE, ORDER1, sr (stubs + silent_run)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(2)

S = es.S
OUT = f"{S}/judge_study"; os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(HERE, "routing", "judge_study.json")
JUDGE_NAME = "cross-encoder/nli-deberta-v3-xsmall"
JUDGE_CUT = 128
LIMIT = 20
by_name, para, sr = es.by_name, es.para, es.sr
cal_names, ev_names, half_B = es.cal_names, es.ev_names, es.half_B
K = len(ev_names)
FIVE, ORDER1 = es.FIVE, es.ORDER1
t0_all = time.time()
def log(*a): print(f"[{time.time() - t0_all:6.0f}s]", *a, flush=True)
RESULT = {"notes": {}}
def dump(): json.dump(RESULT, open(JSON_PATH, "w"), indent=1)
def md5(t): return hashlib.md5(t.encode()).hexdigest()[:16]

# ---------------------------------------------------------------- MiniLM distances (cached)
enc = es.Enc("minilm")
def embed(texts, tag):
    f = f"{OUT}/emb_{tag}.npy"
    if os.path.exists(f):
        a = np.load(f)
        if len(a) == len(texts): return a
    a = enc.encode([enc.ids(t, es.MAXTOK) for t in texts]).astype(np.float64); np.save(f, a); log("embedded", tag, len(texts)); return a
def cdist(A, B): return torch.cdist(torch.from_numpy(np.asarray(A, np.float64)), torch.from_numpy(np.asarray(B, np.float64)), compute_mode="donot_use_mm_for_euclid_dist").numpy()

# ---------------------------------------------------------------- judge A
jtok = AutoTokenizer.from_pretrained(JUDGE_NAME)
jmodel = AutoModelForSequenceClassification.from_pretrained(JUDGE_NAME).eval()
id2label = {int(k): v for k, v in jmodel.config.id2label.items()}
ENT = [k for k, v in id2label.items() if v.lower() == "entailment"][0]
log("judge", JUDGE_NAME, "fast tokenizer", jtok.is_fast, "id2label", id2label, "entailment index", ENT)
_jids = {}
def jids(t):
    if t not in _jids: _jids[t] = jtok(t, add_special_tokens=False, max_length=JUDGE_CUT, truncation=True)["input_ids"]
    return _jids[t]
PAIR_F = f"{OUT}/pair_scores.pkl"
PAIRS = pickle.load(open(PAIR_F, "rb")) if os.path.exists(PAIR_F) else {}
log("pair cache", len(PAIRS))
def dkey(p, h): return (md5(p), md5(h))
def forward(dir_pairs, tokens_per_batch=7000):
    """dir_pairs: list of (premise, hypothesis) -> list of P(entailment), batched by length, no cache."""
    seqs = [[jtok.cls_token_id] + jids(p) + [jtok.sep_token_id] + jids(h) + [jtok.sep_token_id] for p, h in dir_pairs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); res = [0.0] * len(seqs); s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and (e - s + 1) * len(seqs[order[e]]) <= tokens_per_batch and e - s < 128: e += 1
        b = order[s:e]; s = e
        L = max(len(seqs[i]) for i in b)
        ids = torch.full((len(b), L), jtok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b): ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); att[r, :len(seqs[i])] = 1
        with torch.inference_mode():
            pr = torch.softmax(jmodel(input_ids=ids, attention_mask=att).logits.float(), -1)[:, ENT].numpy()
        for r, i in enumerate(b): res[i] = float(pr[r])
    return res
_last_save = [time.time()]
def ensure(pairs):
    """pairs: list of (u, o) strings; computes and caches both directions of every non-identical pair."""
    need = {}
    for u, o in pairs:
        if u == o: continue
        for p, h in ((o, u), (u, o)):
            k = dkey(p, h)
            if k not in PAIRS: need[k] = (p, h)
    if not need: return
    keys = list(need); vals = [need[k] for k in keys]; CH = 4000
    for c in range(0, len(keys), CH):
        out = forward(vals[c:c + CH])
        for k, v in zip(keys[c:c + CH], out): PAIRS[k] = v
        pickle.dump(PAIRS, open(PAIR_F, "wb")); log("judge pairs scored", min(c + CH, len(keys)), "/", len(keys), "(directional, new)")
def EA(u, o):
    if u == o: return 1.0
    return min(PAIRS[dkey(o, u)], PAIRS[dkey(u, o)])

# ---------------------------------------------------------------- judge B (T5 word weights, as ruler_study.py)
from transformers import AutoTokenizer as _AT
ttok = _AT.from_pretrained("t5-base")
def t5_weights():
    f = f"{OUT}/t5_weights.npy"
    if os.path.exists(f): return np.load(f)
    rng = random.Random(0); defs, texts = [], []
    for name in sr.POOL_TASKS:
        data = json.load(open(os.path.join("CL_Benchmark", "Long_Sequence", name, "train.json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
        defs.append(d); texts.append([inst["input"] for inst in data["Instances"]])
    picks = [rng.randrange(len(defs)) for _ in range(sr.BG_N)]
    sources = [sr._source(defs[k], rng.choice(texts[k])) for k in picks]
    rows = ttok(sources, max_length=512, truncation=True)["input_ids"]
    V = 40000; counts = np.zeros(V)
    for r in rows:
        r = np.array(r); r = r[r != ttok.pad_token_id]; counts += np.bincount(r, minlength=V)
    W = sr.SIF_A / (sr.SIF_A + counts / counts.sum()); W[ttok.pad_token_id] = 0.0
    np.save(f, W); return W
WT = t5_weights()
log("T5 weights ready; min/max weight", float(WT.min()), float(WT.max()))
_tids = {}
def tset(t):
    if t not in _tids: _tids[t] = frozenset(ttok(t, add_special_tokens=False)["input_ids"])
    return _tids[t]
def EB(u, o):
    d = tset(u) ^ tset(o)
    return 1.0 - float(max(WT[v] for v in d)) if d else 1.0

# ---------------------------------------------------------------- data: texts
orig = {n: by_name[n]["definition"] for n in by_name}
cal_sorted = es.cal_sorted
mB = len(half_B)
dm = lambda alpha_idx: None
def conf_idx(alpha, m):
    return math.ceil(round((1 - alpha) * (m + 1), 9))
log("half_B m =", mB, "half_A", len(es.half_A), "cal", len(cal_names), "eval", K)

# ---------------------------------------------------------------- step 1 + 2: calibration and choice of judge
Ecal_o = embed([orig[n] for n in cal_sorted], "cal_orig"); cidx = {n: i for i, n in enumerate(cal_sorted)}
Ecal_de = embed([para["tasks"][n]["de"] for n in half_B], "halfB_de")
d_pos = np.linalg.norm(Ecal_de - Ecal_o[[cidx[n] for n in half_B]], axis=1)
idx = conf_idx(0.02, mB); srt = np.sort(d_pos)
r_over = idx > mB; r_w = float(srt[-1] if r_over else srt[idx - 1])
k_tau = math.floor(round(0.08 * (mB + 1), 9))
pos_pairs = [(para["tasks"][n]["de"], orig[n]) for n in half_B]
Dcc = cdist(Ecal_o, Ecal_o); np.fill_diagonal(Dcc, np.inf)
neg_other = [cal_sorted[int(Dcc[cidx[n]].argmin())] for n in half_B]
neg_pairs = [(orig[n], orig[o]) for n, o in zip(half_B, neg_other)]
ensure(pos_pairs + neg_pairs)
sc = {"A": (np.array([EA(*p) for p in pos_pairs]), np.array([EA(*p) for p in neg_pairs])),
      "B": (np.array([EB(*p) for p in pos_pairs]), np.array([EB(*p) for p in neg_pairs]))}
def auc(pos, neg): return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
TAU, tau_info = {}, {}
for j in "AB":
    s = np.sort(sc[j][0])
    if k_tau == 0: TAU[j] = -np.inf
    else: TAU[j] = float(s[k_tau - 1])
AUC = {j: auc(*sc[j]) for j in "AB"}
CHOSEN = "B" if AUC["B"] > AUC["A"] + 0.02 else "A"
RESULT["calibration"] = {
    "m": mB, "r_w": r_w, "r_w_index": idx, "r_w_index_exceeds_m_used_largest": bool(r_over), "k": k_tau, "tau_minus_infinity": k_tau == 0,
    "tau": {j: (None if TAU[j] == -np.inf else TAU[j]) for j in "AB"}, "auc": AUC, "chosen_judge": CHOSEN,
    "judge_A_share_negatives_ge_tau": float((sc["A"][1] >= TAU["A"]).mean()), "judge_A_share_positives_ge_tau": float((sc["A"][0] >= TAU["A"]).mean()),
    "judge_B_share_negatives_ge_tau": float((sc["B"][1] >= TAU["B"]).mean()), "judge_B_share_positives_ge_tau": float((sc["B"][0] >= TAU["B"]).mean()),
    "score_ranges_A": {"pos_min_med_max": [float(sc["A"][0].min()), float(np.median(sc["A"][0])), float(sc["A"][0].max())], "neg_min_med_max": [float(sc["A"][1].min()), float(np.median(sc["A"][1])), float(sc["A"][1].max())]},
    "score_ranges_B": {"pos_min_med_max": [float(sc["B"][0].min()), float(np.median(sc["B"][0])), float(sc["B"][0].max())], "neg_min_med_max": [float(sc["B"][1].min()), float(np.median(sc["B"][1])), float(sc["B"][1].max())]},
    "negatives_with_identical_text": int(sum(a == b for a, b in neg_pairs)),
    "d_pos_min_med_max": [float(d_pos.min()), float(np.median(d_pos)), float(d_pos.max())],
    "judge": JUDGE_NAME, "judge_id2label": {str(k): v for k, v in id2label.items()}, "entailment_index": ENT, "judge_tokenizer_fast": bool(jtok.is_fast)}
RESULT["notes"] = {
    "split": {"calibration": len(cal_names), "evaluation": K, "half_A": len(es.half_A), "half_B": mB},
    "texts": "one instruction text per task and wording; MiniLM distance on whole texts cut at 254 tokens; judge A cut at 128 judge tokens per text",
    "candidates": f"for each text the {LIMIT} nearest evaluation tasks inside r_w are judged, plus its own task if inside r_w and not among them (so up to {LIMIT + 1}); the own pair is always scored (also outside r_w) only for reporting",
    "identical": "E = 1 (judge A) / S empty = 1 (judge B) for identical strings; two tasks with the same definition therefore always claim each other",
    "tau_C": "C, U use the original definitions of the 700 evaluation tasks; own task counted as inside its own zone only for Z/P/M/R, C only looks at OLDER zones",
    "snap": "none (MiniLM d computed in float64; identical texts give d = 0 exactly)"}
dump()
log("r_w", r_w, "tau", TAU, "AUC", AUC, "chosen", CHOSEN, "A neg>=tau", RESULT["calibration"]["judge_A_share_negatives_ge_tau"], "A pos>=tau", RESULT["calibration"]["judge_A_share_positives_ge_tau"])

# ---------------------------------------------------------------- step 3: evaluation set-up
Eo = embed([orig[n] for n in ev_names], "ev_orig")
Ep = {p: embed([para["tasks"][n][p] for n in ev_names], f"ev_{p}") for p in ("fr", "de")}
KIND = {"orig": 0, "de": 1, "fr": 2}
TXT = {"orig": [orig[n] for n in ev_names], "de": [para["tasks"][n]["de"] for n in ev_names], "fr": [para["tasks"][n]["fr"] for n in ev_names]}
EMB = {"orig": Eo, "de": Ep["de"], "fr": Ep["fr"]}
Dm = {k: cdist(EMB[k], Eo) for k in EMB}                     # (K texts, K tasks)
ar = np.arange(K)
class View: pass
V = {}
def prepare(kind):
    v = View(); D = Dm[kind]; inside = D <= r_w
    cand = np.zeros((K, K), bool); over = 0
    for i in range(K):
        ins = np.where(inside[i])[0]
        if len(ins) > LIMIT: over += 1
        top = ins[np.argsort(D[i, ins], kind="stable")[:LIMIT]]
        cand[i, top] = True
        if inside[i, i]: cand[i, i] = True
    v.cand, v.D, v.over, v.inside = cand, D, over, inside
    pairs = [(TXT[kind][i], orig[ev_names[j]]) for i, j in zip(*np.where(cand))]
    pairs += [(TXT[kind][i], orig[ev_names[i]]) for i in range(K)]          # own pair, for reporting
    ensure(pairs)
    SAm = np.full((K, K), np.nan); SBm = np.full((K, K), np.nan)
    ii, jj = np.where(cand | np.eye(K, dtype=bool))
    for i, j in zip(ii, jj):
        u, o = TXT[kind][i], orig[ev_names[j]]; SAm[i, j] = EA(u, o); SBm[i, j] = EB(u, o)
    v.SA, v.SB = SAm, SBm
    v.n_cand = cand.sum(1); v.n_judged = np.array([sum(1 for j in np.where(cand[i])[0] if TXT[kind][i] != orig[ev_names[j]]) for i in range(K)])
    V[kind] = v
    log("prepared", kind, "texts with more than", LIMIT, "inside r_w:", over, "| mean candidates", float(v.n_cand.mean()))
    return v
def zone(kind, judge, tau):
    v = V[kind]
    if judge is None: return v.inside
    s = v.SA if judge == "A" else v.SB
    return v.cand & (np.nan_to_num(s, nan=-1.0) >= tau)
def route_of(Z, D):
    return np.where(Z.any(1), Z.argmax(1), D.argmin(1)), Z.any(1)
def c_u(judge, tau):
    Z = zone("orig", judge, tau); older = ar[None, :] < ar[:, None]
    C = float((Z & older).any(1).mean()); rt, _ = route_of(Z, V["orig"].D)
    return {"C": C, "U": float((rt == ar).mean())}
def para_measures(kinds, judge, tau):
    Z = np.concatenate([zone(k, judge, tau) for k in kinds]); D = np.concatenate([V[k].D for k in kinds]); own = np.concatenate([ar] * len(kinds))
    rt, cert = route_of(Z, D); n = len(own)
    return {"Z": float(Z[np.arange(n), own].mean()), "P": float((cert & (rt == own)).mean()), "M": float((cert & (rt != own)).mean()), "R": float((rt == own).mean()), "n_texts": int(n)}
def table(kset, judge, tau):
    t = {"C_U": c_u(judge, tau)}
    for name, ks in (("French", ["fr"]), ("German", ["de"]), ("both", ["de", "fr"])):
        if all(k in V for k in ks): t[name] = para_measures(ks, judge, tau)
    return t
def verdict(t):
    f = t["French"]; C = t["C_U"]["C"]
    out = "success" if (f["P"] >= 0.75 and f["M"] <= 0.05 and C <= 0.19) else ("against" if (f["P"] < 0.55 or f["M"] > 0.09 or C > 0.25) else "partial")
    return {"P": f["P"], "M": f["M"], "C": C, "outcome": out}
def all_tables():
    T = {"chosen": {"judge": CHOSEN, "tau": TAU[CHOSEN], "table": table(None, CHOSEN, TAU[CHOSEN])}}
    other = "B" if CHOSEN == "A" else "A"
    T["not_chosen_5a"] = {"judge": other, "tau": TAU[other], "table": table(None, other, TAU[other])}
    T["filter_alone_5b"] = {"r_w": r_w, "table": table(None, None, None)}
    return T
def table(kset, judge, tau, _kinds=None):  # noqa: F811  (kinds present so far)
    t = {"C_U": c_u(judge, tau)}
    for name, ks in (("French", ["fr"]), ("German", ["de"]), ("both", ["de", "fr"])):
        if all(k in V for k in ks): t[name] = para_measures(ks, judge, tau)
    return t

prepare("orig"); prepare("fr")
RESULT["stage1_originals_french"] = all_tables()
RESULT["stage1_originals_french"]["verdict_chosen_French"] = verdict(RESULT["stage1_originals_french"]["chosen"]["table"])
RESULT["limit_reached"] = {k: {"texts_with_more_than_20_inside_r_w": int(V[k].over), "texts_with_none_inside_r_w": int((~V[k].inside.any(1)).sum())} for k in V}
dump(); log("stage 1 written", RESULT["stage1_originals_french"]["chosen"]["table"], RESULT["stage1_originals_french"]["verdict_chosen_French"])
prepare("de")
T = all_tables(); T["verdict_chosen_French"] = verdict(T["chosen"]["table"])
RESULT["step3_final"] = T
RESULT["limit_reached"] = {k: {"texts_with_more_than_20_inside_r_w": int(V[k].over), "texts_with_none_inside_r_w": int((~V[k].inside.any(1)).sum())} for k in V}
dump(); log("stage 2 (German) written", T["chosen"]["table"], T["verdict_chosen_French"])

# ---------------------------------------------------------------- 5c: tau sweep (chosen judge, French pivot)
lo, hi = float(sc[CHOSEN][0].min()), float(sc[CHOSEN][0].max())
sweep = []
for tv in np.linspace(lo, hi, 12):
    t = table(None, CHOSEN, float(tv)); f = t["French"]
    sweep.append({"tau": float(tv), "C": t["C_U"]["C"], "Z": f["Z"], "P": f["P"], "M": f["M"], "R": f["R"]})
RESULT["sweep_5c"] = {"judge": CHOSEN, "range": [lo, hi], "curve_French": sweep, "note": "curve only; no value chosen"}
dump(); log("sweep done")

# ---------------------------------------------------------------- 5d: collisions under the paper's rule
mt = json.load(open(os.path.join(HERE, "routing", "many_tasks.json")))
coll = mt["random order, seed 0"]["875"].get("collisions", [])
pairs = [(c["task"], c["older"], c.get("same_definition")) for c in coll if c["task"] in by_name and c["older"] in by_name]
if not coll:
    RESULT["collisions_5d"] = {"note": "many_tasks.json has no collision list; skipped"}
else:
    pp = [(orig[a], orig[b]) for a, b, _ in pairs]; ensure(pp)
    res = {}
    for j in "AB":
        s = np.array([(EA if j == "A" else EB)(*p) for p in pp]); tau = TAU[j]
        diff = np.array([not sd for _, _, sd in pairs])
        res[j] = {"tau": tau, "n_pairs": len(pp), "n_ge_tau": int((s >= tau).sum()), "median_score": float(np.median(s)),
                  "n_pairs_not_same_definition": int(diff.sum()), "n_ge_tau_not_same_definition": int((s[diff] >= tau).sum()),
                  "median_score_not_same_definition": float(np.median(s[diff])) if diff.any() else None}
    RESULT["collisions_5d"] = {"source": "many_tasks.json, 'random order, seed 0', 875 tasks, key 'collisions' (task, older); pair = (original of task, original of older)",
                               "listed": len(coll), "both_in_data": len(pairs), "per_judge": res}
dump(); log("5d done", RESULT["collisions_5d"])

# ---------------------------------------------------------------- 5e: failures
J = CHOSEN; tauJ = TAU[J]
vF = V["fr"]; SJ = vF.SA if J == "A" else vF.SB; ZF = zone("fr", J, tauJ); rtF, certF = route_of(ZF, vF.D)
fails = [i for i in range(K) if not ZF[i, i]][:12]
RESULT["fail_5e"] = {"n_not_in_own_zone_French": int((~ZF[ar, ar]).sum()), "judge": J, "first_12": [
    {"pos": i, "task": ev_names[i], "d": float(vF.D[i, i]), "d_le_r_w": bool(vF.inside[i, i]), "judge_score": float(SJ[i, i]), "judge_A_score": float(vF.SA[i, i]),
     "original_200": orig[ev_names[i]][:200], "paraphrase_200": TXT["fr"][i][:200]} for i in fails]}
ms = [i for i in range(K) if certF[i] and rtF[i] != i][:12]
RESULT["m_cases_5e"] = {"n_M_French": int((certF & (rtF != ar)).sum()), "judge": J, "first_12": [
    {"pos": i, "task": ev_names[i], "routed_to": ev_names[int(rtF[i])], "routed_pos": int(rtF[i]), "d": float(vF.D[i, rtF[i]]), "judge_score": float(SJ[i, rtF[i]]),
     "own_zone_holds": bool(ZF[i, i]), "original_own_160": orig[ev_names[i]][:160], "original_routed_160": orig[ev_names[int(rtF[i])]][:160]} for i in ms]}
dump(); log("5e done")

# ---------------------------------------------------------------- 5f: benchmark rewordings
names15 = sorted(set(FIVE) | set(ORDER1)); data = {}
for n in names15:
    dj = json.load(open(os.path.join("CL_Benchmark", "SuperNI", n, "train.json")))
    data[n] = (dj["Definition"][0] if isinstance(dj["Definition"], list) else dj["Definition"]).strip()
bo = {n: data[n] for n in names15}
Eb = dict(zip(names15, embed([bo[n] for n in names15], "bench_orig")))
wd = {n: [bo[n]] + sr.REWORDED.get(n.split("_")[0], []) for n in names15}
def zone_ok(u, o, d): return d <= r_w and (EA(u, o) if J == "A" else EB(u, o)) >= tauJ
bench_pairs = [(wd[n][l], bo[m]) for n in FIVE for l in range(len(wd[n])) for m in ORDER1 + FIVE]
ensure(bench_pairs)
fe = {}
for n in FIVE:
    for l in range(len(wd[n])): fe[(n, l)] = embed([wd[n][l]], f"bench_{md5(wd[n][l])}")[0]
bres = {"judge": J, "r_w": r_w, "tau": tauJ, "rows": []}
for n in FIVE:
    for l, lev in enumerate(sr.LEVELS[:len(wd[n])]):
        u = wd[n][l]; e = fe[(n, l)]; row = {"task": n, "level": lev, "d_own": float(np.linalg.norm(e - Eb[n])), "judge_A_score": EA(u, bo[n]), "judge_B_score": EB(u, bo[n])}
        row["in_own_zone"] = bool(zone_ok(u, bo[n], row["d_own"]))
        for label, order in (("five", FIVE), ("fifteen", ORDER1)):
            ds = np.array([np.linalg.norm(e - Eb[m]) for m in order]); zs = [zone_ok(u, bo[m], ds[q]) for q, m in enumerate(order)]
            if any(zs): q = zs.index(True); cert = True
            else: q = int(ds.argmin()); cert = False
            row[f"route_{label}"] = {"task": order[q], "certified": cert, "correct": order[q] == n}
        bres["rows"].append(row)
bres["summary"] = {lab: {lev: {"n": sum(1 for r in bres["rows"] if r["level"] == lev), "in_own_zone": sum(r["in_own_zone"] for r in bres["rows"] if r["level"] == lev),
                               "routed_correct_five": sum(r["route_five"]["correct"] for r in bres["rows"] if r["level"] == lev),
                               "routed_correct_fifteen": sum(r["route_fifteen"]["correct"] for r in bres["rows"] if r["level"] == lev),
                               "certified_wrong_five": sum(r["route_five"]["certified"] and not r["route_five"]["correct"] for r in bres["rows"] if r["level"] == lev),
                               "certified_wrong_fifteen": sum(r["route_fifteen"]["certified"] and not r["route_fifteen"]["correct"] for r in bres["rows"] if r["level"] == lev)}
                          for lev in sr.LEVELS} for lab in ("all",)}
RESULT["benchmark_5f"] = bres; dump(); log("5f done")

# ---------------------------------------------------------------- 5g: cost
RESULT["cost_5g"] = {"mean_candidates_per_text_all": float(np.mean([V[k].n_cand.mean() for k in V])),
                     "mean_judged_pairs_per_text_all_non_identical": float(np.mean([V[k].n_judged.mean() for k in V])),
                     "per_kind_mean_candidates": {k: float(V[k].n_cand.mean()) for k in V},
                     "per_kind_mean_judged_non_identical": {k: float(V[k].n_judged.mean()) for k in V}}
sample = [(k, i) for k in ("orig", "fr", "de") for i in range(0, K, 17)]
def one_text(kind, i):
    t0 = time.perf_counter(); u = TXT[kind][i]
    f = enc.encode([enc.ids(u, es.MAXTOK)], use_cache=False).astype(np.float64)
    d = np.linalg.norm(Eo - f, axis=1); ins = np.where(d <= r_w)[0]
    top = ins[np.argsort(d[ins], kind="stable")[:LIMIT]]
    if i in ins and i not in top: top = np.append(top, i)
    pr = [(u, orig[ev_names[j]]) for j in top if u != orig[ev_names[j]]]
    if CHOSEN == "A":
        out = forward([(o, uu) for uu, o in pr] + [(uu, o) for uu, o in pr]) if pr else []
        h = len(pr); s = {j: 1.0 for j in top}
        for q, (uu, o) in enumerate(pr): s[[j for j in top if orig[ev_names[j]] == o][0]] = min(out[q], out[h + q])
    else:
        s = {j: EB(u, orig[ev_names[j]]) for j in top}
    z = [j for j in top if s[j] >= tauJ]; _ = min(z) if z else int(d.argmin())
    return (time.perf_counter() - t0) * 1000, len(pr)
for kind, i in sample[:3]: one_text(kind, i)
tm = [one_text(k, i) for k, i in sample]
RESULT["cost_5g"].update({"timing_sample_texts": len(sample), "chosen_judge_timed": CHOSEN, "mean_ms_per_text_filter_plus_judge": float(np.mean([a for a, _ in tm])),
                          "median_ms": float(np.median([a for a, _ in tm])), "p90_ms": float(np.quantile([a for a, _ in tm], 0.9)), "mean_judged_pairs_in_sample": float(np.mean([b for _, b in tm])),
                          "note": "sequential, one text at a time, 2 threads, no cache; MiniLM encode + distances to 700 originals + judge both directions on candidates + route; originals/French/German sampled every 17th task"})
dump(); log("5g done", RESULT["cost_5g"]); log("done")
