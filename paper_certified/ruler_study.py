"""Second pre-registered experiment (last section of silent_adapters/PLAN.md): measure distance in units of what
paraphrasing does. CPU only, word-vectors only (embedding-table lookups, no forward pass), for the tables of
HuggingFaceTB/SmolLM2-135M and t5-base. Compares the OLD ruler (inverse square root of the shrunk covariance of unrelated
inputs) with the NEW ruler (inverse square root of the shrunk second moment of German-pivot paraphrase displacements of the 175
calibration tasks, beta 0.1 and 0.5), with a common radius (rule A) and with the paper's radius (rule B), on the 700 evaluation
tasks; matched point at C = C_ref (OLD, rule B, kappa 0.5); secondary standard check on the five / fifteen tasks.

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache OMP_NUM_THREADS=4 $S/mm_env/bin/python /path/to/paper_certified/ruler_study.py
Writes routing/ruler_study.json (next to this script). Reuses the conventions of feature_study.py (split, order, weights,
signatures, route rule, measures). Choices not fixed by the brief: see "notes" in the json.
"""
import json, os, random, sys, time, types
import numpy as np

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", f"{S}/hf_cache")
MAX_LEN = 512
KAPPAS = [round(0.1 * i, 1) for i in range(1, 11)]
QGRID = [round(0.0125 * i, 4) for i in range(1, 81)]
BETAS = [0.1, 0.5]
SNAP = 1e-3
TABLES = [("HuggingFaceTB/SmolLM2-135M", "smollm2"), ("t5-base", "t5-base")]
if len(sys.argv) > 1: TABLES = [t for t in TABLES if t[1] in sys.argv[1:]]

import torch
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
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
t_start = time.time()
def log(*a): print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)
def template(definition): return "Definition: " + definition + "\n\nNow complete the following example -\nInput: {0}\nOutput: "

# ---------------------------------------------------------------- routes and measures (as feature_study.py)
def snap(x): x = np.asarray(x, dtype=np.float64).copy(); x[x < SNAP] = 0.0; return x
def radii_r1(d_bg, d_sig, kappa):
    K = len(d_bg); r = np.zeros(K)
    for j in range(K):
        near = d_bg[j] if j == 0 else min(d_bg[j], d_sig[j, :j].min())
        r[j] = kappa * near
    return r
def route(dist, radius):
    inside = dist <= radius[None]
    with np.errstate(divide="ignore", invalid="ignore"): rel = dist / radius[None]
    rel = np.where(np.isnan(rel), 0.0, rel)
    return np.where(inside.any(1), inside.argmax(1), rel.argmin(1)), inside.any(1)
def measures(Du, ou, Dp, op, radius):
    K = len(radius); res = {}
    d = snap(Du); older = np.arange(K)[None, :] < ou[:, None]
    blocked = ((d <= radius[None]) & older).any(1)
    has = np.bincount(ou, minlength=K) > 0
    coll = np.bincount(ou, weights=blocked.astype(float), minlength=K) > 0
    ru, _ = route(d, radius)
    res["C"] = float(coll[has].sum() / has.sum()); res["U"] = float((ru == ou).mean())
    rp, cert = route(snap(Dp), radius); own = rp == op
    res["P"] = float((cert & own).mean()); res["M"] = float((cert & ~own).mean()); res["R"] = float(own.mean())
    res["paraphrased_inputs"] = int(len(op)); res["radius_zero"] = int((radius == 0).sum())
    return res

def match(xs, curve, Cref):
    """xs increasing; curve = list of dicts with C,P,M,R,U. Largest grid point with C <= Cref, then linear interpolation towards
    the next point to C = Cref. Returns dict with x and interpolated P,M,R,U and flags."""
    C = np.array([m["C"] for m in curve]); ok = np.where(C <= Cref + 1e-12)[0]
    if len(ok) == 0:
        return {"x": float(xs[0]), **{k: curve[0][k] for k in "PMRU"}, "flag": "C above C_ref at the smallest radius"}
    i = int(ok.max())
    if i == len(C) - 1:
        return {"x": float(xs[i]), **{k: curve[i][k] for k in "PMRU"}, "flag": "C never exceeds C_ref (end of curve)"}
    t = (Cref - C[i]) / (C[i + 1] - C[i])
    return {"x": float(xs[i] + t * (xs[i + 1] - xs[i])), "t": float(t), "grid_index": i,
            **{k: float(curve[i][k] + t * (curve[i + 1][k] - curve[i][k])) for k in "PMRU"}, "flag": None}

def inv_sqrt(M):
    w, v = np.linalg.eigh(M)
    return (v * w ** -0.5) @ v.T

# ---------------------------------------------------------------- per embedding table
def run_table(model_name, key):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = (T5EncoderModel if "t5" in model_name else AutoModel).from_pretrained(model_name, torch_dtype=torch.float32).eval()
    E = model.get_input_embeddings().weight.detach(); V, D = E.shape
    ids_of = lambda text: tok(text, max_length=MAX_LEN, truncation=True)["input_ids"]
    log(key, "table", tuple(E.shape))

    # --- unrelated inputs, token weights (as unrelated_base)
    rng = random.Random(0); defs, texts = [], []
    for name in sr.POOL_TASKS:
        data = json.load(open(os.path.join("CL_Benchmark", "Long_Sequence", name, "train.json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
        defs.append(d); texts.append([inst["input"] for inst in data["Instances"]])
    picks = [rng.randrange(len(defs)) for _ in range(sr.BG_N)]
    sources = [sr._source(defs[k], rng.choice(texts[k])) for k in picks]
    rows = tok(sources, max_length=MAX_LEN, truncation=True)["input_ids"]
    counts = torch.zeros(V, dtype=torch.double)
    for r in rows:
        r = torch.tensor(r); r = r[r != tok.pad_token_id]; counts += torch.bincount(r, minlength=V).double()
    WEIGHT = sr.SIF_A / (sr.SIF_A + counts / counts.sum()); WEIGHT[tok.pad_token_id] = 0.0

    def cum(ids, nmax):
        T = min(len(ids), int(nmax)); t = torch.tensor(ids[:T])
        w = WEIGHT[t]
        return (torch.cumsum(E[t].double() * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None]).numpy()
    def phi(ids, ns):
        """raw feature at each n in ns (clipped to the length): (len(ns), D) float64"""
        ns = np.asarray(ns); c = cum(ids, ns.max()); return c[np.minimum(ns, len(ids)) - 1]

    # --- OLD ruler
    x64 = np.stack([phi(r[:64], [64])[0] for r in rows])
    mu = x64.mean(0); xc = x64 - mu
    cov = xc.T @ xc / (len(x64) - 1)
    cov = 0.5 * cov + 0.5 * np.eye(D) * np.trace(cov) / D
    W_old = inv_sqrt(cov)
    # --- unrelated signatures
    bg_raw = []
    for d, xs in zip(defs, texts):
        r = tok([sr._source(d, x) for x in xs[:20]], max_length=MAX_LEN, truncation=True)["input_ids"]
        n = sr._prefix_len(r); assert n >= 5, (key, n)
        bg_raw.append(phi(r[0][:n], [n])[0])
    bg_raw = np.stack(bg_raw)

    # --- tasks, split, openings
    tasks = json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"]
    para = json.load(open(os.path.join(HERE, "routing", "paraphrases.json")))
    by_name = {t["name"]: t for t in tasks}; names = sorted(by_name)
    rr = random.Random(0); shuffled = names[:]; rr.shuffle(shuffled)
    cal_names, ev_names = shuffled[:175], shuffled[175:]
    def opening(definition):
        a = ids_of(template(definition).replace("{0}", "A B")); b = ids_of(template(definition).replace("{0}", "zzq xkj"))
        n = min(sr._prefix_len([a, b]), MAX_LEN); return a[:n], n
    ops = {n: opening(by_name[n]["definition"]) for n in names}
    nk = {n: ops[n][1] for n in names}
    sig_raw = {n: phi(ops[n][0], [nk[n]])[0] for n in names}
    log(key, "opening lengths min/median/max", min(nk.values()), int(np.median(list(nk.values()))), max(nk.values()))

    # --- paraphrase displacement (calibration tasks, German pivot, first 3 inputs, n_k of the original opening)
    disp = []
    for n in cal_names:
        t = by_name[n]
        for x in t["inputs"][:3]:
            a = phi(ids_of(template(t["definition"]).replace("{0}", x)), [nk[n]])[0]
            b = phi(ids_of(template(para["tasks"][n]["de"]).replace("{0}", x)), [nk[n]])[0]
            disp.append(b - a)
    disp = np.stack(disp)
    Sig_p = disp.T @ disp / len(disp)
    ev_ = np.linalg.eigvalsh(Sig_p)[::-1]
    shares = ev_ / ev_.sum()
    rank = int((ev_ > ev_[0] * 1e-10).sum())
    diag = {"n_displacements": int(len(disp)), "dim": int(D), "rank": rank, "top10_eigenvalue_shares": [float(s) for s in shares[:10]],
            "top10_cumulative": float(shares[:10].sum()), "trace": float(np.trace(Sig_p))}
    log(key, "displacements", len(disp), "rank", rank, "top10 shares", np.round(shares[:10], 3))
    RULERS = {"OLD": W_old}
    for b in BETAS:
        RULERS[f"NEW_beta{b}"] = inv_sqrt((1 - b) * Sig_p + b * np.trace(Sig_p) / D * np.eye(D))
    RN = list(RULERS)

    # --- evaluation items
    ev_idx = {n: i for i, n in enumerate(ev_names)}
    ev_nk = np.array([nk[n] for n in ev_names]); uniq = np.unique(ev_nk); where = np.searchsorted(uniq, ev_nk)
    ev_sig_raw = np.stack([sig_raw[n] for n in ev_names])
    sigz = {r: (ev_sig_raw - mu) @ W for r, W in RULERS.items()}
    bgz = {r: (bg_raw - mu) @ W for r, W in RULERS.items()}

    def dist_matrix(texts_, rulers, nk_arr, sigz_, uniq_, where_):
        out = {r: np.zeros((len(texts_), len(nk_arr)), dtype=np.float32) for r in rulers}
        for i, tx in enumerate(texts_):
            f = phi(ids_of(tx), uniq_) - mu
            for r in rulers:
                z = (f @ RULERS[r])[where_]
                out[r][i] = np.linalg.norm(z - sigz_[r], axis=1)
        return out
    items = []  # (task column, kind 0 unedited / 1 de / 2 fr, text)
    for n in ev_names:
        t = by_name[n]
        for x in t["inputs"][:6]: items.append((ev_idx[n], 0, template(t["definition"]).replace("{0}", x)))
        for k, p in enumerate(("de", "fr")):
            for x in t["inputs"][:3]: items.append((ev_idx[n], 1 + k, template(para["tasks"][n][p]).replace("{0}", x)))
    owner = np.array([i[0] for i in items]); kind = np.array([i[1] for i in items])
    log(key, "evaluation items", len(items), "unedited", int((kind == 0).sum()), "de", int((kind == 1).sum()), "fr", int((kind == 2).sum()), "unique n", len(uniq))
    DM = dist_matrix([i[2] for i in items], RN, ev_nk, sigz, uniq, where)
    log(key, "distances done")
    mu_k = kind == 0
    sets = {"fr": kind == 2, "de": kind == 1, "both": kind > 0}
    K = len(ev_names)
    out = {"diagnostics_sigma": diag, "rulers": {}}
    for r in RN:
        Du, ou = DM[r][mu_k], owner[mu_k]
        s = sigz[r]; d_sig = snap(np.linalg.norm(s[:, None] - s[None], axis=2))
        d_bg = snap(np.linalg.norm(s[:, None] - bgz[r][None], axis=2).min(1))
        own_d = snap(DM[r][kind > 0][np.arange((kind > 0).sum()), owner[kind > 0]])
        grid_r = [float(np.quantile(own_d, q)) for q in QGRID]
        res = {"rule_A_radii": grid_r, "rule_A_quantiles": QGRID, "rule_A": {}, "rule_B": {}}
        for sname, m in sets.items():
            Dp, op = DM[r][m], owner[m]
            res["rule_A"][sname] = [measures(Du, ou, Dp, op, np.full(K, rr_)) for rr_ in grid_r]
            res["rule_B"][sname] = [measures(Du, ou, Dp, op, radii_r1(d_bg, d_sig, kp)) for kp in KAPPAS]
        # diagnostic ratio: mean distance of paraphrased input from own signature / mean nearest-other-signature distance (eval tasks)
        dd = d_sig + np.diag(np.full(K, np.inf))
        res["ratio_paraphrase_to_nearest_signature"] = float(own_d.mean() / dd.min(1).mean())
        res["mean_own_distance_paraphrased"] = float(own_d.mean()); res["mean_nearest_signature_distance"] = float(dd.min(1).mean())
        res["C_monotone_A"] = bool(np.all(np.diff([m["C"] for m in res["rule_A"]["both"]]) >= -1e-12))
        res["C_monotone_B"] = bool(np.all(np.diff([m["C"] for m in res["rule_B"]["both"]]) >= -1e-12))
        res["P_nondecreasing_A_fr"] = bool(np.all(np.diff([m["P"] for m in res["rule_A"]["fr"]]) >= -1e-12))
        res["_dbg"] = (d_bg, d_sig)
        out["rulers"][r] = res
    # --- reference point and matched points
    ref = out["rulers"]["OLD"]["rule_B"]["both"][KAPPAS.index(0.5)]; C_ref = ref["C"]
    out["reference"] = {"ruler": "OLD", "rule": "B", "kappa": 0.5, **ref}
    out["C_ref"] = C_ref
    out["matched"] = {}
    for r in RN:
        out["matched"][r] = {}
        for sname in ("fr", "both", "de"):
            out["matched"][r][sname] = {"rule_A": match(out["rulers"][r]["rule_A_radii"], out["rulers"][r]["rule_A"][sname], C_ref),
                                        "rule_B": match(KAPPAS, out["rulers"][r]["rule_B"][sname], C_ref)}
    # verdict (rule A, French pivot)
    mo = out["matched"]["OLD"]["fr"]["rule_A"]; mn = out["matched"]["NEW_beta0.1"]["fr"]["rule_A"]
    gain = mn["P"] - mo["P"]
    out["verdict"] = {"P_gain": gain, "P_new": mn["P"], "P_old": mo["P"], "M_new": mn["M"], "M_old": mo["M"],
                      "outcome": "success" if (gain >= 0.20 and mn["M"] <= mo["M"]) else ("against" if (gain < 0.05 or mn["M"] > mo["M"]) else "in between")}
    log(key, "C_ref", C_ref, "verdict", out["verdict"])

    # --- secondary standard check: NEW beta 0.1 with the rule-A radius of the matched point (fr), OLD with rule B kappa 0.5
    names15 = sorted(set(FIVE) | set(ORDER1)); data = {}
    for n in names15:
        dd_ = {}
        for split in ("train", "test"):
            dj = json.load(open(os.path.join("CL_Benchmark", "SuperNI", n, split + ".json")))
            dd_["definition" if split == "train" else "dummy"] = (dj["Definition"][0] if isinstance(dj["Definition"], list) else dj["Definition"]).strip()
            if split == "test": dd_["test"] = [i["input"] for i in dj["Instances"]]
        data[n] = dd_
    defs15 = {n: [data[n]["definition"]] + sr.REWORDED.get(n.split("_")[0], []) for n in names15}
    ops15 = {n: opening(defs15[n][0]) for n in names15}
    nk15 = np.array([ops15[n][1] for n in names15]); u15 = np.unique(nk15); w15 = np.searchsorted(u15, nk15)
    sraw15 = np.stack([phi(ops15[n][0], [ops15[n][1]])[0] for n in names15])
    sz15 = {r: (sraw15 - mu) @ RULERS[r] for r in ("OLD", "NEW_beta0.1")}
    texts15, meta15 = [], []
    for n in names15:
        for li, d in enumerate(defs15[n]):
            for x in data[n]["test"]: texts15.append(template(d).replace("{0}", x)); meta15.append((n, li))
    log(key, "standard check inputs", len(texts15), "levels per task", {n: len(defs15[n]) for n in names15 if len(defs15[n]) != 4})
    D15 = dist_matrix(texts15, ["OLD", "NEW_beta0.1"], nk15, sz15, u15, w15)
    tn = {n: i for i, n in enumerate(names15)}
    r_star = mn["x"]; sec = {"radius_rule_A_new": r_star}
    for r in ("OLD", "NEW_beta0.1"):
        sec[r] = {}
        for label, order in (("five", FIVE), ("fifteen", ORDER1)):
            cols = [tn[n] for n in order]; s = sz15[r][cols]
            d_sig = snap(np.linalg.norm(s[:, None] - s[None], axis=2)); d_bg = snap(np.linalg.norm(s[:, None] - bgz[r][None], axis=2).min(1))
            rad = radii_r1(d_bg, d_sig, 0.5) if r == "OLD" else np.full(len(order), r_star)
            sec[r][label] = {}
            for li, lev in enumerate(sr.LEVELS):
                rowsel = [i for i, (n, l) in enumerate(meta15) if l == li and n in order]
                own = np.array([order.index(meta15[i][0]) for i in rowsel])
                rt, cert = route(snap(D15[r][rowsel][:, cols]), rad)
                sec[r][label][lev] = {"zone": float((cert & (rt == own)).mean()), "certified": float((rt == own).mean()),
                                      "inputs": len(rowsel), "wrong": int((cert & (rt != own)).sum())}
    out["secondary"] = sec
    for r in out["rulers"].values(): r.pop("_dbg")
    return out

if __name__ == "__main__":
    result = {"tables": {}, "notes": {
        "matching": "match(): largest grid point with C <= C_ref, then linear interpolation towards the next point to C = C_ref; flags if out of range.",
        "rule_A_grid": "80 quantiles 0.0125..1.0 of distances of paraphrased inputs (both pivots, 700 evaluation tasks) to own signature, per ruler",
        "diagnostic_ratio": "mean own-signature distance of paraphrased inputs (both pivots) / mean nearest-other-signature distance, among the 700 evaluation signatures",
        "Sigma_p": "uncentred mean of d d^T, d = phi_nk(German paraphrase) - phi_nk(original), first 3 inputs of each of 175 calibration tasks",
        "snap": SNAP, "T5": "T5EncoderModel shared embedding table; tokenizer appends </s>, pad id 0 has weight 0",
        "secondary_radius": "NEW beta 0.1 uses the rule-A matched radius of the French-pivot set (same radius for all tasks)"}}
    for model_name, key in TABLES:
        result["tables"][key] = run_table(model_name, key)
        json.dump(result, open(os.path.join(HERE, "routing", "ruler_study.json"), "w"), indent=1)
    log("done")
