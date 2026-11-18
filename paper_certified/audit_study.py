"""One joint recogniser and a forgetting certificate from held-out inputs (last section of silent_adapters/PLAN.md,
"a different kind of answer, not a shape ... Fixed before any run").
noinstruction_study.py / boundary_study.py run whole studies at import, so the few small definitions needed (task lists, split rule,
shape numbers, random-feature map, minilm encoder) are copied here unchanged; cached minilm features of the earlier study are read.
Run (1 thread each):
  S=$FK_LORA_SCRATCH
  cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 \
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $S/mm_env/bin/python audit_study.py
Optional env: AUDIT_ONLY="lr|minilm+char" (substring filters on variant names; fills caches only) ; AUDIT_COMPUTE_ONLY=1.
Writes routing/audit_study.json (after every step). Caches in $S/audit/. Step B reads fresh_tasks_3.json only after step A is in the JSON."""
import json, math, os, random, sys, time, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "routing", "audit_study.json")
NIC = os.path.join(S, "noinstruction"); AC = os.path.join(S, "audit"); os.makedirs(AC, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]
LONG3 = "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
TASKS = [("SuperNI", n) for n in ORDER1] + [("Long_Sequence", n) for n in LONG3]
KEY = [f"{f}/{n}" for f, n in TASKS]
M = 200; MAXTOK = 256; CAP = 1500; D = 4096
CS = [1, 10, 100]; LAMS = [0.01, 0.1, 1]
FEATS = ["minilm", "minilm+shape", "minilm+char"]
VARIANTS = [("lr", f, C) for f in FEATS for C in CS] + [("analytic", f, l) for f in FEATS[:2] for l in LAMS]
def vname(v): return f"{v[0]}|{v[1]}|{v[2]}"

# ---------------------------------------------------------------- copied definitions
def make_split(n):
    perm = list(range(n)); random.Random(0).shuffle(perm)
    m = M if n >= 2 * M else n // 2
    return np.array(sorted(perm[:m]), dtype=int), np.array(sorted(perm[m:]), dtype=int)

def capped(idx):
    """at most CAP fit inputs per task: random (seed 0) subset, kept in original order"""
    if len(idx) <= CAP: return idx
    return np.array(sorted(random.Random(0).sample(list(map(int, idx)), CAP)), dtype=int)

def mse(v): v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))

def minilm_enc():
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"; tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    def enc(texts):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); res = np.zeros((len(texts), model.config.hidden_size))
        for s in range(0, len(texts), 32):
            b = order[s:s + 32]; t = tok([texts[i] for i in b], max_length=MAXTOK, truncation=True, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = model(**t).last_hidden_state; m = t["attention_mask"][:, :, None].to(h.dtype)
                v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
            res[b] = v.numpy()
        return res
    return enc

def shape_vec(t):
    n = len(t); nn = max(n, 1); words = t.split(); nw = len(words)
    letters = up = dig = ws = other = 0
    for c in t:
        if c.isalpha():
            letters += 1; up += c.isupper()
        elif c.isdigit(): dig += 1
        elif c.isspace(): ws += 1
        else: other += 1
    return [math.log1p(n), math.log1p(nw), (sum(map(len, words)) / nw) if nw else 0.0, up / max(letters, 1), dig / nn, ws / nn, other / nn] \
        + [100.0 * t.count(ch) / nn for ch in ".?!,:"] + [math.log1p(t.count("\n")), float("@" in t), float("#" in t), float("http" in t)]
# NOTE: the plan says "15 numbers" but lists 16 (7 length/share numbers, 5 punctuation rates, line breaks, 3 flags); all 16 listed are used.

def shape_cached(fn, texts):
    if os.path.exists(fn): return np.load(fn)["v"]
    v = np.array([shape_vec(t) for t in texts]); np.savez(fn, v=v); return v

def read_texts(f, n, sp_):
    d = json.load(open(os.path.join("CL_Benchmark", f, n, sp_ + ".json")))
    return [str(i["input"]) for i in d["Instances"]]

# ---------------------------------------------------------------- a "sequence": per task train/test minilm, texts, split, pool
class Seq:
    def __init__(self, tag, names, F, texts, pool_F, pool_texts):
        self.tag, self.names, self.K = tag, names, len(names)
        self.F, self.texts = F, texts                       # F[name] = (train, test) minilm ; texts[name] = (train texts, test texts)
        self.spl = {k: make_split(len(F[k][0])) for k in names}
        self.fit_idx = {k: capped(self.spl[k][1]) for k in names}
        self.pool_F, self.pool_texts = pool_F, pool_texts
        self._cache = {}
        pf = os.path.join(AC, f"shape_pool_{tag}.npz"); v = shape_cached(pf, pool_texts)
        sd = v.std(0); sd[sd == 0] = 1.0; self.sh_m, self.sh_s = v.mean(0), sd; self.pool_shape = v
    def shape_block(self, v): return 0.5 * np.clip((v - self.sh_m) / self.sh_s, -5, 5)
    def shape_task(self, k):
        fn = os.path.join(AC, f"shape_{self.tag}_{k.replace('/', '_')}.npz")
        if os.path.exists(fn): z = np.load(fn); return z["train"], z["test"]
        r = [np.array([shape_vec(t) for t in self.texts[k][s]]) for s in (0, 1)]; np.savez(fn, train=r[0], test=r[1]); return r[0], r[1]
    def char_block(self, texts):
        return normalize(self.tfidf.transform(self.hv.transform([t[:2000] for t in texts])))
    def features(self, feat):
        """returns (fit, cal, test) lists of matrices per task and the pool feature matrix"""
        if feat in self._cache: return self._cache[feat]
        fit, cal, te = [], [], []
        if feat == "minilm+char":
            self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(1, 3), n_features=16384, alternate_sign=False, norm="l2", lowercase=False)
            self.tfidf = TfidfTransformer().fit(self.hv.transform([t[:2000] for t in self.pool_texts]))
        for k in self.names:
            tr, ts = self.F[k]; c, f = self.spl[k]; f = self.fit_idx[k]
            if feat == "minilm": A, B = tr, ts
            elif feat == "minilm+shape":
                s_tr, s_ts = self.shape_task(k); A = np.hstack([tr, self.shape_block(s_tr)]); B = np.hstack([ts, self.shape_block(s_ts)])
            else:
                cf = os.path.join(AC, f"char_{self.tag}_{k.replace('/', '_')}.npz")
                if os.path.exists(cf):
                    z = np.load(cf); Ct, Cs = sp.csr_matrix((z["d0"], z["i0"], z["p0"]), shape=tuple(z["s0"])), sp.csr_matrix((z["d1"], z["i1"], z["p1"]), shape=tuple(z["s1"]))
                else:
                    Ct, Cs = self.char_block(self.texts[k][0]), self.char_block(self.texts[k][1])
                    np.savez(cf, d0=Ct.data, i0=Ct.indices, p0=Ct.indptr, s0=np.array(Ct.shape), d1=Cs.data, i1=Cs.indices, p1=Cs.indptr, s1=np.array(Cs.shape))
                A = sp.hstack([sp.csr_matrix(tr), Ct]).tocsr(); B = sp.hstack([sp.csr_matrix(ts), Cs]).tocsr()
            fit.append(A[f]); cal.append(A[c]); te.append(B)
        if feat == "minilm": P = self.pool_F
        elif feat == "minilm+shape": P = np.hstack([self.pool_F, self.shape_block(self.pool_shape)])
        else: P = None
        self._cache[feat] = (fit, cal, te, P); return self._cache[feat]

def stack(mats): return sp.vstack(mats).tocsr() if sp.issparse(mats[0]) else np.vstack(mats)

# ---------------------------------------------------------------- recognisers; each returns routes[T] = (te routes list for j<=T, cal routes list)
def run_lr(seq, feat, C):
    fit, cal, te, _ = seq.features(feat); K = seq.K; out = {}; notes = []
    for T in range(K):
        if T == 0:
            out[T] = ([np.zeros(te[0].shape[0], int)], [np.zeros(cal[0].shape[0], int)]); continue
        X = stack(fit[:T + 1]); y = np.concatenate([np.full(fit[j].shape[0], j) for j in range(T + 1)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
        notes.append(int(np.max(clf.n_iter_)))
        out[T] = ([clf.predict_proba(te[j]).argmax(1) for j in range(T + 1)], [clf.predict_proba(cal[j]).argmax(1) for j in range(T + 1)])
        log(seq.tag, "lr", feat, C, "T", T + 1, "iters", notes[-1], "warn", len(w))
    return out, {"max_iter_per_arrival": notes}

def make_rff(Pv):
    ps = np.random.default_rng(0).choice(len(Pv), 2000, replace=False); Q = Pv[ps]
    sq = (Q ** 2).sum(1); dm = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Q @ Q.T, 0)); iu = np.triu_indices(2000, 1); sigma = float(np.median(dm[iu]))
    rng = np.random.default_rng(0); W = rng.normal(0.0, 1.0 / sigma, size=(D, Pv.shape[1])); b = rng.uniform(0, 2 * np.pi, size=D)
    return (lambda v: math.sqrt(2.0 / D) * np.cos(v @ W.T + b)), sigma

def run_analytic(seq, feat, lams, order=None, final_only=False):
    """running sums G, Cmat; routes after every arrival for every lam. order: arrival order (indices into the sequence); labels stay the task index."""
    fit, cal, te, P = seq.features(feat); K = seq.K; phi, sigma = make_rff(P)
    order = list(range(K)) if order is None else order
    G = np.zeros((D, D)); Cm = np.zeros((D, K)); out = {l: {} for l in lams}; W_end = {}
    Zte = {}; Zcal = {}
    for step, t in enumerate(order):
        Z = phi(fit[t]); G += Z.T @ Z; Cm[:, t] += Z.sum(0)          # one-hot target of task t: Z^T e_t = column sum
        if not final_only: Zte[t] = phi(te[t]); Zcal[t] = phi(cal[t])
        done = order[:step + 1]
        for l in lams:
            if final_only and step < K - 1: continue
            cho = sla.cho_factor(G + l * np.eye(D)); Wt = sla.cho_solve(cho, Cm[:, done])        # columns = tasks done
            if step == K - 1: W_end[l] = (done, Wt)
            if not final_only:
                # route = task with the largest output (ties to the older task: lower arrival position; columns are in arrival order)
                rt = [np.array(done)[(Zte[j] @ Wt).argmax(1)] for j in done]; rc = [np.array(done)[(Zcal[j] @ Wt).argmax(1)] for j in done]
                out[l][step] = (rt, rc)
        log(seq.tag, "analytic", feat, "arrival", step + 1)
    return out, {"sigma": sigma, "W_end": W_end}

# ---------------------------------------------------------------- measures from routes
def measures(seq, R, path=False):
    """R[T] = (te routes for tasks 0..T, cal routes for tasks 0..T); routes are task indices."""
    K = seq.K; per = {}
    for j, k in enumerate(seq.names):
        te_a, cal_a = R[j][0][j], R[j][1][j]; te_e, cal_e = R[K - 1][0][j], R[K - 1][1][j]
        c = int((cal_e != cal_a).sum()); m = len(cal_a)
        d = {"own_at_arrival": float((te_a == j).mean()), "own_at_end": float((te_e == j).mean()), "changed": float((te_e != te_a).mean()),
             "c": c, "m": m, "certificate": (c + 1) / (m + 1), "heldout_own_at_arrival": float((cal_a == j).mean()), "heldout_own_at_end": float((cal_e == j).mean())}
        d["apriori_bound"] = (1 - d["own_at_arrival"]) + (1 - d["own_at_end"])
        wrong = te_e[te_e != j]
        if len(wrong): bc = np.bincount(wrong, minlength=K); d["top_wrong_task"] = seq.names[int(bc.argmax())]; d["top_wrong_share"] = float(bc.max() / len(wrong))
        per[k] = d
    meas = ["own_at_arrival", "own_at_end", "changed", "certificate", "heldout_own_at_arrival", "heldout_own_at_end", "apriori_bound"]
    mean, se = {}, {}
    for m_ in meas: mean[m_], se[m_] = mse([per[k][m_] for k in seq.names])
    res = {"per_task": per, "mean": mean, "se": se}
    if path:
        P_ = {}
        for T in range(1, K):
            P_[str(T + 1)] = {seq.names[j]: {"changed_test": float((R[T][0][j] != R[j][0][j]).mean()),
                                              "certificate": (int((R[T][1][j] != R[j][1][j]).sum()) + 1) / (len(R[j][1][j]) + 1)} for j in range(T)}
        res["path"] = P_
    return res

def routes_cached(seq, v, extra=None):
    """routes for variant v on seq, cached in $S/audit/. For analytic all three lam values are computed together."""
    rec, feat, par = v
    fn = os.path.join(AC, f"routes_{seq.tag}_{rec}_{feat}_{par}.npz")
    def load():
        z = np.load(fn); K = seq.K
        return {T: ([z[f"te_{T}_{j}"] for j in range(T + 1)], [z[f"cal_{T}_{j}"] for j in range(T + 1)]) for T in range(K)}, json.loads(str(z["meta"]))
    if os.path.exists(fn): return load()
    if rec == "lr": R, meta = run_lr(seq, feat, par)
    else:
        outs, info = run_analytic(seq, feat, LAMS)
        for l in LAMS:
            sv = {"meta": json.dumps({"sigma": info["sigma"]})}
            for T in range(seq.K):
                for j in range(T + 1): sv[f"te_{T}_{j}"] = outs[l][T][0][j]; sv[f"cal_{T}_{j}"] = outs[l][T][1][j]
            np.savez(os.path.join(AC, f"routes_{seq.tag}_analytic_{feat}_{l}.npz"), **sv)
        return load()
    sv = {"meta": json.dumps(meta)}
    for T in range(seq.K):
        for j in range(T + 1): sv[f"te_{T}_{j}"] = R[T][0][j]; sv[f"cal_{T}_{j}"] = R[T][1][j]
    np.savez(fn, **sv); return load()

# ---------------------------------------------------------------- results file
RESULT = json.load(open(OUT)) if os.path.exists(OUT) and not os.environ.get("AUDIT_ONLY") else {}
def dump():
    if os.environ.get("AUDIT_ONLY") or os.environ.get("AUDIT_COMPUTE_ONLY"): return
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(RESULT, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)

def summary(res): return {"mean": res["mean"], "se": res["se"]}

# ---------------------------------------------------------------- development sequences
def dev_seqs():
    pool_texts = [str(x) for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"] if t["name"] not in set(ORDER1) for x in t["inputs"]]
    P = np.load(os.path.join(NIC, "pool_minilm.npy")); assert len(P) == len(pool_texts) == 5165
    seqs = {}
    for tag, ks in (("superni", KEY[:15]), ("long", KEY[15:])):
        F, tx = {}, {}
        for k in ks:
            z = np.load(os.path.join(NIC, "minilm_" + k.replace("/", "_") + ".npz")); F[k] = (z["train"], z["test"])
            f, n = k.split("/"); tx[k] = (read_texts(f, n, "train"), read_texts(f, n, "test"))
            assert len(tx[k][0]) == len(F[k][0]) and len(tx[k][1]) == len(F[k][1]), k
        seqs[tag] = Seq(tag, ks, F, tx, P, pool_texts)
    return seqs

def main():
    only = os.environ.get("AUDIT_ONLY")
    seqs = dev_seqs()
    RESULT["notes"] = [
        "Instances only (the 'input' text); task definitions are never read. Features: cached all-MiniLM-L6-v2 (noinstruction cache). Split per task: seed 0 shuffle, 200 calibration (half if < 400 training inputs), rest fit; calibration inputs are held out and used only for the certificate and held-out measures.",
        f"At most {CAP} fit inputs per task are used (random.Random(0).sample, kept in original order); tasks with more fit inputs than that are subsampled for both recognisers.",
        "Routing: argmax; ties to the older task (lowest index). At T = 1 everything goes to task 1. 'lr': sklearn LogisticRegression(C, max_iter=3000, class_weight='balanced'), retrained from scratch at each arrival.",
        "'analytic': D=4096 random cosine features, sigma = median pairwise Euclidean distance of 2,000 pool features (default_rng(0).choice), W ~ N(0, 1/sigma^2), b ~ U[0, 2 pi], default_rng(0); running G and Cmat; weights = cho_solve(G + lam I, Cmat).",
        "minilm+shape: the plan names 16 shape numbers (while saying 15); all 16 listed are used, standardised with pool mean/std, clipped to [-5, 5], times 0.5. minilm+char: minilm (dense, unit norm) hstacked with l2-normalised tf-idf (fitted on pool texts only) of hashed char_wb 1-3-grams of the text cut at 2,000 characters (l2 normalisation applied to the tf-idf block, the minilm block is already unit norm).",
        "Pool for development: the 5,165 pool inputs of the earlier studies (ni_tasks.json without the 15 SuperNI tasks of order 1); used for shape standardisation, tf-idf and sigma.",
        "Means and standard errors (std ddof=1 / sqrt(#tasks)) over all tasks of the sequence, including task 1 and the last task (changed = 0 by construction there; its certificate is 1/(m+1)).",
        "certificate = (c + 1) / (m + 1) per task, c = held-out inputs routed differently by R_T than by R_j."]
    dump()
    dev = RESULT.setdefault("dev", {"superni": {}, "long": {}}); extra = RESULT.setdefault("dev_meta", {})
    # --- order check of the analytic recogniser (SuperNI, minilm), reversed arrival order
    if not only or "analytic" in only:
        ocf = os.path.join(AC, "order_check.json")
        if os.path.exists(ocf): RESULT["order_check"] = json.load(open(ocf))
        if "order_check" not in RESULT:
            sq = seqs["superni"]; chk = {}
            _, info1 = run_analytic(sq, "minilm", LAMS, final_only=True)  # (kept simple: info holds W_end)
            _, info2 = run_analytic(sq, "minilm", LAMS, order=list(range(sq.K))[::-1], final_only=True)
            for l in LAMS:
                d1, W1 = info1["W_end"][l]; d2, W2 = info2["W_end"][l]
                col1 = {t: W1[:, i] for i, t in enumerate(d1)}; col2 = {t: W2[:, i] for i, t in enumerate(d2)}
                chk[str(l)] = {"max_abs_diff": float(max(np.abs(col1[t] - col2[t]).max() for t in col1)), "max_abs_weight": float(max(np.abs(col1[t]).max() for t in col1))}
            RESULT["order_check"] = {"setting": "SuperNI order 1, minilm, D=4096, arrival order reversed; weights (columns matched by task)", "per_lam": chk,
                                     "largest_abs_difference": max(c["max_abs_diff"] for c in chk.values())}
            json.dump(RESULT["order_check"], open(ocf, "w")); dump(); log("order check", RESULT["order_check"])
    for v in VARIANTS:
        nm = vname(v)
        if only and not any(s in nm for s in only.split(",")): continue
        for tag, sq in seqs.items():
            if nm in dev[tag]: continue
            R, meta = routes_cached(sq, v)
            if os.environ.get("AUDIT_COMPUTE_ONLY"): continue
            dev[tag][nm] = summary(measures(sq, R)); extra.setdefault(tag, {})[nm] = meta
            dump(); log("dev", tag, nm, {k: round(x, 3) for k, x in dev[tag][nm]["mean"].items() if k in ("own_at_end", "changed", "certificate")})
    if os.environ.get("AUDIT_COMPUTE_ONLY") or only: return
    # --- full per-task results for every variant (cheap from cache)
    full = {}
    for v in VARIANTS:
        nm = vname(v)
        for tag, sq in seqs.items():
            R, _ = routes_cached(sq, v); full.setdefault(tag, {})[nm] = measures(sq, R)
    # --- choice
    order_feat = {f: i for i, f in enumerate(FEATS)}
    cand = [v for v in VARIANTS if full["superni"][vname(v)]["mean"]["changed"] <= 0.03]
    if not cand:
        RESULT["chosen"] = None; RESULT["note_no_variant"] = "no variant has a mean changed share <= 0.03 on SuperNI: nothing chosen, step B not run"; dump(); log("nothing chosen"); return
    key = lambda v: (-full["superni"][vname(v)]["mean"]["own_at_end"], -full["long"][vname(v)]["mean"]["own_at_end"],
                     0 if v[0] == "analytic" else 1, order_feat[v[1]], v[2])
    ch = min(cand, key=key); nm = vname(ch)
    RESULT["chosen"] = {"variant": nm, "recogniser": ch[0], "feature": ch[1], "parameter": ch[2], "n_candidates": len(cand),
                        "superni": full["superni"][nm], "long": full["long"][nm]}
    for tag, sq in seqs.items():
        R, _ = routes_cached(sq, ch); RESULT["chosen"][tag + "_path"] = measures(sq, R, path=True)["path"]
    dump(); log("chosen", nm)
    # --- Step B: read ONCE; only now is fresh_tasks_3.json opened
    step_b(ch)
    log("done")

def step_b(ch):
    fresh = json.load(open(os.path.join(S, "ni_fresh", "fresh_tasks_3.json")))["tasks"]
    names = [t["name"] for t in fresh]; assert len(names) == 30 and all(len(t["instances"]) == 1000 for t in fresh)
    fset = set(names); pool_names, pool_texts = [], []
    for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"]:
        if t["name"] in set(ORDER1): continue
        for x in t["inputs"]: pool_names.append(t["name"]); pool_texts.append(str(x))
    assert len(pool_texts) == 5165
    keep = np.array([n not in fset for n in pool_names]); removed_tasks = sorted(set(n for n, kp in zip(pool_names, keep) if not kp))
    Pall = np.load(os.path.join(NIC, "pool_minilm.npy")); P = Pall[keep]; ptexts = [t for t, kp in zip(pool_texts, keep) if kp]
    enc = None; F, tx = {}, {}
    for i, t in enumerate(fresh):
        texts = [str(x["input"]) for x in t["instances"]]; tf = os.path.join(AC, f"minilm_fresh3_{i}.npz")
        if not os.path.exists(tf):
            if enc is None: enc = minilm_enc()
            np.savez(tf, train=enc(texts[:800]), test=enc(texts[800:])); log("minilm fresh3", i, t["name"])
        z = np.load(tf); F[t["name"]] = (z["train"], z["test"]); tx[t["name"]] = (texts[:800], texts[800:])
    seq = Seq("fresh3", names, F, tx, P, ptexts)
    assert all(len(seq.spl[n][0]) == 200 and len(seq.spl[n][1]) == 600 for n in names)
    out = {"tasks": names, "pool_size": int(len(P)), "pool_removed": int((~keep).sum()), "pool_removed_tasks": removed_tasks}
    for label, v in (("chosen", (ch[0], ch[1], ch[2])), ("comparison_lr_C1_minilm", ("lr", "minilm", 1))):
        R, meta = routes_cached(seq, v); res = measures(seq, R, path=(label == "chosen"))
        mn, se = res["mean"], res["se"]
        use = "works" if mn["own_at_end"] >= 0.85 and mn["own_at_arrival"] >= 0.85 else ("does not" if mn["own_at_end"] < 0.70 or mn["own_at_arrival"] < 0.70 else "partial")
        lim = mn["certificate"] + 2 * se["changed"]
        ver = {"use": use, "certificate": "met" if mn["changed"] <= lim else "not met", "mean_own_at_end": mn["own_at_end"], "mean_own_at_arrival": mn["own_at_arrival"],
               "mean_changed": mn["changed"], "se_changed": se["changed"], "mean_certificate": mn["certificate"], "certificate_limit": lim}
        weak = {k: {"own_at_end": d["own_at_end"], "top_wrong_task": d.get("top_wrong_task"), "top_wrong_share_of_wrongly_routed": d.get("top_wrong_share")}
                for k, d in res["per_task"].items() if d["own_at_end"] < 0.7}
        out[label] = {"variant": vname(v), "verdict": ver, "weak_tasks": weak, "mean": mn, "se": se,
                      "per_task": {k: {m: d[m] for m in ("own_at_arrival", "own_at_end", "changed", "certificate", "c", "m")} for k, d in res["per_task"].items()}}
        if label == "chosen": out[label]["path"] = res["path"]
        if label == "chosen": RESULT["verdict_fresh3"] = ver
        RESULT["fresh3"] = out; dump(); log("fresh3", label, ver)
    RESULT["fresh3"] = out; dump()

if __name__ == "__main__":
    main()
