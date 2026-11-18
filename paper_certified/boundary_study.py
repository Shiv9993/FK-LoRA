"""Rule `boundary` (last section of silent_adapters/PLAN.md, "the target ... and the next idea. Fixed before any run").
Zones without instructions: a logistic-regression boundary (task k vs pool + earlier tasks) on frozen MiniLM features plus the knn10 test.
noinstruction_study.py cannot be imported (it runs the whole study at import), so the few functions it defines that are needed
(minilm_enc, qidx/split rules, knn10, ratio10, measures, verdict) are re-implemented here with the same definitions.
Run:  S=$FK_LORA_SCRATCH
  cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=3 \
      $S/mm_env/bin/python boundary_study.py
Writes routing/boundary_study.json after each part. Caches: $S/noinstruction/ (part 1 features, read only), $S/boundary/ (everything else)."""
import json, math, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
from sklearn.linear_model import LogisticRegression

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "routing", "boundary_study.json")
NIC = os.path.join(S, "noinstruction"); BC = os.path.join(S, "boundary"); os.makedirs(BC, exist_ok=True)
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
SEQS = {"superni_order1": list(range(0, 15)), "long_order3": list(range(15, 30)), "both": list(range(30))}
M = 200; MAXTOK = 256; LEVEL = 0.025; ALPHA = 0.05

# ---------------- helpers (same definitions as noinstruction_study.py)
def minilm_enc():
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

def make_split(n):
    perm = list(range(n)); random.Random(0).shuffle(perm)
    m = M if n >= 2 * M else n // 2
    return np.array(sorted(perm[:m]), dtype=int), np.array(sorted(perm[m:]), dtype=int)

def knn10(x, fit):
    return torch.cdist(torch.as_tensor(x), torch.as_tensor(fit)).kthvalue(10, dim=1).values.numpy()
def ratio10(x, fit, d0):
    dk = knn10(x, fit); out = np.full(len(x), np.inf); pos = d0 > 0
    out[pos] = dk[pos] / d0[pos]; out[(~pos) & (dk == 0)] = 0.0
    return out

def qidx(alpha, m): return math.ceil(round((1 - alpha) * (m + 1), 9))
def conf_q(cal, alpha):
    c = np.sort(cal); m = len(c); i = qidx(alpha, m)
    return float(c[min(i, m) - 1])          # if the index exceeds m, the largest score

def mse(v): v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))

def first_or_fb(inside, fb, T):
    ins = inside[:, :T + 1]
    return np.where(ins.any(1), ins.argmax(1), fb[:, :T + 1].argmax(1))

def measures(names, ins_te, fb_te, ins_fit, margin_te=None):
    """ins_te[j]: bool (n_test_j x K) inside zone k; fb_te[j]: float, larger = better fallback; ins_fit[j]: bool (n_fit_j x K)."""
    K = len(names); per = {}
    for j, kj in enumerate(names):
        ins, fb = ins_te[j], fb_te[j]; n = len(ins)
        r_arr = first_or_fb(ins, fb, j); r_end = first_or_fb(ins, fb, K - 1)
        own_in = ins[:, j]; ch = r_arr != r_end
        assert not (ch & own_in).any(), "changed among inside own zone"
        blk = ins[:, :j].any(1) if j else np.zeros(n, bool)
        blk_tr = ins_fit[j][:, :j].any(1) if j else np.zeros(len(ins_fit[j]), bool)
        d = {"coverage": float(own_in.mean()), "blocked": float(blk.mean()), "blocked_train": float(blk_tr.mean()),
             "own_at_arrival": float((r_arr == j).mean()), "own_at_end": float((r_end == j).mean()), "changed": float(ch.mean())}
        if margin_te is not None:
            d["recognition"] = float((margin_te[j].argmax(1) == j).mean())
        if j and blk.any():
            cnt = ins[blk][:, :j].sum(0); o = int(cnt.argmax())
            d["top_blocker"] = names[o]; d["top_blocker_share_of_blocked"] = float(cnt[o] / blk.sum()); d["top_blocker_share_of_test"] = float(ins[:, o].mean())
        per[kj] = d
    meas = [m for m in next(iter(per.values())) if not m.startswith("top_")]
    mean, se = {}, {}
    for m in meas: mean[m], se[m] = mse([per[k][m] for k in names])
    return {"per_task": per, "mean": mean, "se": se}

def verdict(res, lines_works, lines_not):
    mn, se = res["mean"], res["se"]
    claim = mn["changed"] <= ALPHA + 2 * se["changed"] and mn["coverage"] >= 1 - ALPHA - 2 * se["coverage"]
    (ow, bw), (on, bn) = lines_works, lines_not
    use = "works" if mn["own_at_end"] >= ow and mn["blocked"] <= bw else ("does not" if mn["own_at_end"] < on or mn["blocked"] > bn else "partial")
    return {"claim": "holds" if claim else "fails", "use": use, "mean_changed": mn["changed"], "se_changed": se["changed"],
            "changed_limit": ALPHA + 2 * se["changed"], "mean_coverage": mn["coverage"], "se_coverage": se["coverage"],
            "coverage_limit": 1 - ALPHA - 2 * se["coverage"], "mean_own_at_end": mn["own_at_end"], "mean_blocked": mn["blocked"],
            "lines": {"works": {"own_at_end_at_least": ow, "blocked_at_most": bw}, "does_not": {"own_at_end_below": on, "or_blocked_above": bn}}}

def brief(res):
    return {"mean": res["mean"], "se": res["se"]}
def per_task_list(res):
    out = {}
    for k, d in res["per_task"].items():
        e = {m: d[m] for m in ("coverage", "blocked", "own_at_end")}
        if d["blocked"] >= 0.2 and "top_blocker" in d:
            e.update({m: d[m] for m in ("top_blocker", "top_blocker_share_of_blocked", "top_blocker_share_of_test")})
        out[k] = e
    return out

# ---------------- the experiment on one sequence
def run_sequence(tag, names, F, spl, P):
    """F[k] = (train, test) features; spl[k] = (cal idx, fit idx); P = pool features. Returns results for boundary, knn10, ratio10."""
    K = len(names)
    cal = {k: F[k][0][spl[k][0]] for k in names}; fit = {k: F[k][0][spl[k][1]] for k in names}; te = {k: F[k][1] for k in names}
    # --- boundary
    s1_te = [np.zeros((len(te[kj]), K)) for kj in names]; s2_te = [np.zeros((len(te[kj]), K)) for kj in names]
    C1 = [np.zeros((len(cal[kj]), K)) for kj in names]     # a_i of every task's calibration inputs (for the rule `claims`)
    s1_fit = [np.zeros((len(fit[kj]), K)) for kj in names]; s2_fit = [np.zeros((len(fit[kj]), K)) for kj in names]
    q1 = np.zeros(K); q2 = np.zeros(K)
    for i, ki in enumerate(names):
        cf = os.path.join(BC, f"bscores2_{tag}_{i}_{ki.replace('/', '_')}.npz")
        if os.path.exists(cf):
            z = np.load(cf)
        else:
            X = np.vstack([fit[ki], P] + [fit[names[e]] for e in range(i)])
            y = np.r_[np.ones(len(fit[ki])), np.zeros(len(X) - len(fit[ki]))]
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X, y)
            sc1 = lambda x: -clf.decision_function(x)
            d = {"cal1": sc1(cal[ki]), "cal2": knn10(cal[ki], fit[ki])}
            for j, kj in enumerate(names):
                d[f"te1_{j}"] = sc1(te[kj]); d[f"te2_{j}"] = knn10(te[kj], fit[ki])
                d[f"fit1_{j}"] = sc1(fit[kj]); d[f"fit2_{j}"] = knn10(fit[kj], fit[ki]); d[f"calall1_{j}"] = sc1(cal[kj])
            d["iters"] = np.array(int(np.max(clf.n_iter_)))
            np.savez(cf, **d); z = np.load(cf)
            log(tag, "boundary task", i, ki, "n_neg", len(X) - len(fit[ki]), "iters", int(z["iters"]))
        q1[i] = conf_q(z["cal1"], LEVEL); q2[i] = conf_q(z["cal2"], LEVEL)
        for j in range(K):
            s1_te[j][:, i] = z[f"te1_{j}"]; s2_te[j][:, i] = z[f"te2_{j}"]; C1[j][:, i] = z[f"calall1_{j}"]; s1_fit[j][:, i] = z[f"fit1_{j}"]; s2_fit[j][:, i] = z[f"fit2_{j}"]
    ins_te = [(s1_te[j] <= q1) & (s2_te[j] <= q2) for j in range(K)]
    ins_fit = [(s1_fit[j] <= q1) & (s2_fit[j] <= q2) for j in range(K)]
    marg = [q1 - s1_te[j] for j in range(K)]
    out = {"boundary": measures(names, ins_te, marg, ins_fit, margin_te=marg)}
    for j, kj in enumerate(names):
        out["boundary"]["per_task"][kj].update({"q1": float(q1[j]), "q2": float(q2[j])})
    out["claims"] = claims_measures(names, s1_te, C1, cal, LEVEL)
    # --- earlier scores, level 0.05, same data and splits
    d0 = {"cal": {}, "fit": {}, "te": {}}
    for k in names: d0["cal"][k] = knn10(cal[k], P); d0["fit"][k] = knn10(fit[k], P); d0["te"][k] = knn10(te[k], P)
    for score in ("knn10", "ratio10"):
        g = (lambda x, f, k, w: knn10(x, f)) if score == "knn10" else (lambda x, f, k, w: ratio10(x, f, d0[w][k]))
        q = np.array([conf_q(g(cal[k], fit[k], k, "cal"), ALPHA) for k in names])
        Ste = [np.stack([g(te[kj], fit[k], kj, "te") for k in names], 1) for kj in names]
        Sft = [np.stack([g(fit[kj], fit[k], kj, "fit") for k in names], 1) for kj in names]
        out[score] = measures(names, [s <= q for s in Ste], [-(s / q) for s in Ste], [s <= q for s in Sft])
        log(tag, score, "done")
    return out

def claims_measures(names, a_te, C1, cal, level):
    """Rule `claims`: a_k = s1 of the same classifier as `boundary`; q_k = min(own conformal quantile, smallest a_k over the stored calibration inputs of ALL earlier tasks); zone a_k < q_k; newest zone wins."""
    K = len(names); q = np.zeros(K); q_own = np.zeros(K); q_old = np.full(K, np.inf); old_src = [None] * K
    for i in range(K):
        q_own[i] = conf_q(C1[i][:, i], level)
        for j in range(i):
            v = float(C1[j][:, i].min())
            if v < q_old[i]: q_old[i] = v; old_src[i] = names[j]
        q[i] = min(q_own[i], q_old[i])
    def newest(ins, marg, T):
        ii = ins[:, :T + 1]; last = T - ii[:, ::-1].argmax(1)
        return np.where(ii.any(1), last, marg[:, :T + 1].argmax(1))
    per = {}
    for j, kj in enumerate(names):
        a = a_te[j]; ins = a < q; marg = q - a; n = len(a)
        r_arr = newest(ins, marg, j); r_end = newest(ins, marg, K - 1)
        own_in = ins[:, j]; later = ins[:, j + 1:].any(1) if j < K - 1 else np.zeros(n, bool); ch = r_arr != r_end
        assert not (own_in & ~later & ch).any(), "claims: route changed although inside own zone and in no later zone"
        assert (r_arr[own_in] == j).all()
        m = len(cal[kj]); cal_cov = float((C1[j][:, j] < q[j]).mean()); nlater = K - 1 - j
        d = {"coverage_at_arrival": float(own_in.mean()), "own_at_arrival": float((r_arr == j).mean()), "own_at_end": float((r_end == j).mean()),
             "changed": float(ch.mean()), "taken": float(later.mean()),
             "binding": "old" if q_old[j] < q_own[j] else "own", "q": float(q[j]), "q_own": float(q_own[j]), "q_old": None if j == 0 else float(q_old[j]),
             "binding_older_task": old_src[j] if q_old[j] < q_own[j] else None, "old_threshold_source_task": old_src[j],
             "cal_coverage": cal_cov, "n_cal": m, "n_later": nlater,
             "bound": float((1 - own_in.mean()) + nlater / (m + 1)), "bound_cal": float((1 - cal_cov) + nlater / (m + 1))}
        per[kj] = d
    meas = [k for k, v in per[names[0]].items() if isinstance(v, float) or isinstance(v, int)]
    meas = [k for k in meas if k not in ("q", "q_own", "q_old", "n_cal", "n_later")]
    mean, se = {}, {}
    for k in meas: mean[k], se[k] = mse([per[t][k] for t in names])
    mean["share_binding_old"] = float(np.mean([per[t]["binding"] == "old" for t in names]))
    return {"per_task": per, "mean": mean, "se": se}

def claims_verdict(res, lines_works, lines_not):
    mn, se = res["mean"], res["se"]
    guar = mn["changed"] <= mn["bound"] + 2 * se["changed"]
    guar_cal = mn["changed"] <= mn["bound_cal"] + 2 * se["changed"]
    w, nt = lines_works, lines_not
    lo = min(mn["own_at_end"], mn["own_at_arrival"])
    use = "works" if mn["own_at_end"] >= w and mn["own_at_arrival"] >= w else ("does not" if lo < nt else "partial")
    return {"guarantee": "met" if guar else "not met", "guarantee_with_calibration_coverage_bound": "met" if guar_cal else "not met", "use": use,
            "mean_changed": mn["changed"], "se_changed": se["changed"], "mean_bound": mn["bound"], "mean_bound_cal": mn["bound_cal"],
            "mean_own_at_end": mn["own_at_end"], "mean_own_at_arrival": mn["own_at_arrival"], "lines": {"works": w, "does_not_below": nt}}

def claims_per_task(res):
    out = {}
    for k, d in res["per_task"].items():
        e = {m: d[m] for m in ("coverage_at_arrival", "own_at_arrival", "own_at_end", "changed", "taken", "binding")}
        if d["coverage_at_arrival"] < 0.8: e["earlier_task_that_set_the_threshold"] = d["old_threshold_source_task"] if d["binding"] == "old" else None
        out[k] = e
    return out

def summarize(res):
    return {sc: brief(r) for sc, r in res.items()}

RESULT = {"notes": [
    "Rule boundary: LogisticRegression(C=1, class_weight=balanced, max_iter=2000) on unscaled MiniLM features, positives = fit inputs of task k, negatives = pool + fit inputs of earlier tasks of the sequence; score one = minus decision value, score two = knn10 distance to the fit inputs of task k; thresholds = split-conformal quantile at level 0.025 on the task's calibration inputs (index ceil((1-0.025)(m+1)), largest score if it exceeds m); zone = both scores at or below; fallback = largest margin (q1 - score one). Oldest zone wins.",
    "knn10 and ratio10 comparisons: level 0.05, same data/splits/pool, fallback argmin of score/threshold, as in noinstruction_study.py.",
    "Means and standard errors over the tasks of the sequence (std ddof=1/sqrt(n)). Part 1 data have been used before (development); part 2 data are fresh (verdict).",
    "recognition = share of test inputs whose largest margin (q1 - score one) among ALL tasks of the sequence is the own task; no zones.",
    "Part 1 splits: as in noinstruction_study.py (seed 0, 200 calibration or half if fewer than 400 training inputs). Part 1 pool: the 5,165 pool inputs (ni_tasks.json without the 15 SuperNI tasks), cached MiniLM features.",
    "Definitions of tasks are never read.",
    "Rule claims (PLAN.md 06:20): same classifier/scores as boundary (a_k = minus decision value); q_k = min(conformal quantile at 0.025 on own calibration inputs, smallest a_k over ALL stored calibration inputs of ALL earlier tasks); zone a_k < q_k strictly; newest zone wins; fallback largest q_k - a_k. bound_j = (1 - coverage_at_arrival on test) + (#later tasks)/(m_j+1); bound_cal uses the coverage on the calibration inputs. guarantee uses bound (test coverage); the calibration-coverage version is reported too. Calibration inputs are never used for training."]}
def dump():
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(RESULT, open(OUT, "w"), indent=1)

# ================= PART 1
def part1():
    P = np.load(os.path.join(NIC, "pool_minilm.npy")); log("pool", P.shape)
    assert len(P) == 5165
    F = {}
    for k in KEY:
        z = np.load(os.path.join(NIC, "minilm_" + k.replace("/", "_") + ".npz")); F[k] = (z["train"], z["test"])
    spl = {k: make_split(len(F[k][0])) for k in KEY}
    RESULT["part1"] = {"pool_size": len(P), "n": {k: {"n_train": len(F[k][0]), "n_cal": len(spl[k][0]), "n_fit": len(spl[k][1]), "n_test": len(F[k][1])} for k in KEY}, "sequences": {}}
    for sq, idx in SEQS.items():
        names = [KEY[i] for i in idx]
        res = run_sequence("p1_" + sq, names, F, spl, P)
        RESULT["part1"]["sequences"][sq] = {"summary": summarize(res), "per_task": {sc: (claims_per_task(r) if sc == "claims" else per_task_list(r)) for sc, r in res.items()},
                                            "full": res}
        if sq == "superni_order1":
            RESULT["verdict_dev"] = {"sequence": "superni_order1", "rule": "boundary", **verdict(res["boundary"], (0.90, 0.10), (0.75, 0.25))}
            RESULT["verdict_claims_dev"] = {"sequence": "superni_order1", "rule": "claims", **claims_verdict(res["claims"], 0.90, 0.75)}
        RESULT.setdefault("guarantee_claims_by_sequence", {})[sq] = {k: v for k, v in claims_verdict(res["claims"], 0.90, 0.75).items() if k.startswith("guarantee") or k.startswith("mean_")}
        dump(); log("part1 written", sq, {sc: {m: round(r["mean"][m], 3) for m in ("coverage", "coverage_at_arrival", "blocked", "own_at_end", "changed") if m in r["mean"]} for sc, r in res.items()})

# ================= PART 2
def part2():
    fresh = json.load(open(os.path.join(S, "ni_fresh", "fresh_tasks.json")))["tasks"]
    names = [t["name"] for t in fresh]; assert len(names) == 30 and all(len(t["instances"]) == 1000 for t in fresh)
    fset = set(names)
    pool_names, pool_texts = [], []
    for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"]:
        if t["name"] in set(ORDER1): continue
        for x in t["inputs"]: pool_names.append(t["name"]); pool_texts.append(str(x))
    assert len(pool_texts) == 5165, len(pool_texts)
    keep = np.array([n not in fset for n in pool_names]); removed = int((~keep).sum())
    removed_tasks = sorted(set(n for n, kp in zip(pool_names, keep) if not kp))
    log("pool removed", removed, "from", removed_tasks)
    Pall = np.load(os.path.join(NIC, "pool_minilm.npy")); assert len(Pall) == 5165
    P = Pall[keep]
    enc = None; F = {}; spl = {}
    for i, t in enumerate(fresh):
        tf = os.path.join(BC, f"minilm_fresh_{i}.npz")
        if not os.path.exists(tf):
            if enc is None: enc = minilm_enc()
            txt = [str(x["input"]) for x in t["instances"]]
            np.savez(tf, train=enc(txt[:800]), test=enc(txt[800:])); log("minilm fresh", i, t["name"])
        z = np.load(tf); F[t["name"]] = (z["train"], z["test"]); spl[t["name"]] = make_split(800)
    assert all(len(spl[n][0]) == 200 and len(spl[n][1]) == 600 and len(F[n][1]) == 200 for n in names)
    res = run_sequence("p2", names, F, spl, P)
    RESULT["part2"] = {"pool_size": int(len(P)), "pool_removed": removed, "pool_removed_tasks": removed_tasks, "tasks": names,
                       "summary": summarize(res), "per_task": {sc: (claims_per_task(r) if sc == "claims" else per_task_list(r)) for sc, r in res.items()}, "full": res}
    RESULT["verdict_claims_fresh"] = {"rule": "claims", **claims_verdict(res["claims"], 0.85, 0.70)}
    RESULT["verdict_fresh"] = {"rule": "boundary", **verdict(res["boundary"], (0.85, 0.15), (0.70, 0.30))}
    dump(); log("part2 written", {sc: {m: round(r["mean"][m], 3) for m in ("coverage", "coverage_at_arrival", "blocked", "own_at_end", "changed") if m in r["mean"]} for sc, r in res.items()})


# ====================================================================================================================
# STEP A / STEP B (PLAN.md 07:00 "how the next version is chosen"): variants inside the rule `claims`
#   feature: minilm | minilm+shape(w in 0.25, 0.5, 1.0); kind: ovr | pair; r in 1, 2, 3; C in 1, 10, 100.
# Notes on interpretation (not tuned): the shape list in the plan names 16 numbers (7 shares/lengths, 5 punctuation rates, line breaks,
# 3 flags) while saying 15; all 16 listed numbers are used. For kind `pair` the "sum of margins" of a task a in the tie-break is the sum over
# the other tasks b (that have arrived) of the signed pairwise decision value in favour of a. For `ovr` the zone is a < q_old (strict) and
# a <= q_own, the fallback margin is min(q_own, q_old) - a.
# ====================================================================================================================
SHAPE_NAMES = ["log_chars", "log_words", "mean_word_len", "upper_share_of_letters", "digit_share", "space_share", "other_share",
               "per100_period", "per100_question", "per100_exclaim", "per100_comma", "per100_colon", "log_linebreaks", "has_at", "has_hash", "has_http"]
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

def pool_texts_all():
    return [str(x) for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"] if t["name"] not in set(ORDER1) for x in t["inputs"]]

def shape_cached(fn, texts):
    if os.path.exists(fn): return np.load(fn)["v"]
    v = np.array([shape_vec(t) for t in texts]); np.savez(fn, v=v); return v

_POOLSTAT = {}
def pool_shape_stats():
    if not _POOLSTAT:
        v = shape_cached(os.path.join(BC, "shape_pool.npz"), pool_texts_all())
        sd = v.std(0); sd[sd == 0] = 1.0; _POOLSTAT["m"] = v.mean(0); _POOLSTAT["s"] = sd
    return _POOLSTAT["m"], _POOLSTAT["s"]
def shape_block(v, w):
    m, sd = pool_shape_stats(); return w * np.clip((v - m) / sd, -5, 5)

FEATS = [("minilm", None), ("minilm+shape0.25", 0.25), ("minilm+shape0.5", 0.5), ("minilm+shape1.0", 1.0)]
CS = [1, 10, 100]; RS = [1, 2, 3]; KINDS = ["ovr", "pair", "pair-knn", "pair-rff"]

def fit_lr(X, y, C): return LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y)

def seq_scores(tag, names, F, spl, P, C, tf=None):
    """F[k] = (train, test) features of the variant's feature; returns dict with row layout: all test rows of the tasks in order, then all calibration rows."""
    K = len(names); cal = {k: F[k][0][spl[k][0]] for k in names}; fit = {k: F[k][0][spl[k][1]] for k in names}; te = {k: F[k][1] for k in names}
    Xall = np.vstack([te[k] for k in names] + [cal[k] for k in names]); N = len(Xall)
    nte = np.cumsum([0] + [len(te[k]) for k in names]); ncal = len(Xall) - nte[-1]; ccal = nte[-1] + np.cumsum([0] + [len(cal[k]) for k in names])
    ovr = np.zeros((N, K)); pool = np.zeros((N, K)); pair = []
    T = tf if tf is not None else (lambda x: x)
    def fitp(X, y):
        clf = fit_lr(T(X), y, C)
        return lambda Z: np.concatenate([T(Z[a:a + 4000]) @ clf.coef_[0] + clf.intercept_[0] for a in range(0, len(Z), 4000)])
    for i, ki in enumerate(names):
        cf = os.path.join(BC, f"sc4_{tag}_C{C}_{i}_{ki.replace('/', '_')}.npz")
        if os.path.exists(cf): z = np.load(cf)
        else:
            t0 = time.time(); d = {}
            X = np.vstack([fit[ki], P]); y = np.r_[np.ones(len(fit[ki])), np.zeros(len(P))]; d["pool"] = -fitp(X, y)(Xall)
            if i:
                X = np.vstack([fit[ki], P] + [fit[names[e]] for e in range(i)]); y = np.r_[np.ones(len(fit[ki])), np.zeros(len(X) - len(fit[ki]))]
                d["ovr"] = -fitp(X, y)(Xall)
            else: d["ovr"] = d["pool"]
            d["pair"] = np.zeros((N, i))
            for j in range(i):
                X = np.vstack([fit[ki], fit[names[j]]]); y = np.r_[np.ones(len(fit[ki])), np.zeros(len(fit[names[j]]))]
                d["pair"][:, j] = fitp(X, y)(Xall)
            np.savez(cf, **d); z = np.load(cf); log(tag, "C", C, "scores task", i, ki, f"{time.time() - t0:.0f}s")
        ovr[:, i] = z["ovr"]; pool[:, i] = z["pool"]; pair.append(z["pair"])
    return {"ovr": ovr, "pool": pool, "pair": pair, "nte": nte, "ccal": ccal, "m": [len(cal[k]) for k in names]}

def eval_variant(names, Sc, kind, r, level=LEVEL):
    K = len(names); nte, ccal, m = Sc["nte"], Sc["ccal"], Sc["m"]
    calrows = lambda j: slice(ccal[j], ccal[j + 1]); terows = lambda j: slice(nte[j], nte[j + 1])
    if kind == "ovr":
        a = Sc["ovr"]; q_own = np.array([conf_q(a[calrows(i), i], level) for i in range(K)])
        q_old = np.array([min([np.sort(a[calrows(j), i])[r - 1] for j in range(i)], default=np.inf) for i in range(K)])
        qe = np.minimum(q_own, q_old)
        ins_all = (a[:nte[-1]] < q_old) & (a[:nte[-1]] <= q_own); marg_all = qe - a[:nte[-1]]
    else:
        p = Sc["pool"]; q_own = np.array([conf_q(p[calrows(i), i], level) for i in range(K)])
        ins_all = p[:nte[-1]] <= q_own
        for k in range(1, K):
            for j in range(k):
                theta = np.sort(Sc["pair"][k][calrows(j), j])[::-1][r - 1]
                ins_all[:, k] &= Sc["pair"][k][:nte[-1], j] > theta
        marg_all = None
    def fb_pair(D, T):
        sub = D[:, :T + 1, :T + 1]; wins = (sub > 0).sum(2); ssum = sub.sum(2)
        best = wins == wins.max(1, keepdims=True)
        return np.where(best, ssum, -np.inf).argmax(1)
    per = {}
    for j, kj in enumerate(names):
        ins = ins_all[terows(j)]; n = len(ins)
        if kind == "ovr":
            marg = marg_all[terows(j)]; fb = lambda T: marg[:, :T + 1].argmax(1)
        else:
            D = np.zeros((n, K, K))
            for a_ in range(1, K):
                blk = Sc["pair"][a_][terows(j)]; D[:, a_, :a_] = blk; D[:, :a_, a_] = -blk
            fb = lambda T: fb_pair(D, T)
        def route(T):
            ii = ins[:, :T + 1]; last = T - ii[:, ::-1].argmax(1)
            return np.where(ii.any(1), last, fb(T))
        r_arr, r_end = route(j), route(K - 1); own_in = ins[:, j]
        later = ins[:, j + 1:].any(1) if j < K - 1 else np.zeros(n, bool); ch = r_arr != r_end
        assert not (own_in & ~later & ch).any(), "route changed although inside own zone and in no later zone"
        assert (r_arr[own_in] == j).all()
        rec = fb(K - 1); nl = K - 1 - j
        wrong = r_end[r_end != j]
        d = {"coverage_at_arrival": float(own_in.mean()), "own_at_arrival": float((r_arr == j).mean()), "own_at_end": float((r_end == j).mean()),
             "changed": float(ch.mean()), "taken": float(later.mean()), "recognition": float((rec == j).mean()),
             "bound": float((1 - own_in.mean()) + r * nl / (m[j] + 1))}
        if len(wrong): c = np.bincount(wrong, minlength=K); d["top_wrong_destination"] = names[int(c.argmax())]; d["top_wrong_destination_share"] = float(c.max() / len(wrong)); d["top_wrong_destination_is_older"] = bool(c.argmax() < j)
        per[kj] = d
    meas = ["coverage_at_arrival", "own_at_arrival", "own_at_end", "changed", "taken", "recognition", "bound"]
    mean, se = {}, {}
    for k in meas: mean[k], se[k] = mse([per[t][k] for t in names])
    return {"per_task": per, "mean": mean, "se": se}

def variant_features(feat, w, names, base_F, texts):
    """base_F[k] = (train, test) minilm; texts[k] = (train texts, test texts); returns features of the variant."""
    if w is None: return base_F
    out = {}
    for k in names:
        out[k] = tuple(np.hstack([base_F[k][s], shape_block(texts[k][s], w)]) for s in (0, 1))
    return out

def pool_variant(Pm, Pshape, w, keep=None):
    P = Pm if w is None else np.hstack([Pm, shape_block(Pshape, w)])
    return P if keep is None else P[keep]

def part1_inputs():
    """minilm features, split, shape numbers of the part-1 tasks (texts read from CL_Benchmark; the definitions are not read)."""
    F = {}; shp = {}
    for (f, n), k in zip(TASKS, KEY):
        z = np.load(os.path.join(NIC, "minilm_" + k.replace("/", "_") + ".npz")); F[k] = (z["train"], z["test"])
        fn = os.path.join(BC, "shape_p1_" + k.replace("/", "_") + ".npz")
        if os.path.exists(fn): zz = np.load(fn); shp[k] = (zz["train"], zz["test"])
        else:
            tx = [[str(i["input"]) for i in json.load(open(os.path.join("CL_Benchmark", f, n, sp + ".json")))["Instances"]] for sp in ("train", "test")]
            assert len(tx[0]) == len(F[k][0]) and len(tx[1]) == len(F[k][1]), k
            shp[k] = (np.array([shape_vec(t) for t in tx[0]]), np.array([shape_vec(t) for t in tx[1]])); np.savez(fn, train=shp[k][0], test=shp[k][1])
            log("shape", k)
    return F, shp, {k: make_split(len(F[k][0])) for k in KEY}

GRID_SEQS = ("superni_order1",)   # the Long Sequence part of the grid was not computed (decided by the lead for lack of CPU time)
PARTIAL = os.path.join(BC, "devgrid_partial.json")
def make_rff(Pv, D=2048):
    ps = np.random.default_rng(0).choice(len(Pv), 2000, replace=False); Q = torch.as_tensor(Pv[ps])
    dm = torch.cdist(Q, Q); iu = torch.triu_indices(2000, 2000, offset=1); sigma = float(dm[iu[0], iu[1]].median())
    rng = np.random.default_rng(0); W = rng.normal(0.0, 1.0 / sigma, size=(D, Pv.shape[1])); b = rng.uniform(0, 2 * np.pi, size=D)
    return (lambda v: math.sqrt(2.0 / D) * np.cos(v @ W.T + b)), sigma

def seq_scores_knn(tag, names, F, spl, P):
    """pair-knn scores: d_t(x) = distance to the 10th nearest fit input of task t (d_0: pool). pool score a = -(d_0 - d_k), pair f_kj = d_j - d_k."""
    K = len(names); cal = {k: F[k][0][spl[k][0]] for k in names}; fit = {k: F[k][0][spl[k][1]] for k in names}; te = {k: F[k][1] for k in names}
    Xall = np.vstack([te[k] for k in names] + [cal[k] for k in names]); N = len(Xall)
    nte = np.cumsum([0] + [len(te[k]) for k in names]); ccal = nte[-1] + np.cumsum([0] + [len(cal[k]) for k in names])
    def dist(ref):
        return np.concatenate([torch.cdist(torch.as_tensor(Xall[a:a + 3000]), torch.as_tensor(ref)).kthvalue(10, dim=1).values.numpy() for a in range(0, N, 3000)])
    d = []
    for t in range(-1, K):
        cf = os.path.join(BC, f"knn_{tag}_{t}.npy")
        if os.path.exists(cf): d.append(np.load(cf))
        else: d.append(dist(P if t < 0 else fit[names[t]])); np.save(cf, d[-1]); log(tag, "knn distances", t)
    d0, dt = d[0], d[1:]
    pool = np.stack([-(d0 - dt[k]) for k in range(K)], 1); pair = [np.stack([dt[j] - dt[k] for j in range(k)], 1) if k else np.zeros((N, 0)) for k in range(K)]
    return {"ovr": None, "pool": pool, "pair": pair, "nte": nte, "ccal": ccal, "m": [len(cal[k]) for k in names]}

def stepA():
    Pm = np.load(os.path.join(NIC, "pool_minilm.npy")); Pshape = shape_cached(os.path.join(BC, "shape_pool.npz"), pool_texts_all())
    F, shp, spl = part1_inputs(); log("step A inputs ready")
    grid = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    done = grid.setdefault("_done", {})
    def inputs(sq, w):
        names = [KEY[i] for i in SEQS[sq]]
        Fv = {k: tuple(np.hstack([F[k][s], shape_block(shp[k][s], w)]) if w is not None else F[k][s] for s in (0, 1)) for k in names}
        return names, Fv, pool_variant(Pm, Pshape, w)
    def store(feat, kind, r, C, sq, res):
        grid.setdefault(f"{feat}|{kind}|r{r}|C{C}", {})[sq] = res
    def scores(kind, tag, names, Fv, Pv, C):
        if kind in ("ovr", "pair"): return seq_scores(tag, names, Fv, spl, Pv, C)
        if kind == "pair-knn": return seq_scores_knn(tag, names, Fv, spl, Pv)
        tf, sig = make_rff(Pv); log("rff sigma", sig); return seq_scores(tag + "-rff", names, Fv, spl, Pv, C, tf)
    # ---- stage 1: r = 1 for every variant
    for sq in GRID_SEQS:
        for feat, w in FEATS:
            names, Fv, Pv = inputs(sq, w)
            for kind, Cs in (("ovr", CS), ("pair", CS), ("pair-knn", ["na"]), ("pair-rff", CS)):
                for C in Cs:
                    key = f"1|{feat}|{kind}|C{C}|{sq}"
                    if key in done: continue
                    t0 = time.time(); Sc = scores(kind, f"p1_{sq}_{feat}", names, Fv, Pv, C if C != "na" else 0)
                    if kind == "ovr" or True:
                        store(feat, kind, 1, C, sq, eval_variant(names, Sc, "ovr" if kind == "ovr" else "pair", 1))
                    done[key] = True; json.dump(grid, open(PARTIAL, "w")); log("grid", key, f"{time.time() - t0:.0f}s")
    # ---- stage 2 happens in stage2() (r = 2, 3 only if stage 1 was slower than the 25 minute budget; see stepA_main)
    return grid

def stage2(grid, top=None):
    """r = 2 and 3 for the best two (feature, kind, C) on SuperNI at r = 1 (rule of the choice: changed <= 0.03, then own_at_end)."""
    Pm = np.load(os.path.join(NIC, "pool_minilm.npy")); Pshape = shape_cached(os.path.join(BC, "shape_pool.npz"), pool_texts_all())
    F, shp, spl = part1_inputs(); done = grid["_done"]
    c = []
    for vid, v in grid.items():
        if vid == "_done" or not vid.split("|")[2].startswith("r1") or "superni_order1" not in v: continue
        sn = v["superni_order1"]["mean"]; c.append((vid, sn, {"own_at_end": 0.0}))
    ok = sorted([x for x in c if x[1]["changed"] <= 0.03], key=lambda x: (-x[1]["own_at_end"], -x[2]["own_at_end"]))[:2]
    log("stage 2 for", [x[0] for x in ok])
    for vid, _, _ in ok:
        feat, kind, _, CC = vid.split("|"); w = dict(FEATS)[feat]
        for sq in GRID_SEQS:
            names = [KEY[i] for i in SEQS[sq]]
            Fv = {k: tuple(np.hstack([F[k][s], shape_block(shp[k][s], w)]) if w is not None else F[k][s] for s in (0, 1)) for k in names}; Pv = pool_variant(Pm, Pshape, w)
            Cn = 0 if CC == "Cna" else int(CC[1:])
            if kind in ("ovr", "pair"): Sc = seq_scores(f"p1_{sq}_{feat}", names, Fv, spl, Pv, Cn)
            elif kind == "pair-knn": Sc = seq_scores_knn(f"p1_{sq}_{feat}", names, Fv, spl, Pv)
            else: tf, _s = make_rff(Pv); Sc = seq_scores(f"p1_{sq}_{feat}-rff", names, Fv, spl, Pv, Cn, tf)
            for r in (2, 3):
                grid.setdefault(f"{feat}|{kind}|r{r}|{CC}", {})[sq] = eval_variant(names, Sc, "ovr" if kind == "ovr" else "pair", r)
        json.dump(grid, open(PARTIAL, "w")); log("stage 2 written", vid)

def choose(grid):
    cands = []
    for vid, v in grid.items():
        if vid == "_done" or "superni_order1" not in v: continue
        feat, kind, rr, CC = vid.split("|"); r = int(rr[1:]); C = 0 if CC == "Cna" else int(CC[1:]); sn = v["superni_order1"]["mean"]; lg = {"own_at_end": 0.0}   # Long part not computed: tie-break skipped
        w = 0.0 if feat == "minilm" else float(feat.replace("minilm+shape", ""))
        cands.append((vid, sn, lg, r, kind, C, w))
    ok = [c for c in cands if c[1]["changed"] <= 0.03]
    if not ok: return None, cands
    ok.sort(key=lambda c: (-c[1]["own_at_end"], -c[2]["own_at_end"], c[3], c[4] != "ovr", c[5], c[6]))
    return ok[0][0], cands

def finalize_A(grid):
    best, cands = choose(grid)
    RESULT.update(json.load(open(OUT))) if os.path.exists(OUT) else None
    RESULT["notes"].append("The Long Sequence part of the development grid was not computed for lack of CPU time (decided by the lead before any grid number was read), so the tie-break 'higher on the Long Sequence' is skipped and ties go to the next tie-break (smaller r, ovr, smaller C, minilm). Stage 2 (r = 2, 3 for the best two (feature, kind, C) at r = 1) was run on the SuperNI sequence only. The shape feature has the 16 numbers the plan lists (the plan says 15).")
    RESULT["dev_grid"] = {vid: {sq: {"mean": res["mean"], "se": res["se"]} for sq, res in v.items()} for vid, v in grid.items() if vid != "_done"}
    top = sorted(cands, key=lambda c: -c[1]["own_at_end"])[:12]
    RESULT["dev_grid_top12_superni"] = [c[0] for c in top]
    RESULT["chosen"] = None if best is None else {"variant": best, "rule": "highest mean own_at_end on SuperNI among variants with mean changed <= 0.03; ties: Long own_at_end, smaller r, ovr, smaller C, minilm",
                                                  "per_sequence": grid[best], "note": "SuperNI only"}
    dump()
    return best

def stepB():
    RESULT.update(json.load(open(OUT)))
    chosen = RESULT["chosen"]["variant"]; feat, kind, rr, CC = chosen.split("|"); r = int(rr[1:]); C = 0 if CC == "Cna" else int(CC[1:]); w = dict(FEATS)[feat]
    log("step B chosen", chosen)
    fresh = json.load(open(os.path.join(S, "ni_fresh", "fresh_tasks_2.json")))["tasks"]      # read once, here
    names = [t["name"] for t in fresh]; assert len(names) == 30 and all(len(t["instances"]) == 1000 for t in fresh)
    fset = set(names); pool_names = []
    for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"]:
        if t["name"] in set(ORDER1): continue
        pool_names += [t["name"]] * len(t["inputs"])
    assert len(pool_names) == 5165
    keep = np.array([n not in fset for n in pool_names]); removed = int((~keep).sum()); log("pool removed", removed)
    Pm = np.load(os.path.join(NIC, "pool_minilm.npy")); Pshape = shape_cached(os.path.join(BC, "shape_pool.npz"), pool_texts_all())
    enc = None; F = {}; shp = {}; spl = {}
    for i, t in enumerate(fresh):
        txt = [str(x["input"]) for x in t["instances"]]
        tf = os.path.join(BC, f"minilm_fresh2_{i}.npz")
        if not os.path.exists(tf):
            if enc is None: enc = minilm_enc()
            np.savez(tf, train=enc(txt[:800]), test=enc(txt[800:])); log("minilm fresh2", i, t["name"])
        z = np.load(tf); F[t["name"]] = (z["train"], z["test"])
        sf = os.path.join(BC, f"shape_fresh2_{i}.npz")
        if not os.path.exists(sf): np.savez(sf, train=np.array([shape_vec(x) for x in txt[:800]]), test=np.array([shape_vec(x) for x in txt[800:]]))
        z = np.load(sf); shp[t["name"]] = (z["train"], z["test"]); spl[t["name"]] = make_split(800)
    def inputs(w):
        Fv = {k: tuple(np.hstack([F[k][s], shape_block(shp[k][s], w)]) if w is not None else F[k][s] for s in (0, 1)) for k in names}
        return Fv, pool_variant(Pm, Pshape, w, keep)
    Fv, Pv = inputs(w)
    if kind in ("ovr", "pair"): Sc = seq_scores("p3_fresh2_" + feat, names, Fv, spl, Pv, C)
    elif kind == "pair-knn": Sc = seq_scores_knn("p3_fresh2_" + feat, names, Fv, spl, Pv)
    else: tf_, sig = make_rff(Pv); Sc = seq_scores("p3_fresh2_" + feat + "-rff", names, Fv, spl, Pv, C, tf_)
    res = eval_variant(names, Sc, "ovr" if kind == "ovr" else "pair", r)
    # comparison: the plain rule `claims` of before (minilm, ovr, r = 1, C = 1; claims_measures) and its fallback alone
    F0, P0 = inputs(None); Sc0 = seq_scores("p3_fresh2_minilm", names, F0, spl, P0, 1)
    K = len(names); nte, ccal = Sc0["nte"], Sc0["ccal"]
    a_te = [Sc0["ovr"][nte[j]:nte[j + 1]] for j in range(K)]; C1 = [Sc0["ovr"][ccal[j]:ccal[j + 1]] for j in range(K)]
    plain = claims_measures(names, a_te, C1, {names[j]: np.zeros(len(C1[j])) for j in range(K)}, LEVEL)
    plain_v = eval_variant(names, Sc0, "ovr", 1)
    mn, se = res["mean"], res["se"]
    guar = mn["changed"] <= mn["bound"] + 2 * se["changed"]
    use = "works" if mn["own_at_end"] >= 0.85 and mn["own_at_arrival"] >= 0.85 else ("does not" if min(mn["own_at_end"], mn["own_at_arrival"]) < 0.70 else "partial")
    lst = {}
    for k, d in res["per_task"].items():
        e = {m: d[m] for m in ("coverage_at_arrival", "own_at_arrival", "own_at_end", "changed")}
        if d["own_at_end"] < 0.7: e.update({m: d.get(m) for m in ("top_wrong_destination", "top_wrong_destination_share", "top_wrong_destination_is_older")})
        lst[k] = e
    RESULT["fresh2"] = {"variant": chosen, "pool_size": int(keep.sum()), "pool_removed": removed, "tasks": names, "chosen": {"mean": res["mean"], "se": res["se"], "per_task": res["per_task"]},
                        "per_task_list": lst, "plain_claims_ovr_minilm_C1_r1": {"mean": plain["mean"], "se": plain["se"], "per_task": plain["per_task"]},
                        "plain_claims_variant_eval_with_recognition": {"mean": plain_v["mean"], "se": plain_v["se"]}}
    RESULT["verdict_fresh2"] = {"variant": chosen, "guarantee": "met" if guar else "not met", "use": use, "mean_changed": mn["changed"], "se_changed": se["changed"], "mean_bound": mn["bound"],
                                "mean_own_at_end": mn["own_at_end"], "mean_own_at_arrival": mn["own_at_arrival"], "mean_coverage_at_arrival": mn["coverage_at_arrival"],
                                "mean_recognition_fallback_alone": mn["recognition"]}
    dump(); log("step B written", RESULT["verdict_fresh2"])

def stepA_main():
    grid = stepA()
    if os.environ.get("STAGE2", "1") == "1": stage2(grid)
    # wait for the first process (parts 1 and 2) so that its final write cannot overwrite this one
    while os.system("pgrep -f 'boundary_study.py$' > /dev/null") == 0: time.sleep(10)
    best = finalize_A(grid); log("step A written; chosen", best)

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "part12"
    if mode == "part12":
        part1(); part2(); RESULT["run_minutes"] = (time.time() - T0) / 60; dump(); log("done")
    elif mode == "stepA": stepA_main()
    elif mode == "stepB": stepB()
    elif mode == "stepAB": stepA_main(); stepB()
