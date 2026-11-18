"""The frozen model's own hidden states as the feature, under the rules that block nothing (PLAN.md, 2026-11-18 07:23 UTC).

    python hidden_rules_study.py dev        # development on the 15 SuperNI tasks (cached features of noinstruction_study.py)
    python hidden_rules_study.py encode4    # T5 features of the fourth batch of fresh tasks (cached in S/hidden_rules/)
    python hidden_rules_study.py fresh      # the verdict on the fourth fresh batch, read once, for the chosen variant

Instances only: the definition of a task is never read. Feature: mean of the last-layer states of the frozen T5-base encoder over the tokens of
the instance (256 tokens), standardised per dimension with the mean and standard deviation of a pool of unrelated inputs.
Rules: J = joint logistic regression over all tasks learned so far (retrained at every arrival on stored fit features); L = ridge regression on
the features, from sums added up task by task (no stored inputs, the same result in any order of arrival); claims = the rule of PLAN.md 06:20
(one classifier per task against the pool and the earlier tasks, frozen at arrival; the newest zone wins; a newcomer may not take the r-th
smallest-scoring held-out input of any older task). Held-out inputs are never used for fitting; they give the certificate (c + 1)/(m + 1).
"""
import json
import math
import os
import random
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "routing", "hidden_rules_study.json")
NI = os.path.join(S, "noinstruction")
HC = os.path.join(S, "hidden_rules"); os.makedirs(HC, exist_ok=True)
ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]
CS = [0.1, 1, 10]; LAMS = [1, 10, 100]; RS = [1, 2, 3]; LEVEL = 0.025; MAXTOK = 256
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
def mse(v): v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
def conf_q(cal, alpha):
    c = np.sort(cal); m = len(c); i = math.ceil(round((1 - alpha) * (m + 1), 9))
    return float(c[min(i, m) - 1])


# ---------------------------------------------------------------- the pool (for standardising, and as negatives of the rule claims)
def pool_rows():
    """The 2,000 pool inputs of the hidden-state stage of noinstruction_study.py, with the task each one comes from."""
    tasks = json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"]
    src = [t["name"] for t in tasks if t["name"] not in set(ORDER1) for _ in t["inputs"]]
    pidx = sorted(random.Random(0).sample(range(len(src)), 2000))
    return np.load(os.path.join(NI, "pool_t5hidden.npy")), [src[i] for i in pidx]


def standardiser(P):
    mu = P.mean(0); sd = P.std(0); sd[sd < 1e-6] = 1.0
    return lambda X: ((X - mu) / sd).astype(np.float64)


# ---------------------------------------------------------------- the three rules; each returns routes for every time T
def routes_joint(fit, cal, te, C):
    """R_te[j][:, T], R_cal[j][:, T] = route after tasks 0..T have arrived (-1 before the task's own arrival)."""
    K = len(fit); R_te = [np.full((len(te[j]), K), -1) for j in range(K)]; R_cal = [np.full((len(cal[j]), K), -1) for j in range(K)]
    for T in range(K):
        if T == 0:
            pred = lambda X: np.zeros(len(X), int)
        else:
            X = np.vstack(fit[:T + 1]); y = np.concatenate([np.full(len(fit[k]), k) for k in range(T + 1)])
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
            pred = clf.predict
        for j in range(T + 1):
            R_te[j][:, T] = pred(te[j]); R_cal[j][:, T] = pred(cal[j])
    return R_te, R_cal


def ridge_sums(fit):
    """Per task: weighted sums of x x^T and of x (a constant 1 is appended), each task with the same total weight 200."""
    out = []
    for X in fit:
        Z = np.hstack([X, np.ones((len(X), 1))]); w = 200.0 / len(Z)
        out.append((w * Z.T @ Z, w * Z.sum(0)))
    return out


def routes_ridge(fit, cal, te, lam, order=None):
    K = len(fit); sums = ridge_sums(fit); d = sums[0][0].shape[0]
    R_te = [np.full((len(te[j]), K), -1) for j in range(K)]; R_cal = [np.full((len(cal[j]), K), -1) for j in range(K)]
    G = np.zeros((d, d)); Cm = np.zeros((d, K)); W_last = None
    add = lambda X: np.hstack([X, np.ones((len(X), 1))])
    for T in range(K):
        G += sums[T][0]; Cm[:, T] = sums[T][1]
        W = np.linalg.solve(G + lam * (d - 1) * np.eye(d), Cm[:, :T + 1]); W_last = W
        for j in range(T + 1):
            R_te[j][:, T] = (add(te[j]) @ W).argmax(1); R_cal[j][:, T] = (add(cal[j]) @ W).argmax(1)
    return R_te, R_cal, W_last


def routes_claims(fit, cal, te, P, r_list, C=1.0):
    """One classifier per task (own fit inputs against the pool and the fit inputs of earlier tasks), frozen at arrival. a = minus decision value."""
    K = len(fit); a_te = [np.zeros((len(te[j]), K)) for j in range(K)]; a_cal = [np.zeros((len(cal[j]), K)) for j in range(K)]
    for i in range(K):
        X = np.vstack([fit[i], P] + fit[:i]); y = np.r_[np.ones(len(fit[i])), np.zeros(len(X) - len(fit[i]))]
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
        for j in range(K):
            a_te[j][:, i] = -clf.decision_function(te[j]); a_cal[j][:, i] = -clf.decision_function(cal[j])
    out = {}
    for r in r_list:
        q_own = np.array([conf_q(a_cal[i][:, i], LEVEL) for i in range(K)])
        q_old = np.array([min([np.sort(a_cal[j][:, i])[min(r, len(a_cal[j])) - 1] for j in range(i)], default=np.inf) for i in range(K)])
        qe = np.minimum(q_own, q_old)
        def route(a, T):
            ins = ((a < q_old) & (a <= q_own))[:, :T + 1]; last = T - ins[:, ::-1].argmax(1)
            return np.where(ins.any(1), last, (qe - a)[:, :T + 1].argmax(1))
        R_te = [np.full((len(te[j]), K), -1) for j in range(K)]; R_cal = [np.full((len(cal[j]), K), -1) for j in range(K)]
        for j in range(K):
            for T in range(j, K):
                R_te[j][:, T] = route(a_te[j], T); R_cal[j][:, T] = route(a_cal[j], T)
        cover = [float(((a_te[j][:, j] < q_old[j]) & (a_te[j][:, j] <= q_own[j])).mean()) for j in range(K)]
        out[r] = (R_te, R_cal, {"coverage_at_arrival": cover, "binding_old": [bool(q_old[i] < q_own[i]) for i in range(K)]})
    return out


def routes_handicap(fit, cal, te, lam, t_list, floor=False):
    """Rule H (PLAN.md 07:34): the joint ridge recogniser at every arrival; the newcomer takes x only if its margin over the best older task
    exceeds theta = the largest, over older tasks, of the (t+1)-th largest margin among that task's held-out inputs. Newest taker wins."""
    K = len(fit); sums = ridge_sums(fit); d = sums[0][0].shape[0]; add = lambda X: np.hstack([X, np.ones((len(X), 1))])
    G = np.zeros((d, d)); Cm = np.zeros((d, K)); M_te = [np.full((len(te[j]), K), np.inf) for j in range(K)]; M_cal = [np.full((len(cal[j]), K), np.inf) for j in range(K)]
    for l in range(K):
        G += sums[l][0]; Cm[:, l] = sums[l][1]
        if l == 0: continue
        W = np.linalg.solve(G + lam * (d - 1) * np.eye(d), Cm[:, :l + 1])
        mg = lambda X: (lambda Sx: Sx[:, l] - Sx[:, :l].max(1))(add(X) @ W)
        for j in range(K):
            M_te[j][:, l] = mg(te[j]); M_cal[j][:, l] = mg(cal[j])
    out = {}
    for t in t_list:
        theta = np.full(K, -np.inf)
        for l in range(1, K):
            theta[l] = max((np.sort(M_cal[j][:, l])[::-1][t] if t < len(M_cal[j]) else -np.inf) for j in range(l))
            if floor: theta[l] = max(theta[l], 0.0)      # rule H': the newcomer never takes an input that the joint recogniser gives to an older task
        def route(Mx, T):
            r = np.zeros(len(Mx), int)
            for l in range(1, T + 1): r = np.where(Mx[:, l] > theta[l], l, r)
            return r
        R_te = [np.full((len(te[j]), K), -1) for j in range(K)]; R_cal = [np.full((len(cal[j]), K), -1) for j in range(K)]
        for j in range(K):
            for T in range(j, K):
                R_te[j][:, T] = route(M_te[j], T); R_cal[j][:, T] = route(M_cal[j], T)
        take = [1.0] + [float((M_cal[l][:, l] > theta[l]).mean()) for l in range(1, K)]
        out[t] = (R_te, R_cal, {"takes_of_own_heldout": take, "theta": [None] + [float(x) for x in theta[1:]]})
    return out


# ---------------------------------------------------------------- measures
def measures(names, R_te, R_cal, extra=None, held_out_used_by_rule=False, r=None, path=False, handicap_t=None):
    K = len(names); per = {}
    for j, kj in enumerate(names):
        te, ca = R_te[j], R_cal[j]; m = len(ca); n = len(te)
        own_arr = float((te[:, j] == j).mean()); own_end = float((te[:, K - 1] == j).mean())
        ch_end = te[:, j] != te[:, K - 1]; ch_ever = (te[:, j:] != te[:, [j]]).any(1)
        c_end = int((ca[:, j] != ca[:, K - 1]).sum()); c_ever = int((ca[:, j:] != ca[:, [j]]).any(1).sum())
        wrong = te[:, K - 1][te[:, K - 1] != j]
        d = {"n_test": n, "m": m, "own_at_arrival": own_arr, "own_at_end": own_end, "changed": float(ch_end.mean()), "changed_ever": float(ch_ever.mean()),
             "c": c_end, "c_ever": c_ever, "certificate": (c_end + 1) / (m + 1), "certificate_ever": (c_ever + 1) / (m + 1),
             "upper95": float(beta_dist.ppf(0.95, c_end + 1, m - c_end)) if c_end < m else 1.0,
             "heldout_own_at_arrival": float((ca[:, j] == j).mean()), "heldout_own_at_end": float((ca[:, K - 1] == j).mean()),
             "prior_bound": (1 - own_arr) + (1 - own_end)}
        if len(wrong):
            cnt = np.bincount(wrong, minlength=K); d["top_wrong_destination"] = names[int(cnt.argmax())]; d["top_wrong_destination_share_of_test"] = float(cnt.max() / n)
        if r is not None:
            d["claims_bound"] = (1 - extra["coverage_at_arrival"][j]) + r * (K - 1 - j) / (m + 1); d["coverage_at_arrival"] = extra["coverage_at_arrival"][j]
        if handicap_t is not None:
            d["handicap_bound"] = (handicap_t + 1) * (K - 1 - j) / (m + 1); d["takes_of_own_heldout"] = extra["takes_of_own_heldout"][j]
        if path:
            d["path_changed_test"] = [float((te[:, j] != te[:, T]).mean()) for T in range(j, K)]
            d["path_certificate"] = [float(((ca[:, j] != ca[:, T]).sum() + 1) / (m + 1)) for T in range(j, K)]
        per[kj] = d
    keys = ["own_at_arrival", "own_at_end", "changed", "changed_ever", "certificate", "certificate_ever", "upper95", "heldout_own_at_arrival", "heldout_own_at_end", "prior_bound"]
    if r is not None: keys += ["claims_bound", "coverage_at_arrival"]
    if handicap_t is not None: keys += ["handicap_bound", "takes_of_own_heldout"]
    mean, se = {}, {}
    for k in keys: mean[k], se[k] = mse([per[t][k] for t in names])
    return {"per_task": per, "mean": mean, "se": se, "certificate_valid": not held_out_used_by_rule}


def run_all(names, fit, cal, te, P, variants=None, path_for=None):
    res = {}
    want = lambda v: variants is None or v in variants
    for C in CS:
        v = f"J|C{C}"
        if want(v):
            Rt, Rc = routes_joint(fit, cal, te, C); res[v] = measures(names, Rt, Rc, path=(v == path_for)); log(v, res[v]["mean"]["own_at_end"])
    for lam in LAMS:
        v = f"L|lam{lam}"
        if want(v):
            Rt, Rc, _ = routes_ridge(fit, cal, te, lam); res[v] = measures(names, Rt, Rc, path=(v == path_for)); log(v, res[v]["mean"]["own_at_end"])
    rs = [r for r in RS if want(f"claims|r{r}")]
    if rs:
        for r, (Rt, Rc, ex) in routes_claims(fit, cal, te, P, rs).items():
            v = f"claims|r{r}"; res[v] = measures(names, Rt, Rc, extra=ex, held_out_used_by_rule=True, r=r, path=(v == path_for)); log(v, res[v]["mean"]["own_at_end"])
    return res


def brief(r):
    m = r["mean"]
    return {k: round(m[k], 4) for k in ("own_at_arrival", "own_at_end", "changed", "changed_ever", "certificate", "upper95", "prior_bound")}


# ---------------------------------------------------------------- development
def dev_data():
    P, _ = pool_rows(); st = standardiser(P); fit, cal, te = [], [], []
    for n in ORDER1:
        z = np.load(os.path.join(NI, f"t5hidden_SuperNI_{n}.npz")); tr = st(z["train"]); m = 200 if len(tr) >= 400 else len(tr) // 2
        cal.append(tr[:m]); fit.append(tr[m:]); te.append(st(z["test"]))
    return ["SuperNI/" + n for n in ORDER1], fit, cal, te, st(P)


def choose(dev):
    ok = [v for v in dev if dev[v]["mean"]["changed"] <= 0.03]
    rank = lambda v: (-round(dev[v]["mean"]["own_at_end"], 6), {"L": 0, "J": 1, "c": 2}[v[0]], float(v.split("|")[1].lstrip("Clamr")))
    return (sorted(ok, key=rank)[0] if ok else None), ok


def dev():
    names, fit, cal, te, P = dev_data()
    R = {"notes": [
        "Feature: mean of the last-layer states of the frozen T5-base encoder over the non-padding tokens of the instance text (256 tokens, end token included), no adapter; standardised per dimension with the mean and standard deviation of the 2,000 pool inputs. The definition of a task is never read.",
        "Data: the cached reduced data of noinstruction_study.py (at most 400 training inputs per task, the first 400 of a seed-0 shuffle; the first m held out (m = 200, or half if fewer than 400), the rest fit inputs); official test inputs.",
        "Rule J: LogisticRegression(C, class_weight=balanced, max_iter=3000), all tasks learned so far, retrained at every arrival on the fit inputs. Rule L: ridge regression of one-hot task targets on the features plus a constant, from per-task sums (each task weighted to a total of 200), penalty lam * 768 on all coefficients; the same for every order of arrival. Rule claims: PLAN.md 06:20, C = 1, level 0.025, negatives = pool + earlier fit inputs; r = the rank of the smallest held-out score of an older task that a newcomer may not pass (r = 1: none of the held-out inputs can be taken). The plan entry of 07:23 lists r = 0 by mistake; the smallest value that exists is r = 1.",
        "changed = route after the last task differs from the route at the task's own arrival (test inputs); changed_ever = it differs at any time in between. certificate = (c + 1)/(m + 1) with c the number of the task's m held-out inputs whose route changed (arrival against end); upper95 = one-sided 95% Clopper-Pearson bound on the changed share from the same count. For the rule claims the held-out inputs set the thresholds, so its certificate is not valid (certificate_valid false) and its a-priori bound claims_bound = (1 - coverage) + r * (number of later tasks)/(m + 1) is the statement.",
        "prior_bound = (1 - own_at_arrival) + (1 - own_at_end) on the test inputs: a route cannot change unless it is wrong at one of the two times."]}
    res = run_all(names, fit, cal, te, P)
    R["dev"] = {"sequence": "superni_order1", "summary": {v: brief(r) for v, r in res.items()}, "full": res}
    ch, ok = choose(res); R["chosen"] = {"variant": ch, "eligible": ok, "rule": "highest mean own_at_end among variants with mean changed <= 0.03; ties: L, then J, then claims; then the smaller number"}
    if ch:
        full = run_all(names, fit, cal, te, P, variants=[ch], path_for=ch)[ch]
        R["chosen"]["per_task"] = full["per_task"]; R["chosen"]["mean"] = full["mean"]; R["chosen"]["se"] = full["se"]
    # order check of rule L: the weights after all tasks do not depend on the order of arrival
    _, _, W1 = routes_ridge(fit, cal, te, 10); perm = list(range(len(fit)))[::-1]
    _, _, W2 = routes_ridge([fit[i] for i in perm], [cal[i] for i in perm], [te[i] for i in perm], 10)
    R["order_check_rule_L"] = {"largest_abs_difference_of_weights_reversed_order": float(np.abs(W1 - W2[:, ::-1]).max())}
    R["dev_first_sign"] = {"line": "at least 90% of test inputs given to their own task after the last task (the first 'sign' of PLAN.md 06:10)",
                           "chosen_own_at_end": None if not ch else res[ch]["mean"]["own_at_end"], "met": bool(ch and res[ch]["mean"]["own_at_end"] >= 0.90)}
    json.dump(R, open(OUT, "w"), indent=1); log("written", OUT)
    for v, r in res.items(): print(f"{v:12s}", brief(r))
    print("chosen:", ch)
    if ch:
        for k, d in R["chosen"]["per_task"].items(): print(f"  {k[8:45]:38s} arr {d['own_at_arrival']:.2f} end {d['own_at_end']:.2f} changed {d['changed']:.2f} cert {d['certificate']:.3f} c {d['c']}/{d['m']}  -> {d.get('top_wrong_destination', '')[8:30]}")
    print("order check", R["order_check_rule_L"])


# ---------------------------------------------------------------- the fourth fresh batch
FRESH = os.path.join(S, "ni_fresh", "fresh_tasks_4.json")


def fresh_split():
    perm = list(range(800)); random.Random(0).shuffle(perm); sel = perm[:400]
    return sel[:200], sel[200:], list(range(800, 1000))


def encode4():
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    torch.set_num_threads(int(os.environ.get("HR_THREADS", "2")))
    tok = AutoTokenizer.from_pretrained("t5-base"); model = T5EncoderModel.from_pretrained("t5-base").eval()
    def enc(texts, tokens_per_batch=8000):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); res = np.zeros((len(texts), model.config.d_model), np.float32); s0 = 0
        while s0 < len(order):
            e = s0 + 1
            while e < len(order) and e - s0 < 64 and (e - s0 + 1) * min(MAXTOK, len(texts[order[e]]) // 3 + 4) <= tokens_per_batch: e += 1
            b = order[s0:e]; s0 = e
            t = tok([texts[i] for i in b], max_length=MAXTOK, truncation=True, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = model(**t).last_hidden_state; m = t["attention_mask"][:, :, None].to(h.dtype)
                res[b] = ((h * m).sum(1) / m.sum(1)).numpy()
        return res
    tasks = json.load(open(FRESH))["tasks"]; ca, fi, te = fresh_split()
    for i, t in enumerate(tasks):
        f = os.path.join(HC, f"fresh4_{i:02d}_{t['name']}.npz")
        if os.path.exists(f): continue
        x = [str(inst["input"]) for inst in t["instances"]]
        np.savez(f, cal=enc([x[j] for j in ca]), fit=enc([x[j] for j in fi]), test=enc([x[j] for j in te])); log("encoded", i, t["name"])


def fresh():
    R = json.load(open(OUT)); ch = R["chosen"]["variant"]; assert ch, "no chosen variant"
    assert "verdict_fresh4" not in R, "the verdict has been read already"
    tasks = json.load(open(FRESH))["tasks"]; names = [t["name"] for t in tasks]
    P, src = pool_rows(); keep = np.array([s not in set(names) for s in src]); P = P[keep]; st = standardiser(P)
    fit, cal, te = [], [], []
    for i, t in enumerate(tasks):
        z = np.load(os.path.join(HC, f"fresh4_{i:02d}_{t['name']}.npz")); cal.append(st(z["cal"])); fit.append(st(z["fit"])); te.append(st(z["test"]))
    variants = sorted({ch, "J|C1"})
    res = run_all(names, fit, cal, te, st(P), variants=variants, path_for=ch); r = res[ch]; mn, se = r["mean"], r["se"]
    use = "works" if (mn["own_at_end"] >= 0.85 and mn["own_at_arrival"] >= 0.85) else ("does not" if (mn["own_at_end"] < 0.70 or mn["own_at_arrival"] < 0.70) else "partial")
    cert = "met" if mn["changed"] <= mn["certificate"] + 2 * se["changed"] else "not met"
    R["verdict_fresh4"] = {"variant": ch, "use": use, "certificate": cert if r["certificate_valid"] else "not valid for this rule (see claims_bound)",
                           "mean": mn, "se": se, "pool_inputs_kept": int(keep.sum()), "tasks": names,
                           "per_task": r["per_task"], "comparison": {v: brief(res[v]) for v in variants}}
    # the second rule, reported next to the verdict (PLAN.md 07:34 and 07:35 UTC): rule H' with t = 2
    Rt, Rc, ex = routes_handicap(fit, cal, te, 1, [2], floor=True)[2]; h = measures(names, Rt, Rc, extra=ex, held_out_used_by_rule=True, handicap_t=2); hm, hs = h["mean"], h["se"]
    R["verdict_fresh4"]["second_rule_H_prime_t2"] = {
        "keeps_the_recognition": bool(hm["own_at_end"] >= mn["own_at_end"] - 0.02), "bound_met": bool(hm["changed_ever"] <= hm["handicap_bound"] + 2 * hs["changed_ever"]),
        "mean": hm, "se": hs, "per_task": h["per_task"]}
    json.dump(R, open(OUT, "w"), indent=1)
    print("verdict:", use, "| certificate:", R["verdict_fresh4"]["certificate"]); print(ch, brief(r))
    print("H'|t2: own at arrival", round(hm["own_at_arrival"], 4), "at end", round(hm["own_at_end"], 4), "changed", round(hm["changed"], 4), "ever", round(hm["changed_ever"], 4), "bound", round(hm["handicap_bound"], 4),
          "takes of own held-out", round(hm["takes_of_own_heldout"], 4), "| keeps the recognition:", R["verdict_fresh4"]["second_rule_H_prime_t2"]["keeps_the_recognition"], "| bound met:", R["verdict_fresh4"]["second_rule_H_prime_t2"]["bound_met"])
    for v in variants: print("  ", v, brief(res[v]))
    for k, d in r["per_task"].items(): print(f"  {k[:52]:52s} arr {d['own_at_arrival']:.2f} end {d['own_at_end']:.2f} ch {d['changed']:.2f} cert {d['certificate']:.3f} -> {d.get('top_wrong_destination', '')[:40]} {d.get('top_wrong_destination_share_of_test', 0):.2f}")


# ---------------------------------------------------------------- routes for the GPU run on the five-task sequence
FIVE = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification", "task748_glucose_reverse_cause_event_detection",
        "task1687_sentiment140_classification", "task875_emotion_classification"]
ROUTES_OUT = os.path.join(HERE, "..", "silent_adapters", "noinstruction_routes.json")


def routes5():
    """Routes of every test input of the five tasks after 1..5 tasks, for the rules L (chosen), J, claims and 'own'. Written before the GPU run."""
    P, _ = pool_rows(); st = standardiser(P); fit, cal, te = [], [], []
    for n in FIVE:
        z = np.load(os.path.join(NI, f"t5hidden_SuperNI_{n}.npz")); tr = st(z["train"]); m = 200 if len(tr) >= 400 else len(tr) // 2
        cal.append(tr[:m]); fit.append(tr[m:]); te.append(st(z["test"]))
    K = len(FIVE); rules = {}
    Rt, Rc, _ = routes_ridge(fit, cal, te, 1); rules["ridge"] = (Rt, Rc, True, None, None)
    Rt, Rc = routes_joint(fit, cal, te, 1); rules["joint"] = (Rt, Rc, True, None, None)
    Rt, Rc, ex = routes_claims(fit, cal, te, st(P), [1])[1]; rules["claims"] = (Rt, Rc, False, ex, 1)
    Rt, Rc, exh = routes_handicap(fit, cal, te, 1, [2], floor=True)[2]; rules["handicap"] = (Rt, Rc, False, exh, None)
    own_t = [np.tile(np.where(np.arange(K) >= j, j, -1), (len(te[j]), 1)) for j in range(K)]
    own_c = [np.tile(np.where(np.arange(K) >= j, j, -1), (len(cal[j]), 1)) for j in range(K)]
    rules["own"] = (own_t, own_c, True, None, None)
    out = {"about": "Routes for the GPU run of our method without instructions (PLAN.md, 2026-11-18 07:29 UTC). Written before the run. "
                    "routes[rule][task folder][i][str(t)] = 1-based position, in the task order, of the task that test example i (order of test.json) is given to after t tasks. "
                    "ridge = rule L with ridge 1 (the chosen rule); joint = rule J with C = 1; claims = rule claims with r = 1; handicap = rule H' with t = 2 (the joint recogniser at every arrival, the newcomer may take at most 2 held-out inputs of each older task and nothing that the joint recogniser gives to an older task; added at 07:35 UTC, before the run); own = every input to its own task (perfect recognition, for comparison only). "
                    "Feature: frozen T5-base encoder states of the instance text only; 200 fit and 200 held-out training inputs per task; test inputs are never used for fitting.",
           "task_order": FIVE, "routes": {}, "measures": {}}
    for name, (Rt, Rc, valid, ex, r) in rules.items():
        out["routes"][name] = {FIVE[j]: [{str(T + 1): int(Rt[j][i, T]) + 1 for T in range(j, K)} for i in range(len(te[j]))] for j in range(K)}
        ms = measures(FIVE, Rt, Rc, extra=ex, held_out_used_by_rule=not valid, r=r, path=True, handicap_t=(2 if name == "handicap" else None))
        changed_first_four = int(sum((Rt[j][:, j] != Rt[j][:, K - 1]).sum() for j in range(K - 1)))
        out["measures"][name] = {"mean": ms["mean"], "per_task": ms["per_task"], "certificate_valid": ms["certificate_valid"],
                                 "routes_changed_on_the_400_compared_test_inputs": changed_first_four,
                                 "not_own_at_end_of_500": int(sum((Rt[j][:, K - 1] != j).sum() for j in range(K)))}
        print(f"{name:7s} own at arrival {ms['mean']['own_at_arrival']:.3f} at end {ms['mean']['own_at_end']:.3f} | routes changed on the 400 compared inputs: {changed_first_four} | not own at end: {out['measures'][name]['not_own_at_end_of_500']} of 500")
        for k, d in ms["per_task"].items(): print(f"    {k[:44]:44s} arr {d['own_at_arrival']:.2f} end {d['own_at_end']:.2f} changed {d['changed']:.2f} certificate {d['certificate']:.3f} (c {d['c']}/{d['m']}) upper95 {d['upper95']:.3f}" + (f" bound {d['handicap_bound']:.3f} takes {d['takes_of_own_heldout']:.2f}" if name == "handicap" else ""))
    json.dump(out, open(ROUTES_OUT, "w")); print("written", os.path.abspath(ROUTES_OUT))


def dev_handicap():
    names, fit, cal, te, P = dev_data(); R = json.load(open(OUT)); R["dev_rule_H"] = {"note": "Rule H (PLAN.md 07:34 UTC), ridge 1, t = number of held-out inputs of each older task that a newcomer may take. handicap_bound = (t + 1) * (number of later tasks)/(m + 1). The held-out inputs set the thresholds, so the certificate column is not valid for this rule."}
    runs = [(f"H|t{t}", t, v) for t, v in routes_handicap(fit, cal, te, 1, [0, 1, 2]).items()] + [(f"H'|t{t}", t, v) for t, v in routes_handicap(fit, cal, te, 1, [0, 1, 2, 5, 10**6], floor=True).items()]
    R["dev_rule_H"]["note_floor"] = "H' = rule H with the threshold never below zero (added after the development numbers of H showed that a negative threshold lets a newcomer take inputs that the joint recogniser gives to older tasks); t = 1000000 means no protection (the cascade of joint recognisers alone)."
    for tag, t, (Rt, Rc, ex) in runs:
        ms = measures(names, Rt, Rc, extra=ex, held_out_used_by_rule=True, handicap_t=t); R["dev_rule_H"][tag] = ms
        mn = ms["mean"]; print(f"{tag}  own at arrival {mn['own_at_arrival']:.4f} at end {mn['own_at_end']:.4f} changed {mn['changed']:.4f} ever {mn['changed_ever']:.4f} bound {mn['handicap_bound']:.4f} takes of own held-out {mn['takes_of_own_heldout']:.4f}")
        for k, d in ms["per_task"].items(): print(f"    {k[8:46]:38s} arr {d['own_at_arrival']:.2f} end {d['own_at_end']:.2f} changed {d['changed']:.2f} bound {d['handicap_bound']:.3f} takes {d['takes_of_own_heldout']:.2f} -> {d.get('top_wrong_destination', '')[8:30]}")
    json.dump(R, open(OUT, "w"), indent=1)


ORDER2 = ["task748_glucose_reverse_cause_event_detection", "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task639_multi_woz_user_utterance_generation", "task1572_samsum_summary", "task1687_sentiment140_classification", "task591_sciq_answer_generation",
          "task363_sst2_polarity_classification", "task1510_evalution_relation_extraction", "task1729_personachat_generate_next", "task181_outcome_extraction",
          "task511_reddit_tifu_long_text_summarization", "task002_quoref_answer_generation", "task1290_xsum_summarization", "task875_emotion_classification"]


def dev_order2():
    """The chosen rule L (ridge 1) and rule H' (t = 2) on the second official order of the same 15 SuperNI tasks, and on five random orders (seeds 0-4). Development data."""
    names, fit, cal, te, P = dev_data(); R = json.load(open(OUT)); pos = {n: i for i, n in enumerate(names)}
    orders = {"order2": ["SuperNI/" + n for n in ORDER2]}
    for sd in range(5):
        o = list(names); random.Random(sd).shuffle(o); orders[f"random{sd}"] = o
    out = {"note": "Same features, split and rules as the development run; only the order of arrival differs. Rule L ends with the same recogniser for every order, so its share given to the own task after the last task is the same; what changes is the route right after arrival, and so the changed share and the audit."}
    for tag, o in orders.items():
        idx = [pos[n] for n in o]; f, c, t = [fit[i] for i in idx], [cal[i] for i in idx], [te[i] for i in idx]
        Rt, Rc, _ = routes_ridge(f, c, t, 1); mL = measures(o, Rt, Rc)
        Rt, Rc, ex = routes_handicap(f, c, t, 1, [2], floor=True)[2]; mH = measures(o, Rt, Rc, extra=ex, held_out_used_by_rule=True, handicap_t=2)
        out[tag] = {"order": o, "L": {"mean": mL["mean"], "se": mL["se"]}, "H_prime_t2": {"mean": mH["mean"], "se": mH["se"]}}
        a, b = mL["mean"], mH["mean"]
        print(f"{tag:8s} L: arr {a['own_at_arrival']:.4f} end {a['own_at_end']:.4f} changed {a['changed']:.4f} ever {a['changed_ever']:.4f} audit {a['certificate']:.4f} | H' t2: arr {b['own_at_arrival']:.4f} end {b['own_at_end']:.4f} changed {b['changed']:.4f} bound {b['handicap_bound']:.4f} takes {b['takes_of_own_heldout']:.4f}")
    R["dev_other_orders"] = out; json.dump(R, open(OUT, "w"), indent=1)


def routes15():
    """Routes for a GPU run on the 15 SuperNI tasks of order 1 (the tasks the rule was chosen on), same format as routes5."""
    names, fit, cal, te, P = dev_data(); K = len(names); folders = [n.split("/", 1)[1] for n in names]; rules = {}
    Rt, Rc, _ = routes_ridge(fit, cal, te, 1); rules["ridge"] = (Rt, Rc, True, None, None)
    Rt, Rc = routes_joint(fit, cal, te, 1); rules["joint"] = (Rt, Rc, True, None, None)
    Rt, Rc, exh = routes_handicap(fit, cal, te, 1, [2], floor=True)[2]; rules["handicap"] = (Rt, Rc, False, exh, 2)
    own_t = [np.tile(np.where(np.arange(K) >= j, j, -1), (len(te[j]), 1)) for j in range(K)]
    own_c = [np.tile(np.where(np.arange(K) >= j, j, -1), (len(cal[j]), 1)) for j in range(K)]
    rules["own"] = (own_t, own_c, True, None, None)
    out = {"about": "Routes for a GPU run of our method without instructions on the 15 SuperNI tasks of order 1. Written before the run. Same format and rules as noinstruction_routes.json "
                    "(ridge = rule L with ridge 1; joint = rule J with C = 1; handicap = rule H' with t = 2; own = perfect recognition, comparison only). "
                    "These are the 15 tasks on which the rule and its settings were chosen (development data); the features use at most 400 training inputs per task (half held out), test inputs are never used for fitting.",
           "task_order": folders, "routes": {}, "measures": {}}
    for name, (Rt, Rc, valid, ex, ht) in rules.items():
        out["routes"][name] = {folders[j]: [{str(T + 1): int(Rt[j][i, T]) + 1 for T in range(j, K)} for i in range(len(te[j]))] for j in range(K)}
        ms = measures(folders, Rt, Rc, extra=ex, held_out_used_by_rule=not valid, handicap_t=ht, path=True)
        ch = int(sum((Rt[j][:, j] != Rt[j][:, K - 1]).sum() for j in range(K - 1))); n14 = int(sum(len(te[j]) for j in range(K - 1))); nall = int(sum(len(t) for t in te))
        out["measures"][name] = {"mean": ms["mean"], "per_task": ms["per_task"], "certificate_valid": ms["certificate_valid"], "routes_changed_first_14_tasks": ch, "compared": n14,
                                 "not_own_at_end": int(sum((Rt[j][:, K - 1] != j).sum() for j in range(K))), "test_inputs": nall}
        print(f"{name:8s} own at arrival {ms['mean']['own_at_arrival']:.4f} at end {ms['mean']['own_at_end']:.4f} | routes changed {ch} of {n14} | not own at end {out['measures'][name]['not_own_at_end']} of {nall}")
    f = os.path.join(HERE, "..", "silent_adapters", "noinstruction_routes_15.json"); json.dump(out, open(f, "w")); print("written", os.path.abspath(f))


if __name__ == "__main__":
    {"dev": dev, "encode4": encode4, "fresh": fresh, "routes5": routes5, "dev_handicap": dev_handicap, "dev_order2": dev_order2, "routes15": routes15}[sys.argv[1]]()
