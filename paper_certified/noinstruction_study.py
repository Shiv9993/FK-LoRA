"""Pilot A, routes only (last section of silent_adapters/PLAN.md): continual learning without instructions, zones from
split-conformal quantiles of a score on a frozen feature of the instance text only (the definition is never read for the zones).
Run from the folder of the official GainLoRA code:
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2 \
        $S/mm_env/bin/python noinstruction_study.py
Writes routing/noinstruction_study.json (after the primary feature, again after the secondary one). Caches in $S/noinstruction/."""
import json, math, os, random, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
torch.set_num_threads(2)

S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
BENCH = "CL_Benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "routing", "noinstruction_study.json")
CACHE = os.path.join(S, "noinstruction"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "certified_switch"))
from certified_switch import Zones

ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]
LONG3 = "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
TASKS = [("SuperNI", n) for n in ORDER1] + [("Long_Sequence", n) for n in LONG3]   # 30 tasks, sequence 3 order
KEY = [f"{f}/{n}" for f, n in TASKS]
SEQS = {"superni_order1": list(range(0, 15)), "long_order3": list(range(15, 30)), "both": list(range(30))}
UNRELATED = "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")   # the demo's choice
TEMPLATE = "Definition: {d}\n\nNow complete the following example -\nInput: {x}\nOutput: "
ALPHAS = [0.05, 0.1, 0.2]; M = 200; MAXTOK = 256; PRIMARY = ("t5static", "knn10", 0.05)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

def load(folder, name, split):
    d = json.load(open(os.path.join(BENCH, folder, name, split + ".json")))
    return d, [str(i["input"]) for i in d["Instances"]]

data = {k: {} for k in KEY}
for (f, n), k in zip(TASKS, KEY):
    for sp in ("train", "test"):
        data[k][sp] = load(f, n, sp)[1]
log("data read", {k: (len(data[k]["train"]), len(data[k]["test"])) for k in KEY[:2]})

# ---- fixed pool of unrelated inputs (instances only), without the 15 SuperNI tasks of the sequence
POOL_TEXTS = [str(x) for t in json.load(open(os.path.join(S, "ni", "ni_tasks.json")))["tasks"] if t["name"] not in set(ORDER1) for x in t["inputs"]]
log("pool size", len(POOL_TEXTS))

# ---- split per task (random, seed 0): first M of a seeded permutation are calibration, the rest fit
split = {}
for k in KEY:
    n = len(data[k]["train"]); perm = list(range(n)); random.Random(0).shuffle(perm)
    m = M if n >= 2 * M else n // 2          # DEVIATION forced by the data: tasks with fewer than 400 training inputs use half for calibration
    split[k] = (np.array(sorted(perm[:m]), dtype=int), np.array(sorted(perm[m:]), dtype=int))

# ---- features
def t5_zones():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("t5-base"); model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    ids = lambda texts: tok(list(texts), max_length=MAXTOK, truncation=True)["input_ids"]
    unrel = []
    for n in UNRELATED:
        d, _ = load("Long_Sequence", n, "dev"); df = d["Definition"][0] if isinstance(d["Definition"], list) else d["Definition"]
        unrel.append(ids([TEMPLATE.replace("{d}", df.strip()).replace("{x}", str(i["input"])) for i in d["Instances"][:200]]))
    z = Zones(model.get_input_embeddings().weight, tok.pad_token_id); z.fit_background(unrel)
    return z, ids, unrel

def pool_t5():
    fn = os.path.join(CACHE, "pool_t5static.npy")
    if os.path.exists(fn): return np.load(fn)
    z, ids, _ = t5_zones(); P = ((z._mean(ids(POOL_TEXTS)) - z.mu_all) @ z.w_all).numpy().astype(np.float64); np.save(fn, P); return P

def feats_t5():
    fn = os.path.join(CACHE, "t5static.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True); return {k: (z[k + "|train"], z[k + "|test"]) for k in KEY}, z["_tokens"].item(), z["_bg"].item()
    z, ids, unrel = t5_zones()
    out, tokens, save = {}, {}, {}
    for k in KEY:
        r = {}
        for sp in ("train", "test"):
            rows = ids(data[k][sp]); r[sp] = ((z._mean(rows) - z.mu_all) @ z.w_all).numpy().astype(np.float64)
            tokens[k + "|" + sp] = float(np.mean([len(x) for x in rows])); save[k + "|" + sp] = r[sp]
        out[k] = (r["train"], r["test"]); log("t5 feats", k)
    np.savez(fn, _tokens=np.array(tokens, dtype=object), _bg=np.array({"n_unrelated_rows": sum(map(len, unrel))}, dtype=object), **save)
    return out, tokens, {"n_unrelated_rows": sum(map(len, unrel))}

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

def pool_minilm():
    fn = os.path.join(CACHE, "pool_minilm.npy")
    if os.path.exists(fn): return np.load(fn)
    P = minilm_enc()(POOL_TEXTS); np.save(fn, P); return P

def feats_minilm():
    fn = os.path.join(CACHE, "minilm.npz")
    if os.path.exists(fn):
        z = np.load(fn); return {k: (z[k + "|train"], z[k + "|test"]) for k in KEY}
    enc = minilm_enc()
    out, save = {}, {}
    for k in KEY:
        tf = os.path.join(CACHE, "minilm_" + k.replace("/", "_") + ".npz")
        if os.path.exists(tf): z = np.load(tf); out[k] = (z["train"], z["test"])
        else:
            out[k] = (enc(data[k]["train"]), enc(data[k]["test"])); np.savez(tf, train=out[k][0], test=out[k][1])
        save[k + "|train"], save[k + "|test"] = out[k]; log("minilm feats", k)
    np.savez(fn, **save); return out

# ---- scores of every row of every task against every task's zone
def knn10(x, fit):
    d = torch.cdist(torch.as_tensor(x), torch.as_tensor(fit)); return d.kthvalue(10, dim=1).values.numpy()
def centre(x, fit): return np.linalg.norm(x - fit.mean(0), axis=1)
def ratio10(x, fit, d0):
    dk = knn10(x, fit); out = np.full(len(x), np.inf)
    pos = d0 > 0; out[pos] = dk[pos] / d0[pos]; out[(~pos) & (dk == 0)] = 0.0
    return out
SCORES = {"knn10": knn10, "centre": centre, "ratio10": ratio10}

def qidx(alpha, m=M): return math.ceil(round((1 - alpha) * (m + 1), 9))

def analyse(F, score, P, keys=None):
    keys = keys or KEY
    """F: {task: (train, test)}. Returns per task-index dicts of score matrices (rows x 30) and calibration scores."""
    fn = SCORES[score]; cal, fitf = {}, {}
    for k in keys:
        tr = F[k][0]; c, f = split[k]; cal[k] = tr[c]; fitf[k] = tr[f]
    if score == "ratio10":
        d0 = {}
        for k in keys: d0[k, "cal"] = knn10(cal[k], P); d0[k, "fit"] = knn10(fitf[k], P); d0[k, "test"] = knn10(F[k][1], P)
        g = lambda x, f, k, w: ratio10(x, f, d0[k, w])
    else:
        g = lambda x, f, k, w: fn(x, f)
    own_cal = {k: np.sort(g(cal[k], fitf[k], k, "cal")) for k in keys}
    Str, Ste = {}, {}
    for j, kj in enumerate(keys):
        Str[kj] = np.stack([g(fitf[kj], fitf[k], kj, "fit") for k in keys], 1)      # fit inputs of j scored by every task's zone
        Ste[kj] = np.stack([g(F[kj][1], fitf[k], kj, "test") for k in keys], 1)
    return own_cal, Str, Ste

def route(Sm, q, T):
    """Sm: rows x tasks (all), q: tasks; the tasks 0..T (inclusive) have arrived."""
    s, qq = Sm[:, :T + 1], q[:T + 1]; inside = s <= qq
    first = inside.argmax(1); fb = (s / qq).argmin(1)
    return np.where(inside.any(1), first, fb), inside

def mse(v): v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))

def measures(own_cal, Str, Ste, idx, alpha):
    names = [KEY[i] for i in idx]; q = np.array([own_cal[k][qidx(alpha, len(own_cal[k])) - 1] for k in names])
    per = {}; pairs = {}
    for j, kj in enumerate(names):
        sm = Ste[kj][:, idx]; st = Str[kj][:, idx]; n = len(sm)
        r_arr, inside_arr = route(sm, q, j); r_end, _ = route(sm, q, len(idx) - 1)
        own_in = inside_arr[:, j]
        ch = r_arr != r_end
        assert not (ch & own_in).any(), "changed among inside own zone"
        blk = inside_arr[:, :j].any(1) if j else np.zeros(n, bool)
        blk_tr = (st[:, :j] <= q[:j]).any(1) if j else np.zeros(len(st), bool)
        per[kj] = {"coverage": float(own_in.mean()), "blocked": float(blk.mean()), "blocked_train": float(blk_tr.mean()),
                   "own_at_arrival": float((r_arr == j).mean()), "own_at_end": float((r_end == j).mean()),
                   "changed": float(ch.mean()), "changed_among_inside": float((ch & own_in).mean()), "q": float(q[j])}
        pairs[kj] = {names[o]: float(inside_arr[:, o].mean()) for o in range(j)}
    meas = list(next(iter(per.values())).keys()); meas.remove("q")
    mean, se = {}, {}
    for m in meas: mean[m], se[m] = mse([per[k][m] for k in names])
    return {"per_task": per, "mean": mean, "se": se}, pairs

def verdict(res, alpha):
    mn, se = res["mean"], res["se"]
    claim = mn["changed"] <= alpha + 2 * se["changed"] and mn["coverage"] >= 1 - alpha - 2 * se["coverage"]
    use = "works" if mn["own_at_end"] >= 0.90 and mn["blocked"] <= 0.10 else ("does not" if mn["own_at_end"] < 0.75 or mn["blocked"] > 0.25 else "partial")
    return {"claim": "holds" if claim else "fails", "use": use, "alpha": alpha,
            "mean_changed": mn["changed"], "se_changed": se["changed"], "changed_limit": alpha + 2 * se["changed"],
            "mean_coverage": mn["coverage"], "se_coverage": se["coverage"], "coverage_limit": 1 - alpha - 2 * se["coverage"],
            "mean_own_at_end": mn["own_at_end"], "mean_blocked": mn["blocked"]}

RESULT = {"notes": [], "data": {}, "results": {s: {} for s in SEQS}, "verdict": {}, "blocked_pairs": {}, "verdict_ratio": {}, "other_settings_verdicts": {}}
def dump():
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(RESULT, open(OUT, "w"), indent=1)

def run_feature(fname, F, P):
    for score in SCORES:
        own_cal, Str, Ste = analyse(F, score, P)
        for sq, idx in SEQS.items():
            for alpha in ALPHAS:
                res, pairs = measures(own_cal, Str, Ste, idx, alpha)
                RESULT["results"][sq].setdefault(fname, {}).setdefault(score, {})[str(alpha)] = res
                v = verdict(res, alpha)
                if (fname, score, alpha) == PRIMARY:
                    RESULT["verdict"][sq] = v
                    RESULT["blocked_pairs"][sq] = [{"task": KEY[idx[j]], "older_task": KEY[idx[o]], "blocked": b}
                                                   for j in range(len(idx)) for o, b in enumerate(pairs[KEY[idx[j]]].values()) if b >= 0.2]
                else:
                    if (fname, score, alpha) == ("t5static", "ratio10", 0.05):
                        RESULT["verdict_ratio"][sq] = {**v, "pool_size": len(POOL_TEXTS), "median_q": float(np.median([res["per_task"][k]["q"] for k in res["per_task"]])),
                            "per_task": {k: {m: res["per_task"][k][m] for m in ("coverage", "blocked", "own_at_end")} for k in res["per_task"]},
                            "blocked_pairs": [{"task": KEY[idx[j]], "older_task": KEY[idx[o]], "blocked": b}
                                              for j in range(len(idx)) for o, b in enumerate(pairs[KEY[idx[j]]].values()) if b >= 0.2]}
                    RESULT["other_settings_verdicts"].setdefault(sq, {})[f"{fname}|{score}|{alpha}"] = {k: v[k] for k in ("claim", "use", "mean_changed", "mean_coverage", "mean_own_at_end", "mean_blocked")}
    dump(); log("written", fname)


# ======================= stage "t5hidden" (NI_HIDDEN=1): PLAN.md "one more feature ... the frozen model's own hidden states" =========
def hidden_stage():
    global split, RESULT
    torch.set_num_threads(int(os.environ.get("NI_THREADS", "4"))); t_start = time.time()
    RESULT = json.load(open(OUT)); RESULT.setdefault("results_reduced", {}); RESULT.setdefault("verdict_t5hidden", {})
    RESULT["notes"].append("t5hidden stage (PLAN.md 05:43): feature = mean over non-padding tokens of the last-layer states of the frozen T5-base encoder (eval, no adapter), input = instance text only, cut at 256 tokens (end token included), float32. REDUCED DATA: per task at most 400 training inputs (first 400 of the seed-0 shuffle used before), the first m of them calibration (m = 200, or half if the task has fewer than 400), the rest fit; pool = random.Random(0).sample of 2000 of the 5165 pool inputs; official test inputs. 'results_reduced' holds t5static, minilm and t5hidden on exactly this data (SuperNI order 1 only, plus the other sequences for t5hidden when computed). The ratio10/knn10/centre scores and the measures are the same.")
    sel, split = {}, {}
    for k in KEY:
        n = len(data[k]["train"]); perm = list(range(n)); random.Random(0).shuffle(perm)
        sel[k] = np.array(perm[:min(n, 400)], dtype=int); m = 200 if n >= 400 else n // 2
        split[k] = (np.arange(m), np.arange(m, len(sel[k])))
    pidx = np.array(sorted(random.Random(0).sample(range(len(POOL_TEXTS)), 2000)), dtype=int)
    from transformers import T5EncoderModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("t5-base"); model = T5EncoderModel.from_pretrained("t5-base").eval()
    def enc(texts, tokens_per_batch=8000):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); res = np.zeros((len(texts), model.config.d_model), np.float32); s0 = 0
        while s0 < len(order):
            e = s0 + 1
            b = order[s0:e]
            # batch of similar length: grow while estimated padded tokens fit
            while e < len(order) and e - s0 < 64 and (e - s0 + 1) * min(MAXTOK, len(texts[order[e]]) // 3 + 4) <= tokens_per_batch: e += 1
            b = order[s0:e]; s0 = e
            t = tok([texts[i] for i in b], max_length=MAXTOK, truncation=True, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = model(**t).last_hidden_state; m = t["attention_mask"][:, :, None].to(h.dtype)
                res[b] = ((h * m).sum(1) / m.sum(1)).numpy()
        return res
    def task_feats(k):
        tf = os.path.join(CACHE, "t5hidden_" + k.replace("/", "_") + ".npz")
        if os.path.exists(tf): z = np.load(tf); return z["train"], z["test"]
        tr = enc([data[k]["train"][i] for i in sel[k]]); te = enc(data[k]["test"]); np.savez(tf, train=tr, test=te); log("t5hidden", k); return tr, te
    pf = os.path.join(CACHE, "pool_t5hidden.npy")
    if os.path.exists(pf): PH = np.load(pf)
    else: PH = enc([POOL_TEXTS[i] for i in pidx]); np.save(pf, PH)
    log("t5hidden pool", PH.shape)
    def run(keys, F, P, sink, seqs, fname, scores=("ratio10", "knn10", "centre")):
        out = {}
        for score in scores:
            own_cal, Str, Ste = analyse(F, score, P, keys)
            for sq in seqs:
                for alpha in ALPHAS:
                    res, pairs = measures(own_cal, Str, Ste, SEQS[sq], alpha)
                    sink.setdefault(sq, {}).setdefault(fname, {}).setdefault(score, {})[str(alpha)] = res
                    if alpha == 0.05 and fname == "t5hidden" and score in ("ratio10", "knn10") and sink is RESULT["results"]:
                        idx = SEQS[sq]; v = verdict(res, 0.05)
                        RESULT["verdict_t5hidden"].setdefault(sq, {})[score] = {**v, "pool_size": 2000, "median_q": float(np.median([res["per_task"][k]["q"] for k in res["per_task"]])),
                            "per_task": {k: {m_: res["per_task"][k][m_] for m_ in ("coverage", "blocked", "own_at_end")} for k in res["per_task"]},
                            "blocked_pairs": [{"task": KEY[idx[j]], "older_task": KEY[idx[o]], "blocked": bb} for j in range(len(idx)) for o, bb in enumerate(pairs[KEY[idx[j]]].values()) if bb >= 0.2]}
    def stage(keys, seqs):
        FH = {k: task_feats(k) for k in keys}
        run(keys, FH, PH, RESULT["results"], seqs, "t5hidden")
        run(keys, FH, PH, RESULT["results_reduced"], seqs, "t5hidden")
        if "superni_order1" in seqs:
            F5r, _, _ = feats_t5(); Fmr = feats_minilm()
            for fname, FF, PP in (("t5static", F5r, pool_t5()[pidx]), ("minilm", Fmr, pool_minilm()[pidx])):
                Fr = {k: (FF[k][0][sel[k]], FF[k][1]) for k in keys}
                run(keys, Fr, PP, RESULT["results_reduced"], ["superni_order1"], fname)
        dump(); log("written stage", seqs, "elapsed min", (time.time() - t_start) / 60)
    stage(KEY[:15], ["superni_order1"])
    first = time.time() - t_start
    if first < 900: stage(KEY, ["long_order3", "both"] if False else ["superni_order1", "long_order3", "both"])
    else: log("first part took", first, "s: Long Sequence not run")
    RESULT["timing_minutes_t5hidden"] = (time.time() - t_start) / 60; dump()

if os.environ.get("NI_HIDDEN"):
    hidden_stage(); log("done"); sys.exit(0)

F5, tokens, bg = feats_t5()
for k in KEY:
    RESULT["data"][k] = {"n_train": len(data[k]["train"]), "n_fit": len(split[k][1]), "n_cal": len(split[k][0]), "n_test": len(data[k]["test"]),
                         "mean_tokens_train_t5": tokens[k + "|train"], "mean_tokens_test_t5": tokens[k + "|test"]}
RESULT["notes"] = [
    "Instances only ('input'); the Definition of the task files is never read for zones. Zones from the instance text only.",
    "Background for t5static (token weights and rescaling): dev.json of mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic (the demo's choice, first 200 instances each, WITH the definition in the demo's template; the demo itself reads train.json, this study reads dev.json as specified). ALL TEN of these are Long_Sequence tasks that are also in sequences 2 and 3 of this study (overlap, kept as specified: the background only sets token weights and the rescaling). Their dev.json is not train/test data of the tasks.",
    "t5static: T5-base token embeddings, tokenizer cut at 256 tokens (the end token counts), feature (zones._mean(rows) - zones.mu_all) @ zones.w_all. minilm: all-MiniLM-L6-v2, mean pooling, L2-normalised, 256 tokens including special tokens.",
    f"Split per task: random.Random(0) shuffle of the training indices, first {M} calibration, rest fit. q_k = ceil((1-alpha)(m+1))-th smallest calibration score (191, 181, 161 of 200). knn10 uses the fit inputs only (10th nearest); centre uses the mean of the fit features.",
    "DEVIATION FORCED BY THE DATA (stated, not tuned): several training files are small (samsum 128, diplomacy 100, multi_woz 113, cb 200 instances), so m = 200 calibration inputs is impossible there (cb would have no fit input). For every task with fewer than 400 training inputs m = n_train // 2 (rest fit); all other tasks use m = 200. The quantile index is ceil((1-alpha)(m+1)) with that task's m (see data[...].n_cal). Test files of those tasks are also small (about 20).",
    "Means and standard errors (std ddof=1 / sqrt(number of tasks)) are over all tasks of the sequence, including task 1 (blocked = 0 by construction) and the last task (changed = 0 by construction).",
    "blocked_pairs: primary setting, share of the test inputs of task j inside the zone of the older task k when task j arrives, >= 0.2. Sequence 'both' has 30 tasks (SuperNI order 1 then Long order 3), q and routes computed with all 30 zones; the two single sequences use only their own tasks.",
    "Some SuperNI task (task363 sst2) and Long_Sequence sst2 are distinct files; identical texts across files are not removed.",
    "ratio10 (added later, fixed in PLAN.md before computing): a_k = d_k / d_0, d_0 = distance to the 10th nearest feature of a fixed pool of unrelated inputs (instances of silent_adapters ni/ni_tasks.json tasks, without the 15 SuperNI tasks of the sequence; same feature map; same pool for all sequences); d_0 = 0 gives infinity unless d_k = 0 (then 0). Second attempt on data already used, not an independent test.",
    "Verdict rules exactly as in PLAN.md; other_settings_verdicts report the same rules for other settings and are not the verdict.",
    "Test inputs are the official test.json inputs; knn10 scores of test inputs use fit inputs only; blocked_train uses the task's fit inputs scored by older zones."]
run_feature("t5static", F5, pool_t5())
Fm = feats_minilm()
run_feature("minilm", Fm, pool_minilm())
log("done")
