"""Sixth pre-registered experiment (last section of silent_adapters/PLAN.md): a zone decided sentence by sentence.
Zone of task k = all instruction texts u with (1) MiniLM whole-text distance d(u, o_k) <= r and (2) sentence-by-sentence judge score
(u, o_k) >= tau. Sentence pairing: each sentence of u with its most similar sentence of o_k (MiniLM cosine) and each sentence of o_k
with its most similar sentence of u; identical sentences score 1, others min(P_entail(a then b), P_entail(b then a)); the score of
(u, o_k) is the minimum over all pairs (1 for identical texts). Every task within r is judged (no limit). r (alpha 0.10) and tau
(alpha 0.05) from half B of the calibration tasks, German pivot; the judge is chosen on half B among three NLI models.
Steps: 1 calibration + judge choice, 2 evaluation (de, fr, es), 3 Italian pivot + verdict, 4 views. JSON written after each step.

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=3 \
        $S/mm_env/bin/python sentence_judge_study.py
Caches (sentence embeddings, every judged sentence pair as shards) in $S/sentence_judge/; a rerun resumes.
Reuses encoder_study.py (data, split, MiniLM, benchmark constants); measures as in judge_study.py.
"""
import json, os, sys, time, math, re, glob, hashlib, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_study as es
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(3)

S = es.S
OUT = f"{S}/sentence_judge"; os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(HERE, "routing", "sentence_judge_study.json")
MODELS = ["cross-encoder/nli-deberta-v3-xsmall", "cross-encoder/nli-deberta-v3-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]
CUT = 128
by_name, sr = es.by_name, es.sr
ev_names, half_B, cal_names = es.ev_names, es.half_B, es.cal_names
K = len(ev_names); ar = np.arange(K)
FIVE, ORDER1 = es.FIVE, es.ORDER1
t0_all = time.time()
def log(*a): print(f"[{time.time() - t0_all:6.0f}s]", *a, flush=True)
RESULT = {"notes": {}}
def dump(): json.dump(RESULT, open(JSON_PATH, "w"), indent=1)
def md5(t): return hashlib.md5(t.encode()).hexdigest()[:16]
split = lambda d: [s for s in re.split(r"(?<=[.!?])\s+", d.strip()) if s]      # as paraphrase_make.py

# ---------------------------------------------------------------- texts
orig = {n: by_name[n]["definition"] for n in by_name}
PARA = {}
for f, key in (("paraphrases.json", "de"), ("paraphrases.json", "fr"), ("paraphrases_es.json", "es"), ("paraphrases_it.json", "it")):
    PARA[key] = json.load(open(os.path.join(HERE, "routing", f)))["tasks"]
def ptext(p, n): return PARA[p][n][p]
TXT = {"orig": [orig[n] for n in ev_names]}
for p in ("de", "fr", "es", "it"): TXT[p] = [ptext(p, n) for n in ev_names]

# ---------------------------------------------------------------- MiniLM
enc = es.Enc("minilm")
def cdist(A, B): return torch.cdist(torch.from_numpy(np.asarray(A, np.float64)), torch.from_numpy(np.asarray(B, np.float64)), compute_mode="donot_use_mm_for_euclid_dist").numpy()
_te = {}
def embed_texts(texts):
    new = [t for t in dict.fromkeys(texts) if t not in _te]
    if new:
        for t, v in zip(new, enc.encode([enc.ids(t, es.MAXTOK) for t in new])): _te[t] = v.astype(np.float64)
    return np.stack([_te[t] for t in texts])
_se = {}
SE_F = f"{OUT}/sent_emb.pkl"
if os.path.exists(SE_F): _se = pickle.load(open(SE_F, "rb"))
def sent_info(t, use_cache=True):
    ss = split(t)
    if not ss: ss = [t.strip()]
    new = [s for s in dict.fromkeys(ss) if s not in _se] if use_cache else list(dict.fromkeys(ss))
    if new:
        vs = enc.encode([enc.ids(s, es.MAXTOK) for s in new], use_cache=False)
        loc = {s: v for s, v in zip(new, vs)}
    else: loc = {}
    if use_cache:
        _se.update(loc)
        return ss, np.stack([_se[s] for s in ss])
    return ss, np.stack([loc[s] if s in loc else _se[s] for s in ss])
def save_sent_emb(): pickle.dump(_se, open(SE_F, "wb"))

# ---------------------------------------------------------------- judges
class Judge:
    def __init__(self, name):
        self.name = name; tag = name.replace("/", "_")
        try: self.tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            log("fast tokenizer failed", repr(e)[:120]); self.tok = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.ent = [k for k, v in self.id2label.items() if v.lower() == "entailment"][0]
        self.shard_pat = f"{OUT}/pairs_{tag}_*.pkl"; self.cache = {}
        for f in sorted(glob.glob(self.shard_pat)): self.cache.update(pickle.load(open(f, "rb")))
        self.nshard = len(glob.glob(self.shard_pat)); self._ids = {}
        log("judge", name, "fast", self.tok.is_fast, self.id2label, "entailment idx", self.ent, "cached directional", len(self.cache))
    def ids(self, t):
        if t not in self._ids: self._ids[t] = self.tok(t, add_special_tokens=False, max_length=CUT, truncation=True)["input_ids"]
        return self._ids[t]
    def forward(self, dir_pairs, tokens_per_batch=8000):
        seqs = [[self.tok.cls_token_id] + self.ids(p) + [self.tok.sep_token_id] + self.ids(h) + [self.tok.sep_token_id] for p, h in dir_pairs]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); res = [0.0] * len(seqs); s = 0
        while s < len(order):
            e = s + 1
            while e < len(order) and (e - s + 1) * len(seqs[order[e]]) <= tokens_per_batch and e - s < 256: e += 1
            b = order[s:e]; s = e; L = max(len(seqs[i]) for i in b)
            ids = torch.full((len(b), L), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b): ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); att[r, :len(seqs[i])] = 1
            with torch.inference_mode(): pr = torch.softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)[:, self.ent].numpy()
            for r, i in enumerate(b): res[i] = float(pr[r])
        return res
    def ensure(self, pairs):
        """pairs: iterable of (a, b) sentence pairs (a != b); both directions are scored and cached."""
        need = {}
        for a, b in pairs:
            for p, h in ((a, b), (b, a)):
                k = (md5(p), md5(h))
                if k not in self.cache: need[k] = (p, h)
        if not need: return 0
        keys = list(need); vals = [need[k] for k in keys]; CH = 6000
        for c in range(0, len(keys), CH):
            out = self.forward(vals[c:c + CH]); new = {k: v for k, v in zip(keys[c:c + CH], out)}
            self.cache.update(new); self.nshard += 1
            pickle.dump(new, open(f"{OUT}/pairs_{self.name.replace('/', '_')}_{self.nshard:05d}.pkl", "wb"))
            log(self.name.split("/")[-1], "directional pairs scored", min(c + CH, len(keys)), "/", len(keys))
        return len(keys)
    def p(self, a, b): return min(self.cache[(md5(a), md5(b))], self.cache[(md5(b), md5(a))])

def needed(u, o):
    """non-identical sentence pairs (a from u, b from o) of the pairing rule, as a sorted-unique list."""
    if u == o: return []
    su, eu = sent_info(u); so, eo = sent_info(o); sim = eu @ eo.T; out = set()
    for i in range(len(su)): out.add((su[i], so[int(sim[i].argmax())]))
    for j in range(len(so)): out.add((su[int(sim[:, j].argmax())], so[j]))
    return [pr for pr in out if pr[0] != pr[1]]
def score_with(J, pairs):
    """pairs = needed(u, o); returns (score, lowest pair)."""
    if not pairs: return 1.0, None
    sc = [(J.p(a, b), (a, b)) for a, b in pairs]; return min(sc, key=lambda x: x[0])
def canon(pr): return pr if pr[0] <= pr[1] else (pr[1], pr[0])

# ================================================================ step 1: calibration and judge choice
cal_sorted = es.cal_sorted; mB = len(half_B)
E_cal = embed_texts([orig[n] for n in cal_sorted]); cidx = {n: i for i, n in enumerate(cal_sorted)}
E_hB_o = E_cal[[cidx[n] for n in half_B]]
de_hB = [ptext("de", n) for n in half_B]; E_hB_de = embed_texts(de_hB)
d_pos = np.linalg.norm(E_hB_de - E_hB_o, axis=1)
idx_r = math.ceil(round(0.90 * (mB + 1), 9)); srt = np.sort(d_pos); r_over = idx_r > mB
R_ = float(srt[-1] if r_over else srt[idx_r - 1])
k_tau = math.floor(round(0.05 * (mB + 1), 9))
Dneg = cdist(E_hB_de, E_hB_o)                                   # (m paraphrases, m half-B originals)
neg_pairs_idx = [(i, j) for i in range(mB) for j in range(mB) if i != j and Dneg[i, j] <= R_]
neg_rule = "all other half-B instructions within r"
if len(neg_pairs_idx) < 30:
    neg_rule = "fewer than 30 negatives within r: the 5 nearest other half-B instructions of each paraphrase"
    neg_pairs_idx = [(i, int(j)) for i in range(mB) for j in [x for x in np.argsort(Dneg[i]) if x != i][:5]]
pos_pairs = [(de_hB[i], orig[half_B[i]]) for i in range(mB)]
neg_pairs = [(de_hB[i], orig[half_B[j]]) for i, j in neg_pairs_idx]
log("m", mB, "r", R_, "negatives", len(neg_pairs), neg_rule)
def auc(pos, neg): return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
cal_needed = [needed(u, o) for u, o in pos_pairs + neg_pairs]
cal_unique = sorted({canon(pr) for L in cal_needed for pr in L})
save_sent_emb()
JUD = {}; judges = {}
for mn in MODELS:
    J = Judge(mn); J.ensure(cal_unique)
    sc = np.array([score_with(J, L)[0] for L in cal_needed]); ps, ns = sc[:mB], sc[mB:]
    tau = float(np.sort(ps)[k_tau - 1]) if k_tau > 0 else -np.inf
    judges[mn] = {"auc": auc(ps, ns), "tau": None if tau == -np.inf else tau, "share_negatives_ge_tau": float((ns >= tau).mean()), "share_positives_ge_tau": float((ps >= tau).mean()),
                  "n_positives": int(len(ps)), "n_negatives": int(len(ns)), "id2label": {str(k): v for k, v in J.id2label.items()}, "tokenizer_fast": bool(J.tok.is_fast)}
    JUD[mn] = tau; log(mn, judges[mn])
    del J; import gc; gc.collect()
best = max(v["auc"] for v in judges.values())
CHOSEN = [mn for mn in MODELS if judges[mn]["auc"] >= best - 0.005][0]       # MODELS is ordered smallest first
TAU = JUD[CHOSEN]
RESULT["notes"] = {
    "split": {"calibration": len(cal_names), "evaluation": K, "half_B": mB},
    "sentence_split": r"re.split(r'(?<=[.!?])\s+', text.strip()), as paraphrase_make.py (empty result -> the whole text is one sentence)",
    "pairing": "each sentence of u with its most similar sentence of o_k by cosine of MiniLM sentence embeddings (max 254 tokens), and each sentence of o_k with its most similar sentence of u; score = min over all these pairs of min(P_entail(a,b), P_entail(b,a)); identical sentences 1; identical texts 1; ties in argmax -> first",
    "cut": "each sentence cut at 128 judge tokens; input [CLS] a [SEP] b [SEP]; no token_type_ids",
    "r": "split-conformal quantile alpha 0.10 of the MiniLM whole-text distance (cut 254 tokens) between half-B German paraphrase and own instruction: index ceil(0.9 (m+1)) of the sorted values",
    "tau": "alpha 0.05: the k-th smallest positive score, k = floor(0.05 (m+1)), with the chosen judge",
    "judge_choice": f"AUC of sentence score, positives = (German paraphrase, own instruction), negatives = ({neg_rule}); highest AUC; models within 0.005 of the best: the first in the listed order (xsmall, cross-encoder base, MoritzLaurer base) wins, so ties between the two base models go to the cross-encoder",
    "candidates": "every task within r of the text is judged, no limit; own pair is also scored if outside r (reporting only)",
    "R": "text outside every zone goes to the task with the nearest MiniLM whole-text signature (all 700)",
    "C": "share of tasks whose original lies in the zone of an older task; U share with route(o_k) = k",
    "threads": 3}
RESULT["calibration"] = {"m": mB, "r": R_, "r_index": idx_r, "r_index_exceeds_m_used_largest": bool(r_over), "k": k_tau, "tau": TAU, "chosen": CHOSEN, "negatives_rule": neg_rule,
                         "judges": judges, "d_pos_min_med_max": [float(d_pos.min()), float(np.median(d_pos)), float(d_pos.max())]}
dump(); log("step 1 written; chosen", CHOSEN, "r", R_, "tau", TAU)

# ================================================================ step 2/3: evaluation
J = Judge(CHOSEN)
E_ev = {k: embed_texts(TXT[k]) for k in TXT}
DM = {k: cdist(E_ev[k], E_ev["orig"]) for k in TXT}
V = {}
class View: pass
def prepare(kind):
    t0 = time.time(); v = View(); D = DM[kind]; inside = D <= R_; v.D, v.inside = D, inside
    ii, jj = np.where(inside | np.eye(K, dtype=bool))
    pl = {}
    for i, j in zip(ii, jj): pl[(i, j)] = needed(TXT[kind][i], TXT["orig"][j])
    save_sent_emb()
    uniq = sorted({canon(pr) for L in pl.values() for pr in L})
    J.ensure(uniq)
    SC = np.full((K, K), np.nan)
    for (i, j), L in pl.items(): SC[i, j] = score_with(J, L)[0]
    v.S, v.pl = SC, pl
    cnt = inside.sum(1); v.cnt = cnt
    v.n_unique_pairs = len(uniq); v.n_pair_occurrences = int(sum(len(L) for (i, j), L in pl.items() if inside[i, j]))
    v.wall = time.time() - t0; V[kind] = v
    log("prepared", kind, "within r mean/median/max", float(cnt.mean()), float(np.median(cnt)), int(cnt.max()), "unique sentence pairs", len(uniq), "wall s", round(v.wall))
    return v
def zone(kind): v = V[kind]; return v.inside & (np.nan_to_num(v.S, nan=-1.0) >= TAU)
def route_of(Z, D): return np.where(Z.any(1), Z.argmax(1), D.argmin(1)), Z.any(1)
def c_u():
    Z = zone("orig"); older = ar[None, :] < ar[:, None]; rt, _ = route_of(Z, V["orig"].D)
    return {"C": float((Z & older).any(1).mean()), "U": float((rt == ar).mean())}
def para_measures(kind):
    Z = zone(kind); rt, cert = route_of(Z, V[kind].D)
    return {"Z": float(Z[ar, ar].mean()), "P": float((cert & (rt == ar)).mean()), "M": float((cert & (rt != ar)).mean()), "R": float((rt == ar).mean()), "n_texts": K}
def cand_info(kind):
    v = V[kind]; return {"tasks_within_r_mean": float(v.cnt.mean()), "median": float(np.median(v.cnt)), "max": int(v.cnt.max()), "texts_with_none_within_r": int((v.cnt == 0).sum()),
                         "unique_sentence_pairs_judged": v.n_unique_pairs, "sentence_pair_occurrences": v.n_pair_occurrences, "wall_seconds": v.wall,
                         "mean_wall_ms_per_text_cached_batch": 1000 * v.wall / K}
prepare("orig")
RESULT["candidates"] = {"orig": cand_info("orig")}; RESULT["table"] = {"originals": c_u()}
for p, nm in (("de", "German"), ("fr", "French"), ("es", "Spanish")):
    prepare(p); RESULT["candidates"][p] = cand_info(p); RESULT["table"][nm] = {**c_u(), **para_measures(p)}; dump(); log(nm, RESULT["table"][nm])
RESULT["notes"]["step2_written"] = True; dump(); log("step 2 written")
prepare("it"); RESULT["candidates"]["it"] = cand_info("it"); RESULT["table"]["Italian"] = {**c_u(), **para_measures("it")}
t = RESULT["table"]["Italian"]
RESULT["verdict"] = {"pivot": "Italian", "P": t["P"], "M": t["M"], "C": t["C"],
                     "outcome": "success" if (t["P"] >= 0.75 and t["M"] <= 0.05 and t["C"] <= 0.19) else ("against" if (t["P"] < 0.55 or t["M"] > 0.09 or t["C"] > 0.25) else "partial")}
dump(); log("step 3 written", RESULT["table"]["Italian"], RESULT["verdict"])

# ================================================================ step 4: views
# collisions under the paper's rule
mt = json.load(open(os.path.join(HERE, "routing", "many_tasks.json")))
coll = mt["random order, seed 0"]["875"].get("collisions", [])
cp = [(c["task"], c["older"], c.get("same_definition")) for c in coll if c["task"] in by_name and c["older"] in by_name]
Ec = {n: v for n, v in zip(sorted({x for a, b, _ in cp for x in (a, b)}), embed_texts([orig[n] for n in sorted({x for a, b, _ in cp for x in (a, b)})]))}
cl = [needed(orig[a], orig[b]) for a, b, _ in cp]; J.ensure(sorted({canon(pr) for L in cl for pr in L}))
cs = np.array([score_with(J, L)[0] for L in cl]); cd = np.array([np.linalg.norm(Ec[a] - Ec[b]) for a, b, _ in cp]); sd = np.array([bool(x) for _, _, x in cp])
RESULT["collisions"] = {"source": "many_tasks.json, 'random order, seed 0', 875 tasks, key 'collisions' (task, older); pair = (original of task, original of older)",
                        "listed": len(coll), "evaluated": len(cp), "tau": TAU, "r": R_, "n_score_ge_tau": int((cs >= TAU).sum()), "n_in_zone_d_le_r_and_score_ge_tau": int(((cs >= TAU) & (cd <= R_)).sum()),
                        "n_d_le_r": int((cd <= R_).sum()), "median_score": float(np.median(cs)), "n_same_definition": int(sd.sum()), "n_score_ge_tau_not_same_definition": int((cs[~sd] >= TAU).sum()),
                        "n_not_same_definition": int((~sd).sum()), "share_score_ge_tau": float((cs >= TAU).mean()), "share_score_ge_tau_not_same_definition": float((cs[~sd] >= TAU).mean())}
dump(); log("collisions", RESULT["collisions"])

# benchmark
names15 = sorted(set(FIVE) | set(ORDER1)); bo = {}
for n in names15:
    dj = json.load(open(os.path.join("CL_Benchmark", "SuperNI", n, "train.json"))); bo[n] = (dj["Definition"][0] if isinstance(dj["Definition"], list) else dj["Definition"]).strip()
wd = {n: [bo[n]] + sr.REWORDED.get(n.split("_")[0], []) for n in names15}
Eb = {n: embed_texts([bo[n]])[0] for n in names15}
bl = {(n, l, m): needed(wd[n][l], bo[m]) for n in FIVE for l in range(len(wd[n])) for m in set(ORDER1) | set(FIVE)}
J.ensure(sorted({canon(pr) for L in bl.values() for pr in L}))
rows = []
for n in FIVE:
    for l, lev in enumerate(sr.LEVELS[:len(wd[n])]):
        u = wd[n][l]; e = embed_texts([u])[0]; row = {"task": n, "level": lev}
        d_own = float(np.linalg.norm(e - Eb[n])); s_own = score_with(J, bl[(n, l, n)])[0]
        row.update({"d_own": d_own, "score_own": s_own, "in_own_zone": bool(d_own <= R_ and s_own >= TAU)})
        for label, order in (("five", FIVE), ("fifteen", ORDER1)):
            ds = [float(np.linalg.norm(e - Eb[m])) for m in order]; zs = [ds[q] <= R_ and score_with(J, bl[(n, l, m)])[0] >= TAU for q, m in enumerate(order)]
            if any(zs): q = zs.index(True); cert = True
            else: q = int(np.argmin(ds)); cert = False
            row[f"route_{label}"] = {"task": order[q], "certified": cert, "correct": order[q] == n}
        rows.append(row)
summ = {}
for lev in sr.LEVELS:
    rr = [r for r in rows if r["level"] == lev]
    summ[lev] = {"n": len(rr), "in_own_zone": sum(r["in_own_zone"] for r in rr)}
    for lab in ("five", "fifteen"):
        summ[lev][f"routed_right_{lab}"] = sum(r[f"route_{lab}"]["correct"] for r in rr)
        summ[lev][f"certified_wrong_{lab}"] = sum(r[f"route_{lab}"]["certified"] and not r[f"route_{lab}"]["correct"] for r in rr)
RESULT["benchmark"] = {"r": R_, "tau": TAU, "rows": rows, "summary": summ}; dump(); log("benchmark", summ)

# examples: first 12 wrong certificates, Italian pivot
Z = zone("it"); rt, cert = route_of(Z, V["it"].D); ms = [i for i in range(K) if cert[i] and rt[i] != i]
ex = []
for i in ms[:12]:
    j = int(rt[i]); s, pr = score_with(J, V["it"].pl[(i, j)])
    ex.append({"pos": i, "task": ev_names[i], "claimed_by": ev_names[j], "claimed_by_pos": j, "d": float(V["it"].D[i, j]), "score": s, "own_zone_holds": bool(Z[i, i]),
               "lowest_pair_u_sentence": pr[0] if pr else None, "lowest_pair_o_sentence": pr[1] if pr else None})
RESULT["examples"] = {"pivot": "Italian", "n_wrong_certificates": len(ms), "first_12": ex}; dump()

# cost: sequential, one text at a time, no cache of the judge or of the text's sentences
def one_text(kind, i):
    t0 = time.perf_counter(); u = TXT[kind][i]
    f = enc.encode([enc.ids(u, es.MAXTOK)], use_cache=False).astype(np.float64); d = np.linalg.norm(E_ev["orig"] - f, axis=1); ins = np.where(d <= R_)[0]
    su, eu = sent_info(u, use_cache=False); res = {}; allp = []
    for j in ins:
        so, eo = sent_info(orig[ev_names[j]]); sim = eu @ eo.T; ps = set()
        if u == orig[ev_names[j]]: res[j] = []; continue
        for a in range(len(su)): ps.add((su[a], so[int(sim[a].argmax())]))
        for b in range(len(so)): ps.add((su[int(sim[:, b].argmax())], so[b]))
        res[j] = [p for p in ps if p[0] != p[1]]; allp += res[j]
    up = sorted({canon(p) for p in allp}); dirs = [x for p in up for x in (p, (p[1], p[0]))]; out = J.forward(dirs) if dirs else []
    pm = {}
    for q, p in enumerate(up): pm[p] = min(out[2 * q], out[2 * q + 1])
    sc = {j: (min([pm[canon(p)] for p in L]) if L else 1.0) for j, L in res.items()}
    z = [j for j in ins if sc[j] >= TAU]; _ = min(z) if z else int(d.argmin())
    return (time.perf_counter() - t0) * 1000, len(ins), len(up)
sample = [(k, i) for k in ("orig", "fr", "it") for i in range(0, K, 35)]
one_text(*sample[0])
tm = [one_text(k, i) for k, i in sample]
RESULT["cost"] = {"chosen_judge": CHOSEN, "timing_sample_texts": len(sample), "mean_ms_per_text": float(np.mean([a for a, _, _ in tm])), "median_ms": float(np.median([a for a, _, _ in tm])),
                  "p90_ms": float(np.quantile([a for a, _, _ in tm], 0.9)), "mean_tasks_within_r_in_sample": float(np.mean([b for _, b, _ in tm])), "mean_unique_sentence_pairs_in_sample": float(np.mean([c for _, _, c in tm])),
                  "note": "sequential, one text at a time, 3 threads, no judge cache; MiniLM encode of the text and its sentences + distances to 700 originals + all sentence pairs within r in both directions + route; task-side sentence embeddings precomputed; sample = every 35th task of originals, French, Italian",
                  "total_unique_sentence_pairs_per_pivot": {k: V[k].n_unique_pairs for k in V}}
dump(); log("cost", RESULT["cost"]); log("done")
