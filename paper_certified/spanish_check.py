"""One confirmation check, fixed before the Spanish paraphrases are read (last section of silent_adapters/PLAN.md).
(a) the paper's rule (rule B, kappa 0.5, old ruler, T5-base word-vectors) against (b) the MiniLM encoder zone (ruler metric,
alpha = 0.2, ruler from calibration half A, radius from half B, both from German-pivot paraphrases), on the Spanish-pivot
paraphrased inputs of the 700 evaluation tasks. Nothing is fitted or chosen again. Step 0 recomputes the earlier numbers
(radius 49.988, French encoder numbers, reference of the paper's rule) and stops before the Spanish file is opened if they do
not reproduce. Then step 1 (primary, written to JSON at once), then step 2 views (2a other encoder settings, 2b five other
random splits, 2c benchmark tasks).

Run from the folder of the official GainLoRA code (it reads CL_Benchmark from there):
    S=$FK_LORA_SCRATCH
    cd $S/ext_gainlora/repo && HF_HOME=$S/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=3 \
        $S/mm_env/bin/python spanish_check.py
Reuses encoder_study.py by import (Enc, prepare_openings, dist_matrix, measures, conformal_radius, inv_sqrt, view_benchmark) and
copies the T5-word-vector code of ruler_study.py (run_table) for the paper's rule. Caches in $S/spanish_check/.
Items are built as in both scripts: unedited inputs[:6], paraphrased inputs[:3] per pivot; paraphrases identical to the original
are kept (as the earlier scripts keep them).
"""
import json, os, sys, time, random, types
os.environ.setdefault("OMP_NUM_THREADS", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import encoder_study as E
import torch
torch.set_num_threads(3)
from transformers import AutoTokenizer, T5EncoderModel

S = E.S; sr = E.sr; log = E.log
OUTDIR = f"{S}/spanish_check"; os.makedirs(OUTDIR, exist_ok=True)
JSON_PATH = os.path.join(HERE, "routing", "spanish_check.json")
names, by_name, para = E.names, E.by_name, E.para
GRID = E.GRID; NG = len(GRID); ALPHAS = E.ALPHAS
KINDS = {0: "unedited", 1: "de", 2: "fr", 3: "es"}
PIVNAME = {1: "German", 2: "French", 3: "Spanish"}
RES = {"notes": {}}
def dump(): json.dump(RES, open(JSON_PATH, "w"), indent=1)
def r3(x): return round(float(x), 3)

# ------------------------------------------------------------------ split helper (same procedure as the original, seed 0)
def make_split(seed):
    rr = random.Random(seed); sh = names[:]; rr.shuffle(sh)
    cal, ev = sh[:175], sh[175:]
    cs = sorted(cal)
    return {"cal": cal, "ev": ev, "A": cs[0::2], "B": cs[1::2]}
SPL0 = make_split(0)
assert SPL0["ev"] == E.ev_names and SPL0["A"] == E.half_A and SPL0["B"] == E.half_B

# ------------------------------------------------------------------ items and the feature store of the encoder
class Items:
    """Row index for every (task, kind, input index); texts for kind 3 are added after the Spanish file is opened."""
    def __init__(self):
        self.row = {}; self.text = []; self.es_loaded = False
        for n in names:
            t = by_name[n]
            for i, x in enumerate(t["inputs"][:6]): self._add((n, 0, i), E.inputs_text(t["definition"], x))
            for kd, p in ((1, "de"), (2, "fr")):
                for i, x in enumerate(t["inputs"][:3]): self._add((n, kd, i), E.inputs_text(para["tasks"][n][p], x))
    def _add(self, key, text): self.row[key] = len(self.text); self.text.append(text)
    def add_spanish(self, es):
        for n in names:
            t = by_name[n]
            for i, x in enumerate(t["inputs"][:3]): self._add((n, 3, i), E.inputs_text(es["tasks"][n]["es"], x))
        self.es_loaded = True
    def keys(self, n, kd):
        t = by_name[n]; m = min(6 if kd == 0 else 3, len(t["inputs"]))
        return [(n, kd, i) for i in range(m)]
IT = Items()
NMAX = len(IT.text) + 875 * 3 + 5

class Store:
    def __init__(self, enc):
        self.enc = enc; self.F = np.zeros((NMAX, NG, enc.dim), np.float32); self.mask = np.zeros((NMAX, NG), bool)
        self.n_encoded = 0; self.t_encoded = 0.0
        pf = f"{OUTDIR}/store_F.npy"
        if os.path.exists(pf) and os.path.exists(f"{OUTDIR}/store_mask.npy"):
            F = np.load(pf, mmap_mode="r"); m = np.load(f"{OUTDIR}/store_mask.npy")
            if F.shape == self.F.shape: self.F[:] = F; self.mask[:] = m; log("store loaded from disk", int(m.sum()), "filled cells")
    def save(self):
        np.save(f"{OUTDIR}/store_F.npy", self.F); np.save(f"{OUTDIR}/store_mask.npy", self.mask)
    def ensure(self, keys, gpos, chunk=80):
        """encode every (key, grid position) not yet in the store; unique token prefixes of one chunk are encoded once."""
        rows = [IT.row[k] for k in keys]; gpos = sorted(set(int(g) for g in gpos))
        todo = [r for r in rows if any(not self.mask[r, g] for g in gpos)]
        t0 = time.time(); nseq = 0
        for s in range(0, len(todo), chunk):
            rs = todo[s:s + chunk]; seqs = []; slots = []
            for r in rs:
                ids = self.enc.ids(IT.text[r])
                for g in gpos:
                    if not self.mask[r, g]: seqs.append(ids[:GRID[g]]); slots.append((r, g))
            V = self.enc.encode(seqs, use_cache=False)
            for (r, g), v in zip(slots, V): self.F[r, g] = v; self.mask[r, g] = True
            nseq += len(seqs)
        self.n_encoded += nseq; self.t_encoded += time.time() - t0
        if nseq: log(f"encoded {nseq} sequences for {len(todo)} items in {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ encoder pipeline (as run_encoder / measures of encoder_study)
class EncPipe:
    def __init__(self):
        self.enc = E.Enc("minilm"); self.nk, self.Lk, self.ops, self.sig = E.prepare_openings(self.enc)
        self.st = Store(self.enc); self.gi = {n: GRID.index(self.Lk[n]) for n in names}
        self.calib_cache = {}
    def fill_from_old_cache(self):
        """features of the earlier study for the 700 evaluation tasks (kinds 0,1,2) at the grid positions of its U"""
        F = np.load(f"{S}/encoder_study/F_minilm.npy")
        ev_L = np.array([self.Lk[n] for n in E.ev_names]); U = np.unique(ev_L); pos = [GRID.index(int(u)) for u in U]
        r = 0
        for n in E.ev_names:
            for kd in (0, 1, 2):
                for k in IT.keys(n, kd):
                    row = IT.row[k]; self.st.F[row][pos] = F[r]; self.st.mask[row, pos] = True; r += 1
        assert r == len(F), (r, len(F))
        return U, pos
    def calibrate(self, spl, metrics=("plain", "ruler")):
        key = tuple(spl["cal"])
        if key in self.calib_cache: return self.calib_cache[key]
        enc, st = self.enc, self.st
        kA = []
        for n in spl["A"]:
            for i in range(min(3, len(by_name[n]["inputs"]))): kA += [(n, 0, i), (n, 1, i)]
        bn = [n for n in spl["B"] if by_name[n]["inputs"]]
        kB = [(n, 1, 0) for n in bn]
        for g in sorted(set(self.gi[k[0]] for k in kA + kB)):
            st.ensure([k for k in kA + kB if self.gi[k[0]] == g], [g])
        FA = np.stack([st.F[IT.row[k], self.gi[k[0]]] for k in kA]).astype(np.float64)
        disp = FA[1::2] - FA[0::2]
        Sig_p = disp.T @ disp / len(disp)
        Sigma = 0.9 * Sig_p + 0.1 * (np.trace(Sig_p) / enc.dim) * np.eye(enc.dim)
        Rul = E.inv_sqrt(Sigma)
        FB = np.stack([st.F[IT.row[k], self.gi[k[0]]] for k in kB]).astype(np.float64)
        SB = np.stack([self.sig[n] for n in bn]).astype(np.float64)
        T = {"plain": np.eye(enc.dim), "ruler": Rul}
        rad = {}; scores = {}
        for m in metrics:
            sc = E.snap(np.linalg.norm((FB - SB) @ T[m], axis=1)); scores[m] = sc
            rad[m] = {a: E.conformal_radius(sc, a)[0] for a in ALPHAS}
        out = {"Rul": Rul, "T": T, "radius": rad, "n_disp": len(disp), "m": len(bn)}
        self.calib_cache[key] = out
        return out
    def distances(self, ev, kinds, metrics, calib):
        """-> rows' kind, owner (column in ev), and dist[metric] (n, 700)"""
        st = self.st
        keys, kind, owner = [], [], []
        for c, n in enumerate(ev):
            for kd in kinds:
                for k in IT.keys(n, kd): keys.append(k); kind.append(kd); owner.append(c)
        gev = np.array([self.gi[n] for n in ev]); pos = sorted(set(gev.tolist()))
        st.ensure(keys, pos)
        rows = np.array([IT.row[k] for k in keys])
        F = st.F[rows]
        sg = np.stack([self.sig[n] for n in ev])
        D = {m: E.dist_matrix(F, gev, sg, calib["T"][m] if m == "ruler" else None) for m in metrics}
        return np.array(kind), np.array(owner), D

def enc_measures_for(D, kind, owner, r, pivots):
    mu = kind == 0; res = {}
    for kd in pivots:
        sel = kind == kd
        res[kd] = E.measures(D[mu], owner[mu], D[sel], owner[sel], r)
    return res

# ------------------------------------------------------------------ paper's rule (copy of ruler_study.run_table, t5-base, OLD ruler, rule B kappa 0.5)
PSNAP = 1e-3; KAPPA = 0.5
def psnap(x): x = np.asarray(x, dtype=np.float64).copy(); x[x < PSNAP] = 0.0; return x
def radii_r1(d_bg, d_sig, kappa):
    K = len(d_bg); r = np.zeros(K)
    for j in range(K):
        near = d_bg[j] if j == 0 else min(d_bg[j], d_sig[j, :j].min())
        r[j] = kappa * near
    return r
def proute(dist, radius):
    inside = dist <= radius[None]
    with np.errstate(divide="ignore", invalid="ignore"): rel = dist / radius[None]
    rel = np.where(np.isnan(rel), 0.0, rel)
    return np.where(inside.any(1), inside.argmax(1), rel.argmin(1)), inside.any(1)
def pmeasures(Du, ou, Dp, op, radius):
    K = len(radius); res = {}
    d = psnap(Du); older = np.arange(K)[None, :] < ou[:, None]
    blocked = ((d <= radius[None]) & older).any(1)
    has = np.bincount(ou, minlength=K) > 0
    coll = np.bincount(ou, weights=blocked.astype(float), minlength=K) > 0
    ru, _ = proute(d, radius)
    res["C"] = float(coll[has].sum() / has.sum()); res["U"] = float((ru == ou).mean())
    p = psnap(Dp); rp, cert = proute(p, radius); own = rp == op
    res["P"] = float((cert & own).mean()); res["M"] = float((cert & ~own).mean()); res["R"] = float(own.mean())
    res["Z"] = float((p[np.arange(len(op)), op] <= radius[op]).mean()); res["n"] = int(len(op))
    return res

class PaperRule:
    def __init__(self):
        MAX_LEN = 512; self.MAX_LEN = MAX_LEN
        tok = AutoTokenizer.from_pretrained("t5-base")
        model = T5EncoderModel.from_pretrained("t5-base", torch_dtype=torch.float32).eval()
        Emb = model.get_input_embeddings().weight.detach(); V, D = Emb.shape; self.D = D
        ids_of = lambda text: tok(text, max_length=MAX_LEN, truncation=True)["input_ids"]; self.ids_of = ids_of
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
            T = min(len(ids), int(nmax)); t = torch.tensor(ids[:T]); w = WEIGHT[t]
            return (torch.cumsum(Emb[t].double() * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None]).numpy()
        def phi(ids, ns):
            ns = np.asarray(ns); c = cum(ids, ns.max()); return c[np.minimum(ns, len(ids)) - 1]
        self.phi = phi
        x64 = np.stack([phi(r[:64], [64])[0] for r in rows]); self.mu = x64.mean(0); xc = x64 - self.mu
        cov = xc.T @ xc / (len(x64) - 1); cov = 0.5 * cov + 0.5 * np.eye(D) * np.trace(cov) / D
        self.W = E.inv_sqrt(cov)
        bg_raw = []
        for d, xs in zip(defs, texts):
            r = tok([sr._source(d, x) for x in xs[:20]], max_length=MAX_LEN, truncation=True)["input_ids"]
            n = sr._prefix_len(r); assert n >= 5
            bg_raw.append(phi(r[0][:n], [n])[0])
        bg_raw = np.stack(bg_raw)
        def opening(definition):
            a = ids_of(E.template(definition).replace("{0}", "A B")); b = ids_of(E.template(definition).replace("{0}", "zzq xkj"))
            n = min(sr._prefix_len([a, b]), MAX_LEN); return a[:n], n
        ops = {n: opening(by_name[n]["definition"]) for n in names}
        self.nk = np.array([ops[n][1] for n in names]); self.uniq = np.unique(self.nk); self.where = np.searchsorted(self.uniq, self.nk)
        sig_raw = np.stack([phi(ops[n][0], [ops[n][1]])[0] for n in names])
        self.sigz = (sig_raw - self.mu) @ self.W; self.bgz = (bg_raw - self.mu) @ self.W
        self.col = {n: i for i, n in enumerate(names)}
        self.PD = np.zeros((NMAX, 875), np.float32); self.done = np.zeros(NMAX, bool)
        self.n_done = 0; self.t_done = 0.0
        log("paper rule: T5 word-vector pieces ready; opening lengths min/median/max", int(self.nk.min()), int(np.median(self.nk)), int(self.nk.max()))
    def ensure(self, keys):
        t0 = time.time(); n0 = 0
        for k in keys:
            r = IT.row[k]
            if self.done[r]: continue
            f = self.phi(self.ids_of(IT.text[r]), self.uniq) - self.mu
            z = (f @ self.W)[self.where]
            self.PD[r] = np.linalg.norm(z - self.sigz, axis=1); self.done[r] = True; n0 += 1
        self.n_done += n0; self.t_done += time.time() - t0
        if n0: log(f"paper rule: distances for {n0} texts in {time.time() - t0:.0f}s")
    def evaluate(self, ev, kinds):
        """-> dict kind -> measures, with C/U from kind 0; radius from the 700 evaluation signatures of this split"""
        keys, kind, owner = [], [], []
        for c, n in enumerate(ev):
            for kd in (0,) + tuple(kinds):
                for k in IT.keys(n, kd): keys.append(k); kind.append(kd); owner.append(c)
        self.ensure(keys)
        kind = np.array(kind); owner = np.array(owner)
        D = self.PD[np.array([IT.row[k] for k in keys])][:, [self.col[n] for n in ev]]
        s = self.sigz[[self.col[n] for n in ev]]
        d_sig = psnap(np.linalg.norm(s[:, None] - s[None], axis=2))
        d_bg = psnap(np.linalg.norm(s[:, None] - self.bgz[None], axis=2).min(1))
        radius = radii_r1(d_bg, d_sig, KAPPA)
        mu_k = kind == 0; out = {}
        for kd in kinds: out[kd] = pmeasures(D[mu_k], owner[mu_k], D[kind == kd], owner[kind == kd], radius)
        both = np.isin(kind, kinds)
        out["both"] = pmeasures(D[mu_k], owner[mu_k], D[both], owner[both], radius)
        out["radius_zero"] = int((radius == 0).sum())
        return out

def close(a, b, tol=0.0005): return abs(float(a) - float(b)) <= tol

# ================================================================== main
if __name__ == "__main__":
    t_run = time.time()
    RES["notes"] = {
        "split": "original: Random(0) on sorted names of 875 tasks, 175 calibration (halves A/B by sorted alternation), 700 evaluation in shuffled order; seeds 1-5 for the other splits, same procedure",
        "reused": "encoder_study.py imported (Enc, prepare_openings, dist_matrix, measures with SNAP 1e-4, conformal_radius, inv_sqrt, view_benchmark); ruler_study.py run_table copied for the paper's rule (t5-base, OLD ruler, rule B, kappa 0.5, SNAP 1e-3)",
        "features_cache": "F_minilm.npy of the earlier study used for the 700 evaluation tasks (German and French pivots, unedited) at the grid lengths it holds; 150 cells re-encoded as a check; everything else encoded here",
        "items": "unedited inputs[:6], paraphrased inputs[:3] per pivot; paraphrases identical to the original are kept, as in the earlier scripts",
        "Z_paper_rule": "share of paraphrased inputs inside their own task's zone (distance <= that task's radius); the paper's rule has one radius per task",
        "views": {}}

    # ---------------------------------------------------------------- step 0
    log("STEP 0: reproduction (Spanish file not opened)")
    EP = EncPipe()
    U, pos = EP.fill_from_old_cache()
    rng = np.random.RandomState(0); cells = []
    ev_keys = [k for n in E.ev_names for kd in (0, 1, 2) for k in IT.keys(n, kd)]
    for i in rng.choice(len(ev_keys), 150, replace=False): cells.append((ev_keys[i], int(pos[rng.randint(len(pos))])))
    maxdiff = 0.0
    for k, g in cells:
        v = EP.enc.encode([EP.enc.ids(IT.text[IT.row[k]])[:GRID[g]]], use_cache=False)[0]
        maxdiff = max(maxdiff, float(np.abs(v - EP.st.F[IT.row[k], g]).max()))
    log("cache spot check, max abs feature difference", maxdiff)
    assert maxdiff < 1e-4
    cal0 = EP.calibrate(SPL0)
    r02 = cal0["radius"]["ruler"][0.2]
    kind0, owner0, D0 = EP.distances(E.ev_names, (0, 1, 2), ("plain", "ruler"), cal0)
    m0 = enc_measures_for(D0["ruler"], kind0, owner0, r02, (1, 2))
    old = json.load(open(os.path.join(HERE, "routing", "encoder_study.json")))["minilm"]
    stored_fr = old["table"]["ruler"]["0.2"]["French"]; stored_de = old["table"]["ruler"]["0.2"]["German"]
    ok_enc = close(r02, 49.988, 0.0005) and close(r02, old["radius"]["ruler"]["0.2"]["r"], 1e-3)
    for kk in ("C", "U", "P", "M", "R", "Z"):
        ok_enc &= close(m0[2][kk], stored_fr[kk], 5e-4) and close(m0[1][kk], stored_de[kk], 5e-4)
    prompt_fr = {"C": .121, "U": .881, "P": .621, "M": .060, "R": .900, "Z": .680}
    for kk, v in prompt_fr.items(): ok_enc &= close(m0[2][kk], v, 5.1e-4)
    log("encoder reproduction", r02, {k: r3(m0[2][k]) for k in prompt_fr}, "ok", ok_enc)

    PR = PaperRule()
    p0 = PR.evaluate(E.ev_names, (1, 2))
    ref = json.load(open(os.path.join(HERE, "routing", "ruler_study.json")))["tables"]["t5-base"]["reference"]
    ok_p = all(close(p0["both"][k], ref[k], 5e-4) for k in "CUPMR") and p0["both"]["n"] == ref["paraphrased_inputs"]
    for kk, v in {"C": .1686, "P": .4052, "M": .0755, "R": .7542}.items(): ok_p &= close(p0["both"][kk], v, 5.1e-5 if False else 6e-5)
    log("paper rule reproduction", {k: round(p0["both"][k], 4) for k in "CUPMR"}, "ok", ok_p)
    RES["reproduced"] = {"radius_ruler_0.2": r02,
                         "encoder_ruler_0.2_French": {k: m0[2][k] for k in ("C", "U", "P", "M", "R", "Z")},
                         "encoder_ruler_0.2_German": {k: m0[1][k] for k in ("C", "U", "P", "M", "R", "Z")},
                         "paper_rule_both_pivots": {k: p0["both"][k] for k in "CUPMR"},
                         "stored_radius_ruler_0.2": old["radius"]["ruler"]["0.2"]["r"], "stored_reference": {k: ref[k] for k in "CUPMR"},
                         "cache_spot_check_max_abs_diff": maxdiff, "ok": bool(ok_enc and ok_p)}
    dump()
    if not (ok_enc and ok_p):
        log("REPRODUCTION FAILED, stopping before the Spanish file"); sys.exit(2)

    # ---------------------------------------------------------------- step 1 (Spanish file is opened here)
    log("STEP 1: Spanish pivot")
    es = json.load(open(os.path.join(HERE, "routing", "paraphrases_es.json")))
    assert all("es" in es["tasks"][n] for n in names)
    IT.add_spanish(es)
    RES["paraphrase_stats_es"] = es.get("stats")
    ks = (0, 1, 2, 3)
    kind1, owner1, D1 = EP.distances(E.ev_names, (0, 1, 2, 3), ("plain", "ruler"), cal0)
    m_b = enc_measures_for(D1["ruler"], kind1, owner1, r02, (1, 2, 3))
    p_a = PR.evaluate(E.ev_names, (1, 2, 3))
    log("Spanish items", int((kind1 == 3).sum()), "encoder ruler 0.2", {k: r3(m_b[3][k]) for k in "CUPMRZ"}, "paper rule", {k: r3(p_a[3][k]) for k in "PMRZ"})
    RES["paper_rule"] = {"C": p_a[2]["C"], "U": p_a[2]["U"], "radius_zero": p_a["radius_zero"],
                         "French": {k: p_a[2][k] for k in ("P", "M", "R", "Z", "n")}, "German": {k: p_a[1][k] for k in ("P", "M", "R", "Z", "n")},
                         "Spanish": {k: p_a[3][k] for k in ("P", "M", "R", "Z", "n")}}
    def enc_block(m, a, mm, calib, D, kind, owner):
        blk = {"C": mm[3]["C"], "U": mm[3]["U"], "radius": calib["radius"][m][a]}
        for kd in (2, 1, 3): blk[PIVNAME[kd]] = {k: mm[kd][k] for k in ("P", "M", "R", "Z", "tasks_with_unedited")} if False else {**{k: mm[kd][k] for k in ("P", "M", "R", "Z")}, "n": mm[kd]["paraphrased_inputs"]}
        return blk
    RES["encoder"] = {"ruler": {"0.2": enc_block("ruler", 0.2, m_b, cal0, D1, kind1, owner1)}}
    Pa, Ma, Ra = p_a[3]["P"], p_a[3]["M"], p_a[3]["R"]; Pb, Mb, Rb, Zb = m_b[3]["P"], m_b[3]["M"], m_b[3]["R"], m_b[3]["Z"]
    conf = (Pb - Pa >= 0.15) and (Rb - Ra >= 0.08) and (Mb <= Ma)
    RES["verdict"] = {"P_a": Pa, "P_b": Pb, "M_a": Ma, "M_b": Mb, "R_a": Ra, "R_b": Rb, "Z_b": Zb, "P_gain": Pb - Pa, "R_gain": Rb - Ra,
                      "n_spanish_inputs": int(p_a[3]["n"]), "n_spanish_inputs_encoder": int(m_b[3]["paraphrased_inputs"]),
                      "rule": "confirmed if P_b - P_a >= 0.15 and R_b - R_a >= 0.08 and M_b <= M_a",
                      "outcome": "confirmed" if conf else "not confirmed"}
    dump(); log("PRIMARY WRITTEN", RES["verdict"])

    # ---------------------------------------------------------------- step 2a: other encoder settings (cheap)
    log("STEP 2a")
    for metric in ("ruler", "plain"):
        for a in ALPHAS:
            if metric == "ruler" and a == 0.2: continue
            mm = enc_measures_for(D1[metric], kind1, owner1, cal0["radius"][metric][a], (1, 2, 3))
            RES["encoder"].setdefault(metric, {})[str(a)] = enc_block(metric, a, mm, cal0, None, None, None)
    mm = enc_measures_for(D1["plain"], kind1, owner1, cal0["radius"]["plain"][0.2], (1, 2, 3))
    RES["encoder"]["plain"]["0.2"] = enc_block("plain", 0.2, mm, cal0, None, None, None)
    RES["encoder"] = {m: {a: RES["encoder"][m][a] for a in ("0.2", "0.1", "0.05")} for m in ("ruler", "plain")}
    RES["notes"]["views"]["2a"] = "done (six encoder settings on the Spanish pivot, same radii as the original split)"
    dump()

    # ---------------------------------------------------------------- step 2c: benchmark tasks (cheap)
    log("STEP 2c")
    try:
        ns = types.SimpleNamespace(enc=EP.enc, primary="ruler", Rul=cal0["Rul"], radius={"ruler": {0.1: r02}})
        bm = E.view_benchmark(ns)
        bm["alpha"] = 0.2; bm["variant"] = "ruler"; bm["radius"] = r02
        RES["benchmark"] = {"five": bm["five"], "fifteen": bm["fifteen"], "radius": r02, "alpha": 0.2, "variant": "ruler", "note": bm["note"],
                            "keys": "in_own_zone, certified_own, routed_own, certified_wrong (count); wordings original/light/medium/heavy"}
        RES["notes"]["views"]["2c"] = "done (encoder_study.view_benchmark called with the ruler metric and the alpha = 0.2 radius of the original split)"
    except Exception as ex:
        RES["notes"]["views"]["2c"] = f"failed: {ex!r}"
    dump()

    # ---------------------------------------------------------------- step 2b: five other splits
    log("STEP 2b")
    rate = EP.st.n_encoded / max(EP.st.t_encoded, 1e-9)   # sequences per second measured so far
    seeds = [1, 2, 3, 4, 5]
    # how much encoder work is missing: every needed row at the grid positions of its split
    need = 0
    for sd in seeds:
        sp = make_split(sd); gev = sorted(set(EP.gi[n] for n in sp["ev"]))
        for n in sp["ev"]:
            for kd in (0, 2, 3):
                for k in IT.keys(n, kd):
                    need += sum(1 for g in gev if not EP.st.mask[IT.row[k], g])
    est_enc = need / max(rate, 1e-9)
    t_text = PR.t_done / max(PR.n_done, 1)
    est_paper = t_text * 875 * 12
    log(f"2b estimate: {need} encoder sequences (before dedupe) at {rate:.0f}/s -> {est_enc:.0f}s; paper rule ~{est_paper:.0f}s")
    RES["notes"]["views"]["2b_estimate"] = {"encoder_sequences_missing": int(need), "encoder_rate_per_s": rate, "est_encoder_s": est_enc, "est_paper_rule_s": est_paper}
    if est_enc + est_paper > 1500:
        RES["notes"]["views"]["2b"] = "skipped: estimated computing time above the limit of about 10 minutes of work"
        dump()
    else:
        per = {}
        for sd in seeds:
            sp = make_split(sd)
            cal = EP.calibrate(sp)
            r = cal["radius"]["ruler"][0.2]
            kd_, ow_, D_ = EP.distances(sp["ev"], (0, 2, 3), ("ruler",), cal)
            mm = enc_measures_for(D_["ruler"], kd_, ow_, r, (2, 3))
            pp = PR.evaluate(sp["ev"], (2, 3))
            per[str(sd)] = {"paper_rule": {"C": pp[2]["C"], "French": {k: pp[2][k] for k in ("P", "M", "R")}, "Spanish": {k: pp[3][k] for k in ("P", "M", "R")}},
                            "encoder_ruler_0.2": {"C": mm[2]["C"], "radius": r, "French": {k: mm[2][k] for k in ("P", "M", "R", "Z")},
                                                  "Spanish": {k: mm[3][k] for k in ("P", "M", "R", "Z")}}}
            log("seed", sd, "paper", {k: r3(pp[3][k]) for k in "PMR"}, "enc", {k: r3(mm[3][k]) for k in "PMRZ"}, "C", r3(pp[2]["C"]), r3(mm[2]["C"]))
            RES["splits"] = {"seeds": seeds, "per_seed": per}; dump()
        def agg(vals): return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
        summ = {}
        for rule, meas in (("paper_rule", ("P", "M", "R")), ("encoder_ruler_0.2", ("P", "M", "R", "Z"))):
            summ[rule] = {"C": agg([per[str(s)][rule]["C"] for s in seeds])}
            for pv in ("French", "Spanish"): summ[rule][pv] = {k: agg([per[str(s)][rule][pv][k] for s in seeds]) for k in meas}
        summ["encoder_ruler_0.2"]["radius"] = agg([per[str(s)]["encoder_ruler_0.2"]["radius"] for s in seeds])
        # the comparison on each split, as the primary rule would read it (reported, not a second verdict)
        summ["per_seed_spanish_rule"] = {str(s): bool(per[str(s)]["encoder_ruler_0.2"]["Spanish"]["P"] - per[str(s)]["paper_rule"]["Spanish"]["P"] >= 0.15 and
                                                    per[str(s)]["encoder_ruler_0.2"]["Spanish"]["R"] - per[str(s)]["paper_rule"]["Spanish"]["R"] >= 0.08 and
                                                    per[str(s)]["encoder_ruler_0.2"]["Spanish"]["M"] <= per[str(s)]["paper_rule"]["Spanish"]["M"]) for s in seeds}
        RES["splits"]["summary"] = summ
        RES["notes"]["views"]["2b"] = "done (seeds 1-5; ruler and radius refitted on each split's calibration halves, German pivot, alpha 0.2; paper rule's radii refitted from that split's 700 evaluation signatures)"
        dump()
    RES["notes"]["total_run_seconds"] = time.time() - t_run
    try: EP.st.save()
    except Exception as ex: RES["notes"]["store_save"] = repr(ex)
    dump(); log("done")
