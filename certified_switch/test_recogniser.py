"""Tests of recogniser.py: same routes as the research code, order invariance, the handicap, the theorem, Clopper-Pearson.
Run: python test_recogniser.py   (numpy only; the research code needs scipy and sklearn only to be imported, see below)"""
import importlib.util, math, os, sys, time, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recogniser import JointRecogniser, standardiser, clopper_pearson_upper

T0 = time.time()
try:
    import scipy.stats
    HAVE_SCIPY = True
except ImportError:                      # the research file imports scipy and sklearn at the top; stub them if absent
    HAVE_SCIPY = False
    for name in ("scipy", "scipy.stats", "sklearn", "sklearn.linear_model"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["scipy.stats"].beta = types.SimpleNamespace(ppf=lambda *a: float("nan"))
    sys.modules["sklearn.linear_model"].LogisticRegression = None
spec = importlib.util.spec_from_file_location("hidden_rules_study", os.path.join(HERE, "..", "paper_certified", "hidden_rules_study.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)

DIM, K = 12, 6

def make_data(seed, ties=False):
    rng = np.random.default_rng(seed)
    mus = rng.normal(0, 1.0, (K, DIM))
    sizes = rng.integers(25, 80, K)
    fit = [rng.normal(0, 1, (n, DIM)) + mus[i] for i, n in enumerate(sizes)]
    cal = [rng.normal(0, 1, (int(rng.integers(8, 30)), DIM)) + mus[i] for i in range(K)]
    te = [rng.normal(0, 1, (30, DIM)) + mus[i] for i in range(K)]
    if ties:                             # task 3 is a copy of task 1 (exact ties); some inputs repeated
        fit[3] = fit[1].copy(); cal[3] = cal[1].copy(); te[3] = te[1].copy()
        te[4][:5] = te[1][:5]
    return fit, cal, te

def build(fit, cal, nu, order=None, ridge=1.0):
    rec = JointRecogniser(DIM, ridge=ridge, nu=nu)
    for i in (order if order is not None else range(len(fit))):
        rec.add_task(fit[i], cal[i])
    return rec

# (1) equality with the research code
def check_equal():
    for ties in (False, True):
        fit, cal, te = make_data(1 + ties, ties)
        for nu in (None, 0, 1, 2):
            lam = 1.0
            if nu is None:
                Rt, Rc, _ = H.routes_ridge(fit, cal, te, lam)
            else:
                Rt, Rc, _ = H.routes_handicap(fit, cal, te, lam, [nu], floor=True)[nu]
            rec = build(fit, cal, nu, ridge=lam)
            for j in range(K):
                for T in range(j, K):
                    assert (rec.route(te[j], at=T) == Rt[j][:, T]).all(), (ties, nu, j, T)
                    assert (rec.route(cal[j], at=T) == Rc[j][:, T]).all(), (ties, nu, j, T)
            ms = H.measures([str(i) for i in range(K)], Rt, Rc)["per_task"]
            for j in range(K):
                a = rec.audit(j)
                assert (a["changed"], a["changed_ever"], a["m"]) == (ms[str(j)]["c"], ms[str(j)]["c_ever"], ms[str(j)]["m"]), (ties, nu, j)
                if HAVE_SCIPY:
                    assert abs(a["upper"] - ms[str(j)]["upper95"]) < 1e-9
            st = H.standardiser(fit[0]); mine = standardiser(fit[0])
            assert np.allclose(st(te[0]), mine(te[0]), atol=0, rtol=0)
            r2 = JointRecogniser.from_dict(rec.to_dict())
            assert (r2.route(te[0]) == rec.route(te[0])).all() and r2.audit(0) == rec.audit(0)
    print("1 equal to the research code (nu = None, 0, 1, 2; with ties; standardiser; to_dict):", "audit upper95 compared" if HAVE_SCIPY else "upper95 not compared, scipy missing")

# (2) order invariance of the plain rule
def check_order():
    fit, cal, te = make_data(5)
    perm = [3, 0, 5, 1, 4, 2]
    a, b = build(fit, cal, None), build(fit, cal, None, order=perm)
    Wb = np.zeros_like(a.weights())
    for pos, t in enumerate(perm): Wb[:, t] = b.weights()[:, pos]
    assert np.abs(a.weights() - Wb).max() < 1e-9
    for j in range(K):
        assert (a.route(te[j]) == np.array(perm)[b.route(te[j])]).all()
    print("2 plain rule: same final weights and routes for another order of arrival")

# (3) the handicap changes routes only towards the newest task and takes at most nu held-out inputs
def check_handicap():
    for seed in range(50):
        fit, cal, te = make_data(100 + seed)
        nu = seed % 3
        rec = build(fit, cal, nu)
        for l in range(1, K):
            for j in range(l):
                x = np.vstack([cal[j], te[j]])
                before, after = rec.route(x, at=l - 1), rec.route(x, at=l)
                moved = before != after
                assert (after[moved] == l).all()
                assert (rec.route(cal[j], at=l) == l).sum() <= nu
    print("3 handicap: a route changes only to the newest task; at most nu held-out inputs taken (50 draws)")

# (4) the theorem on exchangeable synthetic data
def check_theorem():
    reps, m, nt, later = 2000, 12, 20, 2
    for nu in (0, 1):
        rng = np.random.default_rng(7 + nu)
        shares = []
        for _ in range(reps):
            mus = rng.normal(0, 0.6, (later + 1, 3))
            sizes = [15, 12, 18]
            rec = JointRecogniser(3, ridge=1.0, nu=nu)
            tests = None
            for i in range(later + 1):
                D = lambda n: rng.normal(0, 1, (n, 3)) + mus[i]
                if i == 0: tests = D(nt)
                rec.add_task(D(sizes[i]), D(m))
            ever = np.zeros(nt, bool)
            r0 = rec.route(tests, at=0)
            for T in range(1, later + 1): ever |= rec.route(tests, at=T) != r0
            shares.append(ever.mean())
        shares = np.array(shares)
        bound = later * (nu + 1) / (m + 1)
        se = shares.std(ddof=1) / math.sqrt(reps)
        assert shares.mean() <= bound + 3 * se, (nu, shares.mean(), bound, se)
        print(f"4 theorem, nu={nu}: share that ever changes {shares.mean():.4f} <= bound {bound:.4f} (+3 se {3 * se:.4f})")

# (5) Clopper-Pearson
def check_cp():
    assert abs(clopper_pearson_upper(0, 200) - (1 - 0.05 ** (1 / 200))) < 1e-9
    assert clopper_pearson_upper(200, 200) == 1.0
    assert abs(clopper_pearson_upper(0, 1, 0.9) - 0.9) < 1e-9
    u = clopper_pearson_upper(3, 50)                     # P(X <= 3; 50, p) = 0.05 at p = 0.147837 (checked with exact binomial sums)
    assert abs(u - 0.147837) < 1e-6, u
    if HAVE_SCIPY:
        from scipy.stats import beta
        for c, m in ((1, 30), (7, 200), (0, 5)):
            assert abs(clopper_pearson_upper(c, m) - beta.ppf(0.95, c + 1, m - c)) < 1e-9
    print(f"5 Clopper-Pearson: c=0,m=200 -> {clopper_pearson_upper(0, 200):.6f}; c=m -> 1.0; c=3,m=50 -> {u:.5f}")

for f in (check_equal, check_order, check_handicap, check_theorem, check_cp):
    f()
assert time.time() - T0 < 20
print(f"OK ({time.time() - T0:.1f} s)")
