"""The joint recogniser, its audit and the handicap (paper_certified/sec_audit.tex). numpy only.

Instead of reading an instruction, the system keeps one linear (ridge) recogniser over the features of a
frozen encoder. Every task adds sums, never stored inputs; a newer recogniser sees all tasks so far.
  plain rule  (nu=None): route = argmax of the newest recogniser (ties go to the older task). The sums do not
              depend on the order of arrival, so the final recogniser is the same for every order.
  handicap    (nu=int):  the newcomer l takes an input only if its margin over the best older task is larger
              than a threshold theta_l, the largest over older tasks of the (nu+1)-th largest margin among that
              task's held-out inputs (minus infinity if a task has at most nu; never below zero). The newest
              taker wins. The recogniser and theta_l are stored at arrival and never change.
Held-out inputs are never used for fitting. Routes are the same as hidden_rules_study.py (routes_ridge,
routes_handicap with floor=True)."""
import math
import numpy as np


def standardiser(pool_features):
    """Return a function that standardises features with the mean and std of a pool of unrelated inputs."""
    P = np.asarray(pool_features)
    mu = P.mean(0)
    sd = P.std(0)
    sd[sd < 1e-6] = 1.0
    return lambda X: ((np.asarray(X) - mu) / sd).astype(np.float64)


def _binom_cdf(c, m, p):
    """P(X <= c) for X ~ Binomial(m, p), by log-gamma."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if c >= m else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(m + 1)
    terms = [lg - math.lgamma(i + 1) - math.lgamma(m - i + 1) + i * lp + (m - i) * lq for i in range(c + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson_upper(c, m, confidence=0.95):
    """One-sided upper bound p with P(X <= c; m, p) = 1 - confidence, by bisection. 1.0 if c >= m."""
    if c >= m:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(c, m, mid) > 1.0 - confidence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _add_one(X):
    return np.hstack([X, np.ones((len(X), 1))])


class JointRecogniser:
    """Running sums of a ridge recogniser over tasks, with audit and optional handicap (see module docstring)."""

    def __init__(self, dim, ridge=1.0, nu=None):
        self.dim, self.ridge, self.nu = int(dim), float(ridge), (None if nu is None else int(nu))
        d = self.dim + 1
        self.G = np.zeros((d, d))          # sum of the weighted x x^T of all tasks (constant 1 appended)
        self.C = np.zeros((d, 0))          # one column per task: weighted sum of x
        self.W = []                        # W[l]: recogniser stored at arrival l (d x (l+1))
        self.theta = []                    # theta[l]: threshold stored at arrival l (handicap only)
        self.heldout = []                  # held-out features of every task
        self.hist = []                     # hist[j][:, k] = routes of held-out of task j at arrival j + k

    @property
    def n_tasks(self):
        return len(self.heldout)

    # ---------------------------------------------------------------- state of the newest recogniser
    def _solve(self, G, C):
        d = self.dim + 1
        return np.linalg.solve(G + self.ridge * self.dim * np.eye(d), C)

    def weights(self):
        """The newest recogniser (d x tasks). With the plain rule it does not depend on the order of arrival."""
        return self.W[-1]

    def _margin(self, X, l):
        S = _add_one(np.asarray(X, float)) @ self.W[l]
        return S[:, l] - S[:, :l].max(1)

    def route(self, features, at=None):
        """Task index (0-based) for every row, under the state after arrival `at` (default: the newest)."""
        X = np.asarray(features, float)
        T = self.n_tasks - 1 if at is None else int(at)
        if self.nu is None:
            return (_add_one(X) @ self.W[T]).argmax(1)
        r = np.zeros(len(X), int)
        for l in range(1, T + 1):
            r = np.where(self._margin(X, l) > self.theta[l], l, r)
        return r

    # ---------------------------------------------------------------- adding a task
    def add_task(self, fit_features, heldout_features):
        """Add one task. fit_features (n x dim) enter the sums; heldout_features (m x dim) are only stored."""
        X = np.asarray(fit_features, float)
        l = self.n_tasks
        Z = _add_one(X)
        w = 200.0 / len(Z)                 # every task has the same total weight
        self.G += w * Z.T @ Z
        self.C = np.hstack([self.C, (w * Z.sum(0))[:, None]])
        self.W.append(self._solve(self.G, self.C))
        H = np.asarray(heldout_features, float)
        if self.nu is None:
            self.theta.append(-np.inf)
        elif l == 0:
            self.theta.append(-np.inf)
        else:
            th = -np.inf
            for j in range(l):
                mg = np.sort(self._margin(self.heldout[j], l))[::-1]
                th = max(th, mg[self.nu] if self.nu < len(mg) else -np.inf)
            self.theta.append(max(th, 0.0))
        self.heldout.append(H)
        self.hist.append(np.zeros((len(H), 0), int))
        for j in range(l + 1):             # route of every held-out input at this moment (for the audit)
            self.hist[j] = np.hstack([self.hist[j], self.route(self.heldout[j])[:, None]])
        return l

    # ---------------------------------------------------------------- audit
    def audit(self, task, confidence=0.95):
        """Audit of one task: how many of its held-out inputs now have another route than right after its arrival.
        `valid` is False under the handicap: the held-out inputs then set the thresholds, so the binomial bound
        (which needs them to be untouched by the rule) does not apply; use bound_in_advance instead."""
        h = self.hist[task]
        m = len(h)
        changed = int((h[:, -1] != h[:, 0]).sum())
        ever = int((h != h[:, [0]]).any(1).sum())
        return {"m": m, "changed": changed, "changed_ever": ever, "estimate": changed / m if m else float("nan"),
                "upper": clopper_pearson_upper(changed, m, confidence) if m else 1.0, "valid": self.nu is None}

    def bound_in_advance(self, task):
        """(later tasks) * (nu + 1) / (m + 1) for the handicap rule (Theorem thm:handicap); None for the plain rule."""
        if self.nu is None:
            return None
        return (self.n_tasks - 1 - task) * (self.nu + 1) / (len(self.heldout[task]) + 1)

    def newcomer_report(self):
        """For the last task: share of its own held-out inputs routed to it, and for every older task the number
        of its held-out inputs whose route changed at this arrival. Needs no new adapter."""
        l = self.n_tasks - 1
        own = float((self.hist[l][:, -1] == l).mean())
        moved = [int((self.hist[j][:, -1] != self.hist[j][:, -2]).sum()) for j in range(l)]
        return {"task": l, "own_share": own, "changed_old": moved}

    # ---------------------------------------------------------------- state
    def to_dict(self):
        return {"dim": self.dim, "ridge": self.ridge, "nu": self.nu, "G": self.G.tolist(), "C": self.C.tolist(),
                "W": [w.tolist() for w in self.W], "theta": list(self.theta),
                "heldout": [h.tolist() for h in self.heldout], "hist": [h.tolist() for h in self.hist]}

    @classmethod
    def from_dict(cls, s):
        o = cls(s["dim"], s["ridge"], s["nu"])
        o.G, o.C = np.array(s["G"]), np.array(s["C"]).reshape(s["dim"] + 1, -1)
        o.W = [np.array(w) for w in s["W"]]
        o.theta = [float(t) for t in s["theta"]]
        o.heldout = [np.array(h).reshape(-1, s["dim"]) for h in s["heldout"]]
        o.hist = [np.array(h, dtype=int).reshape(len(o.heldout[i]), -1) for i, h in enumerate(s["hist"])]
        return o
