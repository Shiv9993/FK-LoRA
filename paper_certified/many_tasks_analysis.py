"""The zones on many real tasks (CPU, T5-base word-vectors, final rule). Its rules were written in
silent_adapters/PLAN.md before it was run. Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/many_tasks_analysis.py <file made by many_tasks_fetch.py>

Every task of the default English split of Super-NaturalInstructions gets its zone when it arrives. Measured: how
many tasks arrive with inputs inside an older zone, how many inputs are given to their own task, and how many
routes change later, under `certified` (oldest zone first) and `nearest` (nearest zone, no priority).
Writes routing/many_tasks.json. The scale, the word weights and the signatures of the unrelated tasks are those
of silent_adapters/silent_run.py; the routes are computed here in one piece for speed and are checked against
silent_run.py on the fifteen tasks first.
"""
import collections
import contextlib
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1]
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
with contextlib.redirect_stdout(io.StringIO()):          # sets the model up (and runs the standard check once)
    import routing_check as rc
import torch
sr = rc.sr
KAPPA = 0.5
base, weight = sr._CL["base"], sr._CL["weight"]
EW = sr._CL["emb"] @ base["w_head"]                       # the rescaled word-vector table
MUW = base["mu_head"] @ base["w_head"]
BG = base["bg_sig"]


def tokens(text):
    return torch.tensor(rc.tok(text, max_length=rc.MAX_LEN, truncation=True)["input_ids"])


def prefixes(ids):
    """Rescaled weighted average of the first n tokens, for every n = 1 .. length (row n - 1)."""
    w = weight[ids]
    return torch.cumsum(EW[ids] * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None] - MUW


def pairwise(a, b):
    """Distances between the rows of a and the rows of b, computed directly (no rounding trick: equal rows give exactly 0)."""
    return torch.cat([(a[s:s + 32, None, :] - b[None, :, :]).norm(dim=2) for s in range(0, a.shape[0], 32)])


def prepare(tasks):
    """Signatures, opening lengths, and the distance of every input from every signature (not divided by a radius)."""
    sig, n, owner, feats = [], [], [], []
    for k, t in enumerate(tasks):
        a = rc.tok(rc.template(t["definition"]).replace("{0}", "A B"))["input_ids"]
        b = rc.tok(rc.template(t["definition"]).replace("{0}", "zzq xkj"))["input_ids"]
        nk = min(sr._prefix_len([a, b]), rc.MAX_LEN)
        n.append(nk)
        sig.append(prefixes(torch.tensor(a[:nk]))[nk - 1])
        for x in t["inputs"]:
            feats.append(prefixes(tokens(rc.template(t["definition"]).replace("{0}", x))))
            owner.append(k)
    sig, n = torch.stack(sig), torch.tensor(n)
    dist = torch.stack([(f[(n.clamp(max=f.shape[0]) - 1)] - sig).norm(dim=1) for f in feats])
    return {"sig": sig, "n": n, "owner": torch.tensor(owner), "dist": dist,
            "d_bg": pairwise(sig, BG).min(1).values, "d_sig": pairwise(sig, sig)}


def radii(p, order):
    k = len(order)
    radius = torch.zeros(p["sig"].shape[0], dtype=torch.double)
    seen = torch.zeros(p["sig"].shape[0], dtype=torch.bool)
    for j in order:
        nearest = p["d_bg"][j] if not seen.any() else torch.minimum(p["d_bg"][j], p["d_sig"][j][seen].min())
        radius[j] = KAPPA * nearest
        seen[j] = True
    return radius


def route(dist, radius, pos, allowed, rule):
    """certified: the oldest allowed zone that contains the input, else the nearest allowed zone. nearest: the
    nearest allowed zone. A zone of radius zero holds only inputs at distance zero; ties go to the older task."""
    inside = (dist <= radius[None]) & allowed[None]
    rel = dist / radius[None]
    rel = torch.where(torch.isnan(rel), torch.zeros_like(rel), rel)                  # 0 / 0: at the signature of a zero-radius zone
    rel = torch.where(allowed[None], rel, torch.full_like(rel, float("inf")))
    by_age = torch.argsort(pos)                                                      # columns from oldest to newest
    nearest = by_age[rel[:, by_age].argmin(1)]
    if rule == "nearest":
        return nearest, inside.any(1)
    oldest = by_age[inside[:, by_age].to(torch.int8).argmax(1)]
    return torch.where(inside.any(1), oldest, nearest), inside.any(1)


def words(text):
    return collections.Counter(re.findall(r"\w+", text.lower()))


def alike(a, b):
    wa, wb = words(a), words(b)
    return sum((wa & wb).values()) / max(sum(wa.values()), sum(wb.values()), 1)


def one_order(p, tasks, order, checkpoints, details=False):
    K = p["sig"].shape[0]
    pos = torch.full((K,), 10 ** 9, dtype=torch.long)
    pos[torch.tensor(order)] = torch.arange(len(order))
    radius = radii(p, order)
    own, dist = p["owner"], p["dist"]
    own_pos = pos[own]
    out = {}
    for N in checkpoints:
        rows = own_pos < N                                                           # inputs of the first N tasks
        d, o, op = dist[rows], own[rows], own_pos[rows]
        res = {"tasks": N, "inputs": int(rows.sum()), "radius_zero": int((radius[torch.tensor(order[:N])] == 0).sum()),
               "radius_median": float(radius[torch.tensor(order[:N])].median())}
        # at arrival: is the input inside an older zone? (zones of tasks that arrived strictly earlier)
        older = pos[None] < op[:, None]
        blocked = ((d <= radius[None]) & older).any(1)
        per_task = collections.Counter()
        size = collections.Counter(o.tolist())
        for t, b in zip(o.tolist(), blocked.tolist()):
            per_task[t] += b
        res["tasks_clean"] = sum(per_task[t] == 0 for t in size)
        res["tasks_all_blocked"] = sum(per_task[t] == size[t] for t in size)
        res["tasks_some_blocked"] = N - res["tasks_clean"] - res["tasks_all_blocked"]
        res["inputs_blocked"] = int(blocked.sum())
        for rule in ("certified", "nearest"):
            final, cert = route(d, radius, pos, pos < N, rule)
            then = torch.empty_like(final)
            cert_then = torch.empty_like(cert)
            for value in op.unique().tolist():                                       # routes right after each task arrived
                m = op == value
                then[m], cert_then[m] = route(d[m], radius, pos, pos <= value, rule)
            changed = final != then
            res[rule] = {"right": int((final == o).sum()), "certified_own": int((cert & (final == o)).sum()),
                         "certified_other": int((cert & (final != o)).sum()), "changed": int(changed.sum()),
                         "changed_certified": int((changed & cert_then).sum()),
                         "tasks_with_changed": len(set(o[changed].tolist()))}
            if rule == "certified":
                assert res[rule]["changed_certified"] == 0, "a certified route changed: the guarantee is broken"
        if details and N == checkpoints[-1]:
            pairs = []
            final, _ = route(d, radius, pos, pos < N, "certified")
            for t in sorted(size, key=lambda t: pos[t].item()):
                m = (o == t) & blocked
                if m.any():
                    holder = collections.Counter(final[m].tolist()).most_common(1)[0][0]
                    pairs.append({"task": tasks[t]["name"], "blocked": int(m.sum()), "inputs": size[t], "older": tasks[holder]["name"],
                                  "alike": alike(tasks[t]["definition"], tasks[holder]["definition"]),
                                  "same_definition": tasks[t]["definition"] == tasks[holder]["definition"],
                                  "radius_older": float(radius[holder]), "radius_new": float(radius[t])})
            res["collisions"] = pairs
        out[str(N)] = res
    return out


# ---- check against silent_run.py on the fifteen tasks -----------------------------------------------------
data = [rc.load(nm) for nm in rc.ORDER1]
fifteen = [{"name": nm, "definition": d["definition"], "inputs": d["test"][:20]} for nm, d in zip(rc.ORDER1, data)]
p15 = prepare(fifteen)
r15 = radii(p15, list(range(15)))
sr.KAPPA = KAPPA
sr._CL.update(tasks={}, t=0)
for j, d in enumerate(data):
    sr._CL["tasks"][j + 1] = rc.new_task(d["definition"], d["train"])[0]
    sr._CL["t"] = j + 1
assert max(abs(sr._CL["tasks"][j + 1]["rho"] - float(r15[j])) for j in range(15)) < 1e-6, "radii differ from silent_run.py"
texts = [rc.template(t["definition"]).replace("{0}", x) for t in fifteen for x in t["inputs"]]
ref = torch.cat([sr._routes(rc.ids_of(texts[s:s + 32]), "certified")[0] for s in range(0, len(texts), 32)]).cpu()
mine, _ = route(p15["dist"], r15, torch.arange(15), torch.ones(15, dtype=torch.bool), "certified")
assert (mine + 1 == ref).all(), "routes differ from silent_run.py"
print(f"check against silent_run.py on the fifteen tasks: {len(texts)} routes equal, radii equal", flush=True)

# ---- the many tasks ---------------------------------------------------------------------------------------
raw = json.load(open(DATA))
tasks = [t for t in raw["tasks"] if len(t.get("inputs", [])) >= 2]
print(len(raw["tasks"]), "tasks fetched;", len(tasks), "with at least two inputs;", sum(len(t["inputs"]) for t in tasks), "inputs", flush=True)
p = prepare(tasks)
K = len(tasks)
checkpoints = [c for c in (50, 100, 200, 400) if c < K] + [K]
result = {"commit": raw["commit"], "tasks": K, "inputs": int(p["dist"].shape[0]),
          "opening_tokens": {"min": int(p["n"].min()), "median": int(p["n"].median()), "max": int(p["n"].max())},
          "same_definition_pairs": sum(v - 1 for v in collections.Counter(t["definition"] for t in tasks).values() if v > 1)}
orders = {}
for seed in range(5):
    o = list(range(K))
    random.Random(seed).shuffle(o)
    orders[f"random order, seed {seed}"] = o
number = lambda t: int(re.match(r"task(\d+)", t["name"])[1])
orders["order of the task numbers"] = sorted(range(K), key=lambda k: number(tasks[k]))
for name, o in orders.items():
    result[name] = one_order(p, tasks, o, checkpoints, details=name in ("random order, seed 0", "order of the task numbers"))
    r = result[name][str(K)]
    print(f"{name}: clean {r['tasks_clean']} some {r['tasks_some_blocked']} all {r['tasks_all_blocked']} of {K} tasks | inputs blocked {r['inputs_blocked']} of {r['inputs']}"
          f" | right: certified {r['certified']['right']}, nearest {r['nearest']['right']} | changed later: certified {r['certified']['changed']}"
          f" (certified before: {r['certified']['changed_certified']}), nearest {r['nearest']['changed']} in {r['nearest']['tasks_with_changed']} tasks"
          f" | zero radius {r['radius_zero']}", flush=True)
# ---- added after the first run (not part of the rules written beforehand): what other radius factors would do --------
result["other_radius_factors"] = {}
for kappa in (0.1, 0.2, 0.3, 0.4, 0.5):
    KAPPA = kappa
    r = one_order(p, tasks, orders["random order, seed 0"], [K])[str(K)]
    result["other_radius_factors"][str(kappa)] = r
    print(f"radius factor {kappa}: clean {r['tasks_clean']} some {r['tasks_some_blocked']} all {r['tasks_all_blocked']} | inputs blocked {r['inputs_blocked']}"
          f" | right: certified {r['certified']['right']}, nearest {r['nearest']['right']}", flush=True)
KAPPA = 0.5
holders = collections.Counter(c["older"] for c in result["random order, seed 0"][str(K)]["collisions"])
result["largest_holders"] = holders.most_common(10)
print("older tasks that hold the most later tasks:", holders.most_common(8))
json.dump(result, open(os.path.join(HERE, "routing", "many_tasks.json"), "w"), indent=1)
main = result["random order, seed 0"]
for N in checkpoints:
    r = main[str(N)]
    print(f"  seed 0, first {N:>3} tasks: clean {r['tasks_clean']:>3}, some {r['tasks_some_blocked']:>2}, all {r['tasks_all_blocked']:>3} | inputs {r['inputs']:>4}: "
          f"right certified {r['certified']['right']:>4} nearest {r['nearest']['right']:>4} | changed certified {r['certified']['changed']:>3} nearest {r['nearest']['changed']:>3}"
          f" | median radius {r['radius_median']:.1f}")
pairs = main[str(K)]["collisions"]
bins = collections.Counter("same text" if c["same_definition"] else "0.8 or more" if c["alike"] >= 0.8 else "0.5 to 0.8" if c["alike"] >= 0.5 else "under 0.5" for c in pairs)
print("colliding tasks by how alike the definitions are:", dict(bins), "of", len(pairs))
for c in sorted(pairs, key=lambda c: c["alike"])[:12]:
    print(f"   {c['alike']:.2f}  {c['task'][:44]:<46} held by {c['older'][:44]:<46} {c['blocked']}/{c['inputs']}")
