"""A variant against collisions, "exact claims first" (CPU, T5-base word-vectors, routes only). It was designed after
the test on 875 real tasks; what it is and how it is checked was written in silent_adapters/PLAN.md before this was run.
Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/exact_first_analysis.py <file made by many_tasks_fetch.py>

Rule `exact`: an input that carries the opening of a task token for token goes to that task (the oldest, if several
tasks have that opening); otherwise the oldest zone that contains it; otherwise the nearest zone.
Writes routing/exact_first.json.
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
EW = sr._CL["emb"] @ base["w_head"]
MUW = base["mu_head"] @ base["w_head"]
BG = base["bg_sig"]
RULES = ("exact", "certified", "nearest")


def tokens(text):
    return rc.tok(text, max_length=rc.MAX_LEN, truncation=True)["input_ids"]


def prefixes(ids):
    ids = torch.tensor(ids)
    w = weight[ids]
    return torch.cumsum(EW[ids] * w[:, None], 0) / torch.cumsum(w, 0).clamp(min=1e-12)[:, None] - MUW


def pairwise(a, b):
    """Distances between the rows of a and the rows of b, computed directly (no rounding trick: equal rows give exactly 0)."""
    return torch.cat([(a[s:s + 32, None, :] - b[None, :, :]).norm(dim=2) for s in range(0, a.shape[0], 32)])


def prepare(tasks):
    """tasks: [{"definition", "inputs": [full input texts]}]. Signatures, distances, and which inputs carry which opening."""
    sig, n, openings = [], [], collections.defaultdict(list)
    for k, t in enumerate(tasks):
        a = rc.tok(rc.template(t["definition"]).replace("{0}", "A B"))["input_ids"]
        b = rc.tok(rc.template(t["definition"]).replace("{0}", "zzq xkj"))["input_ids"]
        nk = min(sr._prefix_len([a, b]), rc.MAX_LEN)
        n.append(nk)
        sig.append(prefixes(a[:nk])[nk - 1])
        openings[tuple(a[:min(nk, rc.MAX_LEN - 1)])].append(k)       # the last place of a full-length input holds the end mark
    lengths = sorted({len(o) for o in openings})
    owner, feats, exact = [], [], []
    for k, t in enumerate(tasks):
        for text in t["inputs"]:
            ids = tokens(text)
            feats.append(prefixes(ids))
            owner.append(k)
            row = torch.zeros(len(tasks), dtype=torch.bool)
            for m in lengths:
                if m <= len(ids):
                    for j in openings.get(tuple(ids[:m]), ()):
                        row[j] = True
            exact.append(row)
    sig, n = torch.stack(sig), torch.tensor(n)
    dist = torch.stack([(f[(n.clamp(max=f.shape[0]) - 1)] - sig).norm(dim=1) for f in feats])
    return {"sig": sig, "n": n, "owner": torch.tensor(owner), "dist": dist, "exact": torch.stack(exact),
            "d_bg": pairwise(sig, BG).min(1).values, "d_sig": pairwise(sig, sig)}


def radii(p, order):
    radius = torch.zeros(p["sig"].shape[0], dtype=torch.double)
    seen = torch.zeros(p["sig"].shape[0], dtype=torch.bool)
    for j in order:
        radius[j] = KAPPA * (p["d_bg"][j] if not seen.any() else torch.minimum(p["d_bg"][j], p["d_sig"][j][seen].min()))
        seen[j] = True
    return radius


def route(p, rows, radius, pos, allowed, rule):
    """Returns the route, whether an exact claim decided it, and whether the input is inside an allowed zone."""
    dist, exact = p["dist"][rows], p["exact"][rows] & allowed[None]
    inside = (dist <= radius[None]) & allowed[None]
    rel = dist / radius[None]
    rel = torch.where(torch.isnan(rel), torch.zeros_like(rel), rel)
    rel = torch.where(allowed[None], rel, torch.full_like(rel, float("inf")))
    by_age = torch.argsort(pos)
    nearest = by_age[rel[:, by_age].argmin(1)]
    if rule == "nearest":
        return nearest, torch.zeros(len(nearest), dtype=torch.bool), inside.any(1)
    out = torch.where(inside.any(1), by_age[inside[:, by_age].to(torch.int8).argmax(1)], nearest)
    if rule == "certified":
        return out, torch.zeros(len(out), dtype=torch.bool), inside.any(1)
    claimed = exact.any(1)
    return torch.where(claimed, by_age[exact[:, by_age].to(torch.int8).argmax(1)], out), claimed, inside.any(1)


def one_order(p, order, checkpoints):
    K = p["sig"].shape[0]
    pos = torch.full((K,), 10 ** 9, dtype=torch.long)
    pos[torch.tensor(order)] = torch.arange(len(order))
    radius = radii(p, order)
    own, own_pos = p["owner"], pos[p["owner"]]
    out = {}
    for N in checkpoints:
        rows = own_pos < N
        o, op = own[rows], own_pos[rows]
        res = {"tasks": N, "inputs": int(rows.sum())}
        for rule in RULES:
            final, _, _ = route(p, rows, radius, pos, pos < N, rule)
            then, claimed_then, zone_then = torch.empty_like(final), torch.zeros(len(final), dtype=torch.bool), torch.zeros(len(final), dtype=torch.bool)
            idx = torch.nonzero(rows).squeeze(1)
            for value in op.unique().tolist():
                m = op == value
                sub = torch.zeros_like(rows)
                sub[idx[m]] = True
                then[m], claimed_then[m], zone_then[m] = route(p, sub, radius, pos, pos <= value, rule)
            changed = final != then
            per_task = collections.defaultdict(lambda: [0, 0])
            for t, good in zip(o.tolist(), (then == o).tolist()):
                per_task[t][0] += good
                per_task[t][1] += 1
            res[rule] = {"right": int((final == o).sum()), "changed": int(changed.sum()), "changed_exact": int((changed & claimed_then).sum()),
                         "changed_zone": int((changed & zone_then & ~claimed_then).sum()),
                         "tasks_own_all": sum(a == b for a, b in per_task.values()), "tasks_own_none": sum(a == 0 for a, b in per_task.values())}
            res[rule]["tasks_own_some"] = N - res[rule]["tasks_own_all"] - res[rule]["tasks_own_none"]
            if rule == "exact":
                assert res[rule]["changed_exact"] == 0, "an input with an exact claim changed its route"
            if rule == "certified":
                assert int((changed & zone_then).sum()) == 0, "a certified route changed"
        out[str(N)] = res
    return out


result = {}
# ---- 1. on the five and the fifteen tasks the variant must give the routes of the frozen rule, in every wording ----
for label, names in (("five", rc.FIVE), ("fifteen", rc.ORDER1)):
    data = [rc.load(nm) for nm in names]
    tasks, level_of = [], []
    for nm, d in zip(names, data):
        defs = [d["definition"]] + sr.REWORDED.get(nm.split("_")[0], [])
        inputs = []
        for li, definition in enumerate(defs):
            inputs += [rc.template(definition).replace("{0}", x) for x in d["test"]]
            level_of += [li] * len(d["test"])
        tasks.append({"definition": d["definition"], "inputs": inputs})
    p = prepare(tasks)
    K = len(tasks)
    radius, pos, everything = radii(p, list(range(K))), torch.arange(K), torch.ones(K, dtype=torch.bool)
    all_rows = torch.ones(p["dist"].shape[0], dtype=torch.bool)
    r_exact, claimed, _ = route(p, all_rows, radius, pos, everything, "exact")
    r_cert, _, zone = route(p, all_rows, radius, pos, everything, "certified")
    level_of = torch.tensor(level_of)
    assert (r_exact == r_cert).all(), f"{label}: the variant and the frozen rule give different routes"
    assert bool(claimed[level_of == 0].all()) and not bool(claimed[level_of > 0].any()), "exact claims: only the original wording should have one"
    result[label + "_same_routes"] = int((r_exact == r_cert).sum())
    result[label + "_inputs"] = int(len(r_exact))
    print(f"{label} tasks: the variant gives the route of the frozen rule for {int((r_exact == r_cert).sum())} of {len(r_exact)} test inputs (all wordings); "
          f"inputs with an exact claim: {int(claimed.sum())} (the original wording)", flush=True)

# ---- 2. the stress test: a sixth task that looks like sst2 ----
data = [rc.load(nm) for nm in rc.FIVE]
old = data[1]
defs = [old["definition"]] + sr.REWORDED["task363"]
result["stress"] = {}
for li_new in (1, 2, 3):
    tasks = [{"definition": d["definition"], "inputs": []} for d in data]
    tasks[1]["inputs"] = [rc.template(defs[li]).replace("{0}", x) for li in range(4) for x in old["test"]]
    tasks.append({"definition": defs[li_new], "inputs": [rc.template(defs[li_new]).replace("{0}", x) for x in old["train"]]})
    p = prepare(tasks)
    radius, pos = radii(p, list(range(6))), torch.arange(6)
    old_rows, new_rows = p["owner"] == 1, p["owner"] == 5
    level = torch.arange(4).repeat_interleave(len(old["test"]))
    row = {}
    for rule in RULES:
        before, _, _ = route(p, old_rows, radius, pos, pos < 5, rule)
        after, _, _ = route(p, old_rows, radius, pos, pos < 6, rule)
        own, _, _ = route(p, new_rows, radius, pos, pos < 6, rule)
        row[rule] = {"changed": [int(((before != after) & (level == li)).sum()) for li in range(4)], "new_task_own": int((own == 5).sum()), "new_task_inputs": int(new_rows.sum())}
    result["stress"][sr.LEVELS[li_new]] = row
    print(f"stress, new task with the {sr.LEVELS[li_new]} rewording: " + " | ".join(
        f"{rule}: old routes changed {'/'.join(map(str, row[rule]['changed']))}, new task's own inputs given to it {row[rule]['new_task_own']} of {row[rule]['new_task_inputs']}"
        for rule in RULES), flush=True)

# ---- 3. the 875 tasks ----
raw = json.load(open(DATA))
tasks = [{"name": t["name"], "definition": t["definition"], "inputs": [rc.template(t["definition"]).replace("{0}", x) for x in t["inputs"]]}
         for t in raw["tasks"] if len(t.get("inputs", [])) >= 2]
p = prepare(tasks)
K = len(tasks)
checkpoints = [c for c in (50, 100, 200, 400) if c < K] + [K]
same_opening = sum(int(p["exact"][i].sum()) > 1 for i in range(p["exact"].shape[0]))
result["many"] = {"tasks": K, "inputs": int(p["dist"].shape[0]), "inputs_with_two_exact_claims": same_opening}
orders = {}
for seed in range(5):
    o = list(range(K))
    random.Random(seed).shuffle(o)
    orders[f"random order, seed {seed}"] = o
orders["order of the task numbers"] = sorted(range(K), key=lambda k: int(re.match(r"task(\d+)", tasks[k]["name"])[1]))
for name, o in orders.items():
    result["many"][name] = one_order(p, o, checkpoints)
    r = result["many"][name][str(K)]
    print(f"{name}: " + " | ".join(f"{rule}: right {r[rule]['right']} of {r['inputs']}, tasks with all own inputs at arrival {r[rule]['tasks_own_all']}, none "
                                    f"{r[rule]['tasks_own_none']}, changed later {r[rule]['changed']} (exact {r[rule]['changed_exact']}, zone {r[rule]['changed_zone']})"
                                    for rule in RULES), flush=True)
print("inputs that carry the opening of more than one task:", same_opening)
json.dump(result, open(os.path.join(HERE, "routing", "exact_first.json"), "w"), indent=1)
