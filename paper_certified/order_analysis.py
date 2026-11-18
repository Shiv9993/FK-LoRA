"""Does the order of the tasks matter for the routes? (CPU, T5-base word-vectors, final rule.)
Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/order_analysis.py

The signature of a task does not depend on the order; only its radius does (it is a fixed share of the
distance to the nearest signature known when the task arrives). So the distance of every input from every
signature is computed once, and after that any order costs almost nothing:
  * the five tasks: all 120 orders;
  * the fifteen tasks: the two official orders and 1,000 random orders (seed 0).
Writes routing/orders.json. The functions are those of silent_adapters/silent_run.py.
"""
import contextlib
import io
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
with contextlib.redirect_stdout(io.StringIO()):          # sets the model up (and runs the standard check once)
    import routing_check as rc
import torch
sr = rc.sr
KAPPA = 0.5
ORDER2 = ["task748", "task073", "task1590", "task639", "task1572", "task1687", "task591", "task363", "task1510", "task1729", "task181",
          "task511", "task002", "task1290", "task875"]


def prepare(names):
    """Signatures, and the distance of every test and training input from every signature."""
    data = [rc.load(n) for n in names]
    sr.KAPPA = KAPPA
    sr._CL.update(tasks={}, t=0)
    tasks = {}
    for j, d in enumerate(data):                          # the signature and the opening length do not depend on the order
        a = rc.tok(rc.template(d["definition"]).replace("{0}", "A B"))["input_ids"]
        b = rc.tok(rc.template(d["definition"]).replace("{0}", "zzq xkj"))["input_ids"]
        n = sr._prefix_len([a, b])
        tasks[j] = sr._new_zone(torch.tensor(a[:n]), [rc.ids_of([rc.template(d["definition"]).replace("{0}", "A B")])])[0]
    base = sr._CL["base"]
    sig = torch.stack([tasks[j]["sig"] for j in range(len(names))])
    lengths = sorted({tasks[j]["n"] for j in tasks})

    def dist(texts):
        rows = []
        for s in range(0, len(texts), 32):
            m = sr._means(rc.ids_of(texts[s:s + 32]), lengths)
            rows.append(torch.stack([(((m[tasks[j]["n"]] - base["mu_head"]) @ base["w_head"]) - tasks[j]["sig"]).norm(dim=1)
                                     for j in range(len(names))], 1))
        return torch.cat(rows)

    test, train = {}, {}
    for j, (name, d) in enumerate(zip(names, data)):
        defs = [d["definition"]] + sr.REWORDED.get(name.split("_")[0], [])
        for li, definition in enumerate(defs):
            test[j, li] = dist([rc.template(definition).replace("{0}", x) for x in d["test"]])
        train[j] = dist([rc.template(d["definition"]).replace("{0}", x) for x in d["train"]])
    d_bg = torch.stack([(base["bg_sig"] - s).norm(dim=1).min() for s in sig])
    return {"names": names, "test": test, "train": train, "d_bg": d_bg, "d_sig": torch.cdist(sig, sig)}


def routes(dist, radius, pos, allowed):
    """Final rule: the oldest allowed zone that contains the input, else the nearest allowed zone."""
    rel = dist / radius[None]
    rel = torch.where(allowed[None], rel, torch.full_like(rel, float("inf")))
    inside = rel <= 1.0
    age = torch.where(inside, pos[None].expand_as(rel), torch.full(rel.shape, 10 ** 6))
    oldest = age.argmin(1)
    return torch.where(inside.any(1), oldest, rel.argmin(1)), inside.any(1)


def one_order(p, order):
    k = len(order)
    pos = torch.empty(k, dtype=torch.long)
    pos[torch.tensor(order)] = torch.arange(k)
    radius = torch.empty(k, dtype=torch.double)
    for i, j in enumerate(order):
        earlier = order[:i]
        radius[j] = KAPPA * min([float(p["d_bg"][j])] + [float(p["d_sig"][j, e]) for e in earlier])
    everything = torch.ones(k, dtype=torch.bool)
    out = {"radius_min": float(radius.min()), "blocked": 0}
    for li in range(4):
        total = right = own = wrong = changed = 0
        for j in range(k):
            if (j, li) not in p["test"]:
                continue
            r, c = routes(p["test"][j, li], radius, pos, everything)
            r0, c0 = routes(p["test"][j, li], radius, pos, pos <= pos[j])        # when the task had just been learned
            assert not ((r != r0) & c0).any(), "a certified route changed: the guarantee is broken"
            total += len(r)
            right += int((r == j).sum())
            own += int((c & (r == j)).sum())
            wrong += int((c & (r != j)).sum())
            changed += int((r != r0).sum())
        out[sr.LEVELS[li]] = {"inputs": total, "right": right / total, "own_zone": own / total, "wrong_zone": wrong, "changed": changed}
    for j in range(k):                                    # training inputs of a new task inside an older zone
        if pos[j] > 0:
            out["blocked"] += int(routes(p["train"][j], radius, pos, pos < pos[j])[1].sum())
    return out


def summary(rows):
    out = {"orders": len(rows), "blocked_max": max(r["blocked"] for r in rows), "orders_with_blocked": sum(r["blocked"] > 0 for r in rows)}
    for level in sr.LEVELS:
        for key in ("right", "own_zone", "wrong_zone", "changed"):
            v = sorted(r[level][key] for r in rows)
            out[f"{level}|{key}"] = {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "mean": sum(v) / len(v)}
    return out


def nearest(p):
    """For every task: how far is the nearest signature of an unrelated task, and the nearest signature of another task?"""
    rows = []
    for j, name in enumerate(p["names"]):
        d = p["d_sig"][j].clone()
        d[j] = float("inf")
        rows.append({"task": name, "nearest_unrelated": float(p["d_bg"][j]), "nearest_task": float(d.min()), "which": p["names"][int(d.argmin())]})
    return rows


result = {}
five = prepare(rc.FIVE)
rows = [one_order(five, list(o)) for o in itertools.permutations(range(5))]
result["five_all_120"] = summary(rows)
result["five_official"] = one_order(five, list(range(5)))
fifteen = prepare(rc.ORDER1)
result["fifteen_order1"] = one_order(fifteen, list(range(15)))
result["five_distances"], result["fifteen_distances"] = nearest(five), nearest(fifteen)
short = [n.split("_")[0] for n in rc.ORDER1]
result["fifteen_order2"] = one_order(fifteen, [short.index(n) for n in ORDER2])
rng = random.Random(0)
rows = []
for _ in range(1000):
    o = list(range(15))
    rng.shuffle(o)
    rows.append(one_order(fifteen, o))
result["fifteen_random_1000"] = summary(rows)
result["fifteen_worst_heavy"] = min(rows, key=lambda r: r["heavy"]["right"])
json.dump(result, open(os.path.join(HERE, "routing", "orders.json"), "w"), indent=1)
for key in ("five_official", "fifteen_order1", "fifteen_order2"):
    r = result[key]
    print(key, "| blocked", r["blocked"], "|", " | ".join(f"{l}: right {r[l]['right']:.3f}, own zone {r[l]['own_zone']:.3f}, wrong zone "
                                                           f"{r[l]['wrong_zone']}, changed {r[l]['changed']}" for l in sr.LEVELS))
for key in ("five_all_120", "fifteen_random_1000"):
    r = result[key]
    print(key, "| orders with a blocked input", r["orders_with_blocked"], "| most blocked", r["blocked_max"])
    for l in sr.LEVELS:
        print("   ", l, " | ".join(f"{k}: {r[l + '|' + k]['min']:.3f} / {r[l + '|' + k]['median']:.3f} / {r[l + '|' + k]['max']:.3f}"
                                   for k in ("right", "own_zone", "wrong_zone", "changed")), "(least / middle / most)")
for r in result["fifteen_distances"]:
    print(f"{r['task'][:34]:<36} nearest unrelated signature {r['nearest_unrelated']:.1f} | nearest other task {r['nearest_task']:.1f} ({r['which'][:20]})")
