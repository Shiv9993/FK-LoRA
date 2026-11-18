"""A numerical check of the bound on edits (Proposition 2 of the paper), on inputs of the mechanical edit test.
Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/bound_check.py

For every edited input: the shares of the weight of the opening that the edit removed and added (l, l'), the bound
max(l, l') D, and the distance the input really moved. The bound must hold for every input. Writes routing/bound_check.json.
(The first version of the proposition had (l + l') D; a reviewer pointed out that the same argument gives the maximum.)
"""
import collections
import contextlib
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
with contextlib.redirect_stdout(io.StringIO()):
    import routing_check as rc
import torch
sr = rc.sr
D = json.load(open(os.path.join(HERE, "routing", "edits.json")))["bound"]["diameter"]     # computed by edit_analysis.py


# the edits of edit_analysis.py, copied so that this file runs on its own

def typo(word, rng):
    inner = [i for i in range(1, len(word) - 2) if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]]
    if not inner:
        return word
    i = rng.choice(inner)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def edit(definition, kind, k, rng, pool):
    words = definition.split()
    if kind == "typo":
        long = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 4]
        for i in rng.sample(long, min(k, len(long))):
            words[i] = typo(words[i], rng)
    elif kind == "drop":
        gone = set(rng.sample(range(len(words)), min(k, len(words) - 1)))
        words = [w for i, w in enumerate(words) if i not in gone]
    else:
        for i in rng.sample(range(len(words)), min(k, len(words))):
            words[i] = rng.choice(pool)
    return " ".join(words)


data = [rc.load(n) for n in rc.ORDER1]
pool = [w for d in data for w in d["definition"].split()]
sr.KAPPA = 0.5
sr._CL.update(tasks={}, t=0)
for j, d in enumerate(data):
    sr._CL["tasks"][j + 1] = rc.new_task(d["definition"], d["train"])[0]
    sr._CL["t"] = j + 1
base, weight = sr._CL["base"], sr._CL["weight"]
rows, worst = [], 0.0
for kind in ("typo", "drop", "replace"):
    for k in (1, 2, 4, 8, 16):
        for j, d in enumerate(data):
            task = sr._CL["tasks"][j + 1]
            n = task["n"]
            for i, x in enumerate(d["test"][:10]):
                rng = random.Random(f"0|{kind}-{k}|{j}|{i}")
                a = rc.tok(rc.template(d["definition"]).replace("{0}", x), max_length=rc.MAX_LEN, truncation=True)["input_ids"][:n]
                b = rc.tok(rc.template(edit(d["definition"], kind, k, rng, pool)).replace("{0}", x), max_length=rc.MAX_LEN, truncation=True)["input_ids"][:n]
                ca, cb = collections.Counter(a), collections.Counter(b)
                w = lambda c: sum(float(weight[t]) * m for t, m in c.items())
                common, removed, added = w(ca & cb), w(ca - cb), w(cb - ca)
                lam, lam2 = removed / (common + removed), added / (common + added)
                ids = torch.tensor([b])
                moved = float((((sr._means(ids, [n])[n][0].cpu() - base["mu_head"]) @ base["w_head"]) - task["sig"].cpu()).norm())
                bound = max(lam, lam2) * D
                assert moved <= bound + 1e-6, (kind, k, j, i, moved, bound)
                rows.append({"edit": f"{kind}-{k}", "task": rc.ORDER1[j], "removed": lam, "added": lam2, "moved": moved, "bound": bound,
                             "radius": task["rho"], "inside": moved <= task["rho"], "guaranteed": bound <= task["rho"]})
                worst = max(worst, moved / bound if bound else 0.0)
out = {"inputs": len(rows), "diameter": D, "held": True, "largest_share_of_the_bound": worst,
       "median_share_of_the_bound": sorted(r["moved"] / r["bound"] for r in rows if r["bound"])[len(rows) // 2],
       "inside": sum(r["inside"] for r in rows), "guaranteed": sum(r["guaranteed"] for r in rows)}
json.dump(out, open(os.path.join(HERE, "routing", "bound_check.json"), "w"), indent=1)
print(out)
