"""A mechanical edit test of the routes (CPU, T5-base word-vectors, final rule, fifteen tasks in official order 1).
Its rules were written in silent_adapters/PLAN.md before it was run. Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/edit_analysis.py

Every test input of every task gets its own random edit of the definition (typos, dropped words, replaced words,
lower case, an extra sentence in front, a sentence removed, sentences shuffled). Measured: inside the own zone,
inside the zone of another task, given to the right task. Also computed: the largest distance between two rows of
the rescaled word-vector table, for the worst-case bound on edits. Writes routing/edits.json.
"""
import contextlib
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
with contextlib.redirect_stdout(io.StringIO()):          # sets the model up (and runs the standard check once)
    import routing_check as rc
import torch
sr = rc.sr
KS = (1, 2, 4, 8, 16)

data = [rc.load(n) for n in rc.ORDER1]
sr.KAPPA = 0.5
sr._CL.update(tasks={}, t=0)
for j, d in enumerate(data):
    sr._CL["tasks"][j + 1] = rc.new_task(d["definition"], d["train"])[0]
    sr._CL["t"] = j + 1
POOL = [w for d in data for w in d["definition"].split()]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def typo(word, rng):
    inner = [i for i in range(1, len(word) - 2) if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]]
    if not inner:
        return word
    i = rng.choice(inner)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def edit(definition, kind, k, rng):
    words = definition.split()
    if kind == "typo":
        long = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 4]
        for i in rng.sample(long, min(k, len(long))):
            words[i] = typo(words[i], rng)
        return " ".join(words)
    if kind == "drop":
        gone = set(rng.sample(range(len(words)), min(k, len(words) - 1)))
        return " ".join(w for i, w in enumerate(words) if i not in gone)
    if kind == "replace":
        for i in rng.sample(range(len(words)), min(k, len(words))):
            words[i] = rng.choice(POOL)
        return " ".join(words)
    if kind == "lower":
        return definition.lower()
    s = sentences(definition)
    if kind == "no-first":
        return " ".join(s[1:])
    if kind == "no-last":
        return " ".join(s[:-1])
    if kind == "shuffle":
        t = s[:]
        while t == s:
            rng.shuffle(t)
        return " ".join(t)
    return definition


def measure(kind, k):
    name = kind if k is None else f"{kind}-{k}"
    per_task, used = [], 0
    for j, d in enumerate(data):
        if kind in ("no-first", "no-last", "shuffle") and len(sentences(d["definition"])) < 2:
            continue
        used += 1
        texts = []
        for i, x in enumerate(d["test"]):
            rng = random.Random(f"0|{name}|{j}|{i}")
            text = rc.template(edit(d["definition"], kind, k, rng)).replace("{0}", x)
            texts.append("Please help me with this task. " + text if kind == "polite" else text)
        route, cert = [], []
        for s in range(0, len(texts), 32):
            r, c = sr._routes(rc.ids_of(texts[s:s + 32]), "certified")
            route.append(r)
            cert.append(c)
        route, cert = torch.cat(route), torch.cat(cert)
        per_task.append({"task": rc.ORDER1[j], "inputs": len(texts), "own": int((cert & (route == j + 1)).sum()),
                         "foreign": int((cert & (route != j + 1)).sum()), "right": int((route == j + 1).sum())})
    n = sum(t["inputs"] for t in per_task)
    worst = min(per_task, key=lambda t: t["right"] / t["inputs"])
    out = {"edit": name, "tasks": used, "inputs": n, "own_zone": sum(t["own"] for t in per_task) / n, "foreign_zone": sum(t["foreign"] for t in per_task),
           "right": sum(t["right"] for t in per_task) / n, "worst_task": worst["task"], "worst_right": worst["right"] / worst["inputs"],
           "worst_own": min(t["own"] / t["inputs"] for t in per_task), "per_task": per_task}
    print(f"{name:<12} tasks {used:>2} inputs {n:>4} | own zone {out['own_zone']:.3f} (worst task {out['worst_own']:.2f}) | foreign zone {out['foreign_zone']} | "
          f"right task {out['right']:.3f} (worst: {worst['task'].split('_')[0]} {out['worst_right']:.2f})", flush=True)
    return out


rows = [measure("none", None)]
assert rows[0]["own_zone"] == 1.0, "without an edit every input must be inside its own zone"
for kind in ("typo", "drop", "replace"):
    rows += [measure(kind, k) for k in KS]
rows += [measure(kind, None) for kind in ("lower", "polite", "no-first", "no-last", "shuffle")]

# the worst-case bound: the largest distance between two rows of the rescaled table (tokens the tokenizer can produce)
base = sr._CL["base"]
X = ((sr._CL["emb"][:len(rc.tok)] - base["mu_head"]) @ base["w_head"]).float()
sq = (X * X).sum(1)
D2 = 0.0
for s in range(0, len(X), 2048):
    D2 = max(D2, float((sq[s:s + 2048, None] + sq[None] - 2.0 * X[s:s + 2048] @ X.T).max()))
D = D2 ** 0.5
norms = sq.sqrt()
bound = {"diameter": D, "largest_norm": float(norms.max()), "median_norm": float(norms.median()),
         "tasks": [{"task": rc.ORDER1[k - 1], "opening": v["n"], "radius": v["rho"], "share": v["rho"] / D} for k, v in sorted(sr._CL["tasks"].items())]}
print(f"largest distance between two rows of the rescaled table: {D:.1f} (largest norm {bound['largest_norm']:.1f}, median norm {bound['median_norm']:.1f})")
print("guaranteed share of the weight that may be edited, R_k / D:", " ".join(f"{t['share']:.3f}" for t in bound["tasks"]))
json.dump({"edits": rows, "bound": bound}, open(os.path.join(HERE, "routing", "edits.json"), "w"), indent=1)
