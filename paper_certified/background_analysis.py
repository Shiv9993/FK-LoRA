"""How much do the routes depend on the unrelated inputs? (CPU, T5-base word-vectors, final rule.)
Its rules were written in silent_adapters/PLAN.md before it was run. Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/background_analysis.py

Variants: other random draws of the 4,000 unrelated inputs, fewer unrelated inputs, and only half of the unrelated
tasks. Nothing else is changed. Writes routing/background.json.
"""
import contextlib
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
with contextlib.redirect_stdout(io.StringIO()):          # sets the model up (and runs the standard check once)
    import routing_check as rc
sr = rc.sr
ALL = "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")


class Seeded:                                            # stands in for the `random` module inside silent_run: another draw
    def __init__(self, seed):
        self.seed = seed

    def Random(self, _=None):
        return random.Random(self.seed)


def variant(name, seed=0, n=4000, tasks=ALL):
    sr.random, sr.BG_N, sr.POOL_TASKS = Seeded(seed), n, list(tasks)
    sr._CL.update(weight=None, tasks={}, t=0)
    sr._CL["base"] = sr._claim_base(rc.tok, rc.MAX_LEN)
    out = {"name": name, "seed": seed, "inputs": n, "tasks": list(tasks)}
    with contextlib.redirect_stdout(io.StringIO()):
        for label, order in (("five", rc.FIVE), ("fifteen", rc.ORDER1)):
            out[label] = rc.run(order, label, 0.5)
            radii = [v["rho"] for v in sr._CL["tasks"].values()]
            out[label + "_radius"] = [min(radii), max(radii)]
    f5, f15 = out["five"], out["fifteen"]
    print(f"{name:<34} five: zone " + "/".join(f"{100 * f5[l]['zone']:.0f}" for l in sr.LEVELS) + " right " + "/".join(f"{100 * f5[l]['certified']:.1f}" for l in sr.LEVELS)
          + f" | fifteen: zone(orig) {100 * f15['original']['zone']:.0f} right " + "/".join(f"{100 * f15[l]['certified']:.1f}" for l in sr.LEVELS)
          + f" | wrong zone {sum(v['wrong'] for v in list(f5.values()) + list(f15.values()))} blocked {f5['original']['blocked'] + f15['original']['blocked']}"
          + f" | radii {out['fifteen_radius'][0]:.1f}-{out['fifteen_radius'][1]:.1f}", flush=True)
    return out


rows = [variant("as in the paper (seed 0, 4,000)")]
rows += [variant(f"another draw (seed {s})", seed=s) for s in (1, 2)]
rows += [variant(f"{n:,} unrelated inputs", n=n) for n in (500, 1000, 2000)]
rows += [variant("first five unrelated tasks", tasks=ALL[:5]), variant("last five unrelated tasks", tasks=ALL[5:])]
json.dump(rows, open(os.path.join(HERE, "routing", "background.json"), "w"), indent=1)
