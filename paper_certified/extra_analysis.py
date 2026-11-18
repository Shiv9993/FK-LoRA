"""Two extra analyses for the paper (CPU). Run inside the folder of the official GainLoRA code:

    python /path/to/paper_certified/extra_analysis.py

1. Geometry of the zones (T5-base word-vectors, final rule): for every task and wording, the relative
   distance of the test inputs from their own zone and from the nearest other zone (1 = border of a zone).
2. Cost: time to route one input, and what is stored per task.
Writes routing/geometry.json. The functions are those of silent_adapters/silent_run.py.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "..", "silent_adapters", "routing_check.py"), "--model", "t5-base"]
sys.path.insert(0, os.path.join(HERE, "..", "silent_adapters"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):          # sets the model up (and runs the standard check once)
    import routing_check as rc
import torch
sr = rc.sr


def rel(ids):
    tasks, base = sr._CL["tasks"], sr._CL["base"]
    order = sorted(tasks)
    m = sr._means(ids, sorted({tasks[k]["n"] for k in order}))
    return torch.stack([(((m[tasks[k]["n"]] - base["mu_head"]) @ base["w_head"]) - tasks[k]["sig"]).norm(dim=1) / tasks[k]["rho"]
                        for k in order], 1)


out = {}
for label, order in (("five", rc.FIVE), ("fifteen", rc.ORDER1)):
    data = [rc.load(n) for n in order]
    sr.KAPPA = 0.5
    sr._CL.update(tasks={}, t=0)
    for j, d in enumerate(data):
        sr._CL["tasks"][j + 1] = rc.new_task(d["definition"], d["train"])[0]
        sr._CL["t"] = j + 1
    rows = []
    for j, (name, d) in enumerate(zip(order, data)):
        defs = [d["definition"]] + sr.REWORDED.get(name.split("_")[0], [])
        row = {"task": name, "opening": sr._CL["tasks"][j + 1]["n"], "radius": sr._CL["tasks"][j + 1]["rho"], "wordings": {}}
        for li, definition in enumerate(defs):
            r = rel(rc.ids_of([rc.template(definition).replace("{0}", x) for x in d["test"]]))
            own = r[:, j]
            other = torch.cat([r[:, :j], r[:, j + 1:]], 1).min(1).values
            row["wordings"][sr.LEVELS[li]] = {"own_median": float(own.median()), "own_max": float(own.max()), "other_min": float(other.min()),
                                             "other_median": float(other.median()), "inputs": len(own)}
        rows.append(row)
    out[label] = rows
ids = rc.ids_of([rc.template(d["definition"]).replace("{0}", x) for d in data for x in d["test"][:40]])
t0 = time.time()
for _ in range(3):
    sr._routes(ids, "certified")
per_input = (time.time() - t0) / 3 / len(ids)
d = sr._CL["emb"].shape[1]
out["cost"] = {"seconds_per_input_cpu": per_input, "inputs_timed": len(ids), "tasks": len(sr._CL["tasks"]), "dimensions": d,
               "numbers_per_task": d + 2, "numbers_once": sr._CL["emb"].shape[0] + 2 * d + 2 * d * d}
json.dump(out, open(os.path.join(HERE, "routing", "geometry.json"), "w"), indent=1)
for label in ("five", "fifteen"):
    for row in out[label]:
        print(label, row["task"][:12], "opening", row["opening"], "radius %.1f" % row["radius"],
              " | ".join(f"{k}: own {v['own_median']:.2f} (max {v['own_max']:.2f}), other min {v['other_min']:.2f}" for k, v in row["wordings"].items()))
print("cost:", out["cost"])
