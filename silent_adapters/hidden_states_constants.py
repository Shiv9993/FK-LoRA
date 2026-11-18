"""Computes silent_adapters/hidden_states_constants.json (embedded by make_notebook_hidden.py):
the 90 fresh task names (batches 1-3, in order) with a fingerprint of their first 1000 inputs, and the mean / standard deviation of the
2,000-input T5-base pool (standardiser() of paper_certified/hidden_rules_study.py). Usage: python hidden_states_constants.py <scratch folder S>
Batch 4 is never opened."""
import hashlib, json, os, sys
import numpy as np

S = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))


def fingerprint(inputs):
    return hashlib.sha1("\x1e".join(inputs).encode("utf-8", errors="surrogatepass")).hexdigest()


tasks = []
for f in ("fresh_tasks.json", "fresh_tasks_2.json", "fresh_tasks_3.json"):
    for t in json.load(open(os.path.join(S, "ni_fresh", f)))["tasks"]:
        inp = [str(i["input"]) for i in t["instances"][:1000]]
        tasks.append({"name": t["name"], "n": len(inp), "sha1": fingerprint(inp)})
P = np.load(os.path.join(S, "noinstruction", "pool_t5hidden.npy"))
mu = P.mean(0); sd = P.std(0); sd[sd < 1e-6] = 1.0
out = {"fresh": tasks, "pool_shape": list(P.shape), "pool_dtype": str(P.dtype), "pool_sha1": hashlib.sha1(np.ascontiguousarray(P).tobytes()).hexdigest(),
       "mu": [float(x) for x in mu], "sd": [float(x) for x in sd], "stats_dtype": str(mu.dtype)}
json.dump(out, open(os.path.join(HERE, "hidden_states_constants.json"), "w"))
print(len(tasks), "tasks;", out["pool_sha1"], out["stats_dtype"])
