"""Fetch 30 'fresh' Super-NaturalInstructions tasks (no benchmark task, none used by any earlier test), at the
same fixed commit as many_tasks_fetch.py.

    python3 fresh_tasks_fetch.py <scratch folder S> [--batch B]   # reads S/ni/ni_tasks.json, writes S/ni_fresh/fresh_tasks.json

Selection: random.Random(0) shuffle of the sorted names of the 875 tasks without the 15 benchmark tasks of
SuperNI order 1; in that order, the first 30 with at least 1000 parsed instances (non-empty input), at most 120
candidates tried. With --batch 2 (or 3) the walk continues in the same shuffled list directly after the
candidates tried for the earlier batches (read from their files: kept + skipped), keeps the next 30, has no cap
on candidates (stops when the list ends) and writes fresh_tasks_<B>.json with fields "batch" and
"positions_tried". The first 12 MB of each file are read. The data are not stored in the repository.
"""
import json
import os
import random
import statistics
import subprocess
import sys

from many_tasks_fetch import COMMIT, RAW, objects
import re

S = sys.argv[1]
BATCH = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 1
HEAD = 12 * 1024 * 1024
WANT, MAXTRY, N = 30, 120, 1000
SELECTION = ("random.Random(0) shuffle of the sorted names of the 875 tasks without the 15 benchmark tasks; "
             "first 30 with at least 1000 parsed instances")


def get(path):
    cmd = ["curl", "-sS", "-L", "--max-time", "300", "--retry", "3", "-H", f"Range: bytes=0-{HEAD - 1}", RAW + path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("curl failed: " + r.stderr.decode()[:200])
    return r.stdout.decode("utf-8", errors="ignore")


names = sorted(t["name"] for t in json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"])
bench = set(os.listdir(f"{S}/ext_gainlora/repo/CL_Benchmark/SuperNI/"))
names = [n for n in names if n not in bench]
random.Random(0).shuffle(names)
out_dir = f"{S}/ni_fresh"
os.makedirs(out_dir, exist_ok=True)
out = f"{out_dir}/fresh_tasks.json" if BATCH == 1 else f"{out_dir}/fresh_tasks_{BATCH}.json"
start = 0
for b in range(1, BATCH):
    prev = json.load(open(f"{out_dir}/fresh_tasks.json" if b == 1 else f"{out_dir}/fresh_tasks_{b}.json"))
    start += len(prev["tasks"]) + len(prev["skipped"])
kept, skipped = [], []


def save(final=False):
    d = {"commit": COMMIT, "selection": SELECTION, "tasks": kept, "skipped": skipped}
    if BATCH > 1:
        d = {"commit": COMMIT, "batch": BATCH, "selection": SELECTION + f" (batch {BATCH}: continues after the candidates of the earlier batches)",
             "positions_tried": [start, start + len(kept) + len(skipped)], "tasks": kept, "skipped": skipped}
    json.dump(d, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


for name in (names[:MAXTRY] if BATCH == 1 else names[start:]):
    if len(kept) >= WANT:
        break
    try:
        text = get(f"tasks/{name}.json")
    except Exception as e:
        skipped.append({"name": name, "reason": f"fetch failed: {e}"})
        save()
        continue
    inst = []
    for o in objects(text, "Instances"):
        if isinstance(o.get("input"), str) and o["input"].strip():
            out0 = o.get("output")
            out0 = out0[0] if isinstance(out0, list) and out0 else out0
            if isinstance(out0, str):
                inst.append({"input": o["input"], "output": out0})
    if len(inst) < N:
        skipped.append({"name": name, "reason": f"too few instances ({len(inst)} parsed)"})
        save()
        continue
    m = re.search(r'"Definition"\s*:\s*\[\s*', text)
    try:
        definition = json.JSONDecoder().raw_decode(text, m.end())[0] if m else None
    except ValueError:
        definition = None
    kept.append({"name": name, "definition": definition, "n_instances_parsed": len(inst), "instances": inst[:N]})
    save()
    print(f"{len(kept):2d} {name:70s} {len(inst):6d} {int(statistics.median(len(i['input']) for i in inst[:N])):6d}", flush=True)
save()
print(len(kept), "kept;", len(skipped), "skipped;", len(kept) + len(skipped), "tried")
