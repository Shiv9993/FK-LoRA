"""Fetch the definitions and the first inputs of the tasks of the default English split of
Super-NaturalInstructions (github.com/allenai/natural-instructions, Apache-2.0), at one fixed commit.

    python3 many_tasks_fetch.py <output file .json>

Only the beginning of every task file is read (the definition, the positive examples and the first
instances are at the top). The data are not stored in this repository; this script gets them again.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

COMMIT = "55a365637381ce7f3748fa2eac7aef1a113bbb82"
RAW = f"https://raw.githubusercontent.com/allenai/natural-instructions/{COMMIT}/"
HEAD_BYTES = 131072
MAX_INPUTS = 6
MAX_CHARS = 6000                                          # far more than the 512 tokens that are used


def get(path, first_bytes=None):
    cmd = ["curl", "-sS", "-L", "--max-time", "120", "--retry", "3"]
    if first_bytes:
        cmd += ["-H", f"Range: bytes=0-{first_bytes - 1}"]
    return subprocess.run(cmd + [RAW + path], capture_output=True).stdout.decode("utf-8", errors="ignore")


def objects(text, key):
    """The objects of the list that follows "key": [ ... as far as the fetched part of the file goes."""
    m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), text)
    out, dec = [], json.JSONDecoder()
    if not m:
        return out
    j = m.end()
    while True:
        while j < len(text) and text[j] in " \n\r\t,":
            j += 1
        if j >= len(text) or text[j] != "{":
            return out
        try:
            obj, j = dec.raw_decode(text, j)
        except ValueError:
            return out
        out.append(obj)


def task(name_split):
    name, split = name_split
    text = get(f"tasks/{name}.json", HEAD_BYTES)
    m = re.search(r'"Definition"\s*:\s*\[\s*', text)
    if not m:
        return {"name": name, "split": split, "error": "no definition in the fetched part"}
    try:
        definition, _ = json.JSONDecoder().raw_decode(text, m.end())
    except ValueError:
        return {"name": name, "split": split, "error": "definition not readable"}
    inputs = [o["input"] for o in objects(text, "Instances") if isinstance(o.get("input"), str)]
    source = "instances"
    if len(inputs) < 2:
        inputs, source = [o["input"] for o in objects(text, "Positive Examples") if isinstance(o.get("input"), str)], "positive examples"
    return {"name": name, "split": split, "definition": definition.strip(), "source": source,
            "inputs": [x[:MAX_CHARS] for x in inputs[:MAX_INPUTS]]}


if __name__ == "__main__":
    names = [(n.strip(), split) for split in ("train", "test") for n in get(f"splits/default/{split}_tasks.txt").split("\n") if n.strip()]
    with ThreadPoolExecutor(8) as pool:
        tasks = list(pool.map(task, names))
    json.dump({"commit": COMMIT, "tasks": tasks}, open(sys.argv[1], "w"))
    good = [t for t in tasks if len(t.get("inputs", [])) >= 2]
    print(len(names), "tasks in the two split files;", len(good), "with a definition and at least two inputs;",
          sum(t.get("source") == "instances" for t in good), "of them with inputs from the instances;",
          sum("error" in t for t in tasks), "not readable")
