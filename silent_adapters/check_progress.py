# %% [markdown]
# # Check progress and scores (read-only)
#
# This notebook only **reads** the results that `stage0_colab.ipynb` has saved to Google Drive. It
# changes nothing and needs no GPU, so it can be run while the main notebook is still training.
#
# Runtime → Run all, allow Google Drive, then copy everything it prints and paste it into the chat.

# %%
import os, sys, json, glob

BASE = os.environ.get("CHECK_BASE") or "/content/drive/MyDrive/silent_adapters"
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")

runs = sorted(d for d in glob.glob(os.path.join(BASE, "*", "*")) if os.path.isdir(os.path.join(d, "outputs")))
assert runs, f"nothing found in {BASE}. Is this the same Google account as the training notebook?"

tables = {}
for run in runs:
    out = os.path.join(run, "outputs")
    done = sorted((d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "DONE"))),
                  key=lambda d: int(d.split("-")[0]))
    tasks = [d.split("-", 1)[1] for d in done]
    rows = []                                  # rows[i][j] = score on task j after learning task i
    for i, d in enumerate(done):
        res = json.load(open(os.path.join(out, d, "all_results.json")))
        rows.append([res.get(f"predict_eval_rougeL_for_{tasks[j]}") for j in range(i + 1)])
    prog_path = os.path.join(run, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"tasks": {}}
    name = os.path.basename(run)
    tables[name] = (tasks, rows)
    print("=" * 100)
    print(name)
    print(f"tasks finished: {len(done)}   |   time so far: {sum(t['minutes'] for t in prog['tasks'].values()) / 60:.2f} hours"
          f"   |   batch now: {prog.get('train_batch', 'as set')}")
    if not rows:
        continue
    T = len(rows)
    ap = sum(rows[-1]) / T
    ft = sum(max(r[j] for r in rows[j:-1]) - rows[-1][j] for j in range(T - 1)) / (T - 1) if T > 1 else 0.0
    print(f"AP (average score now, over the {T} finished tasks): {ap:.2f}      FT (forgetting): {ft:.2f}")
    print(f"\n{'task':<46}{'right after':>12}{'best later':>12}{'now':>8}{'change':>9}{'minutes':>9}{'batch':>7}")
    for j, t in enumerate(tasks):
        first, now = rows[j][j], rows[-1][j]
        best = max(r[j] for r in rows[j:])
        p = prog["tasks"].get(done[j], {})
        print(f"{t[:44]:<46}{first:>12.2f}{best:>12.2f}{now:>8.2f}{now - first:>+9.2f}{p.get('minutes', float('nan')):>9.1f}"
              f"{str(p.get('train_batch', '-')):>7}")

# %%
# Every score, so nothing has to be retyped: one line per finished task, listing the scores of all
# tasks learned so far (in order).
for name, (tasks, rows) in tables.items():
    print("\n" + name)
    for i, r in enumerate(rows):
        print(f"after task {i + 1:>2}: " + " ".join(f"{v:.1f}" for v in r))
for res_path in sorted(glob.glob(os.path.join(BASE, "*", "results.json"))):
    print("\n" + res_path, "| SVD step was done with:", json.load(open(res_path)).get("svd_backend"))

# %% [markdown]
# ## `locked` and `certified` runs: are the answers themselves unchanged?
# For every task, the answers written right after it was learned are compared, one by one, with the
# answers written after the last task.

# %%
def answers(path, dataset):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["Dataset"] == dataset:
            out[r["Instance"]["id"] + "|" + r["Instance"]["sentence"]] = r["Prediction"]
    return out


for run in runs:
    if "_locked_" not in os.path.basename(run) and "_certified_" not in os.path.basename(run):
        continue
    out = os.path.join(run, "outputs")
    done = sorted((d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "DONE"))), key=lambda d: int(d.split("-")[0]))
    print("\n" + os.path.basename(run))
    for d in done[:-1]:
        task = d.split("-", 1)[1]
        then = answers(os.path.join(out, d, "predict_eval_predictions.jsonl"), task)
        now = answers(os.path.join(out, done[-1], "predict_eval_predictions.jsonl"), task)
        diff = [k for k in then if then[k] != now.get(k)]
        print(f"{task[:44]:<46} answers compared: {len(then):>4}   changed: {len(diff):>3}")
        for k in diff[:3]:
            print("      then:", then[k][:110]); print("      now :", str(now.get(k))[:110])


# %% [markdown]
# ## The re-test with reworded task definitions
# After every task, runs made with the current notebook answer each test input again with the
# original definition and three rewordings. For `certified` this is done under four rules with the
# same trained adapters: `certified` (protected zones, then nearest task), `nearest` (nearest task
# only), `key` (the exact switch of `locked`), `all` (every adapter on, which is `plain`). The
# published methods are tested as they are (`official`).

# %%
for run in runs:
    out = os.path.join(run, "outputs")
    done = sorted((d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "DONE"))), key=lambda d: int(d.split("-")[0]))
    steps = []
    for d in done:
        if not os.path.exists(os.path.join(out, d, "extra_eval.json")):
            break
        steps.append(json.load(open(os.path.join(out, d, "extra_eval.json"))))
    if not steps:
        continue
    tasks, T = [d.split("-", 1)[1] for d in done], len(steps)
    print("\n" + os.path.basename(run), f"| tasks finished: {T}")
    zones = json.load(open(os.path.join(run, "progress.json")))["tasks"]
    for name in done:
        if zones.get(name, {}).get("zone"):
            print("  ", zones[name]["zone"])
    print(f"{'wording':<10}{'rule':<12}{'tasks':>6}{'learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}{'right task':>12}{'in a zone':>14}")
    for level, last in steps[-1]["levels"].items():
        have = [j for j in range(T) if any(r["dataset"] == tasks[j] for r in last)]
        for rule in steps[-1].get("rules") or ["certified", "nearest", "key", "all"]:
            score, by_id = {}, []
            for i in range(T):
                data = steps[i]["levels"].get(level, [])
                by_id.append({(r["dataset"], r["id"]): r for r in data})
                for j in have:
                    if j <= i:
                        sc = [r[rule]["score"] for r in data if r["dataset"] == tasks[j]]
                        score[i, j] = sum(sc) / len(sc)
            earlier = [j for j in have if j < T - 1]
            changed = total = 0
            for j in earlier:
                for key, r in by_id[j].items():
                    if key[0] == tasks[j]:
                        total += 1
                        changed += r[rule]["pred"] != by_id[-1][key][rule]["pred"]
            ft = sum(max(score[i, j] for i in range(j, T - 1)) - score[T - 1, j] for j in earlier) / len(earlier) if earlier else 0.0
            routed = [r for r in last if r[rule]["route"] is not None]
            right = f"{sum(r[rule]['route'] == tasks.index(r['dataset']) + 1 for r in routed) / len(routed):.3f}" if (routed and rule != "all") else ""
            zone = f"{sum(r['in_zone'] for r in last)} of {len(last)}" if rule == "certified" else ""
            print(f"{level:<10}{rule:<12}{len(have):>6}{sum(score[j, j] for j in have) / len(have):>9.2f}"
                  f"{sum(score[T - 1, j] for j in have) / len(have):>8.2f}{ft:>8.2f}{str(changed) + ' of ' + str(total):>18}{right:>12}{zone:>14}")
    print("re-test equals the official test (same answers, of inputs), per task:", [s_.get("same_as_official") for s_ in steps])
