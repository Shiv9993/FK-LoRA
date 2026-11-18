"""The five-task comparison: the project's certified switch ("ours") and Latent-LoRA ("latent_lora"), same data, same
training loop, same decoding, same scoring.

    python run_five.py --data /path/to/CL_Benchmark --out /path/to/results          (real run: t5-base, 3 seeds, 2 methods)
    LL_TEST=1 python run_five.py --data ... --out ...                                (tiny run for a CPU: t5-small)
    python run_five.py --summary --out /path/to/results                              (only print the table of finished runs)

Order of the tasks and the five tasks follow certified_switch/demo.py. After every task, the test inputs of every task
learned so far are answered with the original and with three reworded definitions; every answer is saved.
  ours         ContinualModel with rule "certified"; from the same trained adapters also "nearest" and "all" (plain adapters)
  latent_lora  rule "soft" (the paper); from the same trained adapters also "sum" (all adapters added, the paper's ablation)
A self-test (8 training examples, 2 optimiser steps per method, learn, answer, save) runs first, on the real device.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                         # latent_lora.py and certified_switch.py next to this file
import torch

TINY = os.environ.get("LL_TEST") == "1"

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.environ.get("LL_DATA", "CL_Benchmark"))
p.add_argument("--out", default=os.environ.get("LL_OUT", os.path.join(HERE, "results_tiny" if TINY else "results")))
p.add_argument("--model", default="t5-small" if TINY else "t5-base")
p.add_argument("--methods", default="ours,latent_lora")
p.add_argument("--seeds", default="42" if TINY else "42,43,44")
p.add_argument("--train", type=int, default=24 if TINY else 800, help="training examples per task (800 = all of train.json)")
p.add_argument("--test", type=int, default=8 if TINY else 100, help="test examples per task")
p.add_argument("--epochs", type=int, default=1 if TINY else 10)
p.add_argument("--batch", type=int, default=8 if TINY else 32, help="examples per optimiser step (effective batch)")
p.add_argument("--ll_batch", type=int, default=8, help="Latent-LoRA examples per optimiser step (the paper's batch size for T5 is 8)")
p.add_argument("--micro", type=int, default=4 if TINY else 16, help="examples per forward pass; batch/micro = gradient accumulation steps")
p.add_argument("--eval_batch", type=int, default=8 if TINY else 32)
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--max_new", type=int, default=50)
p.add_argument("--unrelated_n", type=int, default=30 if TINY else 200, help="inputs per unrelated task (only 'ours' uses them)")
p.add_argument("--ours_lr", type=float, default=3e-4)
p.add_argument("--ours_rank", type=int, default=4)
p.add_argument("--ours_alpha", type=int, default=32)
p.add_argument("--ll_lr", type=float, default=3e-4)
p.add_argument("--ll_rank", type=int, default=32)
p.add_argument("--ll_alpha", type=int, default=16)
p.add_argument("--ll_lambda", type=float, default=0.05)
p.add_argument("--ll_k", type=int, default=5)
p.add_argument("--ll_eps", type=float, default=0.01)
p.add_argument("--ll_ortho_reduce", default="sum", choices=["sum", "mean"])
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--device", default=os.environ.get("LL_DEVICE", ""))
p.add_argument("--summary", action="store_true", help="only print the table of the runs found in --out")
args = p.parse_args()

TASKS = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
         "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
         "task875_emotion_classification"]
UNRELATED = "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
TEMPLATE = "Definition: {d}\n\nNow complete the following example -\nInput: {x}\nOutput: "
METHODS = {"ours": ["certified", "nearest", "all"], "latent_lora": ["soft", "sum"]}      # first rule = the method's own
REWORDED = json.load(open(next(f for f in (os.path.join(HERE, "rewordings.json"),
                                           os.path.join(HERE, "..", "certified_switch", "rewordings.json")) if os.path.exists(f))))
LEVELS = ["original"] + REWORDED["levels"]


# ---- copied unchanged from certified_switch/demo.py: data reading, prompt, score --------------------------------------
def read(folder, name, split):
    data = json.load(open(os.path.join(args.data, folder, name, split + ".json")))
    d = data["Definition"][0] if isinstance(data["Definition"], list) else data["Definition"]
    return d.strip(), [(i["input"], i["output"] if isinstance(i["output"], list) else [i["output"]]) for i in data["Instances"]]


def prompt(definition, x):
    return TEMPLATE.replace("{d}", definition).replace("{x}", x)


def rouge_l(pred, refs):
    """Rouge-L F-measure on lower-case words, best over the reference answers."""
    def lcs(a, b):
        row = [0] * (len(b) + 1)
        for x in a:
            prev = 0
            for j, y in enumerate(b):
                prev, row[j + 1] = row[j + 1], prev + 1 if x == y else max(row[j + 1], row[j])
        return row[-1]
    best, a = 0.0, pred.lower().split()
    for ref in refs:
        b = ref.lower().split()
        n = lcs(a, b)
        if n:
            pr, rc = n / len(a), n / len(b)
            best = max(best, 2 * pr * rc / (pr + rc))
    return 100.0 * best
# -----------------------------------------------------------------------------------------------------------------------


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)                                         # a crash never leaves half a file


# ---------------------------------------------------------------------------------------------- summary of finished runs
def metrics_of(res):
    """Learned, AP, FT, answers changed, right task, per wording and rule, from the saved answers of one run (all tasks done so far)."""
    steps, T, refs = res["steps"], len(res["steps"]), res["references"]
    out = {}
    for level in res["levels"]:
        for rule in res["rules"]:
            s = {}                                                # s[i, j]: score of task j after task i was learned
            for i in range(T):
                for j in range(i + 1):
                    ans = steps[i]["answers"][f"{level}|{rule}|{j + 1}"]["answers"]
                    s[i, j] = sum(rouge_l(a, r) for a, r in zip(ans, refs[j])) / len(refs[j])
            changed = compared = 0
            for j in range(T - 1):
                first = steps[j]["answers"][f"{level}|{rule}|{j + 1}"]["answers"]
                last = steps[T - 1]["answers"][f"{level}|{rule}|{j + 1}"]["answers"]
                changed += sum(a != b for a, b in zip(first, last))
                compared += len(first)
            right = n = 0
            for j in range(T):
                route = steps[T - 1]["answers"][f"{level}|{rule}|{j + 1}"]["route"]
                right += sum(r == j + 1 for r in route)
                n += len(route)
            out[f"{level}|{rule}"] = {
                "tasks_done": T, "learned": sum(s[j, j] for j in range(T)) / T, "AP": sum(s[T - 1, j] for j in range(T)) / T,
                "FT": sum(max(s[i, j] for i in range(j, T - 1)) - s[T - 1, j] for j in range(T - 1)) / (T - 1) if T > 1 else 0.0,
                "FT_vs_learned": sum(s[j, j] - s[T - 1, j] for j in range(T - 1)) / (T - 1) if T > 1 else 0.0,
                "changed": changed, "compared": compared, "right_task": right / n,
                "score_matrix": [[s.get((i, j)) for j in range(T)] for i in range(T)]}
    return out


def summarize(out_dir):
    runs = []
    folder = os.path.join(out_dir, "results")
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if f.endswith(".json"):
            res = json.load(open(os.path.join(folder, f)))
            if res.get("steps"):
                res["metrics"] = metrics_of(res)
                runs.append(res)
    if not runs:
        print("no finished task found in", folder)
        return None
    summary = {}
    names = {"certified": "certified (ours)", "nearest": "nearest zone", "all": "all adapters on", "soft": "soft blend (paper)", "sum": "sum of adapters"}
    print(f"\n{'method':<13}{'rule':<19}{'wording':<10}{'seeds':>6}{'tasks':>6}{'Learned':>9}{'AP':>8}{'FT':>8}{'answers changed':>18}{'right task':>12}")
    for method, rules in METHODS.items():
        mine = [r for r in runs if r["method"] == method]
        if not mine:
            continue
        for rule in rules:
            for level in LEVELS:
                key = f"{level}|{rule}"
                vals = [r["metrics"][key] for r in mine]
                mean = lambda k: sum(v[k] for v in vals) / len(vals)
                row = {"seeds": [r["seed"] for r in mine], "tasks_done": [v["tasks_done"] for v in vals],
                       "per_seed": {str(r["seed"]): {k: v[k] for k in ("learned", "AP", "FT", "FT_vs_learned", "changed", "compared", "right_task")}
                                    for r, v in zip(mine, vals)},
                       "learned": mean("learned"), "AP": mean("AP"), "FT": mean("FT"), "FT_vs_learned": mean("FT_vs_learned"),
                       "changed": sum(v["changed"] for v in vals), "compared": sum(v["compared"] for v in vals), "right_task": mean("right_task")}
                summary[f"{method}|{rule}|{level}"] = row
                print(f"{method:<13}{names[rule]:<19}{level:<10}{len(vals):>6}{min(row['tasks_done']):>6}{row['learned']:>9.2f}{row['AP']:>8.2f}"
                      f"{row['FT']:>8.2f}{str(row['changed']) + ' of ' + str(row['compared']):>18}{row['right_task']:>12.3f}")
    print("\nLearned: mean score of each task right after it was learned. AP: mean score after the last task. FT: mean over tasks 1..T-1 of"
          "\n(best score after learning it, before the last task) minus (score at the end). Answers changed: test inputs of tasks 1..T-1 whose"
          "\nanswer at the end differs from the answer right after their task was learned (summed over seeds). Right task: share of test"
          "\ninputs at the end that were given to their own task (Latent-LoRA: most probable task). Scores: Rouge-L, mean over seeds.")
    write_json({"settings": runs[0]["settings"], "table": summary}, os.path.join(out_dir, "summary.json"))
    print("summary written to", os.path.join(out_dir, "summary.json"))
    return summary


# ---------------------------------------------------------------------------------------------- one method
def load_base(model_name):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)         # on the CPU; the wrappers move it to the device once
    return model, tok


def unrelated_texts(n):
    return [[prompt(d, x) for x, _ in rows[:n]] for d, rows in (read("Long_Sequence", name, "train") for name in UNRELATED)]


def build(method, model, tok, device, unrelated):
    from latent_lora import CertifiedOnDevice, LatentLoRAModel
    if method == "ours":
        return CertifiedOnDevice(model, tok, unrelated, device=device, micro=args.micro, rank=args.ours_rank, alpha=args.ours_alpha,
                                 max_len=args.max_len, max_new=args.max_new, rule="certified")
    return LatentLoRAModel(model, tok, rank=args.ll_rank, alpha=args.ll_alpha, lam=args.ll_lambda, k=args.ll_k, eps=args.ll_eps,
                           max_len=args.max_len, max_new=args.max_new, device=device, micro=args.micro, ortho_reduce=args.ll_ortho_reduce)


def lr_of(method):
    return args.ours_lr if method == "ours" else args.ll_lr


def batch_of(method):
    return args.batch if method == "ours" else args.ll_batch


def check_device(model, device):
    """Every parameter and buffer of the model must be on the device."""
    wrong = {str(t.device) for t in list(model.parameters()) + list(model.buffers()) if t.device.type != torch.device(device).type}
    assert not wrong, f"tensors of the model on {wrong}, expected {device}"


def selftest(methods, device):
    """8 training examples, one optimiser step per task, two tasks, per method: learn, answer (every rule), save. On the real device."""
    log(f"self-test on {device} ({args.model}) ...")
    t0 = time.time()
    tasks = [(name, read("SuperNI", name, "train"), read("SuperNI", name, "test")) for name in TASKS[:2]]
    unrelated = unrelated_texts(min(args.unrelated_n, 20))
    for method in methods:
        try:
            model, tok = load_base(args.model)
            obj = build(method, model, tok, device, unrelated)
            check_device(model, device)
            saved = {}
            for j, (name, (d, train), (_, test)) in enumerate(tasks):
                train, test = train[:8], test[:8]
                obj.micro = min(obj.micro, 4)                           # also exercises gradient accumulation (8 = 2 pieces of 4)
                obj.learn([prompt(d, x) for x, _ in train], [y[0] for _, y in train], epochs=1, batch=8, lr=lr_of(method), seed=0)
                check_device(model, device)
                for rule in METHODS[method]:
                    ans, route, cert = obj.answer([prompt(d, x) for x, _ in test], rule=rule, batch=4)
                    assert len(ans) == 8 and all(isinstance(a, str) for a in ans) and len(route) == 8, "bad answers"
                    saved[f"{j + 1}|{rule}"] = {"answers": ans, "route": route.tolist()}
            trained = [q for q in model.parameters() if q.requires_grad]
            assert trained and all(float(q.detach().abs().sum()) > 0 for q in trained[-8:]), "the newest adapter did not change in training"
            path = os.path.join(args.out, "selftest", f"{method}.json")
            write_json(saved, path)
            assert json.load(open(path)) == saved, "saved file differs from what was written"
            del obj, model
            log(f"  {method}: ok ({time.time() - t0:.0f} s so far)")
        except Exception:
            print("\n" + "=" * 100 + f"\nSELF-TEST FAILED for method '{method}' on device '{device}'. Nothing of the long run was started.\n"
                  "The error (a device mismatch, an out-of-memory error or a missing file shows here):\n" + "=" * 100, flush=True)
            traceback.print_exc()
            sys.exit(2)
    log(f"self-test passed in {time.time() - t0:.0f} s")


def run_job(method, seed, device):
    path = os.path.join(args.out, "results", f"{method}_seed{seed}.json")
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("complete"):
            log(f"{method} seed {seed}: already finished, skipped")
            return
    random.seed(seed)
    torch.manual_seed(seed)
    model, tok = load_base(args.model)
    unrelated = unrelated_texts(args.unrelated_n) if method == "ours" else []
    obj = build(method, model, tok, device, unrelated)
    check_device(model, device)
    n_params = sum(q.numel() for q in model.parameters())
    log(f"{method} seed {seed}: {args.model} ({n_params / 1e6:.0f}M frozen numbers), {len(obj.layers)} adapted layers, device {device}")
    res = {"method": method, "seed": seed, "settings": dict(vars(args), device=str(device)), "tasks": TASKS, "levels": LEVELS,
           "rules": METHODS[method], "references": [], "steps": [], "complete": False}
    tests, t0 = [], time.time()
    for j, name in enumerate(TASKS):
        d, train = read("SuperNI", name, "train")
        _, test = read("SuperNI", name, "test")
        train, test = train[:args.train], test[:args.test]
        defs = [d] + REWORDED["tasks"][name.split("_")[0]]
        tests.append({level: [prompt(defs[li], x) for x, _ in test] for li, level in enumerate(LEVELS)})
        res["references"].append([y for _, y in test])
        t1 = time.time()
        blocked = obj.learn([prompt(d, x) for x, _ in train], [y[0] for _, y in train], epochs=args.epochs, batch=batch_of(method),
                            lr=lr_of(method), seed=seed, log=args.log_every)
        t_train = time.time() - t1
        answers = {}
        for k in range(j + 1):                                          # every task learned so far, every wording, every rule
            for level in LEVELS:
                for rule in METHODS[method]:
                    ans, route, cert = obj.answer(tests[k][level], rule=rule, batch=args.eval_batch)
                    answers[f"{level}|{rule}|{k + 1}"] = {"answers": ans, "route": route.tolist(), "certified": cert.tolist()}
        step = {"task_index": j + 1, "task": name, "blocked_training_inputs": blocked, "train_minutes": t_train / 60,
                "minutes_total": (time.time() - t0) / 60, "loss_last_epoch": obj.info["loss"], "extra_term_last_epoch": obj.info["extra"],
                "micro_batch": obj.micro, "answers": answers}
        if method == "ours":
            step["zone"] = {"rho": obj.zones.tasks[-1]["rho"], "opening_tokens": obj.zones.tasks[-1]["n"]}
        res["steps"].append(step)
        res["complete"] = j == len(TASKS) - 1
        write_json(res, path)                                           # after every task
        m = metrics_of(res)[f"original|{METHODS[method][0]}"]
        log(f"{method} seed {seed}: task {j + 1}/{len(TASKS)} ({name.split('_')[0]}) learned; train {t_train / 60:.1f} min, answers done, "
            f"{(time.time() - t0) / 60:.1f} min so far; loss {obj.info['loss']:.3f}"
            + (f", orthogonal term {obj.info['extra']:.4g}" if obj.info["extra"] is not None else "")
            + f"; original wording, {METHODS[method][0]}: learned {m['learned']:.1f}, now {m['AP']:.1f}, changed {m['changed']} of {m['compared']}")
    res["metrics"] = metrics_of(res)
    write_json(res, path)
    log(f"{method} seed {seed}: finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if args.summary:
        summarize(args.out)
        sys.exit(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "0")) or torch.get_num_threads())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA was asked for but PyTorch sees no GPU.")
    methods = [m for m in args.methods.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    assert all(m in METHODS for m in methods), f"methods must be from {list(METHODS)}"
    os.makedirs(args.out, exist_ok=True)
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else "")
        + f"; methods {methods}; seeds {seeds}; {'TINY MODE' if TINY else 'full settings'}: {args.model}, {args.train} train / {args.test} test per task, "
          f"{args.epochs} epochs, batch {args.batch} for ours and {args.ll_batch} for latent_lora (pieces of {args.micro})")
    selftest(methods, device)
    for seed in seeds:
        for method in methods:
            try:
                run_job(method, seed, device)
            except Exception:
                traceback.print_exc()
                log(f"{method} seed {seed}: FAILED (see above); the tasks finished before the error are in the result file")
                sys.exit(1)
    summarize(args.out)
