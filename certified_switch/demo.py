"""Small demonstration of the certified switch on any Hugging Face model, on a CPU.

    python demo.py --model t5-small --data /path/to/CL_Benchmark

It teaches a model the five tasks of the paper one after another with certified_switch.py (no code of
another paper), then asks every test question again, with the original and with reworded instructions:
  certified   the certified switch
  nearest     nearest zone only (no priority for older zones)
  all         every adapter on (plain adapters, no protection)
The data are the SuperNI and Long_Sequence folders of the GainLoRA repository (CL_Benchmark). Small
models, few examples and a few epochs: the numbers show whether answers are kept, not how good a
model can become.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from certified_switch import ContinualModel

p = argparse.ArgumentParser()
p.add_argument("--model", default="t5-small")
p.add_argument("--data", default="CL_Benchmark")
p.add_argument("--train", type=int, default=300)
p.add_argument("--test", type=int, default=50)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="demo_results")
args = p.parse_args()
torch.manual_seed(args.seed)

TASKS = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
         "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
         "task875_emotion_classification"]
UNRELATED = "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
TEMPLATE = "Definition: {d}\n\nNow complete the following example -\nInput: {x}\nOutput: "
HERE = os.path.dirname(os.path.abspath(__file__))
REWORDED = json.load(open(os.path.join(HERE, "rewordings.json")))
LEVELS = ["original"] + REWORDED["levels"]


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


config = AutoConfig.from_pretrained(args.model)
tok = AutoTokenizer.from_pretrained(args.model)
model = (AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM).from_pretrained(args.model)
unrelated = [[prompt(d, x) for x, _ in rows[:200]] for d, rows in (read("Long_Sequence", n, "train") for n in UNRELATED)]
cm = ContinualModel(model, tok, unrelated)
print(f"{args.model}: {'encoder-decoder' if cm.seq2seq else 'decoder-only'}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M numbers, "
      f"{len(cm.layers)} adapted layers", flush=True)

tests, first, t0 = {}, {}, time.time()
for j, name in enumerate(TASKS):
    d, train = read("SuperNI", name, "train")
    _, test = read("SuperNI", name, "test")
    train, test = train[:args.train], test[:args.test]
    defs = [d] + REWORDED["tasks"][name.split("_")[0]]
    blocked = cm.learn([prompt(d, x) for x, _ in train], [y[0] for _, y in train], epochs=args.epochs, lr=args.lr, seed=args.seed)
    for li, level in enumerate(LEVELS):
        tests[j, level] = ([prompt(defs[li], x) for x, _ in test], [y for _, y in test])
        first[j, level] = cm.answer(tests[j, level][0])       # right after learning this task
    print(f"task {j + 1} learned ({name.split('_')[0]}): zone radius {cm.zones.tasks[-1]['rho']:.1f}, opening of {cm.zones.tasks[-1]['n']} tokens, "
          f"{blocked} of {len(train)} training inputs were inside an older zone; {(time.time() - t0) / 60:.1f} min so far", flush=True)

T, rows, saved = len(TASKS), [], {}
for level in LEVELS:
    for rule in ("certified", "nearest", "all"):
        learned = final = changed = total = right = cert_n = n_all = 0
        forget = []
        for j in range(T):
            texts, refs = tests[j, level]
            ans0, route0, _ = first[j, level]
            if rule == "all" and j == T - 1:
                ans, route, cert = ans0, route0, torch.zeros(len(texts), dtype=torch.bool)
            else:
                ans, route, cert = cm.answer(texts, rule=rule)
            s0 = sum(rouge_l(a, r) for a, r in zip(ans0, refs)) / len(refs)
            s1 = sum(rouge_l(a, r) for a, r in zip(ans, refs)) / len(refs)
            learned += s0 / T
            final += s1 / T
            if j < T - 1:
                forget.append(s0 - s1)
                changed += sum(a != b for a, b in zip(ans0, ans))
                total += len(ans)
            right += int((route == j + 1).sum())
            cert_n += int(cert.sum())
            n_all += len(texts)
            saved[f"{level}|{rule}|{j + 1}"] = {"answers": ans, "route": route.tolist(), "certified": cert.tolist(), "first": ans0}
        rows.append({"wording": level, "rule": rule, "learned": learned, "AP": final, "FT": sum(forget) / len(forget), "changed": changed,
                     "compared": total, "right_task": right / n_all, "certified": cert_n, "inputs": n_all})
        r = rows[-1]
        print(f"{level:<9}{rule:<10} learned {r['learned']:6.2f}  at the end {r['AP']:6.2f}  forgetting {r['FT']:6.2f}  answers changed "
              f"{r['changed']:>3} of {r['compared']}  right task {r['right_task']:.3f}" +
              (f"  certified {r['certified']} of {r['inputs']}" if rule == "certified" else ""), flush=True)
os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
json.dump({"model": args.model, "kind": "encoder-decoder" if cm.seq2seq else "decoder-only", "args": vars(args), "table": rows,
           "minutes": (time.time() - t0) / 60, "answers": saved},
          open(os.path.join(HERE, args.out, args.model.replace("/", "_") + f"_seed{args.seed}.json"), "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
