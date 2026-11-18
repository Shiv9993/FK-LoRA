"""A sibling task: what the two rules of certified_switch.py do when a new task looks like an old one. CPU demo.

    python demo_sibling.py --model t5-small --data /path/to/CL_Benchmark

The five tasks of demo.py are learned first. Then a sixth task arrives that is a sibling of task 2 (sst2): the
same kind of sentences under the lightly reworded definition of task 2, but with other answer words ("good" and
"bad" instead of "POS" and "NEG"), so that one can see which adapter answered. The sixth task is learned twice from
the same starting point, once with each rule:
  certified     the sibling's inputs lie inside the zone of task 2 and stay with task 2: it cannot be learned,
                and this is reported when it arrives. Nothing that task 2 answered changes.
  exact_first   the sibling's inputs carry its instruction word for word and go to it: it is learned. The old
                inputs that carry exactly that instruction now get the sibling's answer; everything else is kept.
Small model, few examples: the numbers show which answers are kept, not how good a model can become.
"""
import argparse
import copy
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from certified_switch import ContinualModel

p = argparse.ArgumentParser()
p.add_argument("--model", default="t5-small")
p.add_argument("--data", default="CL_Benchmark")
p.add_argument("--train", type=int, default=400)
p.add_argument("--test", type=int, default=50)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=5e-3)
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
OTHER_WORDS = {"POS": "good", "NEG": "bad"}


def read(folder, name, split):
    data = json.load(open(os.path.join(args.data, folder, name, split + ".json")))
    d = data["Definition"][0] if isinstance(data["Definition"], list) else data["Definition"]
    return d.strip(), [(i["input"], i["output"] if isinstance(i["output"], list) else [i["output"]]) for i in data["Instances"]]


def prompt(definition, x):
    return TEMPLATE.replace("{d}", definition).replace("{x}", x)


config = AutoConfig.from_pretrained(args.model)
tok = AutoTokenizer.from_pretrained(args.model)
model = (AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM).from_pretrained(args.model)
unrelated = [[prompt(d, x) for x, _ in rows[:200]] for d, rows in (read("Long_Sequence", n, "train") for n in UNRELATED)]
cm = ContinualModel(model, tok, unrelated)
t0 = time.time()
tests = {}
for j, name in enumerate(TASKS):
    d, train = read("SuperNI", name, "train")
    _, test = read("SuperNI", name, "test")
    train, test = train[:args.train], test[:args.test]
    defs = [d] + REWORDED["tasks"][name.split("_")[0]]
    blocked = cm.learn([prompt(d, x) for x, _ in train], [y[0] for _, y in train], epochs=args.epochs, lr=args.lr, seed=args.seed)
    for li, level in enumerate(LEVELS):
        tests[j, level] = [prompt(defs[li], x) for x, _ in test]
    print(f"task {j + 1} learned ({name.split('_')[0]}): {blocked} of {len(train)} training inputs inside an older zone; {(time.time() - t0) / 60:.1f} min so far", flush=True)

# on the five tasks the two rules must give the same routes, in every wording
for key, texts in tests.items():
    rows = cm.ids(texts)
    assert (cm.zones.route(rows, "certified")[0] == cm.zones.route(rows, "exact_first")[0]).all(), key
print("the two rules give the same routes for every test input of the five tasks, in every wording", flush=True)
before = {key: cm.answer(texts)[0] for key, texts in tests.items()}

# the sibling of task 2
d2, train2 = read("SuperNI", TASKS[1], "train")
_, test2 = read("SuperNI", TASKS[1], "test")
sibling_def = REWORDED["tasks"]["task363"][0]                     # the light rewording of task 2's definition
keep = lambda rows: [(x, OTHER_WORDS[y[0]]) for x, y in rows if y[0] in OTHER_WORDS]
sib_train, sib_test = keep(train2[:args.train]), keep(test2[:args.test])
out = {"model": args.model, "args": vars(args), "sibling_definition": sibling_def, "rules": {}}
for rule in ("certified", "exact_first"):
    c = copy.deepcopy(cm)
    c.rule = rule
    blocked = c.learn([prompt(sibling_def, x) for x, _ in sib_train], [y for _, y in sib_train], epochs=args.epochs, lr=args.lr, seed=args.seed)
    ans, route, cert = c.answer([prompt(sibling_def, x) for x, _ in sib_test])
    learned = sum(a.strip().lower() == y for a, (_, y) in zip(ans, sib_test))
    old_words = sum(a.strip() in OTHER_WORDS for a in ans)
    changed = {}
    for level in LEVELS:
        changed[level] = {"changed": 0, "inputs": 0, "task2_changed": 0, "task2_inputs": 0}
        for j in range(len(TASKS)):
            now = c.answer(tests[j, level])[0]
            diff = sum(a != b for a, b in zip(now, before[j, level]))
            changed[level]["changed"] += diff
            changed[level]["inputs"] += len(now)
            if j == 1:
                changed[level]["task2_changed"], changed[level]["task2_inputs"] = diff, len(now)
    out["rules"][rule] = {"blocked": blocked, "train": len(sib_train), "sibling_right": learned, "sibling_inputs": len(sib_test),
                          "sibling_answered_with_old_words": old_words, "sibling_routes": sorted(set(route.tolist())), "changed": changed}
    print(f"{rule:<12} sibling: {blocked} of {len(sib_train)} training inputs inside an older zone; right answers {learned} of {len(sib_test)} "
          f"(answered with the old task's words: {old_words}); routes {sorted(set(route.tolist()))}", flush=True)
    print(f"{'':<12} answers of the five old tasks that changed: " + "; ".join(
        f"{level} {changed[level]['changed']} of {changed[level]['inputs']} (task 2: {changed[level]['task2_changed']} of {changed[level]['task2_inputs']})" for level in LEVELS), flush=True)
out["minutes"] = (time.time() - t0) / 60
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, f"sibling_{args.model.replace('/', '_')}_seed{args.seed}.json"), "w"), indent=1)
print(f"done in {out['minutes']:.1f} min", flush=True)
