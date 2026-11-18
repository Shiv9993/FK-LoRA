"""CPU check of the certified switch: where does every test input go? No training and no GPU.

Run it inside the folder of the official GainLoRA code (it reads CL_Benchmark from there):

    python /path/to/silent_adapters/routing_check.py --model t5-base
    python /path/to/silent_adapters/routing_check.py --model t5-small,t5-base,gpt2 --stress

It uses the functions of silent_run.py (the same code as the GPU runs) and only the tokenizer and
the frozen embedding table of each model, so any model with an embedding table can be checked,
whatever its architecture. Sequences: the five tasks and the 15 tasks of SuperNI order 1. Reworded
task definitions exist for five tasks. It reports
  in own zone    share of a task's test inputs inside that task's protected zone (certified);
  wrong zone     inputs inside the protected zone of another task (blocked for ever);
  right task     share given to their own task after all tasks were added, for three rules:
                 certified (oldest zone, else nearest zone), nearest (nearest zone only),
                 centre (nearest centre of the whole input: the fallback of the first version);
  changed later  inputs whose route changed after their task was added (never possible inside a zone).
--stress adds a sixth task that looks like an old one (the inputs of the sst2 task under a reworded
definition) and counts how many routes of the old task change under each rule.
Set SILENT_SIF=0 to count all words equally, as the first version did.
It says nothing about the answers the model gives: that needs the GPU run.
"""
import argparse
import json
import os
import sys
import types

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="t5-base", help="one or more model names, separated by commas")
parser.add_argument("--kappa", default="0.5", help="radius factor(s), separated by commas")
parser.add_argument("--stress", action="store_true")
parser.add_argument("--json", default="", help="also write the numbers to this file")
ARGS = parser.parse_args()
sys.argv = sys.argv[:1]

import torch
from transformers import AutoModel, AutoTokenizer


class _Official:                                            # stand-ins: only the routing functions are used here,
    def __init__(self, *a, **k): pass                       # so the official training code is not needed
    def forward(self, *a, **k): pass
    def compute_loss(self, *a, **k): pass
    def predict(self, *a, **k): pass


for _name, _attrs in (("t5_inflora", ("LoRALayer", "T5Attention", "T5Stack")), ("cl_trainer_olora", ("OLoRATrainer",)),
                      ("compute_metrics", ())):
    _m = types.ModuleType(_name)
    for _a in _attrs:
        setattr(_m, _a, type(_a, (_Official,), {}))
    sys.modules[_name] = _m
sys.modules["cl_trainer_olora"].skip_instructions = None
sys.modules["compute_metrics"].metric_max_over_ground_truths = sys.modules["compute_metrics"].rougeL_score = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import silent_run as sr

FIVE = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
        "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
        "task875_emotion_classification"]
ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization",
          "task181_outcome_extraction", "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
          "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
          "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
          "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
          "task591_sciq_answer_generation", "task1687_sentiment140_classification", "task875_emotion_classification"]
MAX_LEN = 512
RULES = ("certified", "nearest", "centre")
tok = None


def ids_of(texts):
    rows = tok(texts, max_length=MAX_LEN, truncation=True)["input_ids"]
    out = torch.full((len(rows), max(len(r) for r in rows)), tok.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


def load(name):
    out = {}
    for split in ("train", "test"):
        data = json.load(open(os.path.join("CL_Benchmark", "SuperNI", name, split + ".json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) else data["Definition"]
        out[split] = [i["input"] for i in data["Instances"]]
    out["definition"] = d.strip()
    return out


def template(definition):                                   # as built in src/cl_dataset.py
    return "Definition: " + definition + "\n\nNow complete the following example -\nInput: {0}\nOutput: "


def new_task(definition, train_inputs):
    """The zone and centre of a task with this definition, as silent_run.py makes them."""
    a = tok(template(definition).replace("{0}", "A B"))["input_ids"]
    b = tok(template(definition).replace("{0}", "zzq xkj"))["input_ids"]
    n = sr._prefix_len([a, b])
    train = ids_of([template(definition).replace("{0}", x) for x in train_inputs])
    batches = [train[s:s + 32] for s in range(0, len(train), 32)]
    blocked = sum(int(sr._routes(ids, "certified")[1].sum()) for ids in batches) if sr._CL["tasks"] else 0
    task, _ = sr._new_zone(torch.tensor(a[:n]), batches)
    return task, blocked, len(train)


def run(order, label, kappa):
    data = [load(name) for name in order]
    tests = {}
    for j, (name, d) in enumerate(zip(order, data)):
        defs = [d["definition"]] + sr.REWORDED.get(name.split("_")[0], [])
        for li in range(len(defs)):
            tests[j, li] = ids_of([template(defs[li]).replace("{0}", x) for x in d["test"]])
    sr.KAPPA = kappa
    sr._CL.update(tasks={}, t=0)
    then, blocked_all = {}, 0
    for j, d in enumerate(data):                            # add the tasks one after another
        task, blocked, _ = new_task(d["definition"], d["train"])
        blocked_all += blocked
        sr._CL["tasks"][j + 1] = task
        sr._CL["t"] = j + 1
        for li in range(4):
            if (j, li) in tests:
                then[j, li] = {rule: sr._routes(tests[j, li], rule) for rule in RULES}
    print(f"\n{label}, radius factor {kappa}: zone radii", " ".join(f"{sr._CL['tasks'][k]['rho']:.1f}" for k in sorted(sr._CL["tasks"])),
          f"| training inputs inside an older zone: {blocked_all}")
    out = {}
    for li in range(4):
        js = [j for j in range(len(order)) if (j, li) in tests]
        total = own = wrong = 0
        right = {rule: 0 for rule in RULES}
        changed = dict(right)
        for j in js:
            for rule in RULES:
                route, cert = sr._routes(tests[j, li], rule)
                r0, c0 = then[j, li][rule]
                assert not ((route != r0) & c0).any(), "a certified route changed: the guarantee is broken"
                right[rule] += int((route == j + 1).sum())
                changed[rule] += int((route != r0).sum())
                if rule == "certified":
                    total += len(route)
                    own += int((cert & (route == j + 1)).sum())
                    wrong += int((cert & (route != j + 1)).sum())
        print(f"  {sr.LEVELS[li]:<9} {len(js):>2} tasks, {total:>4} inputs | in own zone {own / total:.3f} | wrong zone {wrong} | right task, and routes "
              f"changed later: " + "; ".join(f"{rule} {right[rule] / total:.3f}, {changed[rule]}" for rule in RULES))
        out[sr.LEVELS[li]] = {"zone": own / total, "wrong": wrong, "blocked": blocked_all,
                              **{rule: right[rule] / total for rule in RULES}, **{rule + "_changed": changed[rule] for rule in RULES}}
    return out


def stress(kappa):
    """After the five tasks, a sixth task arrives that looks like task 2 (sst2): the same kind of inputs under a
    reworded definition. How many routes of the old task change, and is the conflict noticed?"""
    data = [load(name) for name in FIVE]
    old = data[1]
    defs = [old["definition"]] + sr.REWORDED["task363"]
    tests = {li: ids_of([template(defs[li]).replace("{0}", x) for x in old["test"]]) for li in range(4)}
    print(f"\nstress test, radius factor {kappa}: a sixth task that looks like task 2 (sst2). Routes of the 100 old test inputs of task 2:")
    out = {}
    for li_new in (1, 2, 3):
        sr.KAPPA = kappa
        sr._CL.update(tasks={}, t=0)
        for j, d in enumerate(data):
            sr._CL["tasks"][j + 1] = new_task(d["definition"], d["train"])[0]
            sr._CL["t"] = j + 1
        before = {rule: {li: sr._routes(tests[li], rule) for li in range(4)} for rule in ("certified", "nearest")}
        task, blocked, n_train = new_task(defs[li_new], old["train"])
        sr._CL["tasks"][6] = task
        sr._CL["t"] = 6
        print(f"  the new task uses the {sr.LEVELS[li_new]} rewording of task 2's definition: {blocked} of its {n_train} training inputs are "
              f"inside an older zone (noticed when it arrives); its own zone radius {task['rho']:.1f}")
        out[sr.LEVELS[li_new]] = {"blocked": blocked, "train": n_train, "radius": task["rho"]}
        for rule in ("certified", "nearest"):
            cells = []
            for li in range(4):
                route, _ = sr._routes(tests[li], rule)
                r0, c0 = before[rule][li]
                if rule == "certified":
                    assert not ((route != r0) & c0).any(), "a certified route changed: the guarantee is broken"
                out[sr.LEVELS[li_new]][f"{rule}|{sr.LEVELS[li]}"] = {"changed": int((route != r0).sum()), "certified_before": int(c0.sum()),
                                                                 "inputs": len(route)}
                cells.append(f"{sr.LEVELS[li]} {int((route != r0).sum())} (of them certified before: {int(((route != r0) & c0).sum())}; "
                             f"certified before: {int(c0.sum())})")
            print(f"      routes changed under `{rule}`: " + " | ".join(cells))
    return out


summary, stress_out = {}, {}
for name in ARGS.model.split(","):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:                            # models without a padding token: use the end-of-text token
        tok.pad_token = tok.eos_token
    emb = AutoModel.from_pretrained(name).get_input_embeddings().weight.detach().double()
    sr._CL.update(emb=emb, pad=tok.pad_token_id, weight=None, tasks={}, t=0)
    sr._CL["base"] = sr._claim_base(tok, MAX_LEN)
    print(f"\n{'=' * 20} {name}: embedding table {emb.shape[0]} x {emb.shape[1]}; word weights a = {sr.SIF_A} {'=' * 20}")
    for kappa in [float(k) for k in ARGS.kappa.split(",")]:
        summary[name, kappa] = (emb.shape, run(FIVE, "five tasks", kappa), run(ORDER1, "fifteen tasks (SuperNI order 1)", kappa))
        if ARGS.stress:
            stress_out[f"{name}|{kappa}"] = stress(kappa)
    del emb

if len(summary) > 1:
    print("\n\nAll models together. zone: inside own zone; right: given to the own task (certified rule). o/l/m/h: original, light, medium, heavy wording.")
    print(f"{'model':<28}{'table':>14}{'factor':>7} | five tasks: zone o/l/m/h        right o/l/m/h        | fifteen tasks: zone o  right o/l/m/h        wrong zone  blocked  changed")
    for (name, kappa), (shape, five, fifteen) in summary.items():
        f4 = lambda d, k: "/".join(f"{d[l][k] * 100:.0f}" for l in sr.LEVELS)
        print(f"{name:<28}{str(shape[0]) + 'x' + str(shape[1]):>14}{kappa:>7} | {f4(five, 'zone'):>22}  {f4(five, 'certified'):>19}        | "
              f"{fifteen['original']['zone'] * 100:>18.0f}  {f4(fifteen, 'certified'):>19}  {sum(v['wrong'] for v in list(five.values()) + list(fifteen.values())):>10}"
              f"  {five['original']['blocked'] + fifteen['original']['blocked']:>7}  {sum(v['certified_changed'] for v in list(five.values()) + list(fifteen.values())):>7}")

if ARGS.json:
    json.dump({"a": sr.SIF_A, "runs": [{"model": name, "kappa": kappa, "vocabulary": shape[0], "dimensions": shape[1], "five": five,
                                         "fifteen": fifteen} for (name, kappa), (shape, five, fifteen) in summary.items()],
               "stress": stress_out}, open(ARGS.json, "w"), indent=1)
