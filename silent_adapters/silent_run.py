"""Silent adapters: runs the official GainLoRA script `src/run_t5.py` unchanged, with silence training added.

Run it from the folder of the official code, with the same arguments as `src/run_t5.py` and
`--model_name olora --lambda1 0 --lambda2 0` (plain adapters: one new LoRA per task, older ones frozen,
all added together, no gate). This file adds one extra loss while a task is being trained:

    on stand-in inputs, the model WITH the new adapter must give the same next-word probabilities
    as the model WITHOUT it.

Stand-in inputs never use another task of the benchmark:
  kind a: the current task's own inputs, with the task definition replaced by the definition of an
          unrelated task;
  kind b: text of an unrelated task with that task's own definition.
The unrelated tasks come from a different benchmark shipped in the same repository (Long_Sequence),
leaving out its sentiment tasks because the SuperNI sequence contains sentiment tasks.

A third kind of stand-in (kind c) is the task's own input under a "definition" that is a random piece
of unrelated text, so the adapter meets a different wrong definition almost every time.

Separately, SILENT_LOCK=1 adds an exact switch. Every input of this benchmark starts with its task
definition. When a task is learned, the tokens of its definition are stored as a key. An input whose
first tokens equal a stored key is answered with every adapter added AFTER that task switched off, so
the model computes exactly what it computed right after learning that task. Inputs that match no key
are answered with all adapters on.

SILENT_CLAIM=1 is the certified switch, which does not need the exact wording. Each task stores a
signature: the average word-vector of its fixed opening (its task definition), with a radius. The ball
around the signature is the task's protected zone. Word-vectors come from the frozen embedding table
of the base model, so nothing here is trained; in the average, a word counts less the more common it
is in unrelated text (weight a / (a + frequency), SILENT_SIF=a).
An input inside a protected zone belongs to the OLDEST task whose zone contains it, and is answered
with the adapters that existed when that task was learned. Zones are never changed and older tasks
come first, so the answer on such an input can never change again, whatever is learned later
(a "certified" answer). An input outside every zone goes to the nearest zone, measured in units of
that zone's radius (best effort, no guarantee). The same file can also re-test every input with reworded task
definitions and under the other rules (exact key, all adapters on), see _extra_eval.

SILENT_BASELINE=1 changes nothing in the official code: it runs an official method (InfLoRA, GainLoRA)
as it is, and only adds the same re-test with reworded task definitions after the official test.

Settings are read from environment variables so that the official argument parser is left alone.
"""
import json
import os
import random
import runpy
import sys

REPO = os.getcwd()
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from t5_inflora import LoRALayer, T5Attention, T5Stack  # the official model code (used for inflora and olora)
from cl_trainer_olora import OLoRATrainer, skip_instructions   # the official plain-adapter trainer
from compute_metrics import metric_max_over_ground_truths, rougeL_score

LAM = float(os.environ.get("SILENT_LAMBDA", "1.0"))            # weight of the silence loss
N_POOL = int(os.environ.get("SILENT_POOL", "400"))             # stand-in inputs per task
NEG_BATCH = int(os.environ.get("SILENT_NEG_BATCH", "8"))       # stand-in inputs per training step
MAX_LEN = int(os.environ.get("SILENT_MAX_LEN", "256"))         # stand-in inputs are cut to this many tokens
KINDS = os.environ.get("SILENT_KINDS", "a,b").split(",")
LOCK = os.environ.get("SILENT_LOCK", "0") == "1"                # the exact switch
BASELINE = os.environ.get("SILENT_BASELINE", "0") == "1"        # an official method, untouched; only the re-test is added
POOL_TASKS = os.environ.get("SILENT_POOL_TASKS", "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic").split(",")
TEMPLATE = "Definition: {d}\n\nNow complete the following example -\nInput: {x}\nOutput: "   # as in src/cl_dataset.py

# ---- a switch that turns the NEW adapter off (older adapters and the base model are untouched) ----
_official_forward = LoRALayer.forward
LoRALayer._silent_off = False


def _forward(self, x):
    out = _official_forward(self, x)
    return out * 0.0 if (LoRALayer._silent_off and getattr(self, "_is_new", False)) else out


if not BASELINE:
    LoRALayer.forward = _forward


def _source(definition, text):
    return TEMPLATE.replace("{d}", definition.strip()).replace("{x}", text)


def _build_pool(trainer, model):
    """Choose the stand-in inputs and record what the model answers on them before the new adapter learns anything."""
    for m in model.modules():
        if isinstance(m, T5Attention):
            m.lora_q._is_new = True
            m.lora_v._is_new = True
    rng = random.Random(1000 * int(trainer.args.seed) + int(getattr(trainer, "cur_task_id", 0)))
    defs, texts = [], []
    for t in POOL_TASKS:
        data = json.load(open(os.path.join(REPO, "CL_Benchmark", "Long_Sequence", t, "train.json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
        assert isinstance(d, str) and d.strip(), f"unrelated task {t} has no definition"
        defs.append(d)
        texts.append([inst["input"] for inst in data["Instances"]])
    own = trainer.train_dataset
    sources = []
    for i in range(N_POOL):
        kind = KINDS[i % len(KINDS)]
        k = rng.randrange(len(defs))
        if kind == "a":
            sources.append(_source(defs[k], own[rng.randrange(len(own))]["Instance"]["sentence"]))
        elif kind == "c":                                              # a random piece of unrelated text as the definition
            words = rng.choice(texts[k]).split()
            n = rng.randint(8, 40)
            start = rng.randrange(max(1, len(words) - n))
            sources.append(_source(" ".join(words[start:start + n]) or defs[k], own[rng.randrange(len(own))]["Instance"]["sentence"]))
        else:
            sources.append(_source(defs[k], rng.choice(texts[k])))
    tok = trainer.tokenizer
    enc = tok(sources, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    gen_cfg = dict(max_new_tokens=50, num_beams=1, repetition_penalty=1.0, decoder_start_token_id=0,
                   eos_token_id=1, pad_token_id=0)                    # as in the official prediction step
    labels = torch.full((N_POOL, 50), -100, dtype=torch.long)
    was_training = model.training
    model.eval()
    LoRALayer._silent_off = True
    with torch.no_grad():
        for s in range(0, N_POOL, 16):
            ids, att = enc["input_ids"][s:s + 16].to(device), enc["attention_mask"][s:s + 16].to(device)
            out = model.generate(input_ids=ids, generation_config=GenerationConfig(attention_mask=att, **gen_cfg))
            out = out[:, 1:51].cpu()                                   # drop the start token
            out = out.masked_fill(out == 0, -100)                      # padding after the end
            labels[s:s + 16, :out.shape[1]] = out
    LoRALayer._silent_off = False
    model.train(was_training)
    empty = (labels != -100).sum(1) == 0
    labels[empty, 0] = 1                                               # an empty answer is just "end"
    order = list(range(N_POOL))
    rng.shuffle(order)
    print(f"[silent] {N_POOL} stand-in inputs ready (kinds {KINDS}); average answer length "
          f"{(labels != -100).sum(1).float().mean().item():.1f} tokens", flush=True)
    return {"ids": enc["input_ids"], "att": enc["attention_mask"], "labels": labels, "order": order, "pos": 0,
            "calls": 0, "sum": 0.0, "device": device}


def _silence_loss(trainer, model):
    st = getattr(trainer, "_silent_state", None)
    if st is None:
        st = trainer._silent_state = _build_pool(trainer, model)
    if st["pos"] + NEG_BATCH > len(st["order"]):
        random.Random(st["calls"]).shuffle(st["order"])
        st["pos"] = 0
    idx = st["order"][st["pos"]:st["pos"] + NEG_BATCH]
    st["pos"] += NEG_BATCH
    ids, att, labels = (st[k][idx].to(st["device"]) for k in ("ids", "att", "labels"))
    keep = att.sum(0) > 0                                              # drop columns that are padding for the whole batch
    ids, att = ids[:, keep], att[:, keep]
    model.eval()                                                       # no dropout in either pass, so they differ only by the adapter
    LoRALayer._silent_off = True
    with torch.no_grad():
        without = model(input_ids=ids, attention_mask=att, labels=labels).logits
    LoRALayer._silent_off = False
    with_new = model(input_ids=ids, attention_mask=att, labels=labels).logits
    model.train()
    mask = labels != -100
    kl = F.kl_div(F.log_softmax(with_new.float(), -1), F.softmax(without.float(), -1), reduction="none").sum(-1)
    loss = (kl * mask).sum() / mask.sum().clamp(min=1)
    st["calls"] += 1
    st["sum"] += loss.item()
    if st["calls"] == int(os.environ.get("SILENT_SELFCHECK", "0")):
        _selfcheck(model, ids, att, labels)
    if st["calls"] % 50 == 0:
        print(f"[silent] step {st['calls']}: average silence loss over the last 50 steps {st['sum'] / 50:.5f}", flush=True)
        st["sum"] = 0.0
    return loss


def _selfcheck(model, ids, att, labels):
    """After some training: 'new adapter switched off' must equal 'new adapter's weights set to zero'."""
    model.eval()
    with torch.no_grad():
        LoRALayer._silent_off = True
        off = model(input_ids=ids, attention_mask=att, labels=labels).logits
        LoRALayer._silent_off = False
        on = model(input_ids=ids, attention_mask=att, labels=labels).logits
        saved = [(m, m.lora_B.data.clone()) for m in model.modules() if isinstance(m, LoRALayer) and getattr(m, "_is_new", False)]
        for m, _ in saved:
            m.lora_B.data.zero_()
        zeroed = model(input_ids=ids, attention_mask=att, labels=labels).logits
        for m, b in saved:
            m.lora_B.data.copy_(b)
    model.train()
    print(f"[silent] self-check: switched-off vs zeroed weights differ by {(off - zeroed).abs().max().item():.2e} (must be ~0); "
          f"the trained adapter changes these outputs by {(on - off).abs().max().item():.2e}; new adapter layers: {len(saved)}", flush=True)
    assert (off - zeroed).abs().max().item() < 1e-4, "the off-switch does not match zeroed weights"


_official_compute_loss = OLoRATrainer.compute_loss


def _compute_loss(self, model, inputs, return_outputs=False):
    out = _official_compute_loss(self, model, inputs, return_outputs)
    if not model.training or LAM <= 0:
        return out
    extra = LAM * _silence_loss(self, model)
    return (out[0] + extra, out[1]) if return_outputs else out + extra


if not BASELINE:
    OLoRATrainer.compute_loss = _compute_loss


# ---- the exact switch (SILENT_LOCK=1) ----
_LOCKS = {"keys": {}, "t": None, "mask": None, "seen": 0, "unmatched": 0}


def _arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _lock_setup(trainer):
    """Store this task's key (the tokens of its definition) and load the keys of all earlier tasks."""
    tok, t = trainer.tokenizer, int(trainer.cur_task_id) + 1
    template = trainer.train_dataset[0]["Instance"]["instruction"]
    a = tok(template.replace("{0}", "A B"))["input_ids"]
    b = tok(template.replace("{0}", "zzq xkj"))["input_ids"]
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    key = torch.tensor(a[:n], dtype=torch.long)
    assert 5 <= len(key) < len(a), "could not find the task definition at the start of the input"
    keys = {}
    for d in (_arg("--previous_lora_path") or "").split(","):
        if d:
            k = torch.load(os.path.join(d, "task_key.pt"))
            keys[int(k["task_index"])] = k["key"]
    assert sorted(keys) == list(range(1, t)), f"keys of earlier tasks are missing: have {sorted(keys)}, need 1..{t - 1}"
    for k, old in keys.items():
        m = min(len(old), len(key))
        assert not torch.equal(old[:m], key[:m]), f"task {t} has the same definition as task {k}; the switch cannot tell them apart"
    keys[t] = key
    save_dir = os.path.join(trainer.args.output_dir, "saved_weights")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"task_index": t, "key": key}, os.path.join(save_dir, "task_key.pt"))
    _LOCKS.update(keys=keys, t=t)
    print(f"[locked] task {t}: key of {len(key)} tokens stored; {len(keys) - 1} earlier keys loaded", flush=True)


def _lock_mask(input_ids, count):
    """For every input: 1 for the adapters that existed when its task was learned, 0 for later ones."""
    t, B = _LOCKS["t"], input_ids.shape[0]
    matched = torch.zeros(B, dtype=torch.long, device=input_ids.device)
    best = torch.zeros(B, dtype=torch.long, device=input_ids.device)
    for k, key in _LOCKS["keys"].items():
        n = len(key)
        if input_ids.shape[1] >= n:
            hit = (input_ids[:, :n] == key.to(input_ids.device)).all(1) & (n > best)
            matched[hit], best[hit] = k, n
    if count:
        _LOCKS["seen"] += B
        _LOCKS["unmatched"] += int((matched == 0).sum())
    branch_task = torch.arange(t, 0, -1, device=input_ids.device)       # adapters are ordered newest first
    keep = (branch_task[None, :] <= matched[:, None]) | (matched[:, None] == 0)
    return keep.float().unsqueeze(-1)


_official_stack_forward = T5Stack.forward
_official_attention_forward = T5Attention.forward
_official_init = OLoRATrainer.__init__


def _stack_forward(self, *args, **kwargs):
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    if not self.is_decoder and input_ids is not None and _LOCKS["t"]:
        _LOCKS["mask"] = _lock_mask(input_ids, count=not self.training)
    return _official_stack_forward(self, *args, **kwargs)


def _attention_forward(self, *args, **kwargs):
    w, mask = kwargs.get("key_attention_weights"), _LOCKS["mask"]
    if w is not None and mask is not None and w.shape[:2] == mask.shape[:2]:
        kwargs["key_attention_weights"] = w * mask.to(w.dtype)
    return _official_attention_forward(self, *args, **kwargs)


def _init(self, *args, **kwargs):
    _official_init(self, *args, **kwargs)
    _lock_setup(self)


if LOCK:
    T5Stack.forward = _stack_forward
    T5Attention.forward = _attention_forward
    OLoRATrainer.__init__ = _init


# ---- the certified switch (SILENT_CLAIM=1) ----
CLAIM = os.environ.get("SILENT_CLAIM", "0") == "1"
KAPPA = float(os.environ.get("SILENT_KAPPA", "0.5"))            # zone radius = KAPPA x distance to the nearest known signature
EXTRA_EVAL = os.environ.get("SILENT_EVAL", "1") == "1"          # re-test with reworded definitions and the other rules
BG_N = int(os.environ.get("SILENT_BG", "4000"))                # unrelated inputs used to set the scale of distances
SIF_A = float(os.environ.get("SILENT_SIF", "0.001"))            # common words count less: weight a / (a + frequency); 0 = all equal
HEAD = 64
RULES = ["certified", "nearest", "centre", "key", "all"]
LEVELS = ["original", "light", "medium", "heavy"]
# Two switches that are off by default, so that every earlier run behaves exactly as before:
#   SA_NO_BLOCK=1     training and the official test use plain adapters: while task t is learned, adapters 1..t are on for
#                     EVERY training input (no input is given to an older task's adapter set). The re-test then uses the
#                     rules "all" and "centre" (and the table rules below).
#   SA_ROUTE_TABLE=<json file>   extra re-test rules "table:<name>": the route of a test input is read from a table
#                     {"routes": {name: {task folder: [ {str(tasks learned so far): route}, one per test example ]}}}
#                     (route = 1-based position in the task order; adapters 1..route answer the input).
NO_BLOCK = os.environ.get("SA_NO_BLOCK", "0") == "1"
ROUTE_TABLE_PATH = os.environ.get("SA_ROUTE_TABLE")
TABLE = {}
if ROUTE_TABLE_PATH:
    _loaded = json.load(open(ROUTE_TABLE_PATH))
    TABLE = _loaded.get("routes", _loaded)
KEEP_RULES = os.environ.get("SA_KEEP_RULES", "0") == "1"   # with SA_NO_BLOCK=1: keep every rule (the zones are still recorded), "all" first
if NO_BLOCK:
    RULES = ["all", "certified", "nearest", "centre", "key"] if KEEP_RULES else ["all", "centre"]
RULES = RULES + ["table:" + name for name in TABLE]
if KEEP_RULES:                    # mixed rules: the certified route where the input is inside a zone, otherwise the table route
    RULES = RULES + ["zone+table:" + name for name in ("ridge", "handicap") if name in TABLE]
# Reworded task definitions for the five-task test (written for this test; answer formats are unchanged).
REWORDED = {
    "task1510": [
        "Given a phrase that describes the relationship between two words, extract the two words and the lexical relation between them. The relation must be one of the types 'MemberOf', 'MadeOf', 'Synonym', 'Entails', 'HasA', 'HasProperty', 'PartOf', 'Antonym' or 'IsA'. The output should be in the format: word1 relation word2.",
        "You will see a short phrase that tells how two words are related. Find the two words and name the lexical relation that links them. Allowed relations are 'MemberOf', 'MadeOf', 'Synonym', 'Entails', 'HasA', 'HasProperty', 'PartOf', 'Antonym' and 'IsA'. Write your answer as: word1 relation word2.",
        "Read the phrase, pick out the pair of words it talks about, and say which link holds between them ('MemberOf', 'MadeOf', 'Synonym', 'Entails', 'HasA', 'HasProperty', 'PartOf', 'Antonym', 'IsA'). Answer format: word1 relation word2."],
    "task363": [
        'In this task, you are given a sentence from a movie review. Your task is to classify the sentence as "POS" if its sentiment is positive or as "NEG" if its sentiment is negative.',
        'You will read a sentence taken from a film review. Decide whether it expresses a positive or a negative opinion. Answer "POS" for positive and "NEG" for negative.',
        'Here is one line from a review of a movie. Label it "POS" when the writer liked what they saw and "NEG" when they did not.'],
    "task748": [
        'In this task, you are given a short story, and one sentence from the story is selected. Consider the events that happen after that sentence. Is any of them directly caused by it or made possible by it? Write your answer in the form " A >causes/enables> B". Try to use phrases and sentences from the story in your answer when possible. Do not change the selected sentence in your answer.',
        'You will read a short story in which one sentence has been picked out. Think about what happens later in the story and find an event that the picked sentence directly causes or makes possible. Give the answer as " A >causes/enables> B", reusing wording from the story where you can and keeping the picked sentence unchanged.',
        'A brief story follows, with one of its sentences marked. Name a later event that happens because of the marked sentence, or that it allows to happen. Use the format " A >causes/enables> B", copy text from the story when possible, and leave the marked sentence as it is.'],
    "task1687": [
        'In this task, you are given the text of a tweet. Your task is to classify the given tweet text into one of two categories, 1) positive or 2) negative, based on its content.',
        'You will read the text of a tweet. Decide from what it says whether the tweet is positive or negative, and answer with one of these two categories.',
        'Below is a post from Twitter. Say whether its mood is positive or negative.'],
    "task875": [
        "In this task, you are given a sentence that contains a particular emotion. You must classify the sentence into one of these emotions: 'joy', 'love', 'anger', 'fear', or 'surprise'.",
        "You will read a sentence that expresses a feeling. Choose which emotion it shows from this list: 'joy', 'love', 'anger', 'fear', 'surprise'.",
        "Which feeling does the writer of this sentence have? Pick one of 'joy', 'love', 'anger', 'fear', 'surprise'."],
}
_CL = {"t": None, "tasks": {}, "base": None, "emb": None, "weight": None, "pad": 0, "rule": "certified", "force": None,
       "seen": 0, "certified": 0, "trainer": None, "predict": None, "counts": None}
if NO_BLOCK:   # THE BLOCKING IS DECIDED HERE: with rule "certified" (the default above) _claim_mask gives every training input of
    _CL["rule"] = "all"   # task t to the oldest task whose zone contains it, and then adapter t is off and learns nothing. With "all",
                          # _routes returns route = t for every input, so adapters 1..t are on in training, in the dev checks
                          # and in the official test.


def _check_table(task_order, data_dir):
    """Stop with a clear message if the route table lacks an entry. Called before anything is trained."""
    for name, by_task in TABLE.items():
        for t in range(1, len(task_order) + 1):
            for j in range(t):
                task = task_order[j]
                if task not in by_task:
                    raise SystemExit(f"[table:{name}] the route table has no task {task}")
                n = len(json.load(open(os.path.join(data_dir, "SuperNI", task, "test.json")))["Instances"])
                rows = by_task[task]
                if len(rows) != n:
                    raise SystemExit(f"[table:{name}] task {task}: the table has {len(rows)} examples, test.json has {n}")
                for i, row in enumerate(rows):
                    route = row.get(str(t)) if isinstance(row, dict) else None
                    if not isinstance(route, int) or not 1 <= route <= t:
                        raise SystemExit(f"[table:{name}] task {task}, example {i}, after {t} tasks: the route is {route!r}, "
                                         f"it must be a whole number from 1 to {t}")


def _table_route(rule, example, t):
    return TABLE[rule[len("table:"):]][example["Dataset"]][int(example["Instance"]["id"])][str(t)]


def _means(input_ids, lengths):
    """Average word-vector of the first n real tokens of every input, for every n in `lengths` (None: all tokens).
    Each token counts with its weight (all equal if no weights are set)."""
    emb, weight = _CL["emb"], _CL["weight"]
    input_ids = input_ids.to(emb.device)
    out = {n: [] for n in lengths}
    for s in range(0, input_ids.shape[0], 16):
        ids = input_ids[s:s + 16]
        w = (ids != _CL["pad"]).to(emb.dtype)                          # padding adds nothing
        if weight is not None:
            w = w * weight[ids]
        e = emb[ids] * w.unsqueeze(-1)
        for n in lengths:
            part, count = (e, w) if n is None else (e[:, :n], w[:, :n])
            out[n].append(part.sum(1) / count.sum(1).clamp(min=1e-12).unsqueeze(-1))
    return {n: torch.cat(v) for n, v in out.items()}


def _whitener(x):
    """Centre and rescale so that unrelated inputs vary by about the same amount in every direction."""
    x = x.cpu()
    mu = x.mean(0)
    cov = (x - mu).T @ (x - mu) / (x.shape[0] - 1)
    cov = 0.5 * cov + 0.5 * torch.eye(cov.shape[0], dtype=cov.dtype) * torch.trace(cov) / cov.shape[0]
    w, v = torch.linalg.eigh(cov)
    return mu, v @ torch.diag(w ** -0.5) @ v.T


def _prefix_len(rows):
    n = 0
    while all(len(r) > n for r in rows) and len({r[n] for r in rows}) == 1:
        n += 1
    return n


def _claim_base(tok, max_len):
    """The fixed scale of distances and the signatures of unrelated tasks. Made once, from unrelated data only."""
    rng = random.Random(0)
    defs, texts = [], []
    for name in POOL_TASKS:
        data = json.load(open(os.path.join(REPO, "CL_Benchmark", "Long_Sequence", name, "train.json")))
        d = data["Definition"][0] if isinstance(data["Definition"], list) and data["Definition"] else data["Definition"]
        assert isinstance(d, str) and d.strip(), f"unrelated task {name} has no definition"
        defs.append(d)
        texts.append([inst["input"] for inst in data["Instances"]])
    picks = [rng.randrange(len(defs)) for _ in range(BG_N)]
    sources = [_source(defs[k], rng.choice(texts[k])) for k in picks]
    ids = tok(sources, max_length=max_len, truncation=True, padding=True, return_tensors="pt")["input_ids"]
    base = {}
    if SIF_A > 0:                                                      # how common each token is in unrelated text
        counts = torch.bincount(ids[ids != tok.pad_token_id], minlength=_CL["emb"].shape[0]).double()
        base["weight"] = SIF_A / (SIF_A + counts / counts.sum())
        base["weight"][tok.pad_token_id] = 0.0
        _CL["weight"] = base["weight"].to(_CL["emb"].device)
    m = _means(ids, [HEAD, None])
    base["mu_head"], base["w_head"] = _whitener(m[HEAD])
    base["mu_all"], base["w_all"] = _whitener(m[None])
    sigs = []
    for d, xs in zip(defs, texts):
        rows = tok([_source(d, x) for x in xs[:20]], max_length=max_len, truncation=True)["input_ids"]
        n = _prefix_len(rows)
        assert n >= 5, "an unrelated task has no fixed opening"
        sigs.append((_means(torch.tensor([rows[0][:n]]), [n])[n][0].cpu() - base["mu_head"]) @ base["w_head"])
    base["bg_sig"] = torch.stack(sigs)
    base["emb_sum"] = float(_CL["emb"].sum())
    return base


def _routes(input_ids, rule):
    """For every input: the task it is given to (adapters of tasks 1..route are on), and whether that is certified."""
    t, tasks, base = _CL["t"], _CL["tasks"], _CL["base"]
    dev = _CL["emb"].device
    input_ids = input_ids.to(dev)
    B = input_ids.shape[0]
    route = torch.full((B,), t, dtype=torch.long, device=dev)
    cert = torch.zeros(B, dtype=torch.bool, device=dev)
    if rule == "all":                                                  # every adapter on (plain adapters)
        return route, cert
    if rule == "key":                                                  # the exact switch: the stored opening, token for token
        best = torch.zeros(B, dtype=torch.long, device=dev)
        for k in sorted(tasks):
            key = tasks[k]["key"]
            if input_ids.shape[1] >= len(key):
                hit = (input_ids[:, :len(key)] == key).all(1) & (len(key) > best)
                route[hit], best[hit] = k, len(key)
        return route, best > 0
    order = sorted(tasks)
    m = _means(input_ids, sorted({tasks[k]["n"] for k in order}) + [None])
    if rule == "centre":                                               # nearest centre of the whole input (an earlier fallback)
        whole = (m[None] - base["mu_all"]) @ base["w_all"]
        dist = torch.stack([(whole - tasks[k]["centre"]).norm(dim=1) for k in order], 1)
        return torch.tensor(order, device=dev)[dist.argmin(1)], cert
    # distance from every signature, in units of that zone's radius: 1 or less means inside the zone
    rel = torch.stack([(((m[tasks[k]["n"]] - base["mu_head"]) @ base["w_head"]) - tasks[k]["sig"]).norm(dim=1) / tasks[k]["rho"]
                       for k in order], 1)
    nearest = torch.tensor(order, device=dev)[rel.argmin(1)]
    if rule == "certified":
        for i, k in enumerate(order):                                  # oldest first: an older claim is never overruled
            new = (rel[:, i] <= 1.0) & ~cert
            route[new] = k
            cert |= new
    return torch.where(cert, route, nearest), cert


def _new_zone(key, train_batches):
    """The protected zone (signature and radius) and the centre of a new task, from its fixed opening and its
    training inputs. Uses only this task's inputs, the fixed scale, and the signatures already known."""
    base, n = _CL["base"], len(key)
    sig = (_means(key[None], [n])[n][0] - base["mu_head"]) @ base["w_head"]
    known = torch.cat([base["bg_sig"]] + [v["sig"][None] for v in _CL["tasks"].values()])
    nearest_known = float((known - sig).norm(dim=1).min())
    rows = [(_means(ids, [None])[None] - base["mu_all"]) @ base["w_all"] for ids in train_batches]
    return {"n": n, "key": key.to(sig.device), "sig": sig, "rho": KAPPA * nearest_known, "centre": torch.cat(rows).mean(0)}, nearest_known


def _claim_setup(trainer):
    """Make and store this task's protected zone and centre; load those of all earlier tasks."""
    tok, t, model = trainer.tokenizer, int(trainer.cur_task_id) + 1, trainer.model
    weight = model.encoder.embed_tokens.weight
    assert not weight.requires_grad, "the embedding table must stay frozen"
    _CL.update(emb=weight.detach().double(), pad=tok.pad_token_id, trainer=trainer)
    dev = weight.device
    max_len = int(_arg("--max_source_length") or 512)
    tasks, base = {}, None
    for d in (_arg("--previous_lora_path") or "").split(","):
        if d:
            f = torch.load(os.path.join(d, "task_claim.pt"), map_location="cpu")
            tasks[int(f["task_index"])], base = f["task"], f["base"]
    assert sorted(tasks) == list(range(1, t)), f"zones of earlier tasks are missing: have {sorted(tasks)}, need 1..{t - 1}"
    if base is None:
        base = _claim_base(tok, max_len)
    _CL["weight"] = base["weight"].to(dev) if base.get("weight") is not None else None
    assert abs(base["emb_sum"] - float(_CL["emb"].sum())) <= 1e-6 * abs(base["emb_sum"]), "the embedding table has changed"
    _CL["base"] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}
    _CL["tasks"] = {k: {a: (b.to(dev) if torch.is_tensor(b) else b) for a, b in v.items()} for k, v in tasks.items()}
    _CL["t"] = t - 1

    template = trainer.train_dataset[0]["Instance"]["instruction"]
    a = tok(template.replace("{0}", "A B"))["input_ids"]
    b = tok(template.replace("{0}", "zzq xkj"))["input_ids"]
    n = _prefix_len([a, b])
    assert 5 <= n < len(a), "could not find the fixed opening of the input"
    data = trainer.train_dataset                                        # the task's own training inputs, as the model sees them
    batches = [trainer.data_collator([data[i] for i in range(s, min(s + 32, len(data)))])["input_ids"]
               for s in range(0, len(data), 32)]
    blocked = sum(int(_routes(ids, "certified")[1].sum()) for ids in batches) if t > 1 else 0   # already inside an older zone
    task, nearest_known = _new_zone(torch.tensor(a[:n], dtype=torch.long), batches)
    rho = task["rho"]
    _CL["tasks"][t] = task
    _CL["t"] = t
    save_dir = os.path.join(trainer.args.output_dir, "saved_weights")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"task_index": t, "task": {a: (b.cpu() if torch.is_tensor(b) else b) for a, b in task.items()}, "base": base},
               os.path.join(save_dir, "task_claim.pt"))
    print(f"[certified] task {t}: opening of {n} tokens; zone radius {rho:.2f} ({KAPPA} x {nearest_known:.2f}, the distance to the "
          f"nearest known signature); {len(tasks)} earlier zones loaded; training inputs already inside an older zone: "
          f"{blocked} of {len(trainer.train_dataset)}" + (" (not blocked: SA_NO_BLOCK is on, adapters 1..t are on for all of them)" if NO_BLOCK else ""), flush=True)


def _claim_mask(input_ids, count):
    force = _CL["force"]
    if force is not None and force.shape[0] == input_ids.shape[0]:
        route = force.to(input_ids.device)
    else:
        route, cert = _routes(input_ids, _CL["rule"])
        route = route.to(input_ids.device)
        if count:
            _CL["seen"] += input_ids.shape[0]
            _CL["certified"] += int(cert.sum())
    branch_task = torch.arange(_CL["t"], 0, -1, device=input_ids.device)   # adapters are ordered newest first
    return (branch_task[None, :] <= route[:, None]).float().unsqueeze(-1)


def _stack_forward_claim(self, *args, **kwargs):
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    if not self.is_decoder and input_ids is not None and _CL["t"]:
        _LOCKS["mask"] = _claim_mask(input_ids, count=not self.training)
    return _official_stack_forward(self, *args, **kwargs)


def _init_claim(self, *args, **kwargs):
    _official_init(self, *args, **kwargs)
    _claim_setup(self)


_official_predict = OLoRATrainer.predict


def _predict(self, test_dataset, *args, **kwargs):
    _CL["seen"] = _CL["certified"] = 0
    out = _official_predict(self, test_dataset, *args, **kwargs)
    _CL["predict"] = (test_dataset, out.predictions)
    _CL["counts"] = (_CL["seen"], _CL["certified"])
    return out


def _reworded(example, level):
    """The same test example with its task definition reworded (level 0: unchanged). None if there is no rewording."""
    if level == 0:
        return example
    new = REWORDED.get(example["Dataset"].split("_")[0])
    if new is None:
        return None
    inst = dict(example["Instance"])
    cut = inst["instruction"].rfind("Now complete the following example -")
    assert inst["instruction"].startswith("Definition: ") and cut > 0
    inst["instruction"] = "Definition: " + new[level - 1] + "\n\n" + inst["instruction"][cut:]
    return dict(example, Instance=inst)


def _extra_eval(rules):
    """Test every example again: with reworded definitions, and under each rule for choosing the adapters.
    An answer depends only on the input and on which adapters are on, so each (input, route) is generated once.
    With rules == ["official"] the model is used exactly as the official code left it (no route is forced)."""
    trainer, (dataset, official) = _CL["trainer"], _CL["predict"]
    model, tok, t = trainer.model, trainer.tokenizer, _CL["t"]
    device = next(model.parameters()).device
    model.eval()
    examples = [dataset[i] for i in range(len(dataset))]
    refs = {}
    for name in sorted({e["Dataset"] for e in examples}):
        data = json.load(open(os.path.join(_arg("--data_dir"), "SuperNI", name, "test.json")))
        refs[name] = [i["output"] if isinstance(i["output"], list) else [i["output"]] for i in data["Instances"]]
    gen = dict(max_new_tokens=50, num_beams=1, repetition_penalty=1.0, decoder_start_token_id=0, eos_token_id=1,
               pad_token_id=0, synced_gpus=False)                     # as in the official prediction step

    size = int(trainer.args.per_device_eval_batch_size)               # the batch size of the official test

    def batches(exs):
        for s in range(0, len(exs), size):
            b = trainer.data_collator(exs[s:s + size])
            yield b["input_ids"].to(device), b["attention_mask"].to(device)

    def generate(exs, route):
        preds = []
        for ids, att in batches(exs):
            _CL["force"] = None if route is None else torch.full((ids.shape[0],), route, dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    out = model.generate(input_ids=ids, generation_config=GenerationConfig(attention_mask=att, **gen))
            finally:
                _CL["force"] = None
            preds += skip_instructions(model, out.cpu().numpy(), tok)
        return preds

    result = {"task_index": t, "kappa": KAPPA, "sif_a": SIF_A, "rules": rules, "levels": {}}
    failed = {}
    for li, level in enumerate(LEVELS):
        exs = [x for x in (_reworded(e, li) for e in examples) if x is not None]
        if not exs:
            continue
        base_rules = [r for r in rules if "table:" not in r]
        routes = {rule: [] for rule in base_rules}
        certified = []
        for ids, _ in batches(exs):
            for rule in base_rules:
                if rule == "official":
                    routes[rule] += [None] * ids.shape[0]
                    continue
                r, c = _routes(ids, rule)
                routes[rule] += r.tolist()
                if rule == "certified":
                    certified += c.tolist()
        certified = certified or [False] * len(exs)
        preds = {}

        def fill(rule_list):
            for r in sorted({x for rule in rule_list for x in routes[rule]}, key=lambda x: -1 if x is None else x):
                idx = [i for i in range(len(exs)) if any(routes[rule][i] == r for rule in rule_list) and (i, r) not in preds]
                for i, p in zip(idx, generate([exs[i] for i in idx], r)):
                    preds[i, r] = p

        fill(base_rules)
        for rule in [r for r in rules if "table:" in r]:               # a table rule that fails is dropped; the other rules are kept
            try:
                if rule.startswith("zone+"):                           # certified route inside a zone, the table route elsewhere
                    routes[rule] = [routes["certified"][i] if certified[i] else _table_route(rule[len("zone+"):], e, t)
                                    for i, e in enumerate(exs)]
                else:
                    routes[rule] = [_table_route(rule, e, t) for e in exs]
                fill([rule])
            except Exception as err:
                routes.pop(rule, None)
                failed[rule] = f"{type(err).__name__}: {err}"
                print(f"[retest] RULE {rule} FAILED after task {t} and is left out: {failed[rule]}", flush=True)
        used = [r for r in rules if r in routes]
        rows = []
        for i, e in enumerate(exs):
            ref = refs[e["Dataset"]][int(e["Instance"]["id"])]
            row = {"dataset": e["Dataset"], "id": e["Instance"]["id"], "in_zone": bool(certified[i])}
            for rule in used:
                p = preds[i, routes[rule][i]]
                row[rule] = {"route": routes[rule][i], "pred": p,
                             "score": 100.0 * metric_max_over_ground_truths(rougeL_score, prediction=p, ground_truths=ref)}
            rows.append(row)
        result["levels"][level] = rows
        line = " | ".join(f"{rule} {sum(r[rule]['score'] for r in rows) / len(rows):.2f}" for rule in used)
        print(f"[retest] after task {t}, {level} wording, {len(rows)} inputs: average score {line}; "
              f"inside a protected zone {sum(r['in_zone'] for r in rows)}", flush=True)
    if "original" in result["levels"]:                                 # this re-test must reproduce the official test
        mine = [r[rules[0]]["pred"] for r in result["levels"]["original"]]   # rules[0] is never a table rule
        theirs = skip_instructions(model, official, tok)
        same = sum(a == b for a, b in zip(mine, theirs))
        result["same_as_official"] = [same, len(theirs)]
        print(f"[retest] check: this re-test gives the same answers as the official test on {same} of {len(theirs)} inputs", flush=True)
    if failed:                                                         # a rule that failed anywhere is removed everywhere
        result["rules"] = [r for r in rules if r not in failed]
        result["failed_rules"] = failed
        for rows in result["levels"].values():
            for row in rows:
                for r in failed:
                    row.pop(r, None)
    json.dump(result, open(os.path.join(trainer.args.output_dir, "extra_eval.json"), "w"))


if CLAIM:
    assert not LOCK and not BASELINE, "use only one of SILENT_LOCK, SILENT_CLAIM, SILENT_BASELINE"
    T5Stack.forward = _stack_forward_claim
    T5Attention.forward = _attention_forward
    OLoRATrainer.__init__ = _init_claim
    OLoRATrainer.predict = _predict

if BASELINE:                                                           # nothing of the official method is changed:
    assert not LOCK, "use only one of SILENT_LOCK, SILENT_CLAIM, SILENT_BASELINE"
    from transformers import Seq2SeqTrainer                            # only the test set and its answers are noted
    _hf_predict = Seq2SeqTrainer.predict

    def _predict_any(self, test_dataset, *args, **kwargs):
        out = _hf_predict(self, test_dataset, *args, **kwargs)
        _CL.update(trainer=self, predict=(test_dataset, out.predictions), t=int(self.cur_task_id) + 1)
        return out

    Seq2SeqTrainer.predict = _predict_any

if __name__ == "__main__":
    if BASELINE:
        print("[baseline] official method, unchanged; a re-test with reworded definitions follows the official test", flush=True)
    else:
        print(f"[silent] silence training on: weight {LAM}, {N_POOL} stand-in inputs, {NEG_BATCH} per step, kinds {KINDS}",
              flush=True)
    if LOCK:
        print("[locked] exact switch on", flush=True)
    if CLAIM:
        print(f"[certified] certified switch on (radius factor {KAPPA}, word weights a = {SIF_A})", flush=True)
    if TABLE:                                                          # before anything is loaded or trained
        _check_table((_arg("--task_order") or "").split(","), _arg("--data_dir"))
        print(f"[table] route table checked: {', '.join(TABLE)}", flush=True)
    script = os.path.join(REPO, "src", "run_t5.py")
    sys.argv = [script] + sys.argv[1:]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        if LOCK:
            print(f"[locked] inputs seen while testing: {_LOCKS['seen']}; not matched to any task: {_LOCKS['unmatched']}", flush=True)
    if CLAIM and _CL["counts"]:
        print(f"[certified] official test: inputs routed {_CL['counts'][0]}; inside a protected zone {_CL['counts'][1]}", flush=True)
    if (CLAIM or BASELINE) and EXTRA_EVAL and _CL["predict"]:
        _extra_eval(RULES if CLAIM else ["official"])
