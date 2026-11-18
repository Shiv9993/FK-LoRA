"""CPU checks of the code in this folder (run: python test_equivalence.py --data /path/to/CL_Benchmark; t5-small, tiny).
 1. CertifiedOnDevice.learn with micro >= batch gives exactly the same adapters as ContinualModel.learn (certified_switch.py).
 2. With gradient accumulation (micro < batch) the adapters are close (same loss, different rounding).
 3. The out-of-memory fallback halves `micro` and repeats the step.
 4. Latent-LoRA: with a router weight of 1 on one task and 0 on the others, the layer equals base + that task's adapter; the
    adapter is U S R V^T of the SVD; R_t starts at 0; older adapters do not change when a new task is learned; answers are the same
    for any batch composition of the router weights.
 5. rouge_l in run_five.py is the same text as in certified_switch/demo.py.
"""
import argparse
import copy
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="t5-small")
a = ap.parse_args()
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from latent_lora import CertifiedOnDevice, LatentLoRALayer, LatentLoRAModel   # also puts ../certified_switch on the path
from certified_switch import ContinualModel

# 5. the score function is the one of demo.py
def func(path):
    t = open(path).read()
    i = t.index("def rouge_l")
    return t[i:t.index("\n\n\n", i)]
f1, f2 = func(os.path.join(HERE, "run_five.py")), func(os.path.join(HERE, "..", "certified_switch", "demo.py"))
assert f1.split("# ----")[0].strip() == f2.strip(), "rouge_l differs from demo.py"
print("5. rouge_l identical to demo.py: ok")

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForSeq2SeqLM.from_pretrained(a.model, dropout_rate=0.0)   # no dropout: its random masks would differ between piece sizes
import json
def rows(name, split):
    d = json.load(open(f"{a.data}/SuperNI/{name}/{split}.json"))
    dd = d["Definition"][0]
    return [f"Definition: {dd}\n\nNow complete the following example -\nInput: {i['input']}\nOutput: " for i in d["Instances"]], \
           [i["output"][0] if isinstance(i["output"], list) else i["output"] for i in d["Instances"]]
X1, Y1 = rows("task363_sst2_polarity_classification", "train"); X1, Y1 = X1[:12], Y1[:12]
X2, Y2 = rows("task875_emotion_classification", "train"); X2, Y2 = X2[:12], Y2[:12]
unrel = []
for n in ["mnli", "cb", "qqp", "rte", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic"]:
    d = json.load(open(f"{a.data}/Long_Sequence/{n}/train.json"))
    unrel.append([f"Definition: {d['Definition'][0]}\n\nNow complete the following example -\nInput: {i['input']}\nOutput: " for i in d["Instances"][:30]])

def adapters(model_wrapper):
    return [p.detach().clone() for p in model_wrapper.model.parameters() if p.requires_grad]

def train(cls, **kw):
    torch.manual_seed(1)
    m = copy.deepcopy(base)
    cm = cls(m, tok, unrel, rank=4, alpha=32, max_len=256, max_new=8, **kw)
    cm.learn(X1, Y1, epochs=2, batch=8, lr=3e-4, seed=5)
    cm.learn(X2, Y2, epochs=1, batch=8, lr=3e-4, seed=5)
    return cm

ref = train(ContinualModel)
same = train(CertifiedOnDevice, device="cpu", micro=8)
d = max(float((x - y).abs().max()) for x, y in zip(adapters(ref), adapters(same)))
print("1. max difference of adapters, ContinualModel vs CertifiedOnDevice(micro=batch):", d)
assert d == 0.0
acc = train(CertifiedOnDevice, device="cpu", micro=3)
d2 = max(float((x - y).abs().max()) for x, y in zip(adapters(ref), adapters(acc)))
print("2. with accumulation (pieces of 3 of 8): max difference", d2)
assert d2 < 1e-4, d2

# 3. out-of-memory fallback
cm = train(CertifiedOnDevice, device="cpu", micro=8)
orig, fails = cm._step, [2]
def flaky(pick, rows, answers, set_route, micro, extra):
    if fails[0] > 0 and micro > 1:
        fails[0] -= 1
        raise torch.cuda.OutOfMemoryError("fake")
    return orig(pick, rows, answers, set_route, micro, extra)
cm._step = flaky
cm.learn(X1, Y1, epochs=1, batch=8, lr=3e-4, seed=5)
assert cm.micro == 2, cm.micro
print("3. out-of-memory fallback: micro 8 ->", cm.micro, ": ok")

# 4. Latent-LoRA
torch.manual_seed(2)
m = copy.deepcopy(base)
ll = LatentLoRAModel(m, tok, rank=8, alpha=16, lam=0.05, max_len=256, max_new=8, device="cpu", micro=4)
layer = ll.layers[0]
W = layer.base.weight.detach().double()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
assert torch.allclose(layer.U.double(), U[:, :8].float().double()) and torch.allclose(layer.S.double(), S[:8].float().double())
ll.learn(X1, Y1, epochs=2, batch=8, lr=3e-3, seed=5)
snap = [r.detach().clone() for l in ll.layers for r in l.R]
assert all(float(r.abs().sum()) > 0 for r in snap), "R_1 did not train"
ll.learn(X2, Y2, epochs=2, batch=8, lr=3e-3, seed=5)
now1 = [l.R[0].detach() for l in ll.layers]
assert all(torch.equal(a_, b_) for a_, b_ in zip(snap, now1)), "an older adapter changed"
print("4a. R_1 frozen after task 2; extra (orthogonal) term of task 2:", ll.info["extra"])
x = torch.randn(3, 5, layer.base.in_features)
for t in (0, 1):
    p = torch.zeros(3, 2); p[:, t] = 1
    ll.router.p = p
    got = layer(x)
    R = layer.R[t].detach()
    want = layer.base(x) + (x @ layer.V @ R.T * layer.S) @ layer.U.T * layer.scale
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())
p = torch.full((3, 2), 0.5); ll.router.p = p
want = layer.base(x) + (x @ layer.V @ ((layer.R[0] + layer.R[1]).detach() * 0.5).T * layer.S) @ layer.U.T * layer.scale
assert torch.allclose(layer(x), want, atol=1e-5)
ll.router.p = None
print("4b. layer = base + U S R(x) V^T x for one-hot and blended weights: ok")
test = X1[:5] + X2[:5]
a1, r1, _ = ll.answer(test, rule="soft", batch=3)
a2, r2, _ = ll.answer(test, rule="soft", batch=10)
print("4c. routes:", r1.tolist(), "(first five = task 1, last five = task 2); same answers for two batch sizes:", a1 == a2)
post = ll.gmm.posterior(ll.ids(test))
assert torch.allclose(post.sum(1), torch.ones(10, dtype=torch.double))
print("all checks passed")
