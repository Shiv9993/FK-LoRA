"""Continual learning WITHOUT instructions (Pilot B of silent_adapters/PLAN.md), t5-small, CPU.

Same five tasks and adapters as demo.py, but the definition is removed from every input (empty definition in the template).
Two ways to use the same adapters:
  all    every adapter on
  zones  each task claims a zone in a frozen feature space of the whole input (rule "centre" feature); an input is answered
         with the adapters 1..route, route = oldest task whose zone holds it, else smallest score / threshold.
"""
import argparse, json, math, os, time
import random
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel

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
p.add_argument("--tag", default="")
p.add_argument("--score", default="knn", choices=["knn", "ratio", "ridge"])
p.add_argument("--pool", default="")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
args = p.parse_args()
torch.manual_seed(args.seed)
if os.environ.get("OMP_NUM_THREADS"):
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

TASKS = ["task1510_evalution_relation_extraction", "task363_sst2_polarity_classification",
         "task748_glucose_reverse_cause_event_detection", "task1687_sentiment140_classification",
         "task875_emotion_classification"]
UNRELATED = "mnli,cb,qqp,rte,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
TEMPLATE = "Definition: {d}\n\nNow complete the following example -\nInput: {x}\nOutput: "
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
stem = os.path.join(HERE, args.out, f"no_instruction_{args.model.replace('/', '_')}_seed{args.seed}{'' if args.score == 'knn' else '_' + args.score}{args.tag}")
logf = open(stem + ".log", "w")
ALPHA, M_CAL, KNN = 0.05, 100, 10


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


def read(folder, name, split):
    data = json.load(open(os.path.join(args.data, folder, name, split + ".json")))
    return [(i["input"], i["output"] if isinstance(i["output"], list) else [i["output"]]) for i in data["Instances"]]


def prompt(x):                      # the definition is removed
    return TEMPLATE.replace("{d}", "").replace("{x}", x)


def rouge_l(pred, refs):
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


tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
DEV = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
model.to(DEV)
unrelated = [[prompt(x) for x, _ in read("Long_Sequence", n, "train")[:200]] for n in UNRELATED]
cm = ContinualModel(model, tok, unrelated)
Z = cm.zones
log(f"device: {DEV}")
log(f"{args.model}: no instruction, settings {vars(args)}")


def feats(texts):
    return ((Z._mean(cm.ids(texts)) - Z.mu_all) @ Z.w_all).cpu()      # features live on the CPU


def knn_score(f, fit):
    return torch.cdist(f, fit).topk(KNN, dim=1, largest=False).values[:, -1]


pool_f = None
RIDGE = args.score == "ridge"
RULE = "ridge" if RIDGE else "zones"
if args.score == "ratio":
    excl = {n.split("_")[0] for n in TASKS}
    pool_texts = [prompt(x) for t in json.load(open(args.pool))["tasks"] if t["name"].split("_")[0] not in excl for x in t["inputs"]]
    pool_f = feats(pool_texts)
    log(f"pool of {len(pool_texts)} unrelated inputs")


def zscore(f, fit):
    d = knn_score(f, fit)
    if pool_f is None:
        return d
    d0 = knn_score(f, pool_f)
    return torch.where(d0 == 0, torch.where(d == 0, torch.zeros_like(d), torch.full_like(d, float("inf"))), d / d0.clamp(min=1e-300))


def train_task(texts, targets, k=None):
    """Train the newest adapter on all inputs (route = the new task for every input)."""
    rows = cm.ids(texts)
    for layer in cm.layers:
        layer.new_task()
    k = len(zones) if k is None else k
    ans = tok(list(targets), max_length=cm.max_new, truncation=True)["input_ids"]
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(rows), generator=gen).tolist()
        for s in range(0, len(order), 16):
            pick = order[s:s + 16]
            cm.router.route = torch.full((len(pick),), k, dtype=torch.long, device=DEV)
            ids, att = cm._pad([rows[i] for i in pick])
            lab, la = cm._pad([ans[i] for i in pick])
            loss = model(input_ids=ids, attention_mask=att, labels=lab.masked_fill(la == 0, -100)).loss
            loss.backward(); opt.step(); opt.zero_grad()
    cm.router.route = None


@torch.no_grad()
def enc_feats(texts):
    """mean of the last-layer encoder states of the frozen base model (no adapter) over non-padding tokens, raw instance text."""
    rows = tok(list(texts), max_length=256, truncation=True)["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    out = np.zeros((len(rows), enc.config.d_model))
    for s in range(0, len(order), 32):
        pick = order[s:s + 32]
        w = max(len(rows[i]) for i in pick)
        ids = torch.full((len(pick), w), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(pick), w), dtype=torch.long)
        for b, i in enumerate(pick):
            ids[b, :len(rows[i])] = torch.tensor(rows[i]); att[b, :len(rows[i])] = 1
        h = enc(input_ids=ids, attention_mask=att).last_hidden_state.double()
        m = att.unsqueeze(-1).double()
        out[pick] = ((h * m).sum(1) / m.sum(1)).numpy()
    return out


G_list, c_list = [], []


def ridge_route(X, T):
    """X: standardised features (n x D). Route = 1 + argmax over the T columns of [x, 1] @ W_T."""
    D = X.shape[1]
    A = sum(G_list[:T]) + 1.0 * D * np.eye(D + 1)
    W = np.linalg.solve(A, np.stack(c_list[:T], 1))
    return torch.from_numpy(np.argmax(np.hstack([X, np.ones((len(X), 1))]) @ W, axis=1) + 1).long()


if RIDGE:
    enc = T5EncoderModel.from_pretrained(args.model).eval()
    for q in enc.parameters():
        q.requires_grad_(False)
    excl = {n.split("_")[0] for n in TASKS}
    ptexts = [x for t in json.load(open(args.pool))["tasks"] if t["name"].split("_")[0] not in excl for x in t["inputs"]]
    pick = sorted(random.Random(0).sample(range(len(ptexts)), 2000))
    pf = enc_feats([ptexts[i] for i in pick])
    MU, SD = pf.mean(0), pf.std(0)
    SD = np.where(SD < 1e-6, 1.0, SD)
    log(f"ridge: pool of {len(ptexts)} unrelated inputs, 2000 used, feature dim {pf.shape[1]}")


def stdz(f):
    return (f - MU) / SD


zones = []          # per task: {"fit": tensor, "q": float}


def scores(f, T):   # n x T
    return torch.stack([zscore(f, zones[k]["fit"]) for k in range(T)], 1)


def route_of(f, T):
    S = scores(f, T)
    q = torch.tensor([zones[k]["q"] for k in range(T)], dtype=S.dtype)
    inside = S <= q
    first = inside.int().argmax(1)              # oldest zone holding it
    near = (S / q).argmin(1)
    return torch.where(inside.any(1), first, near) + 1, inside


tests, test_f, own_zone, stats = [], [], [], []
first = {"all": [], RULE: []}
arr_te, arr_ho, ho_f = [], [], []
per_stage = []
t0 = time.time()
for j, name in enumerate(TASKS):
    train = read("SuperNI", name, "train")[:args.train]
    test = read("SuperNI", name, "test")[:args.test]
    ttexts = [prompt(x) for x, _ in train]
    if RIDGE:
        f_tr = stdz(enc_feats([x for x, _ in train]))
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(train), generator=g)
        ho, fit = perm[:200].tolist(), perm[200:].tolist()
        Zf = np.hstack([f_tr[fit], np.ones((len(fit), 1))])
        w = 200 / len(fit)
        G_list.append(w * Zf.T @ Zf); c_list.append(w * Zf.sum(0))
        f_te = stdz(enc_feats([x for x, _ in test]))
        train_task(ttexts, [y[0] for _, y in train], k=j + 1)
        tests.append(([prompt(x) for x, _ in test], [y for _, y in test]))
        test_f.append(f_te); ho_f.append(f_tr[ho])
        T = j + 1
        arr_te.append(ridge_route(f_te, T)); arr_ho.append(ridge_route(f_tr[ho], T))
        share = [float((ridge_route(test_f[jj], T) == jj + 1).float().mean()) for jj in range(T)]
        per_stage.append(share)
        first["all"].append(cm.answer(tests[j][0], route=torch.full((len(test),), T, dtype=torch.long))[0])
        first["ridge"].append(cm.answer(tests[j][0], route=arr_te[j])[0])
        log(f"task {T} learned ({name.split('_')[0]}): fit {len(fit)}, held-out {len(ho)}, "
            f"routed to own task {[round(s, 2) for s in share]}; {(time.time() - t0) / 60:.1f} min")
        continue
    f_tr = feats(ttexts)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(train), generator=g)
    m = min(M_CAL, len(train) // 4)
    cal, fit = f_tr[perm[:m]], f_tr[perm[m:]]
    srt = zscore(cal, fit).sort().values
    idx = min(math.ceil((1 - ALPHA) * (m + 1)), m) - 1
    zones.append({"fit": fit, "q": float(srt[idx])})
    f_te = feats([prompt(x) for x, _ in test])
    blocked_tr = blocked_te = 0
    if j:
        blocked_tr = int(torch.stack([zscore(f_tr, zones[k]["fit"]) <= zones[k]["q"] for k in range(j)], 1).any(1).sum())
        blocked_te = int(torch.stack([zscore(f_te, zones[k]["fit"]) <= zones[k]["q"] for k in range(j)], 1).any(1).sum())
    cover = zscore(f_te, zones[j]["fit"]) <= zones[j]["q"]
    stats.append({"task": name.split("_")[0], "q": zones[j]["q"], "blocked_train": blocked_tr, "blocked_test": blocked_te,
                  "coverage_test": int(cover.sum()), "n_train": len(train), "n_test": len(test)})
    train_task(ttexts, [y[0] for _, y in train])
    tests.append(([prompt(x) for x, _ in test], [y for _, y in test]))
    test_f.append(f_te)
    own_zone.append(cover)
    T = j + 1
    share = [float((route_of(test_f[jj], T)[0] == jj + 1).float().mean()) for jj in range(T)]
    per_stage.append(share)
    first["all"].append(cm.answer(tests[j][0], route=torch.full((len(test),), T, dtype=torch.long))[0])
    first["zones"].append(cm.answer(tests[j][0], route=route_of(test_f[j], T)[0])[0])
    log(f"task {T} learned ({name.split('_')[0]}): q={zones[j]['q']:.3f}, train inside older zone {blocked_tr}/{len(train)}, "
        f"test inside older zone {blocked_te}/{len(test)}, coverage {int(cover.sum())}/{len(test)}, "
        f"routed to own task {[round(s, 2) for s in share]}; {(time.time() - t0) / 60:.1f} min")

T = len(TASKS)
res, final_ans = {}, {}
for rule in ("all", RULE):
    learned = final = changed = total = 0
    forget, per_task_first, per_task_final = [], [], []
    for j in range(T):
        texts, refs = tests[j]
        r = torch.full((len(texts),), T, dtype=torch.long) if rule == "all" else (ridge_route(test_f[j], T) if RIDGE else route_of(test_f[j], T)[0])
        ans = cm.answer(texts, route=r)[0]
        final_ans[rule, j] = (ans, r)
        s0 = sum(rouge_l(a, y) for a, y in zip(first[rule][j], refs)) / len(refs)
        s1 = sum(rouge_l(a, y) for a, y in zip(ans, refs)) / len(refs)
        per_task_first.append(s0); per_task_final.append(s1)
        learned += s0 / T; final += s1 / T
        if j < T - 1:
            forget.append(s0 - s1)
            changed += sum(a != b for a, b in zip(first[rule][j], ans)); total += len(ans)
    res[rule] = {"learned": learned, "AP": final, "FT": sum(forget) / len(forget), "changed": changed, "compared": total,
                 "score_after_learning_per_task": per_task_first, "score_final_per_task": per_task_final}
    log(f"{rule:<6} learned {learned:6.2f}  at the end {final:6.2f}  forgetting {res[rule]['FT']:6.2f}  answers changed {changed} of {total}")
    log(f"       per task after learning {[round(s, 1) for s in per_task_first]}  at the end {[round(s, 1) for s in per_task_final]}")

if RIDGE:
    r_, a = res["ridge"], res["all"]
    cert, changed_routes = [], 0
    for j in range(T):
        end_ho = ridge_route(ho_f[j], T)
        c = int((end_ho != arr_ho[j]).sum()); m = 200
        end_te = ridge_route(test_f[j], T)
        cr = int((end_te != arr_te[j]).sum())
        changed_routes += cr
        ca = int(sum(x != y for x, y in zip(first["ridge"][j], final_ans["ridge", j][0]))) if j < T - 1 else 0
        cert.append({"task": TASKS[j].split("_")[0], "c": c, "m": m, "certificate": (c + 1) / (m + 1),
                     "test_routes_changed": cr, "test_routes_changed_share": cr / len(end_te),
                     "test_answers_changed": ca, "test_answers_changed_share": ca / len(end_te),
                     "answers_compared": j < T - 1})
    own_mean = sum(per_stage[-1]) / T
    res["ridge"].update({"routed_to_own_task_after_each_task": per_stage, "routed_to_own_task_final_per_task": per_stage[-1],
                         "routed_to_own_task_final_mean": own_mean, "routes_changed_total": changed_routes,
                         "routes_changed_tasks1to4": changed_routes - cert[-1]["test_routes_changed"],
                         "total_test_inputs": sum(len(tests[j][0]) for j in range(T)), "certificates": cert})
    log(f"ridge: routed to own task at end {[round(s, 2) for s in per_stage[-1]]} mean {own_mean:.3f}; routes changed between arrival and end {changed_routes}")
    for cc in cert:
        log("certificate:", json.dumps(cc))
    part_a = r_["changed"] <= 20 and r_["FT"] <= 3 and a["changed"] > 100
    part_b = r_["learned"] >= 0.9 * a["learned"] and own_mean >= 0.8
    if r_["changed"] > 40 or not part_b:
        verdict = "does not"
    elif part_a and part_b:
        verdict = "works"
    else:
        verdict = "partial"
    res["ridge"]["verdict_part_a"] = bool(part_a); res["ridge"]["verdict_part_b"] = bool(part_b)
    log("verdict parts: (a)", part_a, "(b)", part_b)
    log("verdict:", verdict)
    stats = []
else:
    in_own = chg_in_own = 0
    for j in range(T - 1):
        ans, _ = final_ans["zones", j]
        for i in range(len(ans)):
            if bool(own_zone[j][i]):
                in_own += 1
                chg_in_own += int(ans[i] != first["zones"][j][i])
    assert chg_in_own == 0, f"{chg_in_own} answers inside their own zone changed"
    res["zones"].update({"inside_own_zone_first4": in_own, "changed_inside_own_zone": chg_in_own,
                         "routed_to_own_task_after_each_task": per_stage,
                         "routed_to_own_task_final_per_task": per_stage[-1],
                         "routed_to_own_task_final_mean": sum(per_stage[-1]) / T})
    log(f"zones: inside own zone (tasks 1-4) {in_own} of {sum(len(tests[j][0]) for j in range(T - 1))}, changed among them {chg_in_own}; "
        f"routed to own task at end {[round(s, 2) for s in per_stage[-1]]}")
    log("per task stats:", json.dumps(stats))

    z, a = res["zones"], res["all"]
    if z["changed"] > 40:
        verdict = "does not"
    elif z["changed"] <= 20 and z["FT"] <= 3 and a["changed"] > 100:
        verdict = "works"
    else:
        verdict = "partial"
    log("verdict:", verdict)
json.dump({"model": args.model, "args": vars(args), "instruction": "removed from every input", "alpha": ALPHA, "m_cal": M_CAL,
           "knn": KNN, "rules": res, "per_task": stats, "verdict": verdict, "minutes": (time.time() - t0) / 60},
          open(stem + ".json", "w"), indent=1)
log(f"done in {(time.time() - t0) / 60:.1f} min")
