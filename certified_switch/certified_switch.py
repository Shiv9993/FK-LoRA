"""The certified switch as stand-alone code, for any model that has an embedding table and linear layers.

Nothing here depends on one architecture or on the code of another paper. Three parts:

  Zones          decides, from the frozen word-vectors of an input, which task the input is given to.
                 Each task has a protected zone that is never changed; the oldest zone that contains
                 an input owns it for ever; an input outside every zone goes to the nearest zone.
  RoutedLoRA     one low-rank adapter per task on a linear layer. Older adapters are frozen. For every
                 input only the adapters of tasks 1..route are switched on.
  ContinualModel puts both on a Hugging Face model (encoder-decoder or decoder-only), learns tasks one
                 after another and answers inputs.

Why an answer inside a zone can never change: the route of such an input is the oldest zone that
contains it. Later tasks only add newer zones, so the route stays the same, and the adapters it
switches on are frozen. This holds for any model and any later tasks.

Two rules are offered (`rule=` of ContinualModel):
  "certified"    the rule above. A new task whose instruction lies inside an older zone is refused on those
                 inputs (they stay with the older task); `learn` reports how many.
  "exact_first"  an input that carries the instruction of a task token for token goes to that task (the
                 oldest, if several tasks have the same instruction); otherwise as above. A sibling of an old
                 task can then be learned. The guarantee is weaker: an answer inside a zone stays the same
                 unless a task is learned later whose instruction that input carries exactly. An input that
                 carries a learned instruction exactly keeps its answer without conditions.
On tasks whose instructions do not collide the two rules give the same routes.

What it needs: every task's inputs start with the same text (its instruction). That opening is
what the zone is built from.
"""
import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------------------
# 1. Which task does an input belong to?
# ----------------------------------------------------------------------------------------------
def common_prefix(rows):
    """Number of tokens that all rows share at their start."""
    n = 0
    while all(len(r) > n for r in rows) and len({r[n] for r in rows}) == 1:
        n += 1
    return n


class Zones:
    """Protected zones built from a frozen embedding table.

    table    the model's input embedding matrix (vocabulary x dimensions); it is only read
    kappa    radius of a zone = kappa x distance from its signature to the nearest signature known
             when the task is added (unrelated tasks and earlier tasks)
    a        a token counts with weight a / (a + its share among the tokens of unrelated text)
    """

    def __init__(self, table, pad_id, kappa=0.5, a=1e-3, head=64):
        self.table = table.detach().double()
        self.pad, self.kappa, self.a, self.head = pad_id, kappa, a, head
        self.weight = None
        self.tasks = []                                  # one dict per task, oldest first; never changed

    def _mean(self, rows, n=None):
        """Weighted average word-vector of the first n tokens of every row (all tokens if n is None)."""
        out = []
        for r in rows:
            ids = torch.as_tensor(r[:n] if n else r, dtype=torch.long, device=self.table.device)
            w = (ids != self.pad).double()
            if self.weight is not None:
                w = w * self.weight[ids]
            out.append((self.table[ids] * w[:, None]).sum(0) / w.sum().clamp(min=1e-12))
        return torch.stack(out)

    @staticmethod
    def _whitener(x):
        """Centre and rescale so that unrelated inputs vary by about the same amount in every direction."""
        mu = x.mean(0)
        cov = (x - mu).T @ (x - mu) / (x.shape[0] - 1)
        cov = 0.5 * cov + 0.5 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * torch.trace(cov) / cov.shape[0]
        w, v = torch.linalg.eigh(cov)
        return mu, v @ torch.diag(w ** -0.5) @ v.T

    def fit_background(self, unrelated_tasks):
        """unrelated_tasks: for each unrelated task, a list of token-id rows (its inputs, instruction included).
        Sets the token weights, the scale of distances and the signatures of the unrelated tasks. Done once."""
        rows = [r for task in unrelated_tasks for r in task]
        if self.a > 0:
            counts = torch.zeros(self.table.shape[0], dtype=torch.double, device=self.table.device)
            for r in rows:
                ids = torch.as_tensor([i for i in r if i != self.pad], dtype=torch.long, device=self.table.device)
                counts.index_add_(0, ids, torch.ones(len(ids), dtype=torch.double, device=self.table.device))
            self.weight = self.a / (self.a + counts / counts.sum())
            self.weight[self.pad] = 0.0
        self.mu_head, self.w_head = self._whitener(self._mean(rows, self.head))
        self.mu_all, self.w_all = self._whitener(self._mean(rows))
        self.known = torch.stack([self._signature(task[0][:common_prefix(task[:20])]) for task in unrelated_tasks])

    def _signature(self, opening):
        return (self._mean([opening])[0] - self.mu_head) @ self.w_head

    def add_task(self, train_rows, opening_len=None):
        """Make the zone of a new task from its training inputs. Returns how many of them were already inside
        an older zone (those inputs stay with the older task: the price of the guarantee, visible at once)."""
        n = opening_len or common_prefix(train_rows)
        assert n >= 3, "the task's inputs do not start with a common instruction"
        blocked = int(self.route(train_rows)[1].sum()) if self.tasks else 0
        sig = self._signature(train_rows[0][:n])
        known = torch.cat([self.known] + [t["sig"][None] for t in self.tasks])
        centre = ((self._mean(train_rows) - self.mu_all) @ self.w_all).mean(0)
        self.tasks.append({"n": n, "sig": sig, "rho": self.kappa * float((known - sig).norm(dim=1).min()), "centre": centre,
                           "opening": list(train_rows[0][:n])})
        return blocked

    def route(self, rows, rule="certified"):
        """For every row: its route (1-based task number; adapters 1..route are used) and whether it is certified.
        rule: 'certified' (oldest zone, else nearest zone), 'exact_first' (the oldest task whose instruction the
        input carries token for token, else as 'certified'), 'nearest' (nearest zone), 'centre' (nearest centre of
        the whole input), 'all' (the newest task: every adapter on)."""
        T = len(self.tasks)
        route = torch.full((len(rows),), T, dtype=torch.long)
        cert = torch.zeros(len(rows), dtype=torch.bool)
        if rule == "all" or T == 0:
            return route, cert
        if rule == "centre":
            whole = (self._mean(rows) - self.mu_all) @ self.w_all
            return torch.stack([(whole - t["centre"]).norm(dim=1) for t in self.tasks], 1).argmin(1).cpu() + 1, cert
        means = {n: self._mean(rows, n) for n in {t["n"] for t in self.tasks}}
        rel = torch.stack([(((means[t["n"]] - self.mu_head) @ self.w_head) - t["sig"]).norm(dim=1) / t["rho"]
                           for t in self.tasks], 1).cpu()        # 1 or less: inside that zone
        # A zone of radius zero (its signature equals a known one) holds only inputs at distance zero: 0 / 0 counts as
        # inside (0), any other distance as outside (infinity). Ties go to the older task (argmin takes the first).
        rel = torch.where(torch.isnan(rel), torch.zeros_like(rel), rel)
        nearest = rel.argmin(1) + 1
        if rule in ("certified", "exact_first"):
            for k in range(T):                               # oldest first: an older claim is never overruled
                new = (rel[:, k] <= 1.0) & ~cert
                route[new] = k + 1
                cert |= new
        route = torch.where(cert, route, nearest)
        if rule == "exact_first":
            for i, row in enumerate(rows):                   # an exact claim comes before every zone; the oldest exact claim wins
                for k, t in enumerate(self.tasks):
                    if len(row) >= t["n"] and list(row[:t["n"]]) == t["opening"]:
                        route[i], cert[i] = k + 1, True
                        break
        return route, cert


# ----------------------------------------------------------------------------------------------
# 2. One adapter per task, switched on and off per input
# ----------------------------------------------------------------------------------------------
class Router:
    """Shared by all adapted layers of a model: the route of every input of the current batch."""
    route = None                                             # None: every adapter on


class RoutedLoRA(nn.Module):
    """A linear layer plus one low-rank adapter per task: y = W x + (alpha / r) * sum_k on_k(x) B_k A_k x."""

    def __init__(self, base, router, rank=8, alpha=16):
        super().__init__()
        self.base, self.router, self.rank, self.scale = base, router, rank, alpha / rank
        linear = isinstance(base, nn.Linear)                 # the other kind (GPT-2's Conv1D) stores the weight as in x out
        self.n_in = base.in_features if linear else base.weight.shape[0]
        self.n_out = base.out_features if linear else base.weight.shape[1]
        self.A, self.B = nn.ParameterList(), nn.ParameterList()

    def new_task(self):
        for p in list(self.A) + list(self.B):
            p.requires_grad_(False)                          # older adapters are never changed again
        ref = self.base.weight
        a = torch.empty(self.rank, self.n_in, device=ref.device, dtype=ref.dtype)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.A.append(nn.Parameter(a))
        self.B.append(nn.Parameter(torch.zeros(self.n_out, self.rank, device=ref.device, dtype=ref.dtype)))

    def forward(self, x):
        out = self.base(x)
        route = self.router.route
        for k, (a, b) in enumerate(zip(self.A, self.B)):
            delta = (x @ a.T) @ b.T * self.scale
            if route is not None:                            # adapter k+1 is on for inputs whose route is at least k+1
                on = (route >= k + 1).to(delta.dtype).view(-1, *([1] * (delta.dim() - 1)))
                assert on.shape[0] == delta.shape[0], "one route per input of the batch is needed"
                delta = delta * on
            out = out + delta
        return out


ATTENTION_NAMES = ("q", "v", "q_proj", "v_proj", "query", "value", "c_attn")


def add_adapters(model, router, rank=8, alpha=16, names=ATTENTION_NAMES):
    """Freeze the model and wrap its attention projections. Returns the wrapped layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in names and hasattr(child, "weight") and child.weight.dim() == 2 and not isinstance(child, RoutedLoRA):
                layer = RoutedLoRA(child, router, rank, alpha)
                setattr(parent, name, layer)
                wrapped.append(layer)
    assert wrapped, "no attention projection found; pass the layer names of this model in `names`"
    return wrapped


# ----------------------------------------------------------------------------------------------
# 3. A model that learns tasks one after another
# ----------------------------------------------------------------------------------------------
class ContinualModel:
    def __init__(self, model, tokenizer, unrelated_tasks, kappa=0.5, a=1e-3, rank=8, alpha=16, max_len=256, max_new=32, rule="certified"):
        """unrelated_tasks: a few lists of input texts of tasks that will never be learned (they set the scale).
        rule: "certified" or "exact_first" (see the top of this file)."""
        assert rule in ("certified", "exact_first")
        self.model, self.tok, self.max_len, self.max_new, self.rule = model, tokenizer, max_len, max_new, rule
        self.seq2seq = bool(getattr(model.config, "is_encoder_decoder", False))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.router = Router()
        self.layers = add_adapters(model, self.router, rank, alpha)
        self.zones = Zones(model.get_input_embeddings().weight, tokenizer.pad_token_id, kappa, a)
        self.zones.fit_background([self.ids(texts) for texts in unrelated_tasks])
        self.device = next(model.parameters()).device

    def ids(self, texts):
        return self.tok(list(texts), max_length=self.max_len, truncation=True)["input_ids"]

    def _pad(self, rows, left=False):
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            sl = slice(width - len(r), width) if left else slice(0, len(r))
            ids[i, sl] = torch.tensor(r)
            att[i, sl] = 1
        return ids.to(self.device), att.to(self.device)

    def learn(self, inputs, targets, epochs=3, batch=16, lr=1e-3, seed=0, log=None):
        """Learn one new task. Returns the number of training inputs that an older zone had already claimed."""
        rows = self.ids(inputs)
        blocked = self.zones.add_task(rows)
        for layer in self.layers:
            layer.new_task()
        route = self.zones.route(rows, self.rule)[0]         # inputs that stay with an older task do not train the new adapter
        answers = self.tok(list(targets), max_length=self.max_new, truncation=True, add_special_tokens=self.seq2seq)["input_ids"]
        if not self.seq2seq:
            answers = [a + [self.tok.eos_token_id] for a in answers]
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        gen = torch.Generator().manual_seed(seed)
        self.model.train()
        step = 0
        for _ in range(epochs):
            order = torch.randperm(len(rows), generator=gen).tolist()
            for s in range(0, len(order), batch):
                pick = order[s:s + batch]
                self.router.route = route[pick].to(self.device)
                if self.seq2seq:
                    ids, att = self._pad([rows[i] for i in pick])
                    lab, lab_att = self._pad([answers[i] for i in pick])
                    loss = self.model(input_ids=ids, attention_mask=att, labels=lab.masked_fill(lab_att == 0, -100)).loss
                else:
                    ids, att = self._pad([rows[i] + answers[i] for i in pick])
                    lab = ids.masked_fill(att == 0, -100)
                    for j, i in enumerate(pick):
                        lab[j, :len(rows[i])] = -100         # the loss is on the answer only
                    loss = self.model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                step += 1
                if log and step % log == 0:
                    print(f"      step {step}: loss {loss.item():.3f}", flush=True)
        self.router.route = None
        return blocked

    @torch.no_grad()
    def answer(self, inputs, rule=None, route=None, batch=16):
        """Returns the answers, the route of every input and whether it is certified. rule: the model's own rule
        if not given; any rule of Zones.route can be asked for, to compare."""
        rows = self.ids(inputs)
        if route is None:
            route, cert = self.zones.route(rows, rule or self.rule)
        else:
            cert = torch.zeros(len(rows), dtype=torch.bool)
        self.model.eval()
        out = []
        for s in range(0, len(rows), batch):
            part = rows[s:s + batch]
            ids, att = self._pad(part, left=not self.seq2seq)
            self.router.route = route[s:s + batch].to(self.device)
            gen = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=self.max_new, do_sample=False,
                                      num_beams=1, pad_token_id=self.tok.pad_token_id)
            if not self.seq2seq:
                gen = gen[:, ids.shape[1]:]
            out += [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        self.router.route = None
        return out, route, cert
