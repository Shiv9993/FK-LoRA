"""Latent-LoRA (Azghan et al., arXiv:2607.23837) as a re-implementation from the paper's text, in the style of
certified_switch.py: a class with `learn(inputs, targets, epochs, batch, lr, seed)` and `answer(inputs, rule=...)`.
There is no official code (checked, see README.md), so every detail the paper does not give is a choice, listed in README.md.

Parts:
  LatentLoRALayer   a frozen linear layer plus one trainable r x r matrix R_t per task in the frozen SVD subspace
                    of the layer's weight:  W + (alpha / r) * U_r Sigma_r R(x) V_r^T            (paper Eq. 5 and 15)
  GMMRouter         training-free task recognition: mean-pooled frozen embedding table, K-means mixture per task,
                    one covariance shared by all tasks, uniform prior                           (paper Eq. 11 to 14)
  LatentLoRAModel   learn / answer for an encoder-decoder Hugging Face model (T5). No stored examples, no trained router.
  CertifiedOnDevice the project's ContinualModel (certified_switch.py, not edited) with the model on a GPU, the zone
                    computations kept on the CPU in float64, and gradient accumulation. Same training loop as above.

Devices. Every route / zone / router computation is made on the CPU in float64 and only the final per-batch weights are
moved to the model's device. The model is built on the CPU and moved to the device once, by the constructors here.
"""
import gc
import os
import sys

import torch
import torch.nn as nn

try:
    from certified_switch import ATTENTION_NAMES, ContinualModel
except ImportError:                                           # running inside the repository: ../certified_switch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "certified_switch"))
    from certified_switch import ATTENTION_NAMES, ContinualModel


# ----------------------------------------------------------------------------------------------
# 0. What both methods share: the training loop (same for both, so that the comparison is fair)
# ----------------------------------------------------------------------------------------------
class _Shared:
    """Needs self.model, self.tok, self.max_len, self.max_new, self.device, self.seq2seq, self.micro."""

    def _targets(self, targets):
        """Token ids of the answers, as in ContinualModel.learn (encoder-decoder branch)."""
        return self.tok(list(targets), max_length=self.max_new, truncation=True, add_special_tokens=True)["input_ids"]

    def _step(self, pick, rows, answers, set_route, micro, extra):
        """One optimiser step's gradients for the examples `pick`, in pieces of `micro` examples (gradient accumulation).
        The loss of a piece is the mean over its answer tokens (as in ContinualModel.learn), weighted by its share of the batch."""
        total, n = 0.0, len(pick)
        for c in range(0, n, micro):
            part = pick[c:c + micro]
            set_route(part)
            ids, att = self._pad([rows[i] for i in part])                 # created on self.device
            lab, lab_att = self._pad([answers[i] for i in part])
            loss = self.model(input_ids=ids, attention_mask=att, labels=lab.masked_fill(lab_att == 0, -100)).loss
            (loss * (len(part) / n)).backward()
            total += loss.item() * len(part) / n
        extra_value = None
        if extra is not None:                                             # a term that is added once per step (Latent-LoRA's Eq. 8)
            e = extra()
            e.backward()
            extra_value = e.item()
        return total, extra_value

    def _fit(self, rows, answers, set_route, epochs, batch, lr, seed, log=None, extra=None):
        """AdamW, constant learning rate, `batch` examples per optimiser step (split in pieces of self.micro examples; if the GPU
        runs out of memory the pieces are halved and the step is repeated). Returns the mean loss of the last epoch and, if
        `extra` is given, the mean of that term."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        gen = torch.Generator().manual_seed(seed)
        self.model.train()
        step, last, last_extra = 0, [], []
        for epoch in range(epochs):
            order = torch.randperm(len(rows), generator=gen).tolist()
            last, last_extra = [], []
            for s in range(0, len(order), batch):
                pick = order[s:s + batch]
                while True:
                    failed = False
                    try:
                        value, ev = self._step(pick, rows, answers, set_route, self.micro, extra)
                    except torch.cuda.OutOfMemoryError:
                        failed = True
                    if not failed:
                        break
                    opt.zero_grad(set_to_none=True)                       # outside the except block, so that the memory can be freed
                    gc.collect()
                    torch.cuda.empty_cache()
                    if self.micro <= 1:
                        raise RuntimeError("out of GPU memory even with one example at a time")
                    self.micro = max(1, self.micro // 2)
                    print(f"      GPU out of memory: now {self.micro} examples at a time (batch of {batch} unchanged)", flush=True)
                opt.step()
                opt.zero_grad()
                step += 1
                last.append(value)
                if ev is not None:
                    last_extra.append(ev)
                if log and step % log == 0:
                    print(f"      step {step}: loss {value:.3f}" + (f", orthogonal term {ev:.4g}" if ev is not None else ""), flush=True)
        self.model.eval()
        mean = lambda v: sum(v) / len(v) if v else None
        return mean(last), mean(last_extra)


def to_device_once(model, device):
    """The one place where the base model is moved to the device."""
    model.to(device)
    return torch.device(device) if not isinstance(device, torch.device) else device


# ----------------------------------------------------------------------------------------------
# 1. The project's method, on a device
# ----------------------------------------------------------------------------------------------
class CertifiedOnDevice(_Shared, ContinualModel):
    """ContinualModel (certified_switch.py, unchanged) used with the model on a GPU.

    Why a subclass: ContinualModel builds `Zones` from the model's embedding table at construction, so a model that is already
    on a GPU would make the zone computations run on the GPU in float64, and `self.device` is read at construction. Here the
    model is built on the CPU (zones, background and whitening are made on the CPU in float64), then moved once to the device,
    and `self.device` is set afterwards. Adapters made later by `new_task` are created on the device of the weight they wrap.
    `learn` has the same steps as ContinualModel.learn (zone, new adapters, route of the training inputs, AdamW, same
    batches) but goes through the shared loop with gradient accumulation; with micro >= batch the result is identical
    (checked in test_equivalence.py)."""

    def __init__(self, model, tokenizer, unrelated_tasks, device="cpu", micro=16, **kw):
        model = model.cpu()
        super().__init__(model, tokenizer, unrelated_tasks, **kw)
        assert self.seq2seq, "this wrapper is written for encoder-decoder models (T5)"
        self.device = to_device_once(model, device)
        self.micro = micro

    def learn(self, inputs, targets, epochs=3, batch=16, lr=1e-3, seed=0, log=None):
        rows = self.ids(inputs)
        blocked = self.zones.add_task(rows)                              # CPU, float64
        for layer in self.layers:
            layer.new_task()                                             # on the weight's device
        route = self.zones.route(rows, self.rule)[0]                     # CPU
        answers = self._targets(targets)

        def set_route(part):
            self.router.route = route[part].to(self.device)              # CPU -> device (long tensor, one per input)

        self.info = dict(zip(("loss", "extra"), self._fit(rows, answers, set_route, epochs, batch, lr, seed, log)))
        self.router.route = None
        return blocked


# ----------------------------------------------------------------------------------------------
# 2. Latent-LoRA
# ----------------------------------------------------------------------------------------------
class LatentRouter:
    """Shared by all adapted layers: the weight of every learned task's adapter for each input of the current batch."""
    p = None                                                              # (batch, tasks) on the model's device; None: no adapter


class LatentLoRALayer(nn.Module):
    """A frozen linear layer W (m x n) plus one trainable r x r matrix R_t per task (paper Sec. 3.2, LoRA-XS):
        W ~ U_r diag(S_r) V_r^T  (rank-r truncated SVD, computed once, frozen)
        y = W x + (alpha / r) * U_r diag(S_r) R(x) V_r^T x,      R(x) = sum_t p_t(x) R_t          (Eq. 5, 15)"""

    def __init__(self, base, router, rank=32, alpha=16):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base, self.router, self.rank, self.scale = base, router, rank, alpha / rank
        W = base.weight.detach().cpu().double()                           # the SVD is made once, on the CPU, in float64
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        dt = base.weight.dtype
        self.register_buffer("U", U[:, :rank].to(dt).contiguous())        # m x r   frozen
        self.register_buffer("S", S[:rank].to(dt).contiguous())           # r       frozen
        self.register_buffer("V", Vh[:rank].T.to(dt).contiguous())        # n x r   frozen
        self.R = nn.ParameterList()

    def new_task(self):
        for p in self.R:
            p.requires_grad_(False)                                       # older adapters are snapshots, never changed again
        ref = self.base.weight
        self.R.append(nn.Parameter(torch.zeros(self.rank, self.rank, device=ref.device, dtype=ref.dtype)))

    def ortho(self):
        """Sum over older tasks i of || (S R_i)^T (S R_t) ||_F^2 for the newest task t (paper Eq. 8 and 10)."""
        if len(self.R) < 2:
            return self.S.sum() * 0.0
        new = self.S[:, None] * self.R[-1]
        return sum(((self.S[:, None] * old).T @ new).pow(2).sum() for old in list(self.R)[:-1])

    def forward(self, x):
        out = self.base(x)
        p = self.router.p
        if p is None or len(self.R) == 0:
            return out
        assert p.shape == (x.shape[0], len(self.R)), "one weight per learned task and per input of the batch is needed"
        Rs = torch.stack(list(self.R))                                    # (T, r, r)
        Rx = torch.einsum("bt,tij->bij", p, Rs)                           # R(x), one r x r matrix per input
        z = torch.einsum("blj,bij->bli", x @ self.V, Rx) * self.S         # Sigma R(x) V^T x
        return out + (z @ self.U.T) * self.scale


def add_latent_adapters(model, router, rank=32, alpha=16, names=ATTENTION_NAMES):
    """Freeze the model and wrap its query and value projections (all attention blocks of an encoder-decoder model)."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, nn.Linear) and not isinstance(child, LatentLoRALayer):
                layer = LatentLoRALayer(child, router, rank, alpha)
                setattr(parent, name, layer)
                wrapped.append(layer)
    assert wrapped, "no attention projection found"
    return wrapped


def _kmeans(x, k, iters, gen):
    """K-means with k-means++ seeding and `iters` Lloyd steps. Returns the centres and the share of points of every
    non-empty cluster."""
    n = x.shape[0]
    k = min(k, n)
    first = int(torch.randint(n, (1,), generator=gen))
    idx, d2 = [first], ((x - x[first]) ** 2).sum(1)
    for _ in range(1, k):
        prob = d2 / d2.sum() if float(d2.sum()) > 0 else torch.full_like(d2, 1.0 / n)
        j = int(torch.multinomial(prob, 1, generator=gen))
        idx.append(j)
        d2 = torch.minimum(d2, ((x - x[j]) ** 2).sum(1))
    mu = x[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(x, mu).argmin(1)
        for c in range(k):
            m = assign == c
            if m.any():
                mu[c] = x[m].mean(0)
    counts = torch.bincount(torch.cdist(x, mu).argmin(1), minlength=k).double()
    keep = counts > 0
    return mu[keep], counts[keep] / n


class GMMRouter:
    """Training-free router (paper Sec. 3.3). Everything is on the CPU in float64.
    phi(x) = mean of the frozen input-embedding vectors of the tokens of x (Eq. 11); per task K Gaussians with K-means
    means and weights (Eq. 12), one covariance C for all tasks and components (Eq. 13), posterior with a uniform prior (Eq. 14)."""

    def __init__(self, table, pad_id, k=5, eps=0.01, iters=30):
        self.table = table.detach().cpu().double()                        # a copy: later changes of the model cannot reach it
        self.pad, self.k, self.eps, self.iters = pad_id, k, eps, iters
        self.scatter = None                                               # sum over tasks of S_t (Eq. 13 numerator)
        self.count = 0                                                    # sum over tasks of n_t
        self.tasks = []                                                   # per task: component means (K, d) and log weights (K,)
        self.chol = None

    def phi(self, rows):
        out = []
        for r in rows:
            ids = torch.as_tensor([i for i in r if i != self.pad], dtype=torch.long)
            out.append(self.table[ids].mean(0))
        return torch.stack(out)

    def add_task(self, rows, seed=0):
        """Fit the mixture of a new task on its training inputs and update the shared covariance. Old data are not needed."""
        x = self.phi(rows)
        mu, pi = _kmeans(x, self.k, self.iters, torch.Generator().manual_seed(seed))
        centred = x - x.mean(0)
        s = centred.T @ centred
        self.scatter = s if self.scatter is None else self.scatter + s
        self.count += x.shape[0]
        d = x.shape[1]
        C = self.scatter / self.count + self.eps * torch.eye(d, dtype=x.dtype)
        self.chol = torch.linalg.cholesky(C)
        self.tasks.append((mu, pi.log()))

    def posterior(self, rows):
        """(inputs, tasks) matrix of p(t | x)."""
        x = self.phi(rows)
        L = self.chol
        zx = torch.linalg.solve_triangular(L, x.T, upper=False).T         # whitened inputs, so that Mahalanobis = Euclidean
        logp = []
        for mu, logpi in self.tasks:
            zm = torch.linalg.solve_triangular(L, mu.T, upper=False).T
            d2 = torch.cdist(zx, zm) ** 2                                 # (inputs, K)
            logp.append(torch.logsumexp(logpi[None] - 0.5 * d2, 1))       # constants shared by all tasks cancel in the posterior
        return torch.softmax(torch.stack(logp, 1), 1)


class LatentLoRAModel(_Shared):
    ids = ContinualModel.ids                                              # same tokenisation and padding as the project's method
    _pad = ContinualModel._pad

    def __init__(self, model, tokenizer, rank=32, alpha=16, lam=0.05, k=5, eps=0.01, max_len=256, max_new=32,
                 device="cpu", micro=16, ortho_reduce="sum"):
        """rank, alpha, lam, k, eps: see README.md. ortho_reduce: how the Eq. 8 terms of the many adapted layers are combined."""
        assert ortho_reduce in ("sum", "mean")
        model = model.cpu()
        self.model, self.tok, self.max_len, self.max_new = model, tokenizer, max_len, max_new
        self.seq2seq = bool(getattr(model.config, "is_encoder_decoder", False))
        assert self.seq2seq, "this implementation is written for encoder-decoder models (T5)"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.lam, self.ortho_reduce, self.micro = lam, ortho_reduce, micro
        self.router = LatentRouter()
        self.layers = add_latent_adapters(model, self.router, rank, alpha)
        self.gmm = GMMRouter(model.get_input_embeddings().weight, tokenizer.pad_token_id, k, eps)
        self.device = to_device_once(model, device)                       # the buffers U, S, V move with the model
        self.dtype = next(model.parameters()).dtype

    def _ortho(self):
        terms = torch.stack([layer.ortho() for layer in self.layers])
        return terms.sum() if self.ortho_reduce == "sum" else terms.mean()

    def learn(self, inputs, targets, epochs=3, batch=16, lr=1e-3, seed=0, log=None):
        """Learn one new task: train only R_t (all layers), loss = language-model loss + lam * Eq. 8 (paper Eq. 10); then fit the
        router on the task's training inputs. Returns 0 (no input is ever withheld from a task); self.info has the losses."""
        rows = self.ids(inputs)
        for layer in self.layers:
            layer.new_task()
        T = len(self.layers[0].R)
        answers = self._targets(targets)

        def set_route(part):                                              # while training, only the new adapter is on
            p = torch.zeros(len(part), T, dtype=self.dtype)
            p[:, T - 1] = 1.0
            self.router.p = p.to(self.device)                             # CPU -> device

        extra = (lambda: self.lam * self._ortho()) if T > 1 and self.lam > 0 else None
        self.info = dict(zip(("loss", "extra"), self._fit(rows, answers, set_route, epochs, batch, lr, seed, log, extra)))
        self.router.p = None
        self.gmm.add_task(rows, seed)                                     # after the adapter is trained, on the CPU
        return 0

    @torch.no_grad()
    def answer(self, inputs, rule="soft", route=None, batch=16):
        """Returns the answers, the most probable task of every input (1-based) and zeros (nothing is certified).
        rule: 'soft' (the paper: adapters blended with the posterior, Eq. 15), 'hard' (only the most probable task's
        adapter), 'sum' (all adapters added, the paper's no-router ablation: R(x) = sum_t R_t)."""
        assert rule in ("soft", "hard", "sum")
        rows = self.ids(inputs)
        post = self.gmm.posterior(rows)                                   # CPU, float64
        guess = post.argmax(1)
        if route is not None:                                             # a given route (1-based) is used as a hard choice
            guess = torch.as_tensor(route).long() - 1
            weights = torch.nn.functional.one_hot(guess, post.shape[1]).double()
        elif rule == "soft":
            weights = post
        elif rule == "hard":
            weights = torch.nn.functional.one_hot(guess, post.shape[1]).double()
        else:
            weights = torch.ones_like(post)
        self.model.eval()
        out = []
        for s in range(0, len(rows), batch):
            ids, att = self._pad(rows[s:s + batch])                       # on self.device
            self.router.p = weights[s:s + batch].to(device=self.device, dtype=self.dtype)   # CPU float64 -> device
            gen = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=self.max_new, do_sample=False,
                                      num_beams=1, pad_token_id=self.tok.pad_token_id)
            out += [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        self.router.p = None
        return out, guess + 1, torch.zeros(len(rows), dtype=torch.bool)
