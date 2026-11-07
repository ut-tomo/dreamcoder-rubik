"""Recognition model + dreaming (the paper's "sleep: dreaming" phase).

A small MLP maps task features to the log-weights of a *task-specific* unigram
grammar over the current library (the paper's unigram recognition model; the
paper's best model is the bigram/contextual one, not implemented here).

Training data each iteration (as in DreamCoder):
  * replays   - solved tasks with their frontier programs (weighted by posterior),
  * fantasies - programs sampled from the current generative grammar, turned into
                synthetic tasks by a domain hook  dream(program, rng) -> features | None.
Loss: -log P(program | grammar weights predicted from the task's features), computed
exactly from the grammar's choice events (chosen production vs. the competing set
of type-compatible productions at each node).
"""
from __future__ import annotations

import math
import random

from .program import Abstraction, Application
from .types import Context, UnificationFailure


# ------------------------------------------------------------------ sampling
def sample_program(g, request, rng, max_depth=8, tries=50):
    """Draw a program from the generative grammar g (top-down, depth-limited)."""
    for _ in range(tries):
        try:
            _, p = _sample(g, Context(), [], request, rng, max_depth)
            return p
        except _Fail:
            continue
    return None


class _Fail(Exception):
    pass


def _sample(g, ctx, env, request, rng, depth):
    if depth < 0:
        raise _Fail()
    request = ctx.apply(request)
    if request.is_arrow():
        a, r = request.args
        ctx, body = _sample(g, ctx, [a] + env, r, rng, depth)
        return ctx, Abstraction(body)
    cands = g.candidates(ctx, env, request)
    if not cands:
        raise _Fail()
    if depth == 0:  # only leaves allowed
        cands = [c for c in cands if not c[1].arguments()]
        if not cands:
            raise _Fail()
    z = sum(math.exp(c[0]) for c in cands)
    r = rng.random() * z
    acc = 0.0
    pick = cands[-1]
    for c in cands:
        acc += math.exp(c[0])
        if r <= acc:
            pick = c
            break
    _, t, p, ctx = pick
    for at in t.arguments():
        ctx, a = _sample(g, ctx, env, at, rng, depth - 1)
        p = Application(p, a)
    return ctx, p


# ------------------------------------------------------------------- model
def vocab_of(g):
    return [str(p) for _, _, p in g.productions] + ["$_"]


def events_of(g, request, program, index):
    """List of (chosen_idx, [competitor idxs]) or None if program not in g's support."""
    l, u = g.log_likelihood(request, program)
    if l == -math.inf:
        return None
    out = []
    for chosen, cset in u.get("__events__", []):
        if chosen not in index:
            return None
        out.append((index[chosen], [index[c] for c in cset if c in index]))
    return out


class Recognition:
    def __init__(self, n_features, hidden=128, device=None, seed=0):
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features, self.hidden = n_features, hidden
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU()).to(self.device)
        self.head = None
        self.vocab = None

    def _reset_head(self, vocab):
        """Library changed: new output layer (encoder is kept across iterations)."""
        torch = self.torch
        old = (self.head, self.vocab)
        self.head = torch.nn.Linear(self.hidden, len(vocab)).to(self.device)
        if old[0] is not None:  # carry over rows of productions that still exist
            with torch.no_grad():
                for i, k in enumerate(vocab):
                    if k in old[1]:
                        j = old[1].index(k)
                        self.head.weight[i] = old[0].weight[j]
                        self.head.bias[i] = old[0].bias[j]
        self.vocab = vocab

    def train(self, g, examples, steps=600, batch=64, lr=1e-3, log=print):
        """examples: list of (features, request, program, weight)."""
        torch = self.torch
        vocab = vocab_of(g)
        if vocab != self.vocab:
            self._reset_head(vocab)
        index = {k: i for i, k in enumerate(vocab)}
        data = []
        for f, req, p, w in examples:
            ev = events_of(g, req, p, index)
            if ev:
                data.append((f, ev, w))
        if not data:
            return
        V = len(vocab)
        # pack events: for each example a list of (chosen, mask) rows
        X = torch.tensor([d[0] for d in data], dtype=torch.float32, device=self.device)
        W = torch.tensor([d[2] for d in data], dtype=torch.float32, device=self.device)
        rows_ex, rows_ch, rows_mask = [], [], []
        for i, (_, ev, _) in enumerate(data):
            for c, cs in ev:
                rows_ex.append(i)
                rows_ch.append(c)
                m = [False] * V
                for j in cs:
                    m[j] = True
                rows_mask.append(m)
        E = torch.tensor(rows_ex, device=self.device)
        C = torch.tensor(rows_ch, device=self.device)
        M = torch.tensor(rows_mask, device=self.device)
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = len(data)
        ex_rows = [[] for _ in range(n)]
        for r, i in enumerate(rows_ex):
            ex_rows[i].append(r)
        rng = random.Random(0)
        for step in range(steps):
            idx = rng.sample(range(n), min(batch, n))
            rr = torch.tensor([r for i in idx for r in ex_rows[i]], device=self.device)
            logits = self.head(self.encoder(X))  # small: recompute all, index rows
            lr_ = logits[E[rr]]
            masked = lr_.masked_fill(~M[rr], -1e9)
            ll = lr_.gather(1, C[rr][:, None])[:, 0] - torch.logsumexp(masked, 1)
            loss = -(ll * W[E[rr]]).sum() / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = self.head(self.encoder(X))
            masked = logits[E].masked_fill(~M, -1e9)
            ll = logits[E].gather(1, C[:, None])[:, 0] - torch.logsumexp(masked, 1)
            tot = torch.zeros(n, device=self.device).index_add_(0, E, ll)
        log(f"  [recognition] trained on {n} examples ({len(examples)} given), "
            f"mean -log P(program|task) = {(-tot).mean().item():.2f} nats")

    def grammar_for(self, g, features):
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            w = self.head(self.encoder(x))[0].tolist()
        weights = {k: w[i] for i, k in enumerate(self.vocab[:-1])}
        return g.with_weights(weights, w[-1])


def dream_examples(g, request, dream, n, rng, max_depth=8):
    """Fantasies: sample programs from g; the domain hook builds a task (features) they solve."""
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        p = sample_program(g, request, rng, max_depth)
        if p is None:
            continue
        try:
            f = dream(p, rng)
        except Exception:
            f = None
        if f is not None:
            out.append((f, request, p, 1.0))
    return out
