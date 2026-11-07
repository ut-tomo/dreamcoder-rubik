"""Sleep / abstraction phase: grow the library by compressing solved programs.

A simplified, pure-Python stand-in for the OCaml compressor (compression.ml):
candidate inventions are (a) closed subterms and (b) subterms with exactly one
hole (a one-argument fragment), mined from the frontiers. Candidates are scored
by the MDL objective

    score(L) = sum_tasks max_{p in frontier} log P(p | L)  -  lambda * sum_{inv in L} size(inv)

and added greedily while the score improves. The full version-space refactoring
of the paper is not implemented (see README: "Differences from the paper").
"""
from __future__ import annotations

import math
from collections import defaultdict

from .grammar import Grammar
from .program import Abstraction, Application, Index, Invented, Program, ShiftFailure


class Hole(Program):
    def show(self, paren):
        return "??"

    def free_indices(self, depth=0):
        return set()

    def size(self):
        return 0

    def subterms(self, depth=0):
        yield depth, self


# ------------------------------------------------------------------ mining
def _holed_versions(t, depth=0):
    """All copies of t with one strict sub-term replaced by a Hole.
    Yields (holed_term, binder_depth_at_hole)."""
    if isinstance(t, Application):
        yield Application(Hole(), t.x), depth
        yield Application(t.f, Hole()), depth
        for h, d in _holed_versions(t.f, depth):
            yield Application(h, t.x), d
        for h, d in _holed_versions(t.x, depth):
            yield Application(t.f, h), d
    elif isinstance(t, Abstraction):
        yield Abstraction(Hole()), depth + 1
        for h, d in _holed_versions(t.body, depth + 1):
            yield Abstraction(h), d


def _replace_hole(t, depth_at_hole):
    if isinstance(t, Hole):
        return Index(depth_at_hole)
    if isinstance(t, Application):
        return Application(_replace_hole(t.f, depth_at_hole), _replace_hole(t.x, depth_at_hole))
    if isinstance(t, Abstraction):
        return Abstraction(_replace_hole(t.body, depth_at_hole))
    return t


def _n_prims(t):
    return sum(1 for _, s in t.subterms() if not isinstance(s, (Application, Abstraction, Hole, Index)))


def mine_candidates(programs_by_task, max_size=8, min_tasks=2):
    seen = defaultdict(set)  # fragment string -> tasks
    frag = {}
    for task, progs in programs_by_task.items():
        for p in progs:
            for _, s in p.subterms():
                if isinstance(s, (Index,)) or not isinstance(s, (Application, Abstraction)):
                    continue
                if s.size() > max_size + 1:
                    continue
                if not s.free_indices():
                    if _n_prims(s) >= 2:
                        k = str(s)
                        frag[k] = (s, None)
                        seen[k].add(task)
                for h, d in _holed_versions(s):
                    if h.free_indices() or _n_prims(h) < 2 or h.size() > max_size:
                        continue
                    if isinstance(h, Application) and isinstance(h.f, Hole):
                        continue  # higher-order hole in head position: skip
                    k = str(h)
                    frag[k] = (h, d)
                    seen[k].add(task)
    out = []
    for k, ts in seen.items():
        if len(ts) >= min_tasks:
            f, d = frag[k]
            out.append((len(ts) * _n_prims(f), f, d))
    out.sort(key=lambda x: -x[0])
    return out


def make_invented(fragment, hole_depth):
    if hole_depth is None:
        return Invented(fragment)
    return Invented(Abstraction(_replace_hole(fragment, hole_depth)))


# --------------------------------------------------------------- rewriting
def _match(f, t, k, binding):
    """Match fragment f against term t; k = binder depth inside fragment."""
    if isinstance(f, Hole):
        if any(i < k for i in t.free_indices()):
            return False
        try:
            binding.append(t.shift(-k) if k else t)
        except ShiftFailure:
            return False
        return True
    if type(f) is not type(t):
        return False
    if isinstance(f, Application):
        return _match(f.f, t.f, k, binding) and _match(f.x, t.x, k, binding)
    if isinstance(f, Abstraction):
        return _match(f.body, t.body, k + 1, binding)
    if isinstance(f, Index):
        return f.i == t.i
    return str(f) == str(t)


def rewrite(p, fragment, hole_depth, inv):
    b = []
    if _match(fragment, p, 0, b):
        if hole_depth is None:
            return inv
        return Application(inv, rewrite(b[0], fragment, hole_depth, inv))
    if isinstance(p, Application):
        return Application(rewrite(p.f, fragment, hole_depth, inv), rewrite(p.x, fragment, hole_depth, inv))
    if isinstance(p, Abstraction):
        return Abstraction(rewrite(p.body, fragment, hole_depth, inv))
    return p


# ------------------------------------------------------------ fit & score
def fit_grammar(g, frontiers, pseudo=1.0):
    """frontiers: list of (request, [programs]). Uses/possible estimator (as in EC)."""
    uses, possible = defaultdict(float), defaultdict(float)
    for req, progs in frontiers:
        lls = []
        evs = []
        for p in progs:
            l, u = g.log_likelihood(req, p)
            lls.append(l)
            evs.append(u.get("__events__", []))
        if not lls or max(lls) == -math.inf:
            continue
        m = max(lls)
        z = m + math.log(sum(math.exp(l - m) for l in lls))
        for l, ev in zip(lls, evs):
            w = math.exp(l - z)
            for chosen, cset in ev:
                uses[chosen] += w
                for c in cset:
                    possible[c] += w
    weights = {}
    for _, _, p in g.productions:
        k = str(p)
        weights[k] = math.log((uses[k] + pseudo) / (possible[k] + pseudo))
    lv = math.log((uses["$_"] + pseudo) / (possible["$_"] + pseudo))
    return g.with_weights(weights, lv)


def score(g, frontiers, structure_penalty):
    total = 0.0
    for req, progs in frontiers:
        best = max(g.log_likelihood(req, p)[0] for p in progs)
        if best == -math.inf:
            return -math.inf
        total += best
    lib = sum(p.body.size() for _, _, p in g.productions if isinstance(p, Invented))
    return total - structure_penalty * lib


def compress(g, frontiers_by_task, structure_penalty=1.0, max_new=3, top_candidates=40, log=print):
    """frontiers_by_task: {task_name: (request, [programs])}. Returns (grammar, rewritten frontiers)."""
    fr = dict(frontiers_by_task)
    g = fit_grammar(g, list(fr.values()))
    current = score(g, list(fr.values()), structure_penalty)
    log(f"  [compress] start score {current:.2f} on {len(fr)} tasks")
    for _ in range(max_new):
        cands = mine_candidates({k: v[1] for k, v in fr.items()})[:top_candidates]
        best = None
        for _, f, d in cands:
            try:
                inv = make_invented(f, d)
            except Exception:
                continue
            new_fr = {k: (r, [rewrite(p, f, d, inv) for p in ps]) for k, (r, ps) in fr.items()}
            g2 = fit_grammar(g.add(inv), list(new_fr.values()))
            s = score(g2, list(new_fr.values()), structure_penalty)
            if best is None or s > best[0]:
                best = (s, inv, g2, new_fr)
        if best is None or best[0] <= current:
            break
        current, inv, g, fr = best
        log(f"  [compress] + {inv}  :: {inv.tp}   score {current:.2f}")
    return g, fr
