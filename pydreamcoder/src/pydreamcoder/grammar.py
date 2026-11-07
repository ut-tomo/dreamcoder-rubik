"""Unigram probabilistic grammar over typed programs + top-down enumeration.

Mirrors dreamcoder/grammar.py and dreamcoder/enumeration.py: candidates whose
return type unifies with the request are normalised, programs are enumerated in
decreasing prior probability via iteratively widened description-length windows.
"""
from __future__ import annotations

import math
import time

from .program import Abstraction, Application, Index, Invented
from .types import Context, TCon, TVar, UnificationFailure, canonical


def _canon_with_map(t):
    m, back = {}, []

    def go(t):
        if isinstance(t, TVar):
            if t.i not in m:
                m[t.i] = TVar(len(m))
                back.append(t)
            return m[t.i]
        return TCon(t.name, [go(a) for a in t.args]) if t.args else t

    return go(t), back


def logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Grammar:
    def __init__(self, log_variable, productions):
        # productions: list of (logweight, type, program)
        self.log_variable = log_variable
        self.productions = productions

    @staticmethod
    def uniform(primitives):
        return Grammar(0.0, [(0.0, p.tp if not isinstance(p, Invented) else p.tp, p) for p in primitives])

    @property
    def primitives(self):
        return [p for _, _, p in self.productions]

    def with_weights(self, weights, log_variable):
        return Grammar(log_variable, [(weights.get(str(p), 0.0), t, p) for _, t, p in self.productions])

    def add(self, invented):
        return Grammar(self.log_variable, self.productions + [(0.0, invented.tp, invented)])

    def __str__(self):
        lines = [f"{self.log_variable:7.3f}\t$_"]
        for l, t, p in sorted(self.productions, key=lambda x: -x[0]):
            lines.append(f"{l:7.3f}\t{t}\t{p}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- candidates
    def candidates(self, ctx, env, request, upper=float("inf")):
        # Fast path: ground request and environment -> cache (types stored canonically,
        # re-instantiated with fresh variables in the caller's context).
        # General path: canonicalise (request, env) jointly; cache results expressed
        # over the canonical variables and replay them into the caller's context.
        request = ctx.apply(request)
        envt = [ctx.apply(e) for e in env]
        tup = TCon("*", [request] + envt)
        ckey, back = _canon_with_map(tup)
        cache = self.__dict__.setdefault("_cache", {})
        hit = cache.get(ckey)
        nv = len(back)
        if hit is None:
            c0 = Context(nv)
            hit = []
            for l, t, p, c2 in self._candidates(c0, list(ckey.args[1:]), ckey.args[0]):
                packed = TCon("*", [c2.apply(t)] + [c2.apply(TVar(i)) for i in range(nv)])
                hit.append((l, packed, p))
            cache[ckey] = hit
        if nv == 0:
            out = []
            for l, packed, p in hit:
                if -l >= upper:
                    break
                if packed.free():
                    c2, pk = ctx.instantiate(packed)
                else:
                    c2, pk = ctx, packed
                out.append((l, pk.args[0], p, c2))
            return out
        out = []
        for l, packed, p in hit:
            if -l >= upper:
                break
            c2, pk = ctx.instantiate(packed)
            try:
                for i in range(nv):
                    c2 = c2.unify(pk.args[1 + i], back[i])
            except UnificationFailure:
                continue
            out.append((l, c2.apply(pk.args[0]), p, c2))
        return out

    def _candidates(self, ctx, env, request):
        out = []
        for l, t, p in self.productions:
            try:
                c2, tt = ctx.instantiate(t)
                c2 = c2.unify(tt.returns(), request)
                out.append((l, c2.apply(tt), p, c2))
            except UnificationFailure:
                pass
        vs = []
        for i, vt in enumerate(env):
            try:
                c2 = ctx.unify(vt.returns(), request)
                vs.append((c2.apply(vt), Index(i), c2))
            except UnificationFailure:
                pass
        if vs:
            lv = self.log_variable - math.log(len(vs))
            out += [(lv, t, p, c) for t, p, c in vs]
        if not out:
            return []
        z = logsumexp([o[0] for o in out])
        out = [(l - z, t, p, c) for l, t, p, c in out]
        out.sort(key=lambda o: -o[0])
        return out

    # --------------------------------------------------------------- enumeration
    def enumerate(self, ctx, env, request, upper, lower, depth):
        """Yield (logprior, ctx, program) with lower <= -logprior < upper.

        If self.deadline (unix time) is set, raises EnumerationTimeout once it passes,
        even when no program is being yielded (deep, fully-pruned subtrees)."""
        dl = getattr(self, "deadline", None)
        if dl is not None and time.time() > dl:
            raise EnumerationTimeout()
        if upper <= 0 or depth < 0:
            return
        request = ctx.apply(request)
        if request.is_arrow():
            a, r = request.args
            for l, c2, body in self.enumerate(ctx, [a] + env, r, upper, lower, depth):
                yield l, c2, Abstraction(body)
            return
        for l, t, p, c2 in self.candidates(ctx, env, request, upper):
            mdl = -l
            if mdl >= upper:
                continue
            for al, c3, prog in self._enum_app(c2, env, p, t.arguments(), upper - mdl, lower - mdl, depth):
                yield l + al, c3, prog

    def _enum_app(self, ctx, env, f, arg_types, upper, lower, depth):
        if not arg_types:
            if lower <= 0 < upper:
                yield 0.0, ctx, f
            return
        first, rest = arg_types[0], arg_types[1:]
        for al, c2, a in self.enumerate(ctx, env, ctx.apply(first), upper, 0.0, depth - 1):
            for rl, c3, prog in self._enum_app(c2, env, Application(f, a), rest, upper + al, lower + al, depth):
                yield al + rl, c3, prog

    # ---------------------------------------------------------------- likelihood
    def log_likelihood(self, request, program):
        """Exact log prior of program under this grammar (summaries for refit)."""
        uses = {}
        try:
            _, l = self._ll(_Ctx(), [], request, program, uses)
        except UnificationFailure:
            return -math.inf, uses
        return l, uses

    def _ll(self, ctxw, env, request, p, uses):
        ctx = ctxw.c
        request = ctx.apply(request)
        if request.is_arrow():
            if not isinstance(p, Abstraction):
                raise UnificationFailure()
            a, r = request.args
            return self._ll(ctxw, [a] + env, r, p.body, uses)
        f, args = p.apply_spine()
        cands = self.candidates(ctx, env, request)
        for l, t, cp, c2 in cands:
            if (isinstance(f, Index) and isinstance(cp, Index) and cp.i == f.i) or (
                    not isinstance(f, Index) and cp is f) or (
                    not isinstance(f, Index) and not isinstance(cp, Index) and str(cp) == str(f)):
                ta = t.arguments()
                if len(ta) != len(args):
                    raise UnificationFailure()
                key = "$_" if isinstance(f, Index) else str(f)
                # record the competing set for refitting (inside-outside, approx.)
                uses.setdefault("__events__", []).append((key, tuple(sorted({"$_" if isinstance(x[2], Index) else str(x[2]) for x in cands}))))
                ctxw.c = c2
                total = l
                for at, a in zip(ta, args):
                    _, al = self._ll(ctxw, env, at, a, uses)
                    total += al
                return ctxw, total
        raise UnificationFailure()


class EnumerationTimeout(Exception):
    pass


class _Ctx:
    def __init__(self):
        from .types import Context
        self.c = Context()
