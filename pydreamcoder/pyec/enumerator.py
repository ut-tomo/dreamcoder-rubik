"""Faithful Python port of the official DreamCoder OCaml solver's enumerator
(ellisk42/ec @ cb0e63f: solvers/enumeration.ml `enumerate_programs` / `multicore_enumeration`,
solvers/grammar.ml `unifying_expressions`, solvers/task.ml `enumerate_for_tasks`).

Semantics reproduced exactly:
  * programs are enumerated depth-first in the band  lowerBound <= -log P(p) < upperBound,
    with the same candidate order (variables first, then library sorted stably by arity),
    the same normalisation (lse over unifying candidates; variables share logVariable - log #vars),
    continuation-type pruning, contextual (bigram) grammars (noParent / variableParent / per-argument),
    and no symmetry breaking (violates_symmetry is disabled in the OCaml solver);
  * multicore: best-first expansion into a fringe of shatter = 10*nc partial skeletons, which are
    then completed depth-first in parallel worker processes (fork), exactly as multicore_enumeration;
  * per task, the best maximumFrontier hits by log prior + log likelihood are kept.
Speed: types are tuples/ints with dict substitutions, production types are pre-canonicalised so
instantiation is an integer shift, and candidate lists are memoised per (grammar, request, environment)
when no type variables are involved (the common case), with an exact fallback otherwise.
"""
import math
import os
import time
import heapq

from dreamcoder.program import Application, Abstraction, Index, Primitive, Invented
from dreamcoder.type import TypeConstructor, TypeVariable

NEG_INF = float('-inf')


# ------------------------------------------------------------------ types
# type variable: int ; constructor: (name, arg1, ..., argn) ; arrow: ('->', a, b)

def from_official(t):
    if isinstance(t, TypeVariable):
        return t.v
    return (t.name,) + tuple(from_official(a) for a in t.arguments)


def canonical(t, mapping=None):
    if mapping is None:
        mapping = {}
    if isinstance(t, int):
        if t not in mapping:
            mapping[t] = len(mapping)
        return mapping[t]
    return (t[0],) + tuple(canonical(a, mapping) for a in t[1:])


def shift_type(t, n):
    if isinstance(t, int):
        return t + n
    if len(t) == 1:
        return t
    return (t[0],) + tuple(shift_type(a, n) for a in t[1:])


def is_arrow(t):
    return not isinstance(t, int) and t[0] == '->'


def return_of(t):
    while not isinstance(t, int) and t[0] == '->':
        t = t[2]
    return t


def arguments_of(t):
    out = []
    while not isinstance(t, int) and t[0] == '->':
        out.append(t[1])
        t = t[2]
    return out


def has_vars(t):
    if isinstance(t, int):
        return True
    for a in t[1:]:
        if has_vars(a):
            return True
    return False


def apply_sub(t, sub):
    if isinstance(t, int):
        b = sub.get(t)
        if b is None:
            return t
        b2 = apply_sub(b, sub)
        return b2
    if len(t) == 1:
        return t
    return (t[0],) + tuple(apply_sub(a, sub) for a in t[1:])


def occurs(v, t, sub):
    if isinstance(t, int):
        if t == v:
            return True
        b = sub.get(t)
        return b is not None and occurs(v, b, sub)
    for a in t[1:]:
        if occurs(v, a, sub):
            return True
    return False


class UFail(Exception):
    pass


def _unify(sub, a, b):
    # sub is mutated (caller passes a private copy)
    while isinstance(a, int) and a in sub:
        a = sub[a]
    while isinstance(b, int) and b in sub:
        b = sub[b]
    if isinstance(a, int):
        if a == b:
            return
        if occurs(a, b, sub):
            raise UFail()
        sub[a] = b
        return
    if isinstance(b, int):
        if occurs(b, a, sub):
            raise UFail()
        sub[b] = a
        return
    if a[0] != b[0] or len(a) != len(b):
        raise UFail()
    for x, y in zip(a[1:], b[1:]):
        _unify(sub, x, y)


class Ctx(object):
    __slots__ = ('next', 'sub')

    def __init__(self, nxt=0, sub=None):
        self.next = nxt
        self.sub = sub if sub is not None else {}


EMPTY = Ctx()


def might_unify(a, b):
    # OCaml might_unify: false only when both are constructors with different names/arity
    if isinstance(a, int) or isinstance(b, int):
        return True
    if a[0] != b[0] or len(a) != len(b):
        return False
    for x, y in zip(a[1:], b[1:]):
        if not might_unify(x, y):
            return False
    return True


# ------------------------------------------------------------------ grammars

class Production(object):
    __slots__ = ('program', 'tp', 'nvars', 'ret', 'll', 'arity')

    def __init__(self, program, tp, ll):
        self.program = program
        self.tp = canonical(tp)
        self.nvars = len(_vars(self.tp))
        self.ret = return_of(self.tp)
        self.ll = ll
        self.arity = len(arguments_of(self.tp))


def _vars(t, acc=None):
    if acc is None:
        acc = set()
    if isinstance(t, int):
        acc.add(t)
    else:
        for a in t[1:]:
            _vars(a, acc)
    return acc


class G(object):
    """OCaml `grammar` after strip_recursion (library stably sorted by arity)."""

    def __init__(self, logVariable, productions, continuation_type=None):
        self.logVariable = logVariable
        prods = [Production(p, from_official(t), l) for l, t, p in productions
                 if not (p.isPrimitive and p.name in ("fix1", "fix2"))]
        prods.sort(key=lambda pr: pr.arity)  # python sort is stable, like Core.List.sort
        self.library = prods
        self.continuation_type = None if continuation_type is None else from_official(continuation_type)
        self.cache = {}


def _official_grammar_to_G(g):
    return G(g.logVariable, g.productions, g.continuationType)


class CG(object):
    """OCaml contextual_grammar. Built from an official Grammar (make_dummy_contextual) or ContextualGrammar."""

    def __init__(self, g):
        from dreamcoder.grammar import ContextualGrammar
        if isinstance(g, ContextualGrammar):
            self.no_context = _official_grammar_to_G(g.noParent)
            self.variable_context = _official_grammar_to_G(g.variableParent)
            self.contextual = {p: [_official_grammar_to_G(a) for a in gs] for p, gs in g.library.items()}
        else:
            G0 = _official_grammar_to_G(g)
            self.no_context = G0
            self.variable_context = G0
            self.contextual = {}
            for l, t, p in g.productions:
                self.contextual[p] = [G0] * len(arguments_of(from_official(t)))
        for pr in self.no_context.library:
            if pr.program.isPrimitive and pr.program.name == "REAL":
                raise NotImplementedError("free parameters (REAL) not supported by this port")
        if any(p.isPrimitive and p.name in ("fix1", "fix2") for _, _, p in
               (g.noParent.productions if isinstance(g, ContextualGrammar) else g.productions)):
            raise NotImplementedError("recursion primitives (fix1/fix2 enumeration) not ported")


def lse(xs):
    m = max(xs)
    if m == NEG_INF:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def unifying_expressions(g, environment, request, ctx):
    """[(candidate program, argument types, ctx, normalized ll)] in OCaml order."""
    sub0 = ctx.sub
    out = []
    nvar = 0
    for j, t in enumerate(environment):
        t = apply_sub(t, sub0)
        r = return_of(t)
        if might_unify(r, request):
            sub = dict(sub0)
            try:
                _unify(sub, r, request)
            except UFail:
                continue
            k = Ctx(ctx.next, sub)
            t2 = apply_sub(t, sub)
            out.append([Index(j), arguments_of(t2), k, g.logVariable])
    if out and g.continuation_type is not None and g.continuation_type == request:
        terminal = [c[0].i for c in out if not c[1]]
        if terminal:
            smallest = min(terminal)
            out = [c for c in out if c[1] or c[0].i == smallest]
    if out:
        nv = math.log(len(out))
        for c in out:
            c[3] -= nv
    for pr in g.library:
        if not might_unify(pr.ret, request):
            continue
        n = ctx.next
        sub = dict(sub0)
        if pr.nvars:
            ret = shift_type(pr.ret, n)
            try:
                _unify(sub, ret, request)
            except UFail:
                continue
            k = Ctx(n + pr.nvars, sub)
            args = [shift_type(a, n) for a in arguments_of(pr.tp)]
        else:
            try:
                _unify(sub, pr.ret, request)
            except UFail:
                continue
            k = Ctx(n, sub)
            args = arguments_of(pr.tp)
        out.append([pr.program, args, k, pr.ll])
    if not out:
        return out
    z = lse([c[3] for c in out])
    for c in out:
        c[3] -= z
    return out


def _cand_key(environment, request, ctx):
    """Memo key valid when request & environment are ground (no type variables)."""
    return None


# ------------------------------------------------------------------ depth-first enumeration

class Timeout(Exception):
    pass


def _enc(t, nxt):
    """encode fresh type variables (>= nxt) as negative ints so cached results are next-independent"""
    if isinstance(t, int):
        return -1 - (t - nxt) if t >= nxt else t
    if len(t) == 1:
        return t
    return (t[0],) + tuple(_enc(a, nxt) for a in t[1:])


def _dec(t, nxt):
    if isinstance(t, int):
        return nxt + (-1 - t) if t < 0 else t
    if len(t) == 1:
        return t
    return (t[0],) + tuple(_dec(a, nxt) for a in t[1:])


class Enumerator(object):
    """Exact port of enumerate_programs' / enumerate_contextual_applications.
    Argument sub-enumerations (lower bound 0) are memoised: their result list (in DFS order) is a pure
    function of (grammar, request, environment, substitution, upper bound); fresh type variables are
    stored relative to ctx.next so that the cache is shared across different `next` counters.
    Querying with a smaller upper bound filters the cached list, which preserves the DFS order."""

    MAX_CACHED = 4000000

    def __init__(self, cg, deadline):
        self.cg = cg
        self.deadline = deadline
        self.calls = 0
        self.memo = {}
        self.cmemo = {}
        self.stored = 0

    def check(self):
        self.calls += 1
        if (self.calls & 1023) == 0 and time.time() > self.deadline:
            raise Timeout()

    def candidates(self, g, env, request, ctx):
        """unifying_expressions, memoised on fully resolved (request, environment) types. Because the
        resolved types contain no bound variables, the new bindings produced by unification do not depend
        on the rest of the substitution; fresh variables are stored relative to ctx.next. Exact."""
        sub = ctx.sub
        envR = tuple(apply_sub(t, sub) for t in env) if sub else tuple(env)
        key = (id(g), request, envR)
        nxt = ctx.next
        e = self.cmemo.get(key)
        if e is None:
            if len(self.cmemo) > 1000000:
                self.cmemo.clear()
            raw = unifying_expressions(g, list(envR), request, Ctx(nxt, {}))
            e = [(cand, tuple(_enc(a, nxt) for a in args), c2.next - nxt,
                  tuple((_enc(v, nxt), _enc(t, nxt)) for v, t in c2.sub.items()), ll)
                 for cand, args, c2, ll in raw]
            self.cmemo[key] = e
        out = []
        for cand, args, dn, delta, ll in e:
            if delta:
                s2 = dict(sub)
                for v, t in delta:
                    s2[_dec(v, nxt)] = _dec(t, nxt)
            else:
                s2 = sub
            out.append((cand, [_dec(a, nxt) for a in args], Ctx(nxt + dn, s2), ll))
        return out

    # ---- memoised argument enumeration: all (program, ctx, ll) with 0 <= -ll < ub
    def arg_results(self, g, ctx, request, env, ub):
        self.check()
        if ub < 0.:
            return []
        sub = ctx.sub
        envR = tuple(apply_sub(t, sub) for t in env) if sub else tuple(env)
        key = (id(g), request, envR)
        e = self.memo.get(key)
        nxt = ctx.next
        if e is None or e[0] < ub:
            raw = self._enumerate_list(g, ctx, request, env, ub)
            enc = []
            for p, c2, ll in raw:
                delta = tuple((_enc(v, nxt), _enc(t, nxt)) for v, t in c2.sub.items()
                              if v not in sub)
                enc.append((p, c2.next - nxt, delta, ll))
            if self.stored > self.MAX_CACHED:
                self.memo.clear()
                self.stored = 0
            self.memo[key] = (ub, enc)
            self.stored += len(enc)
            return raw
        out = []
        for p, dn, delta, ll in e[1]:
            if -ll < ub:
                if delta:
                    s2 = dict(sub)
                    for v, t in delta:
                        s2[_dec(v, nxt)] = _dec(t, nxt)
                else:
                    s2 = sub
                out.append((p, Ctx(nxt + dn, s2), ll))
        return out

    def _enumerate_list(self, g, ctx, request, env, ub):
        if is_arrow(request):
            return [(Abstraction(b), c2, ll) for b, c2, ll in
                    self.arg_results(g, ctx, request[2], [request[1]] + env, ub)]
        cg = self.cg
        out = []
        for cand, argtypes, c2, ll in self.candidates(g, env, request, ctx):
            if -ll >= ub:
                continue
            if cand.isIndex:
                reqs = [(a, cg.variable_context) for a in argtypes]
            else:
                reqs = list(zip(argtypes, cg.contextual[cand]))
            for p, c3, al in self.app_list(c2, env, reqs, 0, cand, ub + ll):
                out.append((p, c3, ll + al))
        return out

    def app_list(self, ctx, env, reqs, i, f, ub):
        if ub < 0.:
            return []
        if i == len(reqs):
            return [(f, ctx, 0.)] if 0. < ub else []
        at, ag = reqs[i]
        at = apply_sub(at, ctx.sub)
        out = []
        last = i + 1 == len(reqs)
        for a, c2, ll in self.arg_results(ag, ctx, at, env, ub):
            if last:
                if 0. < ub + ll:
                    out.append((Application(f, a), c2, ll))
            else:
                for p, c3, al in self.app_list(c2, env, reqs, i + 1, Application(f, a), ub + ll):
                    out.append((p, c3, al + ll))
        return out

    # ---- streaming enumeration with a lower bound (top level / around skeletons)
    def programs(self, g, ctx, request, env, lb, ub, depth, k):
        """enumerate_programs': calls k(program, ctx, ll) for lb <= -ll < ub"""
        self.check()
        if ub < 0.:
            return
        if is_arrow(request):
            self.programs(g, ctx, request[2], [request[1]] + env, lb, ub, depth,
                          lambda body, c2, ll: k(Abstraction(body), c2, ll))
            return
        cg = self.cg
        for cand, argtypes, c2, ll in self.candidates(g, env, request, ctx):
            if -ll >= ub:
                continue
            if cand.isIndex:
                reqs = [(a, cg.variable_context) for a in argtypes]
            else:
                reqs = list(zip(argtypes, cg.contextual[cand]))
            self.applications(c2, env, reqs, 0, cand, lb + ll, ub + ll,
                              (lambda ll_: (lambda p, c3, al: k(p, c3, ll_ + al)))(ll))

    def applications(self, ctx, env, reqs, i, f, lb, ub, k):
        if ub < 0.:
            return
        if i == len(reqs):
            if lb <= 0. and 0. < ub:
                k(f, ctx, 0.)
            return
        at, ag = reqs[i]
        at = apply_sub(at, ctx.sub)
        for a, c2, ll in self.arg_results(ag, ctx, at, env, ub):
            self.applications(c2, env, reqs, i + 1, Application(f, a), lb + ll, ub + ll,
                              (lambda ll_: (lambda p, c3, al: k(p, c3, al + ll_)))(ll))


# ------------------------------------------------------------------ best-first fringe (multicore)

class _Hole(object):
    __slots__ = ('tp', 'g')

    def __init__(self, tp, g):
        self.tp, self.g = tp, g


# skeletons: ('hole', tp, g) | ('abs', body) | ('app', f, x) | ('leaf', program)

def _follow(sk, path):
    for step in path:
        if step[0] == 'L':
            sk = sk[1]
        elif step[0] == 'R':
            sk = sk[2]
        else:
            sk = sk[1]
    return sk


def _modify(sk, q, path, i=0):
    if i == len(path):
        assert sk[0] == 'hole'
        return q
    s = path[i]
    if s[0] == 'L':
        return ('app', _modify(sk[1], q, path, i + 1), sk[2])
    if s[0] == 'R':
        return ('app', sk[1], _modify(sk[2], q, path, i + 1))
    return ('abs', _modify(sk[1], q, path, i + 1))


def _unwind(path):
    p = list(path)
    while p:
        last = p.pop()
        if last[0] == 'L':
            p.append(('R',))
            return p
    return p


def _path_env(path):
    return [s[1] for s in path if s[0] == 'A'][::-1]


class State(object):
    __slots__ = ('skeleton', 'context', 'path', 'cost')

    def __init__(self, skeleton, context, path, cost):
        self.skeleton, self.context, self.path, self.cost = skeleton, context, path, cost

    def finished(self):
        return self.skeleton[0] != 'hole' and not self.path

    def __lt__(self, o):
        return self.cost < o.cost


def _successors(cg, state):
    hole = _follow(state.skeleton, state.path)
    assert hole[0] == 'hole'
    ctx = state.context
    request = apply_sub(hole[1], ctx.sub)
    g = hole[2]
    if is_arrow(request):
        return [State(_modify(state.skeleton, ('abs', ('hole', request[2], g)), state.path),
                      ctx, state.path + [('A', request[1])], state.cost)]
    env = _path_env(state.path)
    out = []
    for cand, argtypes, c2, ll in unifying_expressions(g, env, request, ctx):
        if cand.isIndex:
            reqs = [(a, cg.variable_context) for a in argtypes]
        else:
            reqs = list(zip(argtypes, cg.contextual[cand]))
        if not argtypes:
            out.append(State(_modify(state.skeleton, ('leaf', cand), state.path), c2,
                             _unwind(state.path), state.cost - ll))
        else:
            tmpl = ('leaf', cand)
            for a, ag in reqs:
                tmpl = ('app', tmpl, ('hole', a, ag))
            out.append(State(_modify(state.skeleton, tmpl, state.path), c2,
                             state.path + [('L',)] * (len(argtypes) - 1) + [('R',)], state.cost - ll))
    return out


def best_first(cg, request, lb, ub, frontier_size):
    completed = []
    pq = [(0., 0, State(('hole', request, cg.no_context), EMPTY, [], 0.))]
    counter = 1
    while pq and len(pq) < frontier_size:
        _, _, best = heapq.heappop(pq)
        for child in _successors(cg, best):
            if child.finished():
                if lb <= child.cost < ub:
                    completed.append(child)
            elif child.cost < ub:
                heapq.heappush(pq, (child.cost, counter, child))
                counter += 1
    return completed, [s for _, _, s in pq]


def _to_program(sk):
    t = sk[0]
    if t == 'leaf':
        return sk[1]
    if t == 'abs':
        return Abstraction(_to_program(sk[1]))
    return Application(_to_program(sk[1]), _to_program(sk[2]))


def _free(sk):
    t = sk[0]
    if t == 'hole':
        return True
    if t == 'leaf':
        return False
    if t == 'abs':
        return _free(sk[1])
    return _free(sk[1]) or _free(sk[2])


def dfs_around_skeleton(en, cg, lb, ub, state, k):
    """dfs_around_skeleton: calls k(program, ll) for every completion of the skeleton within the band"""
    environment = _path_env(state.path)

    def around(e, depth_abs):
        t = e[0]
        if t == 'abs':
            ab = around(e[1], depth_abs + 1)
            return lambda ctx, l, u, kk: ab(ctx, l, u, lambda body, c2, ll: kk(Abstraction(body), c2, ll))
        if t == 'app':
            ff, xf = _free(e[1]), _free(e[2])
            if xf and not ff:
                fprog = _to_program(e[1])
                ax = around(e[2], depth_abs)
                return lambda ctx, l, u, kk: ax(ctx, l, u, lambda x2, c2, ll: kk(Application(fprog, x2), c2, ll))
            if ff and xf:
                ax = around(e[2], depth_abs)
                af = around(e[1], depth_abs)

                def go(ctx, l, u, kk):
                    def got_f(f2, c2, f_ll):
                        ax(c2, l + f_ll, u + f_ll,
                           lambda x2, c3, x_ll: kk(Application(f2, x2), c3, f_ll + x_ll))
                    af(ctx, 0., u, got_f)
                return go
            assert False
        if t == 'hole':
            g = e[2]
            env = environment[len(environment) - depth_abs:] if depth_abs else []

            def go(ctx, l, u, kk, tp=e[1], g=g, env=env):
                tp2 = apply_sub(tp, ctx.sub)
                en.programs(g, ctx, tp2, env, l, u, 9999, kk)
            return go
        assert False

    f = around(state.skeleton, 0)
    f(state.context, lb - state.cost, ub - state.cost, lambda e, c, ll: k(e, ll - state.cost))


# ------------------------------------------------------------------ task loop (task.ml enumerate_for_tasks)

class Hits(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.heap = []  # min-heap on posterior
        self.n = 0

    def add(self, prior, ll, program, dt):
        self.n += 1
        heapq.heappush(self.heap, (prior + ll, self.n, str(program), prior, ll, dt))
        while len(self.heap) > self.maxsize:
            heapq.heappop(self.heap)


def make_scorer(tasks, timeout):
    """Returns score(p) -> [log likelihood per task], identical to [t.logLikelihood(p, timeout) for t in tasks].
    For plain supervised Tasks (all-or-nothing on examples, no cache) the program is compiled once and
    the SIGVTALRM handler is installed once, instead of once per task; the per-task timeout is unchanged."""
    import signal
    from dreamcoder.task import Task, EvaluationTimeout
    plain = [type(t) is Task and not getattr(t, "cache", False) for t in tasks]

    armed = [False]

    def on_timeout(_1, _2):
        # a late (pending) signal after disarming must be ignored, like the official no-op handler swap
        if armed[0]:
            raise EvaluationTimeout()
    if any(plain) and timeout is not None:
        signal.signal(signal.SIGVTALRM, on_timeout)

    def check(t, f):
        # The timer can fire anywhere (even inside the finally clause), so the outer handler catches it.
        try:
            if timeout is not None:
                armed[0] = True
                signal.setitimer(signal.ITIMER_VIRTUAL, timeout)
            try:
                predict = t.predict
                for x, y in t.examples:
                    try:
                        v = predict(f, x)
                    except BaseException:
                        v = None
                    if v != y:
                        return False
                return True
            finally:
                if timeout is not None:
                    armed[0] = False
                    signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        except EvaluationTimeout:
            armed[0] = False
            if timeout is not None:
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            return False

    def score(p):
        out = []
        compiled = False
        f = None
        for t, pl in zip(tasks, plain):
            if not pl:
                out.append(t.logLikelihood(p, timeout))
                continue
            if not compiled:
                compiled = True
                try:
                    f = p.evaluate([])
                except Exception:
                    f = None
            out.append(0. if (f is not None and check(t, f)) else NEG_INF)
        return out
    return score


def _run_fringe(args):
    (cg, lb, ub, states, tasks, maxfs, eval_timeout, deadline, start) = args
    hits = [Hits(m) for m in maxfs]
    en = Enumerator(cg, deadline)
    count = [0]
    score = make_scorer(tasks, eval_timeout)

    def k(p, prior):
        count[0] += 1
        for j, ll in enumerate(score(p)):
            if ll > NEG_INF and ll == ll:
                hits[j].add(prior, ll, p, time.time() - start)
    try:
        for s in states:
            dfs_around_skeleton(en, cg, lb, ub, s, k)
    except Timeout:
        pass
    return [h.heap for h in hits], count[0]


_POOL_ARGS = None


def _pool_run(i):
    cg, lb, ub, fringe, tasks, maxfs, et, deadline, start = _POOL_ARGS
    return _run_fringe((cg, lb, ub, [fringe[i]], tasks, maxfs, et, deadline, start))


def enumerate_for_tasks(g, tasks, maxFrontiers, lowerBound, upperBound, budgetIncrement, timeout, nc=1,
                        evaluationTimeout=None, shatter_factor=10):
    """Returns ([[hit dicts]] per task, number enumerated)."""
    global _POOL_ARGS
    start = time.time()
    deadline = start + timeout
    cg = CG(g)
    request = from_official(tasks[0].request)
    hits = [Hits(m) for m in maxFrontiers]
    total = 0
    lb = lowerBound
    while time.time() < deadline and any(len(h.heap) < m for h, m in zip(hits, maxFrontiers)) \
            and lb + budgetIncrement <= upperBound:
        ub = lb + budgetIncrement
        shatter = 1 if nc == 1 else shatter_factor * nc
        finished, fringe = best_first(cg, request, lb, ub, shatter)
        fringe.sort(key=lambda s: s.cost)
        results = []
        if nc > 1 and len(fringe) > 1:
            import multiprocessing as mp
            _POOL_ARGS = (cg, lb, ub, fringe, tasks, maxFrontiers, evaluationTimeout, deadline, start)
            with mp.get_context('fork').Pool(nc) as pool:
                results = list(pool.imap_unordered(_pool_run, range(len(fringe)), chunksize=1))
            _POOL_ARGS = None
        else:
            results = [_run_fringe((cg, lb, ub, fringe, tasks, maxFrontiers, evaluationTimeout, deadline, start))]
        # finished programs from the best-first phase are evaluated by the parent
        fin = _run_fringe_finished(finished, tasks, maxFrontiers, evaluationTimeout, start)
        results.append(fin)
        for heaps, n in results:
            total += n
            for j, hp in enumerate(heaps):
                for item in hp:
                    _, _, ps, prior, ll, dt = item
                    if not any(x[2] == ps for x in hits[j].heap):
                        hits[j].n += 1
                        heapq.heappush(hits[j].heap, (prior + ll, hits[j].n, ps, prior, ll, dt))
                        while len(hits[j].heap) > hits[j].maxsize:
                            heapq.heappop(hits[j].heap)
        lb = ub
    out = [[{"program": ps, "logPrior": prior, "logLikelihood": ll, "time": dt}
            for _, _, ps, prior, ll, dt in h.heap] for h in hits]
    return out, total


def _run_fringe_finished(finished, tasks, maxfs, eval_timeout, start):
    hits = [Hits(m) for m in maxfs]
    n = 0
    score = make_scorer(tasks, eval_timeout)
    for s in finished:
        p = _to_program(s.skeleton)
        n += 1
        for j, ll in enumerate(score(p)):
            if ll > NEG_INF and ll == ll:
                hits[j].add(-s.cost, ll, p, time.time() - start)
    return [h.heap for h in hits], n


def count_programs(g, request, lb, ub, timeout=1e9):
    """Benchmark helper: number of programs in the band and elapsed time (single core, no evaluation)."""
    cg = CG(g)
    req = from_official(request)
    en = Enumerator(cg, time.time() + timeout)
    n = [0]
    t = time.time()

    def k(p, c, ll):
        n[0] += 1
    try:
        en.programs(cg.no_context, EMPTY, req, [], lb, ub, 9999, k)
    except Timeout:
        pass
    return n[0], time.time() - t
