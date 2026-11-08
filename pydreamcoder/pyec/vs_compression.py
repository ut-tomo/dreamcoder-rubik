"""Faithful Python port of the official DreamCoder compressor
(ellisk42/ec @ cb0e63f: solvers/versions.ml + solvers/compression.ml).

The algorithm, defaults and scoring follow the OCaml code line-by-line:
  * version spaces with n-step inverse beta-reduction (+ optional inlining),
  * candidate proposal = minimum-cost inhabitants occurring in >= 2 frontiers, `nontrivial` filter,
  * beam search over relative costs (beam_costs), top-I candidates,
  * batched refactoring of the (top-K restricted) frontiers with each candidate,
  * scoring with one inside-outside step + pseudocounts, minus aic*|library| and
    structurePenalty * sum(production sizes),
  * final rewrite of *all* frontier programs with the best invention, repeated until no improvement.
The control flow mirrors `compression_step_master` (which is what the Python frontend
actually invokes, because ocamlInduce hard-codes CPUs=2). The OCaml master/worker split only
partitions frontiers between processes; `nc` worker processes are supported here as well
(see `compress(..., CPUs=...)`), with identical results up to tie-breaking.

Only intentional deviation: OCaml iterates hash tables in an unspecified hash order, which
determines (a) the numbering of version-space nodes and therefore (b) which of several
*equal-cost* refactorings / equal-score candidates is chosen. We iterate in insertion order.
This never changes a score, only which of tied alternatives is returned.
"""
import sys
import math
from collections import OrderedDict

from dreamcoder.program import (Program, Application, Abstraction, Index, Primitive, Invented,
                                InferenceFailure, ShiftFailure)
from dreamcoder.type import Context, UnificationFailure, arrow

sys.setrecursionlimit(100000)

NEG_INF = float('-inf')
INF = float('inf')
EPSILON_COST = 0.01


def lse(xs):
    xs = list(xs)
    if not xs:
        return NEG_INF
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


def is_invalid(x):
    return x != x or x == INF or x == NEG_INF


def minimum_by(f, l):
    # OCaml: List.reduce_exn l ~f:(fun x y -> if f x < f y then x else y)  (ties -> later element)
    it = iter(l)
    best = next(it)
    fb = f(best)
    for y in it:
        fy = f(y)
        if fb < fy:
            continue
        best, fb = y, fy
    return best


def occurs_multiple_times(xs):
    counts = OrderedDict()
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return [k for k, c in counts.items() if c > 1]


class EtaExpandFailure(Exception):
    pass


class DuplicatePrimitive(Exception):
    pass


# ---------------------------------------------------------------- program utilities (program.ml)

def program_size(e):
    if e.isApplication:
        return program_size(e.f) + program_size(e.x)
    if e.isAbstraction:
        return program_size(e.body)
    return 1


def recursively_get_abstraction_body(e):
    while e.isAbstraction:
        e = e.body
    return e


def free_variables(e, d=0):
    if e.isIndex:
        return [e.i - d] if e.i >= d else []
    if e.isApplication:
        return free_variables(e.f, d) + free_variables(e.x, d)
    if e.isAbstraction:
        return free_variables(e.body, d + 1)
    return []


def application_parse(e):
    xs = []
    while e.isApplication:
        xs.append(e.x)
        e = e.f
    xs.reverse()
    return e, xs


def shift_free_variables(p, shift, height=0):
    if p.isIndex:
        if p.i < height:
            return p
        if p.i + shift < 0:
            raise ShiftFailure()
        return Index(p.i + shift)
    if p.isApplication:
        return Application(shift_free_variables(p.f, shift, height), shift_free_variables(p.x, shift, height))
    if p.isAbstraction:
        return Abstraction(shift_free_variables(p.body, shift, height + 1))
    return p


def closed_inference(e):
    try:
        return e.inferType(Context.EMPTY, [], {})[1]
    except InferenceFailure:
        raise UnificationFailure()


def make_invention(i):
    try:
        return Invented(i)
    except InferenceFailure:
        raise UnificationFailure()


def is_arrow(t):
    return t.isArrow()


# ---------------------------------------------------------------- grammar (grammar.ml subset)

class CGrammar(object):
    """OCaml `grammar` record: library = [(program, type, logProbability)], logVariable, continuation_type"""

    def __init__(self, logVariable, library, continuation_type=None):
        self.logVariable = logVariable
        self.library = library
        self.continuation_type = continuation_type
        self._weights = {p: l for p, t, l in library}
        assert len(self._weights) == len(library), "duplicate primitive"

    def primitives(self):
        return [p for p, _, _ in self.library]

    def log_weight(self, p):
        if p.isIndex:
            return self.logVariable
        return self._weights[p]

    def with_weights(self, logVariable, f):
        return CGrammar(logVariable, [(p, t, f(p)) for p, t, _ in self.library], self.continuation_type)


def uniform_grammar(primitives, continuation_type=None):
    n = len(primitives)
    assert len(set(primitives)) == n
    return CGrammar(math.log(0.5), [(p, p.tp if p.isInvented else p.tp, -math.log(n)) for p in primitives],
                    continuation_type)


def unifying_expressions(g, environment, request, context):
    """grammar.ml unifying_expressions: [(candidate, argument types, context, normalized ll)]"""
    variable_candidates = []
    for j, t in enumerate(environment):
        t = t.apply(context)
        try:
            k = context.unify(t.returns(), request)
            t2 = t.apply(k)
            variable_candidates.append((Index(j), t2.functionArguments(), k, g.logVariable))
        except UnificationFailure:
            pass
    if variable_candidates and g.continuation_type is not None and g.continuation_type == request:
        terminal = [p.i for p, ts, _, _ in variable_candidates if ts == []]
        if terminal:
            smallest = min(terminal)
            variable_candidates = [(p, ts, k, l) for p, ts, k, l in variable_candidates
                                   if (not p.isIndex) or ts != [] or p.i == smallest]
    nv = math.log(len(variable_candidates)) if variable_candidates else NEG_INF
    variable_candidates = [(p, ts, k, l - nv) for p, ts, k, l in variable_candidates]
    grammar_candidates = []
    for p, t, l in g.library:
        try:
            k, ti = t.instantiate(context)
            k = k.unify(ti.returns(), request)
            ti = ti.apply(k)
            grammar_candidates.append((p, ti.functionArguments(), k, l))
        except UnificationFailure:
            pass
    candidates = variable_candidates + grammar_candidates
    z = lse([c[3] for c in candidates])
    return [(p, ts, k, l - z) for p, ts, k, l in candidates]


class Summary(object):
    __slots__ = ('normalizer_frequency', 'use_frequency', 'likelihood_constant')

    def __init__(self):
        self.normalizer_frequency = {}
        self.use_frequency = {}
        self.likelihood_constant = 0.


def record_likelihood_event(s, actual, possibles):
    constant = -math.log(sum(1 for p in possibles if p.isIndex)) if actual.isIndex else 0.
    if actual.isIndex:
        actual = Index(0)
    variable_possible = any(p.isIndex for p in possibles)
    ps = set(p for p in possibles if not p.isIndex)
    if variable_possible:
        ps.add(Index(0))
    ps = frozenset(ps)
    s.likelihood_constant += constant
    s.use_frequency[actual] = s.use_frequency.get(actual, 0.) + 1.
    s.normalizer_frequency[ps] = s.normalizer_frequency.get(ps, 0.) + 1.


def summary_likelihood(g, s):
    a = s.likelihood_constant
    for k, d in s.use_frequency.items():
        a += d * g.log_weight(k)
    b = 0.
    for k, d in s.normalizer_frequency.items():
        b += d * lse([g.log_weight(p) for p in k])
    return a - b


def make_likelihood_summary(g, request, expression):
    s = Summary()
    context = [Context.EMPTY]

    def summarize(r, environment, p):
        if r.isArrow():
            if not p.isAbstraction:
                raise Exception("remove_abstractions")
            summarize(r.arguments[1], [r.arguments[0]] + environment, p.body)
            return
        candidates = unifying_expressions(g, environment, r, context[0])
        f, xs = application_parse(p)
        found = None
        for c in candidates:
            if c[0] == f:
                found = c
                break
        if found is None:
            s.likelihood_constant = NEG_INF
            return
        _, argument_types, newContext, _ = found
        context[0] = newContext
        record_likelihood_event(s, f, [c[0] for c in candidates])
        if len(xs) != len(argument_types):
            raise Exception("zip_exn")
        for x, xt in zip(xs, argument_types):
            summarize(xt, environment, x)

    summarize(request, [], expression)
    return s


def likelihood_under_grammar(g, request, program):
    return summary_likelihood(g, make_likelihood_summary(g, request, program))


# ---------------------------------------------------------------- frontier

class CFrontier(object):
    __slots__ = ('request', 'programs')

    def __init__(self, request, programs):
        self.request = request
        self.programs = programs  # [(program, logLikelihood)]


def restrict(topK, g, frontier):
    scored = [(ll + likelihood_under_grammar(g, frontier.request, p), p, ll) for p, ll in frontier.programs]
    # sort_by (fun (posterior,_,_) -> 0 - posterior), stable
    scored.sort(key=lambda x: -x[0])
    return CFrontier(frontier.request, [(p, ll) for _, p, ll in scored[:topK]])


def inside_outside(pseudoCounts, g, frontiers):
    summaries = [[(l, make_likelihood_summary(g, f.request, p)) for p, l in f.programs] for f in frontiers]

    # update g
    use = {}
    norm = {}
    for ss in summaries:
        lws = [l + summary_likelihood(g, s) for l, s in ss]
        z = lse(lws)
        for lw, (_, s) in zip(lws, ss):
            w = math.exp(lw - z)
            for k, d in s.use_frequency.items():
                use[k] = use.get(k, 0.) + d * w
            for k, d in s.normalizer_frequency.items():
                norm[k] = norm.get(k, 0.) + d * w

    def possible(p):
        return sum(d for k, d in norm.items() if p in k)

    def actual(p):
        return use.get(p, 0.)

    lv = math.log(actual(Index(0)) + pseudoCounts) - math.log(possible(Index(0)) + pseudoCounts)
    g2 = g.with_weights(lv, lambda p: math.log(actual(p) + pseudoCounts) - math.log(possible(p) + pseudoCounts))
    total = 0.
    for ss in summaries:
        total += lse([l + summary_likelihood(g2, s) for l, s in ss])
    return g2, total


def production_size(p):
    if p.isPrimitive:
        return 1
    if p.isInvented:
        e = recursively_get_abstraction_body(p.body)
        if e.isApplication and e.f.isApplication and e.f.f.isPrimitive and e.f.f.name == "fix1":
            assert e.f.x.isIndex
            return program_size(e.x)
        return program_size(e)
    raise Exception("Element of grammar is neither primitive nor invented")


def grammar_induction_score(aic, structurePenalty, pseudoCounts, frontiers, g):
    g, ll = inside_outside(pseudoCounts, g, frontiers)
    return g, ll - aic * len(g.library) - structurePenalty * sum(production_size(p) for p, _, _ in g.library)


# ---------------------------------------------------------------- version spaces (versions.ml)

UNIVERSE_T = ('U',)
VOID_T = ('V',)


class VersionTable(object):
    def __init__(self):
        self.s2i = {}
        self.i2s = []
        self.recursive_inversion_table = []
        self.n_step_table = {}
        self.substitution_table = {}
        self.void = self.incorporate_space(VOID_T)
        self.universe = self.incorporate_space(UNIVERSE_T)
        assert self.void == 0 and self.universe == 1

    def __len__(self):
        return len(self.i2s)

    def clear_dynamic_programming_tables(self):
        self.n_step_table.clear()
        self.substitution_table.clear()

    def incorporate_space(self, v):
        i = self.s2i.get(v)
        if i is not None:
            return i
        i = len(self.i2s)
        self.s2i[v] = i
        self.i2s.append(v)
        self.recursive_inversion_table.append(None)
        return i

    # constructors
    def apply(self, f, x):
        if f == 0 or x == 0:
            return 0
        return self.incorporate_space(('A', f, x))

    def abstract(self, b):
        if b == 0:
            return 0
        return self.incorporate_space(('L', b))

    def index(self, i):
        return self.incorporate_space(('I', i))

    def terminal(self, e):
        return self.incorporate_space(('T', e))

    def union(self, vs):
        if self.universe in vs:
            return self.universe
        acc = set()
        for v in vs:
            s = self.i2s[v]
            tag = s[0]
            if tag == 'u':
                acc.update(s[1])
            elif tag == 'V':
                pass
            elif tag == 'U':
                assert False
            else:
                acc.add(v)
        if not acc:
            return 0
        if len(acc) == 1:
            return next(iter(acc))
        return self.incorporate_space(('u', tuple(sorted(acc))))

    def incorporate(self, e):
        if e.isIndex:
            return self.index(e.i)
        if e.isAbstraction:
            return self.abstract(self.incorporate(e.body))
        if e.isApplication:
            return self.apply(self.incorporate(e.f), self.incorporate(e.x))
        return self.terminal(e)

    def extract(self, j):
        s = self.i2s[j]
        tag = s[0]
        if tag == 'u':
            return [p for u in s[1] for p in self.extract(u)]
        if tag == 'A':
            fs = self.extract(s[1])
            xs = self.extract(s[2])
            return [Application(f, x) for f in fs for x in xs]
        if tag == 'I':
            return [Index(s[1])]
        if tag == 'V':
            return []
        if tag == 'T':
            return [s[1]]
        if tag == 'L':
            return [Abstraction(b) for b in self.extract(s[1])]
        raise Exception("extract universe")

    def extract1(self, j):
        r = self.extract(j)
        assert len(r) == 1
        return r[0]

    def shift_free(self, n, index, c=0):
        if n == 0:
            return index
        s = self.i2s[index]
        tag = s[0]
        if tag == 'u':
            return self.union([self.shift_free(n, i, c) for i in s[1]])
        if tag == 'I':
            i = s[1]
            if i < c:
                return index
            if i >= n + c:
                return self.index(i - n)
            return 0
        if tag == 'A':
            return self.apply(self.shift_free(n, s[1], c), self.shift_free(n, s[2], c))
        if tag == 'L':
            return self.abstract(self.shift_free(n, s[1], c + 1))
        return index

    def shift_versions(self, n, index, c=0):
        if n == 0:
            return index
        s = self.i2s[index]
        tag = s[0]
        if tag == 'u':
            return self.union([self.shift_versions(n, i, c) for i in s[1]])
        if tag == 'I':
            i = s[1]
            if i < c:
                return index
            if i + n >= 0:
                return self.index(i + n)
            return 0
        if tag == 'A':
            return self.apply(self.shift_versions(n, s[1], c), self.shift_versions(n, s[2], c))
        if tag == 'L':
            return self.abstract(self.shift_versions(n, s[1], c + 1))
        return index

    def intersection(self, a, b):
        sa = self.i2s[a]
        sb = self.i2s[b]
        ta, tb = sa[0], sb[0]
        if ta == 'U':
            return b
        if tb == 'U':
            return a
        if ta == 'V' or tb == 'V':
            return 0
        if ta == 'u' and tb == 'u':
            return self.union([self.intersection(x, y) for x in sa[1] for y in sb[1]])
        if ta == 'u':
            return self.union([self.intersection(x, b) for x in sa[1]])
        if tb == 'u':
            return self.union([self.intersection(x, a) for x in sb[1]])
        if ta == 'L' and tb == 'L':
            return self.abstract(self.intersection(sa[1], sb[1]))
        if ta == 'A' and tb == 'A':
            return self.apply(self.intersection(sa[1], sb[1]), self.intersection(sa[2], sb[2]))
        if ta == 'I' and tb == 'I' and sa[1] == sb[1]:
            return a
        if ta == 'T' and tb == 'T' and sa[1] == sb[1]:
            return a
        return 0

    def have_intersection(self, a, b, table=None):
        if a == b:
            return True
        if a > b:
            a, b = b, a
        if table is not None:
            r = table.get((a, b))
            if r is not None:
                return r
        sa = self.i2s[a]
        sb = self.i2s[b]
        ta, tb = sa[0], sb[0]
        if ta == 'V' or tb == 'V':
            r = False
        elif ta == 'U' or tb == 'U':
            r = True
        elif ta == 'u' and tb == 'u':
            r = any(self.have_intersection(x, y, table) for x in sa[1] for y in sb[1])
        elif ta == 'u':
            r = any(self.have_intersection(x, b, table) for x in sa[1])
        elif tb == 'u':
            r = any(self.have_intersection(x, a, table) for x in sb[1])
        elif ta == 'L' and tb == 'L':
            r = self.have_intersection(sa[1], sb[1], table)
        elif ta == 'A' and tb == 'A':
            r = self.have_intersection(sa[1], sb[1], table) and self.have_intersection(sa[2], sb[2], table)
        elif ta == 'I' and tb == 'I':
            r = sa[1] == sb[1]
        elif ta == 'T' and tb == 'T':
            r = sa[1] == sb[1]
        else:
            r = False
        if table is not None:
            table[(a, b)] = r
        return r

    def inline(self, j):
        def il(arguments, j):
            s = self.i2s[j]
            tag = s[0]
            if tag == 'A':
                return il([s[2]] + arguments, s[1])
            if tag == 'u':
                return self.union([il(arguments, v) for v in s[1]])
            if tag == 'T' and s[1].isInvented:
                body = s[1].body

                def make_substitution(used, unused, body):
                    while True:
                        if not unused:
                            if body.isAbstraction:
                                return None
                            return used, body
                        if body.isAbstraction:
                            used = [unused[0]] + used
                            unused = unused[1:]
                            body = body.body
                        else:
                            return used, body

                def apply_substitution(k, arguments, e):
                    if e.isIndex:
                        i = e.i
                        if i < k:
                            return self.index(i)
                        if i - k < len(arguments):
                            return self.shift_versions(k, arguments[i - k])
                        return self.index(i - len(arguments))
                    if e.isApplication:
                        return self.apply(apply_substitution(k, arguments, e.f), apply_substitution(k, arguments, e.x))
                    if e.isAbstraction:
                        return self.abstract(apply_substitution(k + 1, arguments, e.body))
                    return self.incorporate(e)

                ms = make_substitution([], arguments, body)
                if ms is None:
                    return 0
                used, body = ms
                f = apply_substitution(0, used, body)
                for a in arguments[len(used):]:
                    f = self.apply(f, a)
                return f
            return 0
        return il([], j)

    def substitutions(self, index, n=0):
        key = (index, n)
        m = self.substitution_table.get(key)
        if m is not None:
            return m
        m = OrderedDict()
        s = self.shift_free(n, index)
        if s != 0:
            m[s] = self.index(n)
        sp = self.i2s[index]
        tag = sp[0]
        if tag == 'T':
            if self.universe not in m:
                m[self.universe] = index
        elif tag == 'I':
            i = sp[1]
            if self.universe not in m:
                m[self.universe] = index if i < n else self.index(1 + i)
        elif tag == 'L':
            for k, d in self.substitutions(sp[1], n + 1).items():
                assert k not in m
                m[k] = self.abstract(d)
        elif tag == 'u':
            new_mapping = OrderedDict()
            for x in sp[1]:
                for v, b in self.substitutions(x, n).items():
                    new_mapping.setdefault(v, []).append(b)
            for k, d in new_mapping.items():
                m[k] = self.union(d)
        elif tag == 'A':
            new_mapping = OrderedDict()
            fm = self.substitutions(sp[1], n)
            xm = self.substitutions(sp[2], n)
            for v1, f in fm.items():
                for v2, x in xm.items():
                    if self.have_intersection(v1, v2):
                        v = self.intersection(v1, v2)
                        a = self.apply(f, x)
                        new_mapping.setdefault(v, []).append(a)
            for k, d in new_mapping.items():
                m[k] = self.union(d)
        self.substitution_table[key] = m
        return m

    def _top_inversions(self, j):
        out = []
        for v, b in self.substitutions(j).items():
            if v == self.universe or self.i2s[b] == ('I', 0):
                continue
            out.append(self.apply(self.abstract(b), v))
        return out

    def inversion(self, j):
        return self.union(self._top_inversions(j))

    def recursive_inversion(self, j):
        ri = self.recursive_inversion_table[j]
        if ri is not None:
            return ri
        s = self.i2s[j]
        if s[0] == 'u':
            ri = self.union([self.recursive_inversion(u) for u in s[1]])
        else:
            top = self._top_inversions(j)
            if s[0] == 'A':
                child = [self.apply(self.recursive_inversion(s[1]), s[2]),
                         self.apply(s[1], self.recursive_inversion(s[2]))]
            elif s[0] == 'L':
                child = [self.abstract(self.recursive_inversion(s[1]))]
            else:
                child = []
            ri = self.union(child + top)
        self.recursive_inversion_table[j] = ri
        return ri

    def beta_pruning(self, j):
        def bp(j, isApplied, canBeta):
            s = self.i2s[j]
            tag = s[0]
            if tag == 'A':
                f2 = bp(s[1], True, canBeta)
                x2 = bp(s[2], False, canBeta)
                return self.apply(f2, x2)
            if tag == 'L':
                if isApplied and not canBeta:
                    return 0
                if isApplied and canBeta:
                    return self.abstract(bp(s[1], False, False))
                return self.abstract(bp(s[1], False, canBeta))
            if tag == 'u':
                return self.union([bp(u, isApplied, canBeta) for u in s[1]])
            return j
        return bp(j, False, True)

    def n_step_inversion(self, j, n, inline=False):
        key = (n, j)
        ns = self.n_step_table.get(key)
        if ns is not None:
            return ns

        def step(v):
            if inline:
                i = self.inline(v)
                return self.union([self.recursive_inversion(v), i])
            return self.recursive_inversion(v)

        def n_step(current):
            out = []
            completed = 0
            while True:
                out.append(self.beta_pruning(current))
                if completed == n:
                    return out
                current = step(current)
                completed += 1

        def visit(j):
            s = self.i2s[j]
            tag = s[0]
            if tag in ('u', 'V', 'U'):
                assert False
            if tag == 'A':
                children = self.apply(visit(s[1]), visit(s[2]))
            elif tag == 'L':
                children = self.abstract(visit(s[1]))
            else:
                children = j
            return self.union([children] + n_step(j))

        ns = self.beta_pruning(visit(j))
        self.n_step_table[key] = ns
        return ns

    def reachable_versions(self, indices):
        visited = set()
        order = []
        stack = list(indices)
        while stack:
            j = stack.pop()
            if j in visited:
                continue
            visited.add(j)
            order.append(j)
            s = self.i2s[j]
            tag = s[0]
            if tag == 'L':
                stack.append(s[1])
            elif tag == 'A':
                stack.append(s[1])
                stack.append(s[2])
            elif tag == 'u':
                stack.extend(s[1])
        return order


# ---------------------------------------------------------------- cost tables

class CostTable(object):
    def __init__(self, v):
        self.v = v
        self.function_cost = {}
        self.argument_cost = {}

    def minimum_cost_inhabitants(self, j, canBeLambda=True, given=None):
        table = self.argument_cost if canBeLambda else self.function_cost
        c = table.get(j)
        if c is not None:
            return c
        v = self.v
        if given is not None and v.have_intersection(given, j):
            c = (1., [given])
        else:
            s = v.i2s[j]
            tag = s[0]
            if tag in ('U', 'V'):
                assert False
            if tag in ('I', 'T'):
                c = (1., [j])
            elif tag == 'u':
                children = [self.minimum_cost_inhabitants(u, canBeLambda, given) for u in s[1]]
                cmin = min(cc for cc, _ in children)
                if is_invalid(cmin):
                    c = (cmin, [])
                else:
                    c = (cmin, [p for cc, ps in children if cc == cmin for p in ps])
            elif tag == 'L':
                if canBeLambda:
                    cost, children = self.minimum_cost_inhabitants(s[1], True, given)
                    c = (cost + EPSILON_COST, [v.abstract(ch) for ch in children])
                else:
                    c = (INF, [])
            else:  # 'A'
                fc, fs = self.minimum_cost_inhabitants(s[1], False, given)
                xc, xs = self.minimum_cost_inhabitants(s[2], True, given)
                if is_invalid(fc) or is_invalid(xc):
                    c = (INF, [])
                else:
                    c = (fc + xc + EPSILON_COST, [v.apply(f2, x2) for f2 in fs for x2 in xs])
        cost, indices = c
        c = (cost, sorted(set(indices)))
        table[j] = c
        return c


class CheapCostTable(object):
    def __init__(self, v):
        self.v = v
        self.function_cost = {}
        self.argument_cost = {}

    def minimal_inhabitant_cost(self, j, canBeLambda=True, given=None, intersectionTable=None):
        table = self.argument_cost if canBeLambda else self.function_cost
        c = table.get(j)
        if c is not None:
            return c
        v = self.v
        if given is not None and v.have_intersection(given, j, intersectionTable):
            c = 1.
        else:
            s = v.i2s[j]
            tag = s[0]
            if tag in ('U', 'V'):
                assert False
            if tag in ('I', 'T'):
                c = 1.
            elif tag == 'u':
                c = min(self.minimal_inhabitant_cost(u, canBeLambda, given, intersectionTable) for u in s[1])
            elif tag == 'L':
                c = (EPSILON_COST + self.minimal_inhabitant_cost(s[1], True, given, intersectionTable)) \
                    if canBeLambda else INF
            else:
                c = EPSILON_COST + self.minimal_inhabitant_cost(s[1], False, given, intersectionTable) + \
                    self.minimal_inhabitant_cost(s[2], True, given, intersectionTable)
        table[j] = c
        return c

    def minimal_inhabitant(self, j, canBeLambda=True, given=None, intersectionTable=None):
        c = self.minimal_inhabitant_cost(j, canBeLambda, given, intersectionTable)
        if is_invalid(c):
            return None
        v = self.v
        if c == 1. and given is not None and v.have_intersection(given, j, intersectionTable):
            return v.extract1(given)
        s = v.i2s[j]
        tag = s[0]
        if tag in ('I', 'T'):
            return v.extract1(j)
        if tag == 'u':
            best = minimum_by(lambda u: self.minimal_inhabitant_cost(u, canBeLambda, given, intersectionTable), s[1])
            return self.minimal_inhabitant(best, canBeLambda, given, intersectionTable)
        if tag == 'L':
            return Abstraction(self.minimal_inhabitant(s[1], True, given, intersectionTable))
        if tag == 'A':
            return Application(self.minimal_inhabitant(s[1], False, given, intersectionTable),
                               self.minimal_inhabitant(s[2], True, given, intersectionTable))
        assert False


class Beam(object):
    __slots__ = ('default_function_cost', 'default_argument_cost', 'relative_function', 'relative_argument')

    def __init__(self, dfc, dac):
        self.default_function_cost = dfc
        self.default_argument_cost = dac
        self.relative_function = OrderedDict()
        self.relative_argument = OrderedDict()

    def rf(self, i):
        return self.relative_function.get(i, self.default_function_cost)

    def ra(self, i):
        return self.relative_argument.get(i, self.default_argument_cost)


def _relax(table, key, data):
    old = table.get(key)
    if old is None or old > data:
        table[key] = data


def _narrow(bs, b):
    def narrow(bm):
        if len(bm) > bs:
            items = sorted(bm.items(), key=lambda kv: kv[1])
            return OrderedDict(items[:bs])
        return bm
    b.relative_function = narrow(b.relative_function)
    b.relative_argument = narrow(b.relative_argument)


def calculate_candidate_costs(v, candidates):
    return {k: 1. + len(set(free_variables(v.extract1(k)))) for k in candidates}


def beam_costs_table(ct, bs, candidates, frontier_indices):
    v = ct.v
    cand_set = set(candidates)
    candidate_cost = calculate_candidate_costs(v, candidates)
    caching = {}

    def calculate_costs(j):
        bm = caching.get(j)
        if bm is not None:
            return bm
        dac, inhabitants = ct.minimum_cost_inhabitants(j, True)
        dfc, _ = ct.minimum_cost_inhabitants(j, False)
        bm = Beam(dfc, dac)
        for cand in inhabitants:
            if cand in cand_set:
                cost = candidate_cost[cand]
                bm.relative_function[cand] = cost
                bm.relative_argument[cand] = cost
        s = v.i2s[j]
        tag = s[0]
        if tag == 'L':
            child = calculate_costs(s[1])
            for k, d in list(child.relative_argument.items()):
                _relax(bm.relative_argument, k, d + EPSILON_COST)
        elif tag == 'A':
            fb = calculate_costs(s[1])
            xb = calculate_costs(s[2])
            domain = list(fb.relative_function.keys()) + list(xb.relative_argument.keys())
            for i in domain:
                c = EPSILON_COST + fb.rf(i) + xb.ra(i)
                _relax(bm.relative_function, i, c)
                _relax(bm.relative_argument, i, c)
        elif tag == 'u':
            for u in s[1]:
                child = calculate_costs(u)
                for k, d in list(child.relative_function.items()):
                    _relax(bm.relative_function, k, d)
                for k, d in list(child.relative_argument.items()):
                    _relax(bm.relative_argument, k, d)
        _narrow(bs, bm)
        caching[j] = bm
        return bm

    for f in frontier_indices:
        for j in f:
            calculate_costs(j)
    return caching


def beam_costs_scores(ct, bs, candidates, frontier_indices):
    caching = beam_costs_table(ct, bs, candidates, frontier_indices)
    frontier_beams = [[caching[j] for j in f] for f in frontier_indices]

    def score(i):
        return sum(min(min(b.ra(i), b.rf(i)) for b in bs_) for bs_ in frontier_beams)
    return [score(i) for i in candidates]


def batched_refactor(ct, candidates, frontier_indices):
    caching = beam_costs_table(ct, len(candidates), candidates, frontier_indices)
    v = ct.v

    def refactor(canBeLambda, i, j):
        inhabitants = ct.minimum_cost_inhabitants(j, True)[1]
        if i in inhabitants:
            return v.extract1(i)
        s = v.i2s[j]
        tag = s[0]
        if tag == 'L':
            assert canBeLambda
            return Abstraction(refactor(True, i, s[1]))
        if tag == 'A':
            return Application(refactor(False, i, s[1]), refactor(True, i, s[2]))
        if tag == 'u':
            def cost(u2):
                bm = caching[u2]
                return bm.ra(i) if canBeLambda else bm.rf(i)
            return refactor(canBeLambda, i, minimum_by(cost, s[1]))
        if tag == 'I':
            return Index(s[1])
        if tag == 'T':
            return s[1]
        assert False

    return [[[refactor(True, i, j) for j in f] for f in frontier_indices] for i in candidates]


# ---------------------------------------------------------------- compression.ml

def eta_long(request, e):
    context = [Context.EMPTY]

    def make_long(e, request):
        if request.isArrow():
            return Abstraction(Application(shift_free_variables(e, 1), Index(0)))
        return None

    def visit(request, environment, e):
        if e.isAbstraction and request.isArrow():
            return Abstraction(visit(request.arguments[1], [request.arguments[0]] + environment, e.body))
        if e.isAbstraction:
            raise EtaExpandFailure()
        e2 = make_long(e, request)
        if e2 is not None:
            return visit(request, environment, e2)
        f, xs = application_parse(e)
        if f.isIndex:
            ft = environment[f.i].apply(context[0])
        elif f.isPrimitive or f.isInvented:
            context[0], ft = f.tp.instantiate(context[0])
        else:
            assert False, "not in beta long form"
        context[0] = context[0].unify(request, ft.returns())
        ft = ft.apply(context[0])
        xt = ft.functionArguments()
        if len(xs) != len(xt):
            raise EtaExpandFailure()
        xs2 = [visit(t.apply(context[0]), environment, x) for x, t in zip(xs, xt)]
        r = f
        for x in xs2:
            r = Application(r, x)
        return r

    return visit(request, [], e)


def normalize_invention(i):
    fv = sorted(set(free_variables(i)))
    mapping = {v: k for k, v in enumerate(fv)}

    def visit(d, e):
        if e.isIndex:
            if e.i < d:
                return e
            return Index(d + mapping[e.i - d])
        if e.isAbstraction:
            return Abstraction(visit(d + 1, e.body))
        if e.isApplication:
            return Application(visit(d, e.f), visit(d, e.x))
        return e

    renamed = visit(0, i)
    for _ in fv:
        renamed = Abstraction(renamed)
    return make_invention(renamed)


def rewrite_with_invention(i):
    fv = sorted(set(free_variables(i)))
    closed = normalize_invention(i)
    applied = closed
    for k in range(len(fv) - 1, -1, -1):
        applied = Application(applied, Index(fv[k]))

    def visit(e):
        if e == i:
            return applied
        if e.isApplication:
            return Application(visit(e.f), visit(e.x))
        if e.isAbstraction:
            return Abstraction(visit(e.body))
        return e

    def rewriter(request, e):
        try:
            return eta_long(request, visit(e))
        except (UnificationFailure, InferenceFailure):
            raise EtaExpandFailure()
    return rewriter


def nontrivial(e):
    indices = set()
    duplicated = [0]
    primitives = [0]

    def visit(d, e):
        if e.isIndex:
            i = e.i - d
            if i in indices:
                duplicated[0] += 1
            else:
                indices.add(i)
        elif e.isApplication:
            visit(d, e.f)
            visit(d, e.x)
        elif e.isAbstraction:
            visit(d + 1, e.body)
        else:
            primitives[0] += 1
    visit(0, e)
    return primitives[0] > 1 or (primitives[0] == 1 and duplicated[0] > 0)


def _try_normalize(c):
    try:
        return normalize_invention(c)
    except (UnificationFailure, InferenceFailure):
        return None


# ---- worker-side computations (one worker per frontier partition, as in compression_worker)

class Worker(object):
    def __init__(self, g, frontiers, topK, arity, bs, inline):
        self.g = g
        self.original_frontiers = frontiers
        self.frontiers = [restrict(topK, g, f) for f in frontiers]
        self.arity, self.bs, self.inline = arity, bs, inline
        self.v = VersionTable()
        self.frontier_indices = [[self.v.n_step_inversion(self.v.incorporate(p), arity, inline)
                                  for p, _ in f.programs] for f in self.frontiers]

    def propose(self):
        v = self.v
        self.cost_table = CostTable(v)
        out = []
        for idxs in self.frontier_indices:
            reach = v.reachable_versions(idxs)
            inh = sorted(set(k for j in reach for k in self.cost_table.minimum_cost_inhabitants(j)[1]))
            progs = [v.extract1(k) for k in inh]
            out.append([p for p in progs if nontrivial(p)])
        return out

    def beam(self, candidates):
        idx = [self.v.incorporate(c) for c in candidates]
        s = beam_costs_scores(self.cost_table, self.bs, idx, self.frontier_indices)
        self.cost_table = None
        return s

    def batched_rewrite(self, inventions):
        v = self.v
        inv_idx = [v.incorporate(i) for i in inventions]
        fidx = [[v.n_step_inversion(v.incorporate(p), self.arity, self.inline) for p, _ in f.programs]
                for f in self.frontiers]
        v.clear_dynamic_programming_tables()
        refactored = batched_refactor(CostTable(v), inv_idx, fidx)
        result = []
        for new_programs_per_frontier, invention in zip(refactored, inventions):
            rewriter = rewrite_with_invention(invention)
            fs = []
            for new_programs, frontier in zip(new_programs_per_frontier, self.frontiers):
                ps = []
                for program, (orig, ll) in zip(new_programs, frontier.programs):
                    try:
                        p2 = rewriter(frontier.request, program)
                    except EtaExpandFailure:
                        p2 = orig
                    ps.append((p2, ll))
                fs.append(CFrontier(frontier.request, ps))
            result.append(fs)
        return result

    def final_rewrite(self, invention):
        self.v = None
        frontiers = self.original_frontiers
        v = VersionTable()
        inv = {}
        for f in frontiers:
            for p, _ in f.programs:
                inv[v.incorporate(p)] = v.n_step_inversion(v.incorporate(p), self.arity, self.inline)
        v.clear_dynamic_programming_tables()
        i = v.incorporate(invention)
        ct = CheapCostTable(v)
        rewriter = rewrite_with_invention(invention)
        out = []
        for f in frontiers:
            ps = []
            for orig, ll in f.programs:
                index = inv[v.incorporate(orig)]
                program = ct.minimal_inhabitant(index, given=i)
                try:
                    p2 = rewriter(f.request, program)
                except EtaExpandFailure:
                    p2 = orig
                ps.append((p2, ll))
            out.append(CFrontier(f.request, ps))
        return out


def divide_work_fairly(nc, xs):
    nt = len(xs)
    base = nt // nc
    residual = nt - base * nc
    out = []
    i = 0
    while i < nt:
        cnt = base + (1 if residual > 0 else 0)
        out.append(xs[i:i + cnt])
        i += cnt
        residual -= 1
    return out


def _worker_main(conn, g, frontiers, topK, arity, bs, inline):
    w = Worker(g, frontiers, topK, arity, bs, inline)
    conn.send(w.propose())
    while True:
        cmd, arg = conn.recv()
        if cmd == 'beam':
            conn.send(w.beam(arg))
        elif cmd == 'batched':
            conn.send(w.batched_rewrite(arg))
        elif cmd == 'final':
            conn.send(w.final_rewrite(arg))
        elif cmd == 'kill':
            conn.close()
            return


class _InProc(object):
    def __init__(self, *a):
        self.w = Worker(*a)
        self.proposals = self.w.propose()

    def call(self, cmd, arg):
        return {'beam': self.w.beam, 'batched': self.w.batched_rewrite, 'final': self.w.final_rewrite}[cmd](arg)


def compression_step_master(g, frontiers, nc, structurePenalty, aic, pseudoCounts, arity, bs, topI, topK,
                            inline=True, log=None):
    log = log or (lambda *a: None)
    parts = [p for p in divide_work_fairly(nc, frontiers) if p]
    if nc > 1 and len(parts) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        workers = []
        for part in parts:
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker_main, args=(b, g, part, topK, arity, bs, inline))
            pr.start()
            workers.append((a, pr))
        proposals = [a.recv() for a, _ in workers]

        def broadcast(cmd, arg):
            for a, _ in workers:
                a.send((cmd, arg))
            return [a.recv() for a, _ in workers]

        def finish():
            for a, pr in workers:
                a.send(('kill', None))
                pr.join()
    else:
        w = _InProc(g, frontiers, topK, arity, bs, inline)
        proposals = [w.proposals]

        def broadcast(cmd, arg):
            return [w.call(cmd, arg)]

        def finish():
            pass

    candidates = occurs_multiple_times([c for wp in proposals for fl in wp for c in fl])
    log("Total number of candidates: %d" % len(candidates))
    scores = broadcast('beam', candidates)
    summed = [sum(s[k] for s in scores) for k in range(len(candidates))]
    ranked = sorted(zip(candidates, summed), key=lambda cs: cs[1])  # stable
    candidates = [c for c, _ in ranked][:topI]
    prims = g.primitives()
    kept = []
    for c in candidates:
        n = _try_normalize(c)
        if n is not None and n not in prims:
            kept.append(c)
    candidates = kept
    log("Trimmed down the beam, have only %d best candidates" % len(candidates))
    if not candidates:
        finish()
        return None

    per_worker = broadcast('batched', candidates)
    new_frontiers = [[f for wres in per_worker for f in wres[k]] for k in range(len(candidates))]

    def score(frontiers_, candidate):
        ng = uniform_grammar([normalize_invention(candidate)] + prims)
        return grammar_induction_score(aic, structurePenalty, pseudoCounts, frontiers_, ng)

    _, initial_score = grammar_induction_score(aic, structurePenalty, pseudoCounts,
                                               [restrict(topK, g, f) for f in frontiers], g)
    log("Initial score: %f" % initial_score)
    scored = [(score(fs, c), c) for c, fs in zip(candidates, new_frontiers)]
    (g2, best_score), best_candidate = minimum_by(lambda x: -x[0][1], scored)
    if best_score < initial_score:
        log("No improvement possible.")
        finish()
        return None
    newp = g2.library[0][0]
    log("Improved score to %f (dScore=%f) w/ new primitive\n\t%s : %s" %
        (best_score, best_score - initial_score, newp, newp.tp))
    frontiers2 = [f for wres in broadcast('final', best_candidate) for f in wres]
    finish()
    g3 = inside_outside(pseudoCounts, g2, frontiers2)[0]
    return g3, frontiers2


def compression_loop(g, frontiers, nc=1, structurePenalty=1., inline=True, aic=1., topK=2, pseudoCounts=30.,
                     arity=3, bs=1000000, topI=300, iterations=1000, log=None):
    log = log or (lambda *a: None)
    for _ in range(iterations):
        r = compression_step_master(g, frontiers, nc, structurePenalty, aic, pseudoCounts, arity, bs, topI, topK,
                                    inline=inline, log=log)
        if r is None:
            return g, frontiers
        g, frontiers = r
    log("Exiting compression because of iteration bound.")
    return g, frontiers


# ---------------------------------------------------------------- JSON interface (same message as ./compression)

def strip_grammar(g):
    return g


def run_message(j, log=None):
    """Takes the JSON message the Python frontend sends to ./compression; returns the response dict."""
    from dreamcoder.type import Type
    def parse(s):
        return Program.parse(s)
    dsl = j["DSL"]
    lib = []
    for p in dsl["productions"]:
        e = parse(p["expression"])
        t = e.tp if (e.isPrimitive or e.isInvented) else closed_inference(e)
        lib.append((e, t, p["logProbability"]))
    ct = Type.fromjson(dsl["continuationType"]) if dsl.get("continuationType") else None
    g = CGrammar(dsl["logVariable"], lib, ct)
    frontiers = [CFrontier(Type.fromjson(f["request"]),
                           [(parse(e["program"]), e["logLikelihood"]) for e in f["programs"]])
                 for f in j["frontiers"]]
    aic = j["aic"]
    if aic > 500:
        g2, f2 = g, frontiers
    else:
        g2, f2 = compression_loop(g, frontiers, nc=j.get("CPUs", 1), structurePenalty=j["structurePenalty"],
                                  inline=j.get("inline", True), aic=aic, topK=j["topK"],
                                  pseudoCounts=j["pseudoCounts"], arity=j["arity"], bs=j["bs"], topI=j["topI"],
                                  iterations=j.get("iterations", 1000), log=log)
    return {"DSL": {"logVariable": g2.logVariable,
                    "productions": [{"expression": str(p), "logProbability": l} for p, _, l in g2.library]},
            "frontiers": [{"request": f.request.json(),
                           "programs": [{"program": str(p), "logLikelihood": ll} for p, ll in f.programs]}
                          for f in f2]}
