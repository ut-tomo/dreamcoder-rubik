"""List-processing domain from the DreamCoder paper (bootstrapTarget_extra primitives,
official tasks in dreamcoder-ec/data/list_tasks.json, copied to repro/data/)."""
import json
import math
import os

from pydreamcoder.dreamcoder import io_task
from pydreamcoder.program import Primitive
from pydreamcoder.types import arrow, t0, t1, tbool, tint, tlist

c = lambda f: (lambda a: lambda b: f(a, b))


def _fold(l):
    def g(x0):
        def h(f):
            acc = x0
            for x in reversed(l):
                acc = f(x)(acc)
            return acc
        return h
    return g


def _unfold(x):
    def a(stop):
        def b(f):
            def d(nxt):
                out, v = [], x
                while not stop(v):
                    out.append(f(v))
                    v = nxt(v)
                    if len(out) > 100:
                        raise ValueError("unfold too long")
                return out
            return d
        return b
    return a


def _is_prime(n):
    return n > 1 and all(n % k for k in range(2, int(math.isqrt(n)) + 1))


def primitives():
    return [
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), lambda f: lambda l: [f(x) for x in l]),
        Primitive("unfold", arrow(t0, arrow(t0, tbool), arrow(t0, t1), arrow(t0, t0), tlist(t1)), _unfold),
        Primitive("range", arrow(tint, tlist(tint)), lambda n: list(range(n)) if n < 100 else (_ for _ in ()).throw(ValueError())),
        Primitive("index", arrow(tint, tlist(t0), t0), c(lambda i, l: l[i] if 0 <= i < len(l) else (_ for _ in ()).throw(IndexError()))),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("if", arrow(tbool, t0, t0, t0), lambda b: lambda x: lambda y: x if b else y),
        Primitive("+", arrow(tint, tint, tint), c(lambda a, b: a + b)),
        Primitive("-", arrow(tint, tint, tint), c(lambda a, b: a - b)),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), c(lambda x, l: [x] + l)),
        Primitive("car", arrow(tlist(t0), t0), lambda l: l[0]),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), lambda l: l[1:] if l else (_ for _ in ()).throw(IndexError())),
        Primitive("empty?", arrow(tlist(t0), tbool), lambda l: not l),
        Primitive("0", tint, 0),
        Primitive("1", tint, 1),
        Primitive("*", arrow(tint, tint, tint), c(lambda a, b: a * b)),
        Primitive("mod", arrow(tint, tint, tint), c(lambda a, b: a % b)),
        Primitive("gt?", arrow(tint, tint, tbool), c(lambda a, b: a > b)),
        Primitive("eq?", arrow(tint, tint, tbool), c(lambda a, b: a == b)),
        Primitive("is-prime", arrow(tint, tbool), _is_prime),
        Primitive("is-square", arrow(tint, tbool), lambda n: n >= 0 and math.isqrt(n) ** 2 == n),
    ]


_T = {"list-of-int": tlist(tint), "int": tint, "bool": tbool}


def tasks(path=os.path.join(os.path.dirname(__file__), "data", "list_tasks.json")):
    out = []
    for d in json.load(open(path)):
        ti, to = d["type"]["input"], d["type"]["output"]
        if ti not in _T or to not in _T:
            continue
        ex = [((e["i"],), e["o"]) for e in d["examples"]]
        out.append(io_task(d["name"], arrow(_T[ti], _T[to]), ex))
    return out
