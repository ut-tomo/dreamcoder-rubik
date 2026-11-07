"""Lambda-calculus programs with de Bruijn indices (mirrors dreamcoder/program.py)."""
from __future__ import annotations

from .types import Context, arrow, canonical, UnificationFailure


class Program:
    def __str__(self):
        return self.show(False)

    __repr__ = __str__

    def __eq__(self, o):
        return type(o) is type(self) and str(o) == str(self)

    def __hash__(self):
        return hash(str(self))

    def is_application(self):
        return isinstance(self, Application)

    def apply_spine(self):
        """f x y z -> (f, [x, y, z])"""
        args, p = [], self
        while isinstance(p, Application):
            args.append(p.x)
            p = p.f
        return p, args[::-1]

    def infer_type(self):
        ctx, t = self.infer(Context(), [])
        return canonical(ctx.apply(t))


class Index(Program):
    __slots__ = ("i",)

    def __init__(self, i):
        self.i = i

    def show(self, paren):
        return f"${self.i}"

    def evaluate(self, env):
        return env[self.i]

    def infer(self, ctx, env):
        return ctx, env[self.i]

    def shift(self, d, c=0):
        if self.i >= c:
            if self.i + d < 0:
                raise ShiftFailure()
            return Index(self.i + d)
        return self

    def free_indices(self, depth=0):
        return {self.i - depth} if self.i >= depth else set()

    def size(self):
        return 1

    def subterms(self, depth=0):
        yield depth, self


class ShiftFailure(Exception):
    pass


class Abstraction(Program):
    __slots__ = ("body", "_s")

    def __init__(self, body):
        self.body = body
        self._s = None

    def show(self, paren):
        return f"(lambda {self.body.show(False)})"

    def evaluate(self, env):
        b = self.body
        return lambda x: b.evaluate([x] + env)

    def infer(self, ctx, env):
        ctx, a = ctx.fresh()
        ctx, r = self.body.infer(ctx, [a] + env)
        return ctx, arrow(a, r)

    def shift(self, d, c=0):
        return Abstraction(self.body.shift(d, c + 1))

    def free_indices(self, depth=0):
        return self.body.free_indices(depth + 1)

    def size(self):
        return self.body.size()

    def subterms(self, depth=0):
        yield depth, self
        yield from self.body.subterms(depth + 1)


class Application(Program):
    __slots__ = ("f", "x")

    def __init__(self, f, x):
        self.f, self.x = f, x

    def show(self, paren):
        f, args = self.apply_spine()
        s = " ".join([f.show(True)] + [a.show(True) for a in args])
        return f"({s})"

    def evaluate(self, env):
        return self.f.evaluate(env)(self.x.evaluate(env))

    def infer(self, ctx, env):
        ctx, ft = self.f.infer(ctx, env)
        ctx, xt = self.x.infer(ctx, env)
        ctx, r = ctx.fresh()
        ctx = ctx.unify(ft, arrow(xt, r))
        return ctx, r

    def shift(self, d, c=0):
        return Application(self.f.shift(d, c), self.x.shift(d, c))

    def free_indices(self, depth=0):
        return self.f.free_indices(depth) | self.x.free_indices(depth)

    def size(self):
        return self.f.size() + self.x.size()

    def subterms(self, depth=0):
        yield depth, self
        yield from self.f.subterms(depth)
        yield from self.x.subterms(depth)


class Primitive(Program):
    __slots__ = ("name", "tp", "value")

    def __init__(self, name, tp, value):
        self.name, self.tp, self.value = name, tp, value

    def show(self, paren):
        return self.name

    def evaluate(self, env):
        return self.value

    def infer(self, ctx, env):
        return ctx.instantiate(self.tp)

    def shift(self, d, c=0):
        return self

    def free_indices(self, depth=0):
        return set()

    def size(self):
        return 1

    def subterms(self, depth=0):
        yield depth, self


class Invented(Program):
    """A learned library routine: a closed program used like a primitive."""
    __slots__ = ("body", "tp", "_v")

    def __init__(self, body):
        self.body = body
        self.tp = body.infer_type()
        self._v = None

    def show(self, paren):
        return "#" + self.body.show(False)

    def evaluate(self, env):
        if self._v is None:
            self._v = self.body.evaluate([])
        return self._v

    def infer(self, ctx, env):
        return ctx.instantiate(self.tp)

    def shift(self, d, c=0):
        return self

    def free_indices(self, depth=0):
        return set()

    def size(self):
        return 1

    def subterms(self, depth=0):
        yield depth, self


def well_typed(p):
    try:
        p.infer_type()
        return True
    except UnificationFailure:
        return False
