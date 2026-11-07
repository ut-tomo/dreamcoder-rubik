"""Hindley-Milner style types (mirrors dreamcoder/type.py, simplified)."""
from __future__ import annotations


class UnificationFailure(Exception):
    pass


class Type:
    def is_arrow(self):
        return isinstance(self, TCon) and self.name == "->"

    def arguments(self):
        t, out = self, []
        while t.is_arrow():
            out.append(t.args[0])
            t = t.args[1]
        return out

    def returns(self):
        t = self
        while t.is_arrow():
            t = t.args[1]
        return t


class TCon(Type):
    __slots__ = ("name", "args")

    def __init__(self, name, args=()):
        self.name, self.args = name, tuple(args)

    def __eq__(self, o):
        return isinstance(o, TCon) and o.name == self.name and o.args == self.args

    def __hash__(self):
        return hash((self.name, self.args))

    def __repr__(self):
        if self.name == "->":
            a = self.args[0]
            sa = f"({a})" if a.is_arrow() else str(a)
            return f"{sa} -> {self.args[1]}"
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"

    def free(self):
        s = set()
        for a in self.args:
            s |= a.free()
        return s


class TVar(Type):
    __slots__ = ("i",)

    def __init__(self, i):
        self.i = i

    def __eq__(self, o):
        return isinstance(o, TVar) and o.i == self.i

    def __hash__(self):
        return hash(("v", self.i))

    def __repr__(self):
        return f"t{self.i}"

    def free(self):
        return {self.i}


def arrow(*ts):
    if len(ts) == 1:
        return ts[0]
    return TCon("->", (ts[0], arrow(*ts[1:])))


tint = TCon("int")
tbool = TCon("bool")


def tlist(t):
    return TCon("list", (t,))


t0, t1, t2 = TVar(0), TVar(1), TVar(2)


class Context:
    """Immutable-ish substitution. Methods return new contexts."""
    __slots__ = ("next", "sub")

    def __init__(self, next_=0, sub=None):
        self.next = next_
        self.sub = sub or {}

    def fresh(self):
        return Context(self.next + 1, self.sub), TVar(self.next)

    def apply(self, t):
        if isinstance(t, TVar):
            if t.i in self.sub:
                r = self.apply(self.sub[t.i])
                return r
            return t
        if not t.args:
            return t
        return TCon(t.name, [self.apply(a) for a in t.args])

    def instantiate(self, t):
        """Rename all type variables of t to fresh ones."""
        m = {}
        ctx = self

        def go(t):
            nonlocal ctx
            if isinstance(t, TVar):
                if t.i not in m:
                    ctx, v = ctx.fresh()
                    m[t.i] = v
                return m[t.i]
            if not t.args:
                return t
            return TCon(t.name, [go(a) for a in t.args])

        r = go(t)
        return ctx, r

    def unify(self, a, b):
        a, b = self.apply(a), self.apply(b)
        if isinstance(a, TVar):
            if a == b:
                return self
            if a.i in b.free():
                raise UnificationFailure()
            s = dict(self.sub)
            s[a.i] = b
            return Context(self.next, s)
        if isinstance(b, TVar):
            return self.unify(b, a)
        if a.name != b.name or len(a.args) != len(b.args):
            raise UnificationFailure()
        ctx = self
        for x, y in zip(a.args, b.args):
            ctx = ctx.unify(x, y)
        return ctx


def canonical(t):
    """Rename variables to 0..n in order of appearance."""
    m = {}

    def go(t):
        if isinstance(t, TVar):
            if t.i not in m:
                m[t.i] = TVar(len(m))
            return m[t.i]
        return TCon(t.name, [go(a) for a in t.args]) if t.args else t

    return go(t)
