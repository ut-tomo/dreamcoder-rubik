"""DSL v1: algs are group elements (permutations), with group/symmetry operators.

Type `alg`. A program is a closed term of type alg; running it means applying
that permutation to the task's start state (then re-orienting the whole cube).

  base turns   R, U, F            : alg          (variant "ruf")
               R                  : alg          (variant "r": every other face
                                                  turn must come from symmetry)
  group ops    seq   : alg -> alg -> alg          a then b
               inv   : alg -> alg                 a^-1
               setup : alg -> alg -> alg          a b a^-1        (conjugate)
               comm  : alg -> alg -> alg          a b a^-1 b^-1   (commutator)
  symmetry     x, y, z, mirror : alg -> alg       g^-1 a g  (same alg, other side
                                                  / mirrored)
Nothing about cube colours, pieces or human recognition is built in.
"""
from pydreamcoder.program import Primitive
from pydreamcoder.types import TCon, arrow

from . import perm as P

talg = TCon("alg")
c2 = lambda f: (lambda a: lambda b: f(a, b))


def primitives(variant="ruf"):
    base = {"ruf": "RUF", "r": "R"}[variant]
    ps = [Primitive(m, talg, P.move(m)) for m in base]
    ps += [
        Primitive("seq", arrow(talg, talg, talg), c2(P.seq)),
        Primitive("inv", arrow(talg, talg), P.inv),
        Primitive("setup", arrow(talg, talg, talg), c2(P.setup)),
        Primitive("comm", arrow(talg, talg, talg), c2(P.comm)),
    ]
    ps += [Primitive(s, arrow(talg, talg), (lambda s: lambda a: P.sym(s, a))(s))
           for s in ("x", "y", "z", "mirror")]
    return ps


def to_moves(program):
    """Move sequence a closed alg program performs (face names after symmetry)."""
    return " ".join(program.evaluate([]).moves) or "(id)"
