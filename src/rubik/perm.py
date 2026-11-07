"""Permutation view of the 2x2x2 (DSL v1).

An *alg* is an element of the cube group, represented as a sticker permutation
(24-tuple, convention of cube2: new_state[i] = old_state[perm[i]]), paired with a
human-readable move string used only for display.

Invariance: the whole-cube symmetries (24 rotations, x 2 with the L/R mirror = 48)
act on algs by conjugation, g . A = g^-1 A g. "A seen from another side" is
therefore one primitive application instead of a separate routine to discover.

States are compared *up to whole-cube orientation*: before any goal check the
cube is re-oriented so that the DLB cubie sits in its home slot with home
orientation. On a 2x2x2 this makes L == R + rotation etc., so algs may use any face.
"""
from __future__ import annotations

import itertools

from . import cube2

ID = tuple(range(24))
N = 24


def compose(p, q):
    """p then q"""
    return tuple(p[i] for i in q)


def inverse(p):
    r = [0] * N
    for i, j in enumerate(p):
        r[j] = i
    return tuple(r)


# ---------------------------------------------------------------- geometry
def _mat_perm(m):
    """Sticker permutation induced by an orthogonal 3x3 integer matrix (all stickers move)."""
    perm = [0] * N
    for src, (pos, n) in enumerate(cube2.STICKERS):
        perm[cube2.INDEX[(cube2._mv(m, pos), cube2._mv(m, n))]] = src
    return tuple(perm)


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


_ROT_X = cube2._rot(0, 1)   # x  (like R)
_ROT_Y = cube2._rot(1, 1)   # y  (like U)
_ROT_Z = cube2._rot(2, 1)   # z  (like F)
_MIRROR = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]   # left-right reflection

ROT = {"x": _mat_perm(_ROT_X), "y": _mat_perm(_ROT_Y), "z": _mat_perm(_ROT_Z)}
MIRROR = _mat_perm(_MIRROR)


def _closure(gens):
    seen = {ID}
    frontier = [ID]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                b = compose(a, g)
                if b not in seen:
                    seen.add(b)
                    nxt.append(b)
        frontier = nxt
    return seen


ROTATIONS = sorted(_closure(list(ROT.values())))          # 24
assert len(ROTATIONS) == 24
SYMMETRIES = sorted(_closure(list(ROT.values()) + [MIRROR]))  # 48
assert len(SYMMETRIES) == 48

# all 18 face turns (6 faces) as permutations, for naming / display
FACE_TURNS = {}
for f in "RLUDFB":
    b = cube2.BASE[f]
    FACE_TURNS[f] = b
    FACE_TURNS[f + "2"] = compose(b, b)
    FACE_TURNS[f + "'"] = compose(compose(b, b), b)
NAME_OF = {p: m for m, p in FACE_TURNS.items()}


def conj(g, a):
    """g^-1 a g : 'do a, as seen after the whole-cube transform g'."""
    return compose(compose(inverse(g), a), g)


# ------------------------------------------------------- orientation quotient
_DLB = (-1, -1, -1)
_DLB_STICKERS = [cube2.INDEX[(p, n)] for (p, n) in cube2.STICKERS if p == _DLB]


def normalize(state):
    """Re-orient the whole cube so the DLB cubie is home (unique rotation)."""
    for r in ROTATIONS:
        s = compose(state, r)
        if all(s[i] == i for i in _DLB_STICKERS):
            return s
    raise AssertionError("no rotation fixes DLB")


# ----------------------------------------------------------- symbolic moves
def _face_map(g):
    """How conjugation by g renames each face turn."""
    return {m: NAME_OF[conj(g, p)] for m, p in FACE_TURNS.items()}


SYM_NAMES = {"x": ROT["x"], "y": ROT["y"], "z": ROT["z"], "mirror": MIRROR}
RENAME = {k: _face_map(g) for k, g in SYM_NAMES.items()}
_AMT = {"": 1, "2": 2, "'": 3}
_SUF = {1: "", 2: "2", 3: "'"}


def simplify(moves):
    """Merge consecutive turns of the same face (R R -> R2, R R' -> nothing)."""
    out = []
    for m in moves:
        if out and out[-1][0] == m[0]:
            k = (_AMT[out[-1][1:]] + _AMT[m[1:]]) % 4
            out.pop()
            if k:
                out.append(m[0] + _SUF[k])
        else:
            out.append(m)
    return tuple(out)


def inv_moves(ms):
    return tuple(m[0] + {"": "'", "'": "", "2": "2"}[m[1:]] for m in reversed(ms))


# ---------------------------------------------------------------- alg values
class Alg(tuple):
    """(perm, moves). Equality/hash by perm only."""
    __slots__ = ()

    def __new__(cls, perm, moves):
        return super().__new__(cls, (perm, moves))

    perm = property(lambda s: s[0])
    moves = property(lambda s: s[1])

    def __eq__(self, o):
        return isinstance(o, Alg) and self[0] == o[0]

    def __hash__(self):
        return hash(self[0])

    def __repr__(self):
        return " ".join(self[1]) or "(id)"


def move(m):
    return Alg(FACE_TURNS[m], (m,))


def seq(a, b):
    return Alg(compose(a.perm, b.perm), simplify(a.moves + b.moves))


def inv(a):
    return Alg(inverse(a.perm), inv_moves(a.moves))


def setup(a, b):
    """a b a^-1  (conjugate: setup, alg, undo setup)"""
    return seq(seq(a, b), inv(a))


def comm(a, b):
    """a b a^-1 b^-1"""
    return seq(seq(a, b), seq(inv(a), inv(b)))


def sym(name, a):
    g = SYM_NAMES[name]
    return Alg(conj(g, a.perm), simplify(tuple(RENAME[name][m] for m in a.moves)))


def apply_to(state, a):
    return normalize(compose(state, a.perm))


def self_test():
    # rotation renaming is geometrically right: y takes R to a side face, mirror R -> L'
    assert RENAME["mirror"]["R"] == "L'", RENAME["mirror"]["R"]
    assert RENAME["y"]["U"] == "U"
    # sexy move has order 6
    s = comm(move("R"), move("U"))
    p = ID
    for k in range(1, 7):
        p = compose(p, s.perm)
    assert p == ID
    # normalization: L == R x' on a 2x2 => L then R' is a pure rotation => normalizes to solved
    assert apply_to(ID, seq(move("L"), move("R'"))) == ID
    return True
