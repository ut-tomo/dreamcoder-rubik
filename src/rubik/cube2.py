"""2x2x2 cube as a permutation of 24 stickers, built from 3D geometry.

A sticker is (cubie position in {-1,1}^3, outward normal). A face turn rotates
every sticker whose position lies in the turning layer by -90 deg about the
face axis (clockwise when looking at the face). x = R, y = U, z = F.
The DLB corner (-1,-1,-1) is never moved by R/U/F, so the cube needs no
re-orientation and "solved" means "identity permutation".
"""
from __future__ import annotations

import itertools
import random


FACES = {"R": (0, 1), "L": (0, -1), "U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1)}

STICKERS = []  # (pos, normal) tuples
for pos in itertools.product((-1, 1), repeat=3):
    for ax in range(3):
        n = [0, 0, 0]
        n[ax] = pos[ax]
        STICKERS.append((pos, tuple(n)))
INDEX = {s: i for i, s in enumerate(STICKERS)}
FACE_OF = []
for _, n in STICKERS:
    ax = next(i for i in range(3) if n[i])
    FACE_OF.append(next(f for f, (a, s) in FACES.items() if a == ax and s == n[ax]))


def _rot(axis, sign):
    """Matrix for clockwise quarter turn of the face (axis, sign) seen from outside."""
    c, s = 0, -1 * sign  # rotate by -90deg about +axis for +face; +90 about +axis for -face
    m = [[int(r == k) for k in range(3)] for r in range(3)]
    i, j = (axis + 1) % 3, (axis + 2) % 3  # right-handed cyclic order
    m[i][i], m[i][j], m[j][i], m[j][j] = c, -s, s, c
    return m


def _mv(m, v):
    return tuple(sum(m[r][k] * v[k] for k in range(3)) for r in range(3))


def _face_perm(face):
    axis, sign = FACES[face]
    m = _rot(axis, sign)
    perm = list(range(24))  # new_state[dst] = old_state[src]
    for src, (pos, n) in enumerate(STICKERS):
        if pos[axis] != sign:
            continue
        p2 = _mv(m, pos)
        n2 = _mv(m, n)
        perm[INDEX[(p2, n2)]] = src
    return tuple(perm)


def compose(p, q):
    """apply p then q"""
    return tuple(p[i] for i in q)


def power(p, k):
    r = tuple(range(24))
    for _ in range(k % 4):
        r = compose(r, p)
    return r


SOLVED = tuple(range(24))
BASE = {f: _face_perm(f) for f in FACES}
MOVES = {}
for f in "RUF":
    MOVES[f] = BASE[f]
    MOVES[f + "2"] = power(BASE[f], 2)
    MOVES[f + "'"] = power(BASE[f], 3)


def apply(state, move):
    perm = MOVES[move]
    return tuple(state[i] for i in perm)


def apply_seq(state, seq):
    for m in seq:
        state = apply(state, m)
    return state


def parse(s):
    return s.split()


def inverse_seq(seq):
    inv = {"": "'", "'": "", "2": "2"}
    return [m[0] + inv[m[1:]] for m in reversed(seq)]


def random_scramble(k, rng=random):
    """k moves, no two consecutive moves on the same face."""
    seq, last = [], None
    while len(seq) < k:
        m = rng.choice(list(MOVES))
        if m[0] == last:
            continue
        seq.append(m)
        last = m[0]
    return seq


# ------------------------------------------------------------------ predicates
CORNERS = sorted({pos for pos, _ in STICKERS})


def corner_solved(state, pos):
    return all(state[INDEX[(pos, n)]] == INDEX[(pos, n)] for (p, n) in STICKERS if p == pos)


D_CORNERS = [p for p in CORNERS if p[1] == -1]


def d_layer_solved(state):
    return all(corner_solved(state, p) for p in D_CORNERS)


def u_face_oriented(state):
    """All U-layer stickers facing up show the U colour."""
    return all(FACE_OF[state[i]] == "U" for i, (p, n) in enumerate(STICKERS) if n == (0, 1, 0))


def solved(state):
    return state == SOLVED


def n_d_corners_solved(state):
    return sum(corner_solved(state, p) for p in D_CORNERS)
