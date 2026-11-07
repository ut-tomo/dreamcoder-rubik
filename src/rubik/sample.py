"""Uniform sampling of 2x2x2 states with some corners held solved (cubie level).

Each corner slot lists its 3 stickers in right-handed order (n0 x n1 = n2).
Placing cubie c in slot p with twist t maps c's i-th sticker to p's (i+t)-th.
Reachable states (DLB fixed) = any permutation of the other 7 corners, with the
total twist (measured on the U/D sticker) = 0 mod 3. Checked against random
scrambles in self_test().
"""
from __future__ import annotations

import random

from . import cube2


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rh(pos):
    ns = []
    for ax in range(3):
        n = [0, 0, 0]
        n[ax] = pos[ax]
        ns.append(tuple(n))
    # put the y-axis (U/D) sticker first, then fix handedness
    a, b, c = ns[1], ns[0], ns[2]
    if _cross(a, b) != c:
        b, c = c, b
    return [cube2.INDEX[(pos, n)] for n in (a, b, c)]


SLOTS = {p: _rh(p) for p in cube2.CORNERS}
DLB = (-1, -1, -1)


def state_from(placement):
    """placement: slot -> (cubie_home_slot, twist). Missing slots are solved."""
    s = list(range(24))
    for slot, (cub, t) in placement.items():
        dst, src = SLOTS[slot], SLOTS[cub]
        for i in range(3):
            s[dst[(i + t) % 3]] = src[i]
    return tuple(s)


def twist_of(state, slot):
    """Twist of the cubie currently in slot (where its U/D sticker went)."""
    dst = SLOTS[slot]
    ud = {SLOTS[c][0] for c in SLOTS}
    for i in range(3):
        if state[dst[i]] in ud:
            return i
    raise AssertionError


def sample(fixed, rng, fixed_twist_free=()):
    """Random reachable state with the corners in `fixed` solved (DLB always fixed).

    fixed_twist_free: slots whose cubie stays in place but may be twisted
    (unused for now; kept for OLL-style variants)."""
    fixed = set(fixed) | {DLB}
    free = [p for p in cube2.CORNERS if p not in fixed]
    while True:
        cubies = free[:]
        rng.shuffle(cubies)
        tw = [rng.randrange(3) for _ in free]
        tw[-1] = (-sum(tw[:-1])) % 3
        s = state_from({p: (c, t) for p, c, t in zip(free, cubies, tw)})
        if s != cube2.SOLVED or not free:
            return s


def sample_with(fixed, rng, cond):
    """Like sample(), but also require cond(state) (rejection sampling)."""
    while True:
        s = sample(fixed, rng)
        if cond(s):
            return s


def self_test(n=300, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        s = cube2.apply_seq(cube2.SOLVED, cube2.random_scramble(12, rng))
        assert sum(twist_of(s, p) for p in cube2.CORNERS) % 3 == 0
        # every scramble is expressible as a placement
        pl = {}
        for p in cube2.CORNERS:
            stick = s[SLOTS[p][0]]
            home = next(c for c in SLOTS if stick in SLOTS[c])
            pl[p] = (home, (-SLOTS[home].index(stick)) % 3)
        assert state_from(pl) == s, "sticker ordering inconsistent with move geometry"
    return True
