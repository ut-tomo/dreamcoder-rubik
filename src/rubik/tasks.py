"""Task sets for DSL v1 (programs of type alg).

Chained subgoals ("stages"): each stage starts from a state in which the
previous stages are already done, and asks for the next one only.

  stage  start state (random, uniform over reachable states)    goal
  1      anything                                               DRF corner solved
  2      DRF solved                                             + DRB solved
  3      DRF, DRB solved                                        + whole D layer
  4      D layer solved                                         + U face one colour
  5      D layer solved, U face one colour                      solved

(DLB is the reference corner: the cube is re-oriented so it is always home.)

Full solve: scrambles of length 1..max_len (curriculum), goal = solved.
"""
import random

from pydreamcoder.dreamcoder import Task

from . import cube2, perm as P, sample
from .dsl import talg

DRF, DRB, DLF = (1, -1, 1), (1, -1, -1), (-1, -1, 1)

STAGES = [
    ("1-corner", [], lambda s: cube2.corner_solved(s, DRF)),
    ("2-corners", [DRF], lambda s: cube2.corner_solved(s, DRF) and cube2.corner_solved(s, DRB)),
    ("3-Dlayer", [DRF, DRB], cube2.d_layer_solved),
    ("4-Uorient", [DRF, DRB, DLF], lambda s: cube2.d_layer_solved(s) and cube2.u_face_oriented(s)),
    ("5-solved", [DRF, DRB, DLF], cube2.solved),
]


def _task(name, start, goal, meta):
    def check(a):
        return goal(P.apply_to(start, a))
    return Task(name, talg, check, meta=meta)


def stage_tasks(per_stage=24, seed=0, exclude=()):
    """per_stage distinct start states per stage, none of them in `exclude`.
    A stage whose state space is smaller than per_stage gets fewer tasks
    (stage 5 has only 23 possible starts)."""
    rng = random.Random(seed)
    exclude = set(exclude)
    out = []
    for i, (name, fixed, goal) in enumerate(STAGES):
        prev = STAGES[i - 1][2] if i else (lambda s: True)
        seen, n, misses = set(), 0, 0
        while n < per_stage and misses < 5000:
            s = sample.sample_with(fixed, rng, lambda s: prev(s) and not goal(s))
            if s in seen or s in exclude:
                misses += 1
                continue
            seen.add(s)
            out.append(_task(f"{name}#{n}", s, goal, {"stage": name, "stage_idx": i, "start": s}))
            n += 1
    return out


def random_states(n, seed, exclude=()):
    """n distinct, uniformly random, unsolved reachable states (DLB home)."""
    rng = random.Random(seed)
    exclude, out = set(exclude), []
    while len(out) < n:
        s = sample.sample([], rng)
        if s not in exclude and not cube2.solved(s):
            exclude.add(s)
            out.append(s)
    return out


def full_random_tasks(n=120, seed=0, exclude=()):
    """Full task only: uniformly random cubes, goal = solved."""
    return [_task(f"full#{j}", s, cube2.solved, {"stage": "full", "stage_idx": 4, "start": s})
            for j, s in enumerate(random_states(n, seed, exclude))]


def stage_task(k, start, name):
    name_k, _, goal = STAGES[k]
    return _task(name, start, goal, {"stage": name_k, "stage_idx": k, "start": start})


def full_tasks(lengths=range(1, 9), per_length=6, seed=0):
    rng = random.Random(seed)
    out = []
    for k in lengths:
        for j in range(per_length):
            scr = cube2.random_scramble(k, rng)
            s = cube2.apply_seq(cube2.SOLVED, scr)
            if cube2.solved(s):
                continue
            out.append(_task(f"full|k{k}#{j}|{' '.join(scr)}", s, cube2.solved,
                             {"k": k, "scramble": scr, "stage_idx": 4, "start": P.normalize(s)}))
    return out


# ---------------------------------------------------------- recognition hooks
N_FEATURES = 24 * 6 + len(STAGES)
_FACE_IDX = {f: i for i, f in enumerate("RLUDFB")}
_GOAL_FIXED = [[DRF], [DRF, DRB], [DRF, DRB, DLF], [DRF, DRB, DLF], None]


def features(start, stage_idx):
    """One-hot colour of each of the 24 sticker slots + one-hot stage (goal) id.
    Start states are already orientation-normalized (DLB home)."""
    f = [0.0] * N_FEATURES
    for slot, home in enumerate(start):
        f[slot * 6 + _FACE_IDX[cube2.FACE_OF[home]]] = 1.0
    f[24 * 6 + stage_idx] = 1.0
    return f


def task_features(t):
    return features(t.meta["start"], t.meta["stage_idx"])


def make_dream(stage_ids):
    """Fantasy tasks: program p -> pick a goal-satisfying end state G, start = G . p^-1."""
    def dream(p, rng):
        a = p.evaluate([])
        k = rng.choice(stage_ids)
        goal = STAGES[k][2]
        G = cube2.SOLVED if _GOAL_FIXED[k] is None else sample.sample_with(_GOAL_FIXED[k], rng, goal)
        s = P.compose(G, P.inverse(a.perm))
        if P.normalize(s) != s or goal(s) or not goal(P.apply_to(s, a)):
            return None
        return features(s, k)
    return dream
