"""2x2x2 cube domain for the official DreamCoder frontend (vendored in pydreamcoder/ec).

Same DSL, tasks and features as v3/v4 (see dsl.py / tasks.py), expressed with the official classes:
  * primitives: dreamcoder.program.Primitive over base type `alg`;
  * tasks: CubeTask(dreamcoder.task.Task), all-or-nothing likelihood (goal reached after applying the program);
  * recognition: CubeFeatureExtractor (24 sticker slots x 6 colours one-hot + stage id) with the official
    RecognitionModel on top; Helmholtz "dreams" via taskOfProgram (program p -> goal state G -> start G.p^-1).
Importable under PyPy (no torch at module level)."""
import random

from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import baseType, arrow

from . import cube2, perm as P, sample

talg = baseType("alg")
NEG_INF = float("-inf")

DRF, DRB, DLF = (1, -1, 1), (1, -1, -1), (-1, -1, 1)
STAGES = [
    ("1-corner", [], lambda s: cube2.corner_solved(s, DRF)),
    ("2-corners", [DRF], lambda s: cube2.corner_solved(s, DRF) and cube2.corner_solved(s, DRB)),
    ("3-Dlayer", [DRF, DRB], cube2.d_layer_solved),
    ("4-Uorient", [DRF, DRB, DLF], lambda s: cube2.d_layer_solved(s) and cube2.u_face_oriented(s)),
    ("5-solved", [DRF, DRB, DLF], cube2.solved),
]
_GOAL_FIXED = [[DRF], [DRF, DRB], [DRF, DRB, DLF], [DRF, DRB, DLF], None]


def _c2(f):
    return lambda a: lambda b: f(a, b)


def _sym(s):
    return lambda a: P.sym(s, a)


_REGISTERED = {}


def primitives(variant="ruf"):
    """Registers (once) and returns the cube primitives, in the same order as dsl.primitives."""
    if variant in _REGISTERED:
        return _REGISTERED[variant]
    base = {"ruf": "RUF", "r": "R"}[variant]
    ps = [Primitive(m, talg, P.move(m)) for m in base]
    ps += [Primitive("seq", arrow(talg, talg, talg), _c2(P.seq)),
           Primitive("inv", arrow(talg, talg), P.inv),
           Primitive("setup", arrow(talg, talg, talg), _c2(P.setup)),
           Primitive("comm", arrow(talg, talg, talg), _c2(P.comm))]
    ps += [Primitive(s, arrow(talg, talg), _sym(s)) for s in ("x", "y", "z", "mirror")]
    _REGISTERED[variant] = ps
    return ps


def register_all():
    """Entry point for the PyPy solver/compressor processes (PYEC_PRIMITIVE_MODULES=rubik.ec_domain:register_all)."""
    primitives("ruf")
    register_helmholtz()


def cube_helmholtz_hash(timeout, request, inputs):
    """Special Helmholtz behaviour for --biasOptimal (the cube analogue of the official tower/LOGO hashes in
    Dreaming.ml): a program's behaviour is the permutation it applies; programs with the same permutation
    share one dream frontier (exact sticker permutation, not reduced modulo whole-cube rotation, like the
official hashes which use raw outputs). The identity has no use as a dream and is dropped."""
    from pyec.helmholtz import run_with_timeout

    def h(p):
        ok, a = run_with_timeout(timeout, lambda: p.evaluate([]))
        if not ok or a is None:
            return None
        key = tuple(a.perm)
        if key == P.ID:
            return None
        return (key,), 0.
    return h


def register_helmholtz():
    try:
        from pyec.helmholtz import register_special_helmholtz
    except ImportError:
        return
    register_special_helmholtz("cube", cube_helmholtz_hash)


class CubeTask(Task):
    def __init__(self, name, start, stage_idx):
        super(CubeTask, self).__init__(name, talg, [], features=features(start, stage_idx), cache=False)
        self.start = start
        self.stage_idx = stage_idx
        self.stage = STAGES[stage_idx][0]

    def goal(self, s):
        return STAGES[self.stage_idx][2](s)

    def logLikelihood(self, e, timeout=None):
        try:
            a = e.evaluate([])
        except Exception:
            return NEG_INF
        return 0. if self.goal(P.apply_to(self.start, a)) else NEG_INF

    def __getstate__(self):
        d = dict(self.__dict__)
        return d


def stage_tasks(per_stage=24, seed=0, exclude=()):
    """Identical sampling to tasks.stage_tasks (same RNG calls) -> same task sets as v3/v4."""
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
            out.append(CubeTask("%s#%d" % (name, n), s, i))
            n += 1
    return out


def random_states(n, seed, exclude=()):
    rng = random.Random(seed)
    exclude, out = set(exclude), []
    while len(out) < n:
        s = sample.sample([], rng)
        if s not in exclude and not cube2.solved(s):
            exclude.add(s)
            out.append(s)
    return out


def full_random_tasks(n=120, seed=0, exclude=()):
    return [CubeTask("full#%d" % j, s, 4) for j, s in enumerate(random_states(n, seed, exclude))]


# ---------------------------------------------------------------- recognition
N_FEATURES = 24 * 6 + len(STAGES)
_FACE_IDX = {f: i for i, f in enumerate("RLUDFB")}


def features(start, stage_idx):
    f = [0.0] * N_FEATURES
    for slot, home in enumerate(start):
        f[slot * 6 + _FACE_IDX[cube2.FACE_OF[home]]] = 1.0
    f[24 * 6 + stage_idx] = 1.0
    return f


DREAM_STAGES = [0, 1, 2, 3, 4]  # set by the experiment script (full-task curriculum: [4])


def dream_task(p, rng=random):
    """Helmholtz fantasy: program p -> random goal-satisfying end state G -> task starting at G.p^-1."""
    try:
        a = p.evaluate([])
    except Exception:
        return None
    k = rng.choice(DREAM_STAGES)
    goal = STAGES[k][2]
    G = cube2.SOLVED if _GOAL_FIXED[k] is None else sample.sample_with(_GOAL_FIXED[k], rng, goal)
    s = P.compose(G, P.inverse(a.perm))
    if P.normalize(s) != s or goal(s) or not goal(P.apply_to(s, a)):
        return None
    return CubeTask("dream", s, k)


def make_feature_extractor():
    import torch.nn as nn
    from dreamcoder.recognition import variable

    register_helmholtz()

    class CubeFeatureExtractor(nn.Module):
        special = "cube"  # pyec.helmholtz behaviour hash (only used with --biasOptimal)

        def __init__(self, tasks, testingTasks=[], cuda=False):
            super(CubeFeatureExtractor, self).__init__()
            self.outputDimensionality = N_FEATURES
            self.recomputeTasks = True  # fantasies are stochastic (random goal state)
            self.use_cuda = cuda

        def featuresOfTask(self, t):
            return variable(features(t.start, t.stage_idx), cuda=self.use_cuda).float()

        def featuresOfTasks(self, ts):
            return variable([features(t.start, t.stage_idx) for t in ts], cuda=self.use_cuda).float()

        def taskOfProgram(self, p, tp):
            return dream_task(p)

    return CubeFeatureExtractor
