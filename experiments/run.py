"""v2: the full 2x2 ablation on the 2x2x2 (DSL v1, permutation/symmetry primitives).

    arms: --recognition on/off  x  library learning on/off (--no-compress)

Compute is matched: per-task wake (recognition) gets --per-task seconds per task,
shared wake (no recognition) gets --per-task * n_tasks seconds in total.
A held-out test set (fresh states, same stages) is solved once at the end.

    python experiments/run.py --recognition --out results/v2_nn_lib_s0.json
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "pydreamcoder", "src"))
sys.setrecursionlimit(10000)

from pydreamcoder.dreamcoder import run  # noqa: E402
from pydreamcoder.program import Invented  # noqa: E402
from rubik import perm, sample  # noqa: E402
from rubik.dsl import primitives, to_moves  # noqa: E402
from rubik.tasks import N_FEATURES, full_tasks, make_dream, stage_tasks, task_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["stages", "full"], default="stages")
ap.add_argument("--dsl", choices=["ruf", "r"], default="ruf")
ap.add_argument("--iterations", type=int, default=8)
ap.add_argument("--per-task", type=float, default=1.0, help="wake seconds per task (compute-matched)")
ap.add_argument("--per-stage", type=int, default=24)
ap.add_argument("--test-per-stage", type=int, default=12)
ap.add_argument("--max-len", type=int, default=8)
ap.add_argument("--per-length", type=int, default=6)
ap.add_argument("--frontier", type=int, default=3)
ap.add_argument("--max-new", type=int, default=3)
ap.add_argument("--structure-penalty", type=float, default=1.0)
ap.add_argument("--no-compress", action="store_true")
ap.add_argument("--recognition", action="store_true")
ap.add_argument("--n-dreams", type=int, default=2000)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

assert perm.self_test() and sample.self_test(50)
if a.kind == "stages":
    tasks = stage_tasks(a.per_stage, a.seed)
    test = stage_tasks(a.test_per_stage, 1000 + a.seed)
    stage_ids = [0, 1, 2, 3, 4]
else:
    tasks = full_tasks(range(1, a.max_len + 1), a.per_length, a.seed)
    test = full_tasks(range(1, a.max_len + 1), max(1, a.per_length // 2), 1000 + a.seed)
    stage_ids = [4]
for t in test:
    t.name = "test:" + t.name
rec = None
if a.recognition:
    rec = dict(features=task_features, dream=make_dream(stage_ids), n_features=N_FEATURES,
               n_dreams=a.n_dreams, steps=a.steps, seed=a.seed)
print(f"{len(tasks)} train / {len(test)} test tasks ({a.kind}), DSL {a.dsl}, "
      f"recognition={a.recognition}, library={not a.no_compress}", flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
g, hist, fr = run(primitives(a.dsl), tasks, iterations=a.iterations, per_task_timeout=a.per_task,
                  compress_on=not a.no_compress, structure_penalty=a.structure_penalty,
                  frontier_size=a.frontier, max_new=a.max_new, out=a.out, log=lambda s: print(s, flush=True),
                  recognition=rec, test_tasks=test)
tf = run.test_frontiers


def group(n):
    n = n.split(":")[-1]
    return n.split("#")[0] if a.kind == "stages" else n.split("#")[0].split("|")[1]


tot = collections.Counter(group(t.name) for t in tasks)
ok = collections.Counter(group(k) for k in fr)
ttot = collections.Counter(group(t.name) for t in test)
tok = collections.Counter(group(k) for k, v in tf.items() if v)
print("\n=== solved per group (train / test)")
for k in tot:
    print(f"  {k:10s} {ok[k]}/{tot[k]}   {tok[k]}/{ttot[k]}")
lib = []
for _, _, p in g.productions:
    if isinstance(p, Invented):
        lib.append({"program": str(p), "type": str(p.tp), "moves": to_moves(p) if not p.tp.is_arrow() else ""})
if a.out:
    d = json.load(open(a.out))
    d["solutions_moves"] = {k: to_moves(v[1][0]) for k, v in fr.items()}
    d["test_moves"] = {k: to_moves(v[0]) for k, v in tf.items() if v}
    d["per_group"] = {k: [ok[k], tot[k]] for k in tot}
    d["test_per_group"] = {k: [tok[k], ttot[k]] for k in ttot}
    d["library_readable"] = lib
    d["args"] = vars(a)
    # search policy: probability of each production at the root of a program (normalised
    # over all productions), averaged per stage. Recognition arms: task-conditioned;
    # plain arms: the single generative grammar used for every task.
    import math
    def dist(gr):
        ws = [(l, str(pp)) for l, _, pp in gr.productions]
        z = math.log(sum(math.exp(l) for l, _ in ws))
        return {k: math.exp(l - z) for l, k in ws}
    pol = {}
    for t in tasks + test:
        gr = run.model.grammar_for(g, task_features(t)) if run.model is not None and run.model.vocab else g
        st = t.meta["stage"]
        acc = pol.setdefault(st, {"n": 0, "p": {}})
        acc["n"] += 1
        for k, v in dist(gr).items():
            acc["p"][k] = acc["p"].get(k, 0.0) + v
    d["policy"] = {st: sorted(((k, v / acc["n"]) for k, v in acc["p"].items()), key=lambda x: -x[1])[:12]
                   for st, acc in pol.items()}
    json.dump(d, open(a.out, "w"), indent=1)
