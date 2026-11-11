"""v3: 4 arms x 2 training strategies on the 2x2x2, scored end-to-end on random cubes.

    arms:       --recognition on/off  x  library learning on/off (--no-compress)
    strategies: --kind stages  (curriculum of 5 chained subtasks)
                --kind full    (full task only: random cube -> solved)

Main metric: E2E_N uniformly random cubes (fixed set, excluded from every training
set) must end up solved. "stages" chains its stage solvers (budget per stage = T);
"full" searches for one solving program (budget 5T), so both get 5T per cube.

Compute is matched: per-task wake (recognition) gets --per-task seconds per task,
shared wake (no recognition) gets --per-task * n_tasks seconds in total.

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

from pydreamcoder.dreamcoder import run, wake, wake_per_task  # noqa: E402
from pydreamcoder.program import Invented  # noqa: E402
from rubik import perm, sample  # noqa: E402
from rubik.dsl import primitives, to_moves  # noqa: E402
from rubik import cube2  # noqa: E402
from rubik.tasks import (N_FEATURES, STAGES, full_random_tasks, make_dream, random_states,  # noqa: E402
                         stage_task, stage_tasks, task_features)

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["stages", "full"], default="stages")
ap.add_argument("--dsl", choices=["ruf", "r"], default="ruf")
ap.add_argument("--iterations", type=int, default=8)
ap.add_argument("--per-task", type=float, default=1.0, help="wake seconds per task (compute-matched)")
ap.add_argument("--per-stage", type=int, default=24)
ap.add_argument("--n-full", type=int, default=120)
ap.add_argument("--e2e-n", type=int, default=100)
ap.add_argument("--e2e-seed", type=int, default=9999)
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
E2E = random_states(a.e2e_n, a.e2e_seed)  # same cubes for every arm / strategy / seed
if a.kind == "stages":
    tasks = stage_tasks(a.per_stage, a.seed, exclude=E2E)
    stage_ids = [0, 1, 2, 3, 4]
else:
    tasks = full_random_tasks(a.n_full, a.seed, exclude=E2E)
    stage_ids = [4]
rec = None
if a.recognition:
    rec = dict(features=task_features, dream=make_dream(stage_ids), n_features=N_FEATURES,
               n_dreams=a.n_dreams, steps=a.steps, seed=a.seed)
print(f"{len(tasks)} train tasks, {a.e2e_n} e2e cubes ({a.kind}), DSL {a.dsl}, "
      f"recognition={a.recognition}, library={not a.no_compress}", flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
g, hist, fr = run(primitives(a.dsl), tasks, iterations=a.iterations, per_task_timeout=a.per_task,
                  compress_on=not a.no_compress, structure_penalty=a.structure_penalty,
                  frontier_size=a.frontier, max_new=a.max_new, out=a.out, log=lambda s: print(s, flush=True),
                  recognition=rec)


def export_model(path):
    """Final model for the in-browser search: grammar (exact log-weights) + recognition MLP."""
    m = {"grammar": {"log_variable": g.log_variable,
                      "productions": [[l, str(t), str(p)] for l, t, p in g.productions]},
         "recognition": None}
    if run.model is not None and run.model.vocab is not None:
        enc, head = run.model.encoder, run.model.head
        lin = [enc[0], enc[2], head]
        m["recognition"] = {"vocab": run.model.vocab, "activation": "relu",
                            "layers": [{"W": l.weight.detach().cpu().tolist(),
                                        "b": l.bias.detach().cpu().tolist()} for l in lin]}
    json.dump(m, open(path, "w"))


if a.out:
    export_model(a.out[:-5] + ".model.json")




def solve_batch(tasks_, budget):
    """Final model, no learning: best program per task within `budget` s per task."""
    if run.model is not None and run.model.vocab is not None:
        fr = wake_per_task(g, tasks_, lambda t: run.model.grammar_for(g, task_features(t)), budget,
                           frontier_size=1, log=lambda s: None)
    else:
        fr = wake(g, tasks_, budget * len(tasks_), frontier_size=1, log=lambda s: None)
    return {k: v[0] for k, v in fr.items() if v}


print(f"\n=== end-to-end on {len(E2E)} random cubes ({a.kind})", flush=True)
state = list(E2E)
alive = set(range(len(E2E)))
moves = {i: [] for i in alive}
reached = collections.Counter()  # how many cubes got through each stage
if a.kind == "stages":
    for k, (name, _, goal) in enumerate(STAGES):
        todo = [i for i in sorted(alive) if not goal(state[i])]
        sol = solve_batch([stage_task(k, state[i], f"e2e{i}") for i in todo], a.per_task)
        for i in todo:
            p = sol.get(f"e2e{i}")
            if p is None:
                alive.discard(i)
                continue
            state[i] = perm.apply_to(state[i], p.evaluate([]))
            moves[i].append(to_moves(p))
        reached[name] = len(alive)
        print(f"  after {name}: {len(alive)}/{len(E2E)}", flush=True)
else:
    sol = solve_batch([stage_task(4, s, f"e2e{i}") for i, s in enumerate(E2E)], 5 * a.per_task)
    for i in list(alive):
        p = sol.get(f"e2e{i}")
        if p is None:
            alive.discard(i)
            continue
        state[i] = perm.apply_to(state[i], p.evaluate([]))
        moves[i].append(to_moves(p))
solved_e2e = [i for i in alive if cube2.solved(state[i])]
assert len(solved_e2e) == len(alive)
lens = [len(" ".join(moves[i]).split()) for i in solved_e2e]
print(f"  E2E SOLVED {len(solved_e2e)}/{len(E2E)}  mean length "
      f"{sum(lens) / max(1, len(lens)):.1f}", flush=True)
tot = collections.Counter(t.meta["stage"] for t in tasks)
ok = collections.Counter(t.meta["stage"] for t in tasks if t.name in fr)
lib = []
for _, _, p in g.productions:
    if isinstance(p, Invented):
        lib.append({"program": str(p), "type": str(p.tp), "moves": to_moves(p) if not p.tp.is_arrow() else ""})
if a.out:
    d = json.load(open(a.out))
    d["solutions_moves"] = {k: to_moves(v[1][0]) for k, v in fr.items()}
    d["e2e"] = {"n": len(E2E), "solved": len(solved_e2e), "mean_len": sum(lens) / max(1, len(lens)),
                "reached": dict(reached), "moves": {str(i): moves[i] for i in solved_e2e}}
    d["per_group"] = {k: [ok[k], tot[k]] for k in tot}
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
    for t in tasks:
        gr = run.model.grammar_for(g, task_features(t)) if run.model is not None and run.model.vocab else g
        st = t.meta["stage"]
        acc = pol.setdefault(st, {"n": 0, "p": {}})
        acc["n"] += 1
        for k, v in dist(gr).items():
            acc["p"][k] = acc["p"].get(k, 0.0) + v
    d["policy"] = {st: sorted(((k, v / acc["n"]) for k, v in acc["p"].items()), key=lambda x: -x[1])[:12]
                   for st, acc in pol.items()}
    json.dump(d, open(a.out, "w"), indent=1)
