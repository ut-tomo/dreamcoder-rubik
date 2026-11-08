"""v5: the 8 cube arms on the official DreamCoder frontend (validated dreamcoder-pyrepro/ec = ellisk42/ec @ cb0e63f, Python)
with the faithful pure-Python solver (pyfast) and compressor (pyvs).

    arms: recognition (contextual/bigram network, official) on/off  x  library learning (compression) on/off
    kind: stages (5 chained subtask types, 24 tasks each) | full (120 random cubes -> solved)

Hyperparameters = the paper's full-system configuration (ec official_experiments, "no batch" variant):
  --structurePenalty 1.5 --pseudoCounts 30 --aic 1 --topK 2 --arity 3 --maximumFrontier 5 --contextual
  --auxiliary -RS 5000 -R 3600 --helmholtzRatio 0.5, activation tanh, hidden [64].
Not used: --biasOptimal (needs the OCaml Helmholtz enumerator to execute domain programs; impossible for a
Python-only domain in official ec as well -> official random-Helmholtz path), taskBatchSize (all tasks every
iteration, as in the official "no_batch" runs). Enumeration timeout per task -t 5 (v3/v4 budget).

End-to-end evaluation: the same 100 random cubes (seed 9999, excluded from training) for every arm; held-out
enumeration with the final model (official testing protocol: one job per task, `-t` seconds each);
stages: 5 chained stage tasks with 5 s each; full: one 25 s search.
"""
import argparse, collections, json, math, os, random, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PD = os.path.abspath(os.environ.get("DREAMCODER_BACKEND_ROOT", os.path.join(ROOT, "..", "dreamcoder-pyrepro")))
if not os.path.isfile(os.path.join(PD, "pyrepro", "backend.py")):
    raise RuntimeError("DREAMCODER_BACKEND_ROOT must point to the validated dreamcoder-pyrepro checkout")
for p in (os.path.join(ROOT, "src"), PD, os.path.join(PD, "ec")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYREPRO_PRIMITIVE_MODULES", "rubik.ec_domain:register_all")
os.environ["PYREPRO_EXTRA_PATH"] = os.path.join(ROOT, "src")
sys.setrecursionlimit(20000)

from dreamcoder.dreamcoder import ecIterator  # noqa: E402
from dreamcoder.grammar import Grammar  # noqa: E402
from dreamcoder.enumeration import multicoreEnumeration  # noqa: E402
from dreamcoder.program import Invented  # noqa: E402
from rubik import ec_domain as D, perm as P, cube2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["stages", "full"], default="stages")
ap.add_argument("--iterations", type=int, default=20)
ap.add_argument("-t", "--enumerationTimeout", type=float, default=5)
ap.add_argument("-c", "--CPUs", type=int, default=4)
ap.add_argument("--per-stage", type=int, default=24)
ap.add_argument("--n-full", type=int, default=120)
ap.add_argument("--e2e-n", type=int, default=100)
ap.add_argument("--e2e-seed", type=int, default=9999)
ap.add_argument("--no-consolidation", action="store_true")
ap.add_argument("--no-recognition", action="store_true")
ap.add_argument("-RS", "--recognitionSteps", type=int, default=5000)
ap.add_argument("-R", "--recognitionTimeout", type=int, default=3600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--biasOptimal", action="store_true",
                help="bias-optimal recognition training with Helmholtz dreams from pyrepro.helmholtz")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.biasOptimal:
    os.environ["PYREPRO_HELMHOLTZ"] = "1"
else:
    os.environ.pop("PYREPRO_HELMHOLTZ", None)

def seed_everything(seed):
    """Python, NumPy and PyTorch RNGs (the official frontend only seeds its task batcher). Wall-clock
    enumeration budgets still make runs nondeterministic; seeds fix tasks, dreams' RNG and NN init."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))


seed_everything(a.seed)
assert P.self_test()
E2E = D.random_states(a.e2e_n, a.e2e_seed)
if a.kind == "stages":
    tasks = D.stage_tasks(a.per_stage, a.seed, exclude=E2E)
    D.DREAM_STAGES = [0, 1, 2, 3, 4]
else:
    tasks = D.full_random_tasks(a.n_full, a.seed, exclude=E2E)
    D.DREAM_STAGES = [4]
prims = D.primitives("ruf")
g0 = Grammar.uniform(prims)
rec = not a.no_recognition
lib = not a.no_consolidation
print("%d train tasks, %d e2e cubes (%s), recognition=%s, library=%s" % (len(tasks), a.e2e_n, a.kind, rec, lib),
      flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
ckdir = a.out[:-5] + "_ckpt"
os.makedirs(ckdir, exist_ok=True)

params = dict(cuda=False, solver="pyfast", compressor="pyvs", useRecognitionModel=rec, noConsolidation=not lib,
              contextual=rec, biasOptimal=rec and a.biasOptimal, auxiliaryLoss=rec, recognitionSteps=a.recognitionSteps,
              recognitionTimeout=a.recognitionTimeout, helmholtzRatio=0.5,
              featureExtractor=D.make_feature_extractor() if rec else None, activation="tanh",
              topK=2, maximumFrontier=5, pseudoCounts=30.0, aic=1.0, structurePenalty=1.5, arity=3,
              evaluationTimeout=None, enumerationTimeout=a.enumerationTimeout, testingTimeout=0,
              iterations=a.iterations, CPUs=a.CPUs, seed=a.seed, outputPrefix=os.path.join(ckdir, "cube"),
              storeTaskMetrics=False)
def write_manifest():
    import hashlib, platform, subprocess as sp
    commit = os.environ.get("SOURCE_COMMIT")  # the shared server copy is not a git checkout
    if not commit:
        try:
            commit = sp.run(["git", "-C", ROOT, "rev-parse", "HEAD"], stdout=sp.PIPE, stderr=sp.PIPE,
                            universal_newlines=True).stdout.strip() or None
        except Exception:
            commit = None
    m = {"args": vars(a), "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                      for k, v in params.items()},
         "train_tasks": [t.name for t in tasks],
         "train_tasks_sha1": hashlib.sha1("\n".join(D.state_key(t) for t in tasks).encode()).hexdigest()
         if hasattr(D, "state_key") else None,
         "e2e_sha1": hashlib.sha1(repr(E2E).encode()).hexdigest(),
         "git_commit": commit, "backend_root": PD,
         "backend_commit": sp.check_output(["git", "-C", PD, "rev-parse", "HEAD"]).decode().strip(), "host": platform.node(), "python": sys.version,
         "pypy": os.environ.get("PYREPRO_PYPY"), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(m, open(a.out[:-5] + ".manifest.json", "w"), indent=1)


write_manifest()
hist = []
t0 = time.time()
result = None
for j, result in enumerate(ecIterator(g0, tasks, testingTasks=[], **params)):
    g = result.grammars[-1]
    solved = [t.name for t, f in result.allFrontiers.items() if not f.empty]
    hist.append({"iteration": j + 1, "solved": len(solved), "n_tasks": len(tasks),
                 "library": [str(p) for p in g.primitives if p.isInvented],
                 "hits_top_down": result.hitsAtEachWake[-1] if result.hitsAtEachWake else None,
                 "elapsed": time.time() - t0})
    print("ITER %d solved %d/%d library %d elapsed %.0fs" % (j + 1, len(solved), len(tasks),
          len(hist[-1]["library"]), hist[-1]["elapsed"]), flush=True)
    json.dump({"history": hist, "args": vars(a)}, open(a.out, "w"), indent=1)

g = result.grammars[-1]
recognizer = result.recognitionModel if rec else None


def grammar_for(t):
    return recognizer.grammarOfTask(t).untorch() if recognizer is not None else g


def solve(tasks_, budget):
    if not tasks_:
        return {}
    fs, _ = multicoreEnumeration({t: grammar_for(t) for t in tasks_}, tasks_, solver="pyfast",
                                 maximumFrontier=1, enumerationTimeout=budget, CPUs=a.CPUs,
                                 evaluationTimeout=None, testing=True)
    return {f.task.name: f.bestPosterior.program for f in fs if not f.empty}


def to_moves(p):
    return " ".join(p.evaluate([]).moves) or "(id)"


print("\n=== end-to-end on %d random cubes (%s)" % (len(E2E), a.kind), flush=True)
state = list(E2E)
alive = set(range(len(E2E)))
moves = {i: [] for i in alive}
reached = {}
T = a.enumerationTimeout
if a.kind == "stages":
    for k, (name, _, goal) in enumerate(D.STAGES):
        todo = [i for i in sorted(alive) if not goal(state[i])]
        sol = solve([D.CubeTask("e2e%d" % i, state[i], k) for i in todo], T)
        for i in todo:
            p = sol.get("e2e%d" % i)
            if p is None:
                alive.discard(i)
                continue
            state[i] = P.apply_to(state[i], p.evaluate([]))
            moves[i].append(to_moves(p))
        reached[name] = len(alive)
        print("  after %s: %d/%d" % (name, len(alive), len(E2E)), flush=True)
else:
    sol = solve([D.CubeTask("e2e%d" % i, s, 4) for i, s in enumerate(E2E)], 5 * T)
    for i in list(alive):
        p = sol.get("e2e%d" % i)
        if p is None:
            alive.discard(i)
            continue
        state[i] = P.apply_to(state[i], p.evaluate([]))
        moves[i].append(to_moves(p))
solved_e2e = [i for i in alive if cube2.solved(state[i])]
assert len(solved_e2e) == len(alive)
lens = [len(" ".join(moves[i]).split()) for i in solved_e2e]
print("  E2E SOLVED %d/%d  mean length %.1f" % (len(solved_e2e), len(E2E), sum(lens) / max(1, len(lens))), flush=True)

tot = collections.Counter(t.stage for t in tasks)
ok = collections.Counter(t.stage for t, f in result.allFrontiers.items() if not f.empty)
library = []
for l, tp, p in g.productions:
    if p.isInvented:
        library.append({"program": str(p), "type": str(tp), "logProbability": l,
                        "moves": to_moves(p) if not tp.isArrow() else ""})


def root_dist(gr):
    gg = gr.noParent if hasattr(gr, "noParent") else gr
    ws = [(l, str(pp)) for l, _, pp in gg.productions]
    z = math.log(sum(math.exp(l) for l, _ in ws))
    return {k: math.exp(l - z) for l, k in ws}


pol = {}
for t in tasks:
    acc = pol.setdefault(t.stage, {"n": 0, "p": {}})
    acc["n"] += 1
    for k, v in root_dist(grammar_for(t)).items():
        acc["p"][k] = acc["p"].get(k, 0.0) + v

out = {"history": hist, "args": vars(a), "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                       for k, v in params.items()},
       "grammar": {"logVariable": g.logVariable, "productions": [[l, str(t), str(p)] for l, t, p in g.productions]},
       "solutions_moves": {t.name: to_moves(f.bestPosterior.program) for t, f in result.allFrontiers.items() if not f.empty},
       "solutions": {t.name: str(f.bestPosterior.program) for t, f in result.allFrontiers.items() if not f.empty},
       "e2e": {"n": len(E2E), "solved": len(solved_e2e), "mean_len": sum(lens) / max(1, len(lens)),
               "reached": reached, "moves": {str(i): moves[i] for i in solved_e2e}},
       "per_group": {k: [ok[k], tot[k]] for k in tot},
       "library_readable": library,
       "policy": {st: sorted(((k, v / acc["n"]) for k, v in acc["p"].items()), key=lambda x: -x[1])[:12]
                  for st, acc in pol.items()}}
json.dump(out, open(a.out, "w"), indent=1)

# final model for the in-browser search: grammar + recognition network weights (official architecture)
model = {"grammar": out["grammar"], "recognition": None}
if recognizer is not None:
    sd = {k: v.detach().cpu().tolist() for k, v in recognizer.state_dict().items()}
    gb = recognizer.grammarBuilder
    model["recognition"] = {"state_dict": sd, "activation": "tanh", "n_features": D.N_FEATURES,
                            "library_rows": {str(p): js for p, js in gb.library.items()},
                            "n_grammars": gb.n_grammars,
                            "productions": [str(p) for _, _, p in gb.grammar.productions]}
json.dump(model, open(a.out[:-5] + ".model.json", "w"))
print("wrote", a.out, flush=True)
