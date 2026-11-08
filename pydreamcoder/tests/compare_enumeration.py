"""Check pyec.enumerator against (a) the official Python enumerator with symmetry breaking disabled
(exact set of programs + priors in a band) and (b) the OCaml solver (program counts, hits)."""
import sys, os, json, time, subprocess, math
EC = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
sys.path.insert(0, EC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dreamcoder.grammar as G
from dreamcoder.grammar import Grammar
from dreamcoder.type import Context, arrow, tlist, tint, tbool
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra, McCarthyPrimitives, primitives
from pyec import enumerator as E

G.violatesSymmetry = lambda f, x, i: False  # OCaml solver: violates_symmetry is disabled
prims = bootstrapTarget_extra()
g = Grammar.uniform(prims)
req = arrow(tlist(tint), tlist(tint))
UB = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0

SKIP = len(sys.argv) > 2
t = time.time()
ref = {}
for l, _, p in ([] if SKIP else g.enumeration(Context.EMPTY, [], req, upperBound=UB, maximumDepth=99, lowerBound=0.)):
    ref[str(p)] = -l
tr = time.time() - t
if not SKIP: print("official python enumerator: %d programs in %.1fs (%.0f/s)" % (len(ref), tr, len(ref) / tr))

cg = E.CG(g)
en = E.Enumerator(cg, time.time() + 1e9)
mine = {}
t = time.time()
en.programs(cg.no_context, E.EMPTY, E.from_official(req), [], 0., UB, 9999,
            lambda p, c, ll: mine.__setitem__(str(p), -ll))
tm = time.time() - t
print("pyec enumerator: %d programs in %.1fs (%.0f/s)" % (len(mine), tm, len(mine) / tm))
if not SKIP: print("same program set:", set(ref) == set(mine), "| only official:", len(set(ref) - set(mine)), "only pyec:", len(set(mine) - set(ref)))
if not SKIP: print("max |dMDL|:", max(abs(ref[k] - mine[k]) for k in set(ref) & set(mine)))

# OCaml solver: count programs in the same band with one impossible task
msg = {"DSL": g.json(), "tasks": [{"examples": [{"inputs": [[1, 2]], "output": [9, 9, 9, 9, 9, 9, 9, 9]}], "name": "impossible",
        "request": req.json(), "maximumFrontier": 10}],
       "programTimeout": 0.0005, "nc": 1, "timeout": 600, "lowerBound": 0., "upperBound": UB, "budgetIncrement": UB,
       "verbose": False, "shatter": 10}
env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
t = time.time()
p = subprocess.run([EC + "/solver"], input=json.dumps(msg).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
to = time.time() - t
r = json.loads(p.stdout.decode())
print("OCaml solver (with evaluation): %d programs in %.1fs (%.0f/s)" % (r["number_enumerated"], to, r["number_enumerated"] / to))
