"""Benchmark one realistic solver call (enumeration + evaluation on list tasks) for pyec vs the OCaml solver."""
import sys, os, time, json, subprocess
EC = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.main import retrieveJSONTasks
from pyec.enumerator import enumerate_for_tasks
os.chdir(os.path.join(os.path.dirname(__file__), "..", "ec"))
g = Grammar.uniform(bootstrapTarget_extra())
tasks = [t for t in retrieveJSONTasks("data/list_tasks.json") if str(t.request) == "list(int) -> list(int)"]
task = [t for t in tasks if "drop-k" in t.name or "evens" in t.name][:1] or tasks[:1]
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TS = tasks[:NT]
task = TS[0]
lb, ub = float(sys.argv[1]), float(sys.argv[2])
prof = os.environ.get("PROF")
t = time.time()
if prof:
    import cProfile, pstats
    cProfile.run("r = enumerate_for_tasks(g, TS, [10]*len(TS), lb, ub, ub - lb, 60, nc=1, evaluationTimeout=0.0005)", "/tmp/bs.prof")
    pstats.Stats("/tmp/bs.prof").sort_stats("tottime").print_stats(15)
else:
    r = enumerate_for_tasks(g, TS, [10]*len(TS), lb, ub, ub - lb, 60, nc=1, evaluationTimeout=0.0005)
dt = time.time() - t
print("tasks", len(TS), "pyec: %d programs in %.2fs = %.0f/s, hits %d" % (r[1], dt, r[1] / dt, sum(len(h) for h in r[0])))
msg = {"DSL": g.json(), "tasks": [{"examples": [{"inputs": list(xs), "output": y} for xs, y in t.examples],
       "name": t.name, "request": t.request.json(), "maximumFrontier": 10} for t in TS],
       "programTimeout": 0.0005, "nc": 1, "timeout": 60, "lowerBound": lb, "upperBound": ub, "budgetIncrement": ub - lb,
       "verbose": False, "shatter": 10}
env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
t = time.time()
p = subprocess.run([EC + "/solver"], input=json.dumps(msg).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
dt = time.time() - t
o = json.loads(p.stdout.decode())
print("ocaml: %d programs in %.2fs = %.0f/s, hits %d" % (o["number_enumerated"], dt, o["number_enumerated"] / dt, sum(len(o[t.name]) for t in TS)))
