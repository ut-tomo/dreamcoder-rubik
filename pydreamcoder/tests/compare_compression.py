"""Run the official OCaml ./compression and pyec.vs_compression on the same saved message; compare."""
import sys, json, time, subprocess, os
sys.path.insert(0, os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.domains.list.listPrimitives import primitives, McCarthyPrimitives, bootstrapTarget_extra, basePrimitives
for f in (basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra):
    try: f()
    except Exception as e: print("prims", f.__name__, e)
from pyec.vs_compression import run_message

fn = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "both"
j = json.load(open(fn))
if len(sys.argv) > 3: j["iterations"] = int(sys.argv[3])
print("frontiers", len(j["frontiers"]), "topK", j["topK"], "topI", j["topI"], "arity", j["arity"], "CPUs", j["CPUs"], "pc", j["pseudoCounts"], "iters", j.get("iterations"))
def inventions(resp):
    return [p["expression"] for p in resp["DSL"]["productions"] if p["expression"].startswith("#")]
res = {}
if mode in ("both", "ocaml"):
    t = time.time()
    root = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
    p = subprocess.run([root + "/compression"], input=json.dumps(j).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    ro = json.loads(p.stdout.decode())
    res["ocaml"] = ro
    print("OCAML %.1fs" % (time.time() - t))
    for l in p.stderr.decode().splitlines():
        if "Improved score" in l or "Initial score" in l or "candidates" in l.lower(): print("  ocaml:", l)
    for i in inventions(ro): print("  O", i)
if mode in ("both", "python"):
    t = time.time()
    rp = run_message(j, log=lambda *a: print("  py:", *a, flush=True))
    res["python"] = rp
    print("PYTHON %.1fs" % (time.time() - t))
    for i in inventions(rp): print("  P", i)
if len(res) == 2:
    a, b = inventions(res["ocaml"]), inventions(res["python"])
    print("SAME INVENTIONS (as sets):", set(a) == set(b), "| ocaml", len(a), "python", len(b))
    lo = {p["expression"]: p["logProbability"] for p in res["ocaml"]["DSL"]["productions"]}
    lp = {p["expression"]: p["logProbability"] for p in res["python"]["DSL"]["productions"]}
    common = set(lo) & set(lp)
    print("max |dlogP| over common productions: %.2e" % max(abs(lo[k] - lp[k]) for k in common))
    fo = [sorted(e["program"] for e in f["programs"]) for f in res["ocaml"]["frontiers"]]
    fp = [sorted(e["program"] for e in f["programs"]) for f in res["python"]["frontiers"]]
    print("identical rewritten frontiers: %d/%d" % (sum(x == y for x, y in zip(fo, fp)), len(fo)))
