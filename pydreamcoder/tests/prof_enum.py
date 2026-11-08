import sys, os, time, cProfile, pstats
EC = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
sys.path.insert(0, EC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.grammar import Grammar
from dreamcoder.type import arrow, tlist, tint
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from pyec import enumerator as E
g = Grammar.uniform(bootstrapTarget_extra())
req = arrow(tlist(tint), tlist(tint))
UB = float(sys.argv[1])
cg = E.CG(g)
en = E.Enumerator(cg, time.time() + 1e9)
n = [0]; maxsub = [0]
def k(p, c, ll):
    n[0] += 1; maxsub[0] = max(maxsub[0], len(c.sub))
t = time.time()
if len(sys.argv) > 2:
    cProfile.run("en.programs(cg.no_context, E.EMPTY, E.from_official(req), [], 0., UB, 9999, k)", "/tmp/prof.out")
    pstats.Stats("/tmp/prof.out").sort_stats("tottime").print_stats(12)
else:
    en.programs(cg.no_context, E.EMPTY, E.from_official(req), [], 0., UB, 9999, k)
print(n[0], "programs", time.time() - t, "s", "calls", en.calls, "max |sub|", maxsub[0])
