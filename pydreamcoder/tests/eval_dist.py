import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.main import retrieveJSONTasks
from pyec import enumerator as E
os.chdir(os.path.join(os.path.dirname(__file__), "..", "ec"))
g = Grammar.uniform(bootstrapTarget_extra())
tasks = [t for t in retrieveJSONTasks("data/list_tasks.json") if str(t.request) == "list(int) -> list(int)"][:60]
cg = E.CG(g)
en = E.Enumerator(cg, time.time() + 1e9)
progs = []
en.programs(cg.no_context, E.EMPTY, E.from_official(tasks[0].request), [], 0., 15., 9999, lambda p, c, ll: progs.append(p))
score = E.make_scorer(tasks, 0.0005)
for p in progs[:500]: score(p)  # warm
times = []
for p in progs:
    t = time.time(); score(p); times.append((time.time() - t, str(p)))
times.sort(reverse=True)
tot = sum(t for t, _ in times)
print("programs", len(progs), "total %.2fs, per program-task %.1f us" % (tot, tot / len(progs) / len(tasks) * 1e6))
print("top 1%% of programs take %.0f%% of time" % (100 * sum(t for t, _ in times[:len(times) // 100]) / tot))
for t, s in times[:8]: print("  %.1f ms  %s" % (t * 1e3, s))
