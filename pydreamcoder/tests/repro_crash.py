import sys, os, time, faulthandler
faulthandler.enable()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.main import retrieveJSONTasks
from pyec.enumerator import enumerate_for_tasks
os.chdir(os.path.join(os.path.dirname(__file__), "..", "ec"))
g = Grammar.uniform(bootstrapTarget_extra())
req = sys.argv[1]
tasks = [t for t in retrieveJSONTasks("data/list_tasks.json") if str(t.request) == req]
for t in tasks:
    t0 = time.time()
    r = enumerate_for_tasks(g, [t], [10], float(sys.argv[2]), float(sys.argv[3]), 1.5, float(sys.argv[4]), nc=1, evaluationTimeout=0.0005)
    print(t.name, r[1], "programs", "%.1fs" % (time.time() - t0), len(r[0][0]), "hits", flush=True)
