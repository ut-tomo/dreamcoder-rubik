import sys, os, time, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamcoder.program import Program
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.main import retrieveJSONTasks
os.chdir(os.path.join(os.path.dirname(__file__), "..", "ec"))
bootstrapTarget_extra()
tasks = [t for t in retrieveJSONTasks("data/list_tasks.json") if str(t.request) == "list(int) -> list(int)"][:60]
p = Program.parse("(lambda (map (lambda (+ $0 1)) (cdr $0)))")
N = 3000
def h(*a): pass
signal.signal(signal.SIGVTALRM, h)
for name, fn in [
    ("setitimer pair", lambda: (signal.setitimer(signal.ITIMER_VIRTUAL, 0.0005), signal.setitimer(signal.ITIMER_VIRTUAL, 0))),
    ("evaluate([])", lambda: p.evaluate([])),
]:
    for _ in range(2):
        t = time.time()
        for i in range(N): fn()
        print(name, "%.2f us" % ((time.time() - t) / N * 1e6))
f = p.evaluate([])
for _ in range(3):
    t = time.time()
    for i in range(N // 10):
        for tk in tasks:
            x, y = tk.examples[0]
            try: v = tk.predict(f, x)
            except BaseException: v = None
            v != y
    print("predict first example per task: %.2f us" % ((time.time() - t) / (N // 10) / len(tasks) * 1e6))
print("examples per task", sum(len(t.examples) for t in tasks) / len(tasks), "avg input len", sum(len(t.examples[0][0][0]) for t in tasks) / len(tasks))
