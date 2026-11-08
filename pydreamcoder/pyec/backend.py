"""Glue between the official DreamCoder Python frontend (vendored, see ../ec) and the pure-Python
ports of the OCaml solver (pyec.enumerator) and compressor (pyec.vs_compression).

solveForTask_pyfast  : same signature / return value as dreamcoder.enumeration.solveForTask_ocaml
pyvsInduce           : same signature / return value as dreamcoder.compression.ocamlInduce
Both run the port in a PyPy subprocess when PYEC_PYPY points to a pypy3 binary (default: in-process).
"""
import os, sys, json, time, subprocess, datetime

from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.utilities import eprint

PYEC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _primitive_modules():
    s = os.environ.get("PYEC_PRIMITIVE_MODULES", "")
    return [x.strip() for x in s.split(",") if x.strip()]


def _pypy():
    return os.environ.get("PYEC_PYPY")


def _env():
    env = dict(os.environ)
    ec = os.environ.get("PYEC_EC_ROOT", os.path.join(PYEC_ROOT, "ec"))
    extra = os.environ.get("PYEC_EXTRA_PATH", "")
    env["PYTHONPATH"] = ":".join(x for x in [PYEC_ROOT, ec, extra, env.get("PYTHONPATH", "")] if x)
    return env


def _dumps(obj):
    """dill, but primitives are pickled by name (their Python values differ across interpreters)."""
    import dill, io
    from dreamcoder.program import Primitive

    class P(dill.Pickler):
        def persistent_id(self, o):
            if isinstance(o, Primitive):
                return ("prim", o.name)
            return None
    buf = io.BytesIO()
    P(buf, protocol=4).dump(obj)
    return buf.getvalue()


def _loads(data):
    import dill, io
    from dreamcoder.program import Primitive

    class U(dill.Unpickler):
        def persistent_load(self, pid):
            assert pid[0] == "prim"
            return Primitive.GLOBALS[pid[1]]
    return U(io.BytesIO(data)).load()


_SERVER = {"path": None, "proc": None}


def start_server(g=None, tasks=None, evaluationTimeout=None, seconds=None):
    """Start the persistent PyPy fork-server (called once, in the main process, before solver workers fork)."""
    if not _pypy() or _SERVER["path"] or os.environ.get("PYEC_NO_SERVER"):
        return
    import tempfile, socket, atexit
    path = os.path.join(tempfile.gettempdir(), "pyec_%d_%d.sock" % (os.getpid(), int(time.time() * 1000)))
    env = _env()
    env["PYEC_SERVER_PARENT"] = str(os.getpid())
    proc = subprocess.Popen([_pypy(), "-m", "pyec.pypy_server", path], env=env)
    for _ in range(600):
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                s.close()
                break
            except OSError:
                pass
        time.sleep(0.05)
    _SERVER["path"], _SERVER["proc"] = path, proc
    seconds = float(os.environ.get("PYEC_WARMUP", "8")) if seconds is None else seconds
    if tasks and seconds > 0:
        # JIT warm-up on the real grammar/tasks (one job per request type), outside any timed job
        from pyec.pypy_server import send_msg, recv_msg
        if not isinstance(g, dict):
            g = {t: g for t in tasks}
        jobs, seen = [], set()
        for t in tasks:
            if t.request not in seen:
                seen.add(t.request)
                jobs.append((g[t], [t]))
        t0 = time.time()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(path)
        send_msg(s, b"WARMUP" + _dumps({"jobs": jobs[:6], "seconds": seconds,
                                         "evaluationTimeout": evaluationTimeout}))
        recv_msg(s)
        s.close()
        eprint("(pyec) PyPy solver server warmed up in %.1fs" % (time.time() - t0))
    owner = os.getpid()

    def stop():
        if os.getpid() == owner:
            proc.terminate()
    atexit.register(stop)


def solveForTask_pyfast(_=None, elapsedTime=0., CPUs=1, g=None, tasks=None, lowerBound=None, upperBound=None,
                        budgetIncrement=None, timeout=None, testing=None, likelihoodModel=None,
                        evaluationTimeout=None, maximumFrontiers=None):
    mfs = [maximumFrontiers[t] for t in tasks]
    # Interpreter choice is a pure speed decision (identical algorithm & results): PyPy's JIT is ~2x faster
    # at enumeration but ~5x slower than CPython at evaluating freshly-built program closures, so jobs that
    # evaluate every program on many tasks run in-process under CPython.
    use_pypy = _pypy() and len(tasks) <= int(os.environ.get("PYEC_PYPY_MAX_TASKS", "4"))
    if use_pypy and _SERVER["path"]:
        import socket
        from pyec.pypy_server import send_msg, recv_msg
        msg = _dumps({"g": g, "tasks": tasks, "maximumFrontiers": mfs, "lowerBound": lowerBound,
                      "upperBound": upperBound, "budgetIncrement": budgetIncrement, "timeout": timeout,
                      "nc": CPUs, "evaluationTimeout": evaluationTimeout, "sent_at": time.time()})
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(_SERVER["path"])
        send_msg(s, msg)
        hits, pc = _loads(recv_msg(s))
        s.close()
    elif use_pypy:
        msg = _dumps({"g": g, "tasks": tasks, "maximumFrontiers": mfs, "lowerBound": lowerBound,
                          "upperBound": upperBound, "budgetIncrement": budgetIncrement, "timeout": timeout,
                          "nc": CPUs, "evaluationTimeout": evaluationTimeout,
                          "primitive_modules": _primitive_modules()})
        p = subprocess.run([_pypy(), "-m", "pyec.solver_cli"], input=msg, stdout=subprocess.PIPE,
                           env=_env())
        if p.returncode != 0:
            raise RuntimeError("pyec solver failed")
        hits, pc = _loads(p.stdout)
    else:
        from pyec.enumerator import enumerate_for_tasks
        hits, pc = enumerate_for_tasks(g, tasks, mfs, lowerBound, upperBound, budgetIncrement, timeout, nc=CPUs,
                                       evaluationTimeout=evaluationTimeout)
    frontiers, searchTimes = {}, {}
    for t, solutions in zip(tasks, hits):
        frontier = Frontier([FrontierEntry(program=p, logLikelihood=e["logLikelihood"],
                                           logPrior=g.logLikelihood(t.request, p))
                             for e in solutions for p in [Program.parse(e["program"])]], task=t)
        frontiers[t] = frontier
        if frontier.empty:
            searchTimes[t] = None
        else:
            searchTimes[t] = min((e["logLikelihood"] + e["logPrior"], e["time"]) for e in solutions)[1] + elapsedTime
    return frontiers, searchTimes, pc


def pyvsInduce(g, frontiers, _=None, topK=1, pseudoCounts=1.0, aic=1.0, structurePenalty=0.001, a=0, CPUs=1,
               bs=1000000, topI=300, factored=False, iteration=None, topk_use_only_likelihood=None):
    """Mirror of dreamcoder.compression.ocamlInduce, calling the Python port instead of ./compression."""
    CPUs = 2  # ocamlInduce hard-codes CPUs = 2 (compression_step_master with two workers)
    iterations = 99
    while True:
        g0 = g
        originalFrontiers = frontiers
        t2f = {f.task: f for f in frontiers}
        frontiers = [f for f in frontiers if not f.empty]
        message = {"arity": a, "topK": topK, "pseudoCounts": float(pseudoCounts), "aic": aic, "bs": bs,
                   "topI": topI, "structurePenalty": float(structurePenalty), "CPUs": CPUs, "DSL": g.json(),
                   "factored_apply": factored, "iterations": iterations,
                   "frontiers": [f.json() for f in frontiers]}
        message = json.dumps(message)
        os.makedirs("compressionMessages", exist_ok=True)
        fn = "compressionMessages/%s" % datetime.datetime.now().isoformat()
        with open(fn, "w") as f:
            f.write(message)
        eprint("Compression message saved to:", fn)
        if _pypy():
            env = _env()
            p = subprocess.run([_pypy(), "-m", "pyec.compress_cli"], input=message.encode(),
                               stdout=subprocess.PIPE, env=env)
            if p.returncode != 0:
                raise RuntimeError("pyec compression failed")
            response = json.loads(p.stdout.decode())
        else:
            from pyec.vs_compression import run_message
            response = run_message(json.loads(message), log=eprint)
        gj = response["DSL"]
        g = Grammar(gj["logVariable"],
                    [(l, p.infer(), p) for production in gj["productions"]
                     for l in [production["logProbability"]] for p in [Program.parse(production["expression"])]],
                    continuationType=g0.continuationType)
        frontiers = {original.task: Frontier([FrontierEntry(p, logLikelihood=e["logLikelihood"],
                                                            logPrior=g.logLikelihood(original.task.request, p))
                                              for e in new["programs"] for p in [Program.parse(e["program"])]],
                                             task=original.task)
                     for original, new in zip(frontiers, response["frontiers"])}
        frontiers = [frontiers.get(f.task, t2f[f.task]) for f in originalFrontiers]
        if iterations == 1 and len(g) > len(g0):
            eprint("Grammar changed - running another round of consolidation.")
            continue
        eprint("Finished consolidation.")
        return g, frontiers
