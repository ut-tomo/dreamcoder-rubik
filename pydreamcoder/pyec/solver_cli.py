"""Drop-in replacement for the official ./solver binary, run under PyPy.
stdin: dill-pickled dict(g, tasks, maximumFrontiers, lowerBound, upperBound, budgetIncrement, timeout, nc,
evaluationTimeout, primitive_modules, extra_imports); stdout: dill-pickled (hits, number_enumerated)."""
import sys, os, time, importlib
T0 = time.time()


def main():
    data = sys.stdin.buffer.read()
    from pyec.compress_cli import load_primitives
    load_primitives()
    from pyec.backend import _loads, _dumps
    req = _loads(data)
    from pyec.enumerator import enumerate_for_tasks
    timeout = req["timeout"] - (time.time() - T0) - req.get("spawn_overhead", 0.)
    hits, n = enumerate_for_tasks(req["g"], req["tasks"], req["maximumFrontiers"], req["lowerBound"],
                                  req["upperBound"], req["budgetIncrement"], max(0., timeout), nc=req["nc"],
                                  evaluationTimeout=req["evaluationTimeout"])
    sys.stdout.buffer.write(_dumps((hits, n)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
