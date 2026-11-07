"""Wake-sleep loop: wake (enumeration), sleep-abstraction (compression), sleep-dreaming (recognition).

Ablations: compress_on=False -> no library learning; recognition=None -> no recognition model.
Compute is matched: shared enumeration (no recognition) gets per_task_timeout * n_tasks seconds,
per-task enumeration (recognition) gets per_task_timeout seconds for each task.
"""
from __future__ import annotations

import heapq
import json
import math
import signal
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .grammar import EnumerationTimeout, Grammar
from .library import compress
from .types import Context, canonical


@dataclass
class Task:
    name: str
    request: Any                       # Type
    check: Callable[[Any], bool]      # evaluated program -> solved?
    meta: dict = field(default_factory=dict)


def io_task(name, request, examples, n_args=1):
    """Programming-by-example task: examples = [(inputs_tuple, output)]."""
    def check(f):
        for xs, y in examples:
            try:
                v = f
                for x in xs:
                    v = v(x)
                if v != y:
                    return False
            except Exception:
                return False
        return True
    return Task(name, request, check)


class EvalTimeout(BaseException):  # not swallowed by task checks
    pass


def _alarm(signum, frame):
    raise EvalTimeout()


def _run_limited(fn, limit):
    """Run fn() with a wall-clock limit (SIGALRM; main thread on Unix only)."""
    if limit <= 0 or threading.current_thread() is not threading.main_thread():
        return fn()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def wake(g, tasks, timeout, frontier_size=5, step=1.5, max_depth=12, eval_limit=0.05, log=print):
    """Enumerate programs in order of prior, shared across tasks with the same request."""
    by_req = defaultdict(list)
    for t in tasks:
        by_req[str(canonical(t.request))].append(t)
    frontiers = {t.name: [] for t in tasks}   # heap of (logprior, str, program)
    n_enum = 0
    n_timeouts = 0
    deadline = time.time() + timeout
    for _, ts in by_req.items():
        req = ts[0].request
        lower, upper = 0.0, step
        per_req_deadline = time.time() + timeout / len(by_req)
        g.deadline = min(deadline, per_req_deadline)
        try:
            while True:
                for l, _, p in g.enumerate(Context(), [], req, upper, lower, max_depth):
                    n_enum += 1
                    pending = [t for t in ts if not (len(frontiers[t.name]) >= frontier_size
                                                     and l <= frontiers[t.name][0][0])]
                    if not pending:
                        continue

                    def check_all(p=p, pending=pending):
                        f = p.evaluate([])
                        hits = []
                        for t in pending:
                            try:
                                if t.check(f):
                                    hits.append(t)
                            except EvalTimeout:
                                raise
                            except Exception:
                                pass
                        return hits
                    try:
                        hits = _run_limited(check_all, eval_limit * max(1, len(pending)) ** 0.5)
                    except EvalTimeout:
                        n_timeouts += 1
                        hits = []
                    except Exception:
                        hits = []
                    for t in hits:
                        fr = frontiers[t.name]
                        item = (l, str(p), p)
                        if len(fr) < frontier_size:
                            heapq.heappush(fr, item)
                        else:
                            heapq.heappushpop(fr, item)
                    if all(len(frontiers[t.name]) >= frontier_size for t in ts):
                        raise EnumerationTimeout()
                lower, upper = upper, upper + step
        except EnumerationTimeout:
            pass
        finally:
            g.deadline = None
    log(f"  [wake] enumerated {n_enum} programs, solved {sum(1 for v in frontiers.values() if v)}/{len(tasks)}, {n_timeouts} eval timeouts")
    return {k: [p for _, _, p in sorted(v, reverse=True)] for k, v in frontiers.items()}


def wake_per_task(g, tasks, grammar_for, per_task_timeout, frontier_size=5, step=1.5, max_depth=12,
                  eval_limit=0.05, log=print):
    """Enumerate separately for each task under its own (recognition-predicted) grammar."""
    out = {}
    n_enum = 0
    for t in tasks:
        gt = grammar_for(t)
        fr = []
        gt.deadline = time.time() + per_task_timeout
        lower, upper = 0.0, step
        try:
            while True:
                for l, _, p in gt.enumerate(Context(), [], t.request, upper, lower, max_depth):
                    n_enum += 1
                    if len(fr) >= frontier_size and l <= fr[0][0]:
                        continue
                    try:
                        ok = _run_limited(lambda p=p: t.check(p.evaluate([])), eval_limit)
                    except EvalTimeout:
                        ok = False
                    except Exception:
                        ok = False
                    if ok:
                        item = (l, str(p), p)
                        if len(fr) < frontier_size:
                            heapq.heappush(fr, item)
                        else:
                            heapq.heappushpop(fr, item)
                        if len(fr) >= frontier_size:
                            raise EnumerationTimeout()
                lower, upper = upper, upper + step
        except EnumerationTimeout:
            pass
        finally:
            gt.deadline = None
        out[t.name] = [p for _, _, p in sorted(fr, reverse=True)]
    log(f"  [wake/per-task] enumerated {n_enum} programs, solved {sum(1 for v in out.values() if v)}/{len(tasks)}")
    return out


def run(primitives, tasks, iterations=5, timeout=60, structure_penalty=1.0, compress_on=True,
        frontier_size=5, max_new=3, out=None, log=print,
        recognition=None, per_task_timeout=None, test_tasks=None):
    """recognition: None, or dict(features=task->list, dream=(program, rng)->features|None,
    n_features=int, n_dreams=int, steps=int, seed=int). If per_task_timeout is given, the shared
    wake uses per_task_timeout * len(tasks) seconds (compute-matched to per-task wake)."""
    import random as _random
    if per_task_timeout is not None:
        timeout = per_task_timeout * len(tasks)
    model = None
    if recognition is not None:
        from .recognition import Recognition, dream_examples
        model = Recognition(recognition["n_features"], seed=recognition.get("seed", 0))
        drng = _random.Random(recognition.get("seed", 0))
    g = Grammar.uniform(primitives)
    req_of = {t.name: t.request for t in tasks}
    history = []
    all_frontiers = {}
    for it in range(iterations):
        log(f"=== iteration {it}  (library size {len(g.productions)})")
        t0 = time.time()
        if model is not None and model.vocab is not None:
            fr = wake_per_task(g, tasks, lambda t: model.grammar_for(g, recognition["features"](t)),
                               per_task_timeout or timeout / len(tasks), frontier_size=frontier_size, log=log)
        else:
            fr = wake(g, tasks, timeout, frontier_size=frontier_size, log=log)
        for k, v in fr.items():
            if v:
                # keep the best programs found so far, expressed in the current library
                all_frontiers[k] = (req_of[k], v)
        solved = sorted(k for k, v in fr.items() if v)
        rec = {"iteration": it, "solved_this_iter": len(solved), "solved_ever": len(all_frontiers),
               "n_tasks": len(tasks), "library": [str(p) for _, _, p in g.productions if p.__class__.__name__ == "Invented"],
               "wake_seconds": round(time.time() - t0, 1)}
        if compress_on and all_frontiers:
            g, all_frontiers = compress(g, all_frontiers, structure_penalty=structure_penalty, max_new=max_new, log=log)
        if model is not None:
            t1 = time.time()
            req = tasks[0].request
            replays = []
            byname = {t.name: t for t in tasks}
            for k, (r, progs) in all_frontiers.items():
                for p in progs:
                    replays.append((recognition["features"](byname[k]), r, p, 1.0 / len(progs)))
            dreams = dream_examples(g, req, recognition["dream"], recognition.get("n_dreams", 1000), drng)
            model.train(g, replays + dreams, steps=recognition.get("steps", 600), log=log)
            rec["n_replays"], rec["n_dreams"] = len(replays), len(dreams)
            rec["dream_seconds"] = round(time.time() - t1, 1)
        history.append(rec)
        log(f"  solved {rec['solved_this_iter']}/{len(tasks)} this iteration, {rec['solved_ever']} ever")
        if out:
            with open(out, "w") as fh:
                json.dump({"history": history, "grammar": str(g),
                           "frontiers": {k: [str(p) for p in v[1]] for k, v in all_frontiers.items()}}, fh, indent=1)
    test = None
    if test_tasks:
        log(f"=== held-out test ({len(test_tasks)} tasks)")
        if model is not None and model.vocab is not None:
            tf = wake_per_task(g, test_tasks, lambda t: model.grammar_for(g, recognition["features"](t)),
                               per_task_timeout or timeout / len(tasks), frontier_size=1, log=log)
        else:
            tt = (per_task_timeout * len(test_tasks)) if per_task_timeout else timeout * len(test_tasks) / len(tasks)
            tf = wake(g, test_tasks, tt, frontier_size=1, log=log)
        test = {k: [str(p) for p in v] for k, v in tf.items() if v}
        log(f"  test solved {len(test)}/{len(test_tasks)}")
        if out:
            d = json.load(open(out))
            d["test"] = test
            d["test_n"] = len(test_tasks)
            json.dump(d, open(out, "w"), indent=1)
    run.model = model
    run.test_frontiers = {k: v for k, v in (tf.items() if test_tasks else [])}
    return g, history, all_frontiers
