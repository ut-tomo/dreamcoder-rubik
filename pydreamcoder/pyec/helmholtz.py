"""Port of the official OCaml ./helmholtz binary (solvers/Helmholtz.ml + Dreaming.ml) used for
bias-optimal recognition training (--biasOptimal). Same message in, same JSON out.

helmholtz_enumeration: enumerate programs in bands of 1.5 nats (enumeration.ml order, via pyec.enumerator),
hash each program to a behaviour (outputs on the task inputs, or a registered special hash) and keep, per
behaviour, the programs with minimal extra cost and maximal prior (ties: union).
remove_bad_dreams: drop behaviours whose non-failing outputs are all reproduced by some other behaviour.
output_job: uniform subsample to at most 50000 behaviours.

Evaluation uses the official Python primitive implementations (the OCaml binary uses its own OCaml ones);
the default per-program evaluation timeout is the OCaml default, 0.001 s.
"""
import json
import random
import signal
import sys
import time

from pyec import enumerator as E

SPECIAL = {}


def register_special_helmholtz(name, factory):
    """factory(timeout, request, inputs) -> hash(program) -> (behaviour tuple, extra cost) or None"""
    SPECIAL[name] = factory


class _EvalTimeout(Exception):
    pass


def _tuplify(v):
    if isinstance(v, (list, tuple)):
        return tuple(_tuplify(x) for x in v)
    return v


def run_with_timeout(timeout, thunk):
    """run_for_interval: returns (True, value) or (False, None) on timeout/exception"""
    def handler(_1, _2):
        raise _EvalTimeout()
    old = signal.signal(signal.SIGVTALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_VIRTUAL, timeout)
        try:
            return True, thunk()
        finally:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    except _EvalTimeout:
        return False, None
    except RecursionError:
        return False, None
    except Exception:
        return False, None
    finally:
        signal.signal(signal.SIGVTALRM, old)


def default_hash(timeout, request, inputs):
    def h(program):
        try:
            f = program.evaluate([])
        except Exception:
            f = None
        outputs = []
        for xs in inputs:
            if f is None:
                outputs.append(None)
                continue

            def run(xs=xs):
                v = f
                for x in xs:
                    v = v(x)
                return v
            ok, v = run_with_timeout(timeout, run)
            outputs.append(_tuplify(v) if ok else None)
        if any(o is not None for o in outputs):
            return tuple(outputs), 0.
        return None
    return h


def helmholtz_enumeration(behavior_hash, g, request, timeout, maximumSize=sys.maxsize, upperBound=None):
    """behaviour -> (ll, extra_cost, sorted program strings). upperBound (port-only, for tests) stops
    after the band containing it instead of at the timeout."""
    table = {}
    deadline = time.time() + timeout
    cg = E.CG(g)
    req, nrec = E.recursive_request(cg, E.from_official(request))
    post = E.make_post(cg, nrec, 99)
    stop = [False]

    def k(p, _ctx, ll):
        if stop[0]:
            return
        if len(table) > maximumSize:
            stop[0] = True
            raise E.Timeout()
        p = post(p)
        if p is None:
            return
        r = behavior_hash(p)
        if r is None:
            return
        key, extra = r
        old = table.get(key)
        if old is None or extra < old[1] or (extra == old[1] and ll > old[0]):
            table[key] = (ll, extra, [str(p)])
        elif extra == old[1] and ll == old[0]:
            table[key] = (ll, extra, sorted(set(old[2]) | {str(p)}))
    lb = 0.
    while time.time() < deadline and not stop[0] and (upperBound is None or lb < upperBound):
        en = E.Enumerator(cg, deadline)
        try:
            en.programs(cg.no_context, E.EMPTY, req, [], lb, lb + 1.5, 9999, k)
        except E.Timeout:
            break
        lb += 1.5
    return table


def remove_bad_dreams(table):
    """[(behaviour, (ll, programs))] without behaviours dominated by another behaviour"""
    items = [(key, (v[0], v[2])) for key, v in table.items()]
    if not items:
        return []
    n = len(items[0][0])
    containers = [dict() for _ in range(n)]
    for idx, (key, _) in enumerate(items):
        assert len(key) == n
        for j, o in enumerate(key):
            if o is not None:
                containers[j].setdefault(o, set()).add(idx)
    sweet = []
    for idx, (key, data) in enumerate(items):
        dominating = None
        for j, o in enumerate(key):
            if o is None:
                continue
            others = containers[j][o]
            dominating = set(others) if dominating is None else dominating & others
        if len(dominating) > 1:
            continue
        sweet.append((key, data))
    print("Removed %d nightmares." % (len(items) - len(sweet)), file=sys.stderr)
    return sweet


def output_job(results, maxExamples=50000, rng=None):
    if len(results) >= maxExamples:
        rng = rng or random.Random(0)
        p = maxExamples / len(results)
        results = [r for r in results if rng.random() < p]
    return [{"ll": ll, "programs": list(ps)} for _, (ll, ps) in results]


def grammar_from_json(d):
    """inverse of Grammar.json() (non-contextual DSL, as sent by dreaming.helmholtzEnumeration)"""
    from dreamcoder.grammar import Grammar
    from dreamcoder.program import Program
    from dreamcoder.type import Type
    prods = []
    for p in d["productions"]:
        e = Program.parse(p["expression"])
        prods.append((p["logProbability"], e.infer(), e))
    ct = d.get("continuationType")
    return Grammar(d["logVariable"], prods,
                   continuationType=Type.fromjson(ct) if ct else None)


def run_message(j, g=None):
    from dreamcoder.type import Type
    request = Type.fromjson(j["request"])
    timeout = j["timeout"]
    et = j.get("evaluationTimeout", 0.001)
    maximumSize = j.get("maximumSize", sys.maxsize)
    if g is None:
        g = grammar_from_json(j["DSL"])
    special = j.get("special")
    if special is None:
        h = default_hash(et, request, j["extras"])
    else:
        if special not in SPECIAL:
            raise KeyError("Could not find special Helmholtz enumerator: %s" % special)
        h = SPECIAL[special](et, request, j["extras"])
    table = helmholtz_enumeration(h, g, request, timeout, maximumSize=maximumSize,
                                  upperBound=j.get("pyecUpperBound"))
    return output_job(remove_bad_dreams(table))


def helmholtzEnumeration(g, request, inputs, timeout, _=None, special=None, evaluationTimeout=None):
    """Drop-in for dreamcoder.dreaming.helmholtzEnumeration: returns the JSON response as bytes."""
    message = {"request": request.json(), "timeout": timeout, "DSL": g.json(), "extras": inputs}
    if evaluationTimeout:
        message["evaluationTimeout"] = evaluationTimeout
    if special:
        message["special"] = special
    return json.dumps(run_message(json.loads(json.dumps(message)), g=g)).encode("utf-8")
