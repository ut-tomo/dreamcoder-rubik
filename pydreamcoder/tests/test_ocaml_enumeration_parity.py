"""Differential checks of pyec.enumerator against the official OCaml ./solver binary.

A task with no examples is solved by every well-typed program in both implementations, so with a large
maximumFrontier the OCaml solver returns the complete enumerated band with priors. This exercises the
parts the official Python enumerator does not implement: the top-level fix1/fix2 rewrite of
enumeration.ml and the maxFreeParameters bound.

Needs EC_SOLVER (default: $EC_ROOT/solver, EC_ROOT defaults to the shared dreamcoder-ec checkout).
Run: PYTHONPATH=pydreamcoder:pydreamcoder/ec python -m unittest pydreamcoder/tests/test_ocaml_enumeration_parity.py
"""
import json
import os
import subprocess
import unittest

from dreamcoder.grammar import Grammar
from dreamcoder.program import Program, Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist, tbool, baseType
from dreamcoder.domains.list.listPrimitives import McCarthyPrimitives, primitiveRecursion2
from pyec import enumerator as port

EC_ROOT = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
SOLVER = os.environ.get("EC_SOLVER", os.path.join(EC_ROOT, "solver"))


def ocaml_band(grammar, request, ub, max_parameters=None):
    msg = {"DSL": grammar.json(),
           "tasks": [{"examples": [], "name": "anything", "request": request.json(),
                      "maximumFrontier": 10 ** 7}],
           "programTimeout": 1.0, "nc": 1, "timeout": 600, "lowerBound": 0., "upperBound": ub,
           "budgetIncrement": ub, "verbose": False, "shatter": 10}
    if max_parameters is not None:
        msg["maxParameters"] = max_parameters
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
    p = subprocess.run([SOLVER], input=json.dumps(msg).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, check=True)
    hits = json.loads(p.stdout.decode())["anything"]
    # normalise the printing through the official parser
    return {str(Program.parse(h["program"])): -h["logPrior"] for h in hits}


def port_band(grammar, request, ub, max_parameters=None):
    task = Task("anything", request, [])
    if max_parameters is not None:
        task.maxParameters = max_parameters
    hits, _ = port.enumerate_for_tasks(grammar, [task], [10 ** 7], 0., ub, ub, 600.,
                                       evaluationTimeout=1.0)
    return {str(Program.parse(h["program"])): -h["logPrior"] for h in hits[0]}


@unittest.skipUnless(os.path.exists(SOLVER), "OCaml solver not available")
class OCamlEnumerationParity(unittest.TestCase):
    def compare(self, grammar, request, ub, **kw):
        ref = ocaml_band(grammar, request, ub, **kw)
        mine = port_band(grammar, request, ub, **kw)
        self.assertGreater(len(ref), 10, "a vacuous parity check is not evidence")
        self.assertEqual(set(ref) - set(mine), set(), "missing in port")
        self.assertEqual(set(mine) - set(ref), set(), "extra in port")
        for p in ref:
            self.assertAlmostEqual(ref[p], mine[p], places=6, msg=p)
        return ref

    def test_fix1(self):
        g = Grammar.uniform(McCarthyPrimitives())
        ref = self.compare(g, arrow(tlist(tint), tint), 9.)
        self.assertTrue(any("fix1" in p for p in ref), "band must contain recursive programs")
        ref = self.compare(g, arrow(tlist(tint), tlist(tint)), 9.)
        self.assertTrue(any("fix1" in p for p in ref))

    def test_fix2(self):
        g = Grammar.uniform(McCarthyPrimitives() + [primitiveRecursion2])
        ref = self.compare(g, arrow(tint, tlist(tint), tlist(tint)), 10.)
        self.assertTrue(any("fix2" in p for p in ref))

    def test_no_recursion_for_other_arity(self):
        # fix1 only applies to one-argument requests; a two-argument request must not use it
        g = Grammar.uniform(McCarthyPrimitives())
        ref = self.compare(g, arrow(tint, tint, tbool), 8.)
        self.assertFalse(any("fix" in p for p in ref))

    def test_max_free_parameters(self):
        treal = baseType("real")
        prims = [Primitive("+.", arrow(treal, treal, treal), None),
                 Primitive("*.", arrow(treal, treal, treal), None),
                 Primitive("REAL", treal, None)]
        g = Grammar.uniform(prims)
        for mfp in (0, 1, 2):
            with self.subTest(maxParameters=mfp):
                ref = self.compare(g, arrow(treal, treal), 9., max_parameters=mfp)
                self.assertTrue(all(p.count("REAL") <= mfp for p in ref))


if __name__ == "__main__":
    unittest.main()
