"""pyec.helmholtz vs the official OCaml ./helmholtz binary.

Exact comparison needs both implementations to reach the same set of behaviours; this holds when the
behaviour space is finite and is exhausted well before the timeout (deeper programs can then never
improve a behaviour's prior). remove_bad_dreams is also checked against its specification directly.
Run: PYTHONPATH=pydreamcoder:pydreamcoder/ec python -m unittest pydreamcoder/tests/test_helmholtz_parity.py
"""
import json
import os
import subprocess
import unittest

from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.type import arrow, tint, tbool
from dreamcoder.domains.list.listPrimitives import McCarthyPrimitives
from pyec import helmholtz as H

EC_ROOT = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
BINARY = os.environ.get("EC_HELMHOLTZ", os.path.join(EC_ROOT, "helmholtz"))


def ocaml(message):
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
    p = subprocess.run([BINARY], input=json.dumps(message).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, check=True)
    return json.loads(p.stdout.decode())


def canon(resp):
    return sorted((round(e["ll"], 6), tuple(sorted(str(Program.parse(p)) for p in e["programs"])))
                  for e in resp)


class RemoveBadDreams(unittest.TestCase):
    def test_specification(self):
        table = {(1, None, 3): (-1., 0., ["a"]),   # dominated by (1, 2, 3)
                 (1, 2, 3): (-2., 0., ["b"]),
                 (None, 2, 4): (-1., 0., ["c"]),   # only (None,2,4) has output 4 at position 2
                 (5, None, None): (-1., 0., ["d"])}
        kept = {k for k, _ in H.remove_bad_dreams(table)}
        self.assertEqual(kept, {(1, 2, 3), (None, 2, 4), (5, None, None)})


@unittest.skipUnless(os.path.exists(BINARY), "OCaml helmholtz binary not available")
class HelmholtzParity(unittest.TestCase):
    def test_saturated_finite_behaviours(self):
        # total primitives only: partial ones (car/cdr) create None outputs, and then the dominance filter
        # depends on how deep each implementation enumerated, so the comparison would not be exact
        g = Grammar.uniform([p for p in McCarthyPrimitives() if p.name not in ("car", "cdr", "fix1")])
        request = arrow(tint, tint, tbool)
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]
        msg = {"request": request.json(), "timeout": 5., "DSL": g.json(), "extras": inputs,
               "evaluationTimeout": 0.01}
        ref = canon(ocaml(msg))
        mine = canon(H.run_message(dict(msg, timeout=120., pyecUpperBound=19.5)))
        self.assertEqual(len(ref), 32, "all 2^5 boolean behaviours should be reached")
        self.assertGreater(len(ref), 5, "a vacuous parity check is not evidence")
        self.assertEqual([x[0] for x in ref], [x[0] for x in mine])
        self.assertEqual(ref, mine)


if __name__ == "__main__":
    unittest.main()
