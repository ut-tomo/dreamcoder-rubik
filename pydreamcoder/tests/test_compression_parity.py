"""Assertion-based differential test: pyec.vs_compression vs the official OCaml ./compression binary on
saved compression messages (the exact JSON the Python frontend sends, saved by the official code under
<run>/compressionMessages/).

PYEC_COMPRESSION_MESSAGES: comma-separated glob(s) of message files (required; test is skipped otherwise).
PYEC_COMPRESSION_CACHE: directory to cache OCaml responses (OCaml is deterministic for a given message).
Run: PYTHONPATH=pydreamcoder:pydreamcoder/ec python -m unittest pydreamcoder/tests/test_compression_parity.py
"""
import glob
import hashlib
import json
import os
import subprocess
import unittest

EC_ROOT = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
COMPRESSOR = os.environ.get("EC_COMPRESSOR", os.path.join(EC_ROOT, "compression"))
MESSAGES = sorted(f for g in os.environ.get("PYEC_COMPRESSION_MESSAGES", "").split(",") if g
                  for f in glob.glob(g))
CACHE = os.environ.get("PYEC_COMPRESSION_CACHE")
# Production weights are compared with a tolerance: OCaml breaks exact ties between equal-posterior programs
# at the topK cut (restrict) by float noise from summing over a Core Hashtbl.Poly in hash order, so the
# programs used for parameter estimation can differ among mathematically tied ones (see debug_ties.py).
# Inventions and rewritten frontiers must still be identical.
LOGP_TOL = float(os.environ.get("PYEC_COMPRESSION_LOGP_TOL", "1e-2"))


def load_primitives():
    from dreamcoder.domains.list.listPrimitives import (primitives, McCarthyPrimitives,
                                                         bootstrapTarget_extra, basePrimitives)
    for f in (basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra):
        f()
    for m in os.environ.get("PYEC_PRIMITIVE_MODULES", "").split(","):
        if m.strip():
            __import__(m.strip())


def ocaml(message_bytes):
    key = hashlib.sha1(message_bytes).hexdigest()
    path = os.path.join(CACHE, key + ".json") if CACHE else None
    if path and os.path.exists(path):
        return json.load(open(path))
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))
    p = subprocess.run([COMPRESSOR], input=message_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, check=True)
    r = json.loads(p.stdout.decode())
    if path:
        os.makedirs(CACHE, exist_ok=True)
        json.dump(r, open(path, "w"))
    return r


def port(message_bytes):
    """pyec in PyPy (PYEC_PYPY, as in the training runs) or in-process."""
    pypy = os.environ.get("PYEC_PYPY")
    if pypy:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ec = os.environ.get("PYEC_EC_ROOT", os.path.join(root, "ec"))
        env = dict(os.environ, PYTHONPATH=":".join([root, ec, os.environ.get("PYTHONPATH", "")]))
        p = subprocess.run([pypy, "-m", "pyec.compress_cli"], input=message_bytes, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=env, check=True)
        return json.loads(p.stdout.decode())
    from pyec.vs_compression import run_message
    return run_message(json.loads(message_bytes.decode()))


def normalise(resp):
    from dreamcoder.program import Program
    prods = [(str(Program.parse(p["expression"])), p["logProbability"]) for p in resp["DSL"]["productions"]]
    frontiers = [sorted((str(Program.parse(e["program"])), e["logLikelihood"]) for e in f["programs"])
                 for f in resp["frontiers"]]
    return resp["DSL"]["logVariable"], prods, frontiers


@unittest.skipUnless(MESSAGES and os.path.exists(COMPRESSOR), "no saved messages / OCaml compressor")
class CompressionParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_primitives()

    def test_messages(self):
        for fn in MESSAGES:
            with self.subTest(message=fn):
                raw = open(fn, "rb").read()
                lv_o, prods_o, fr_o = normalise(ocaml(raw))
                lv_p, prods_p, fr_p = normalise(port(raw))
                inv_o = [e for e, _ in prods_o if e.startswith("#")]
                inv_p = [e for e, _ in prods_p if e.startswith("#")]
                self.assertEqual(set(inv_o), set(inv_p), "different inventions")
                self.assertEqual(len(inv_o), len(inv_p))
                self.assertEqual(sorted(e for e, _ in prods_o), sorted(e for e, _ in prods_p))
                lo, lp = dict(prods_o), dict(prods_p)
                worst = max([abs(lo[e] - lp[e]) for e in lo] + [abs(lv_o - lv_p)])
                print("\n%s: %d inventions, max |dlogP| = %.2e" % (os.path.basename(fn), len(inv_o), worst))
                self.assertLessEqual(worst, LOGP_TOL)
                self.assertEqual(len(fr_o), len(fr_p))
                for i, (a, b) in enumerate(zip(fr_o, fr_p)):
                    self.assertEqual([x for x, _ in a], [x for x, _ in b], "frontier %d rewritten differently" % i)


if __name__ == "__main__":
    unittest.main()
