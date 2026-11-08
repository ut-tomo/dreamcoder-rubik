"""Bounded differential checks against the unchanged official Python enumerator.

Run with PYTHONPATH=pydreamcoder:pydreamcoder/ec python -m unittest discover
-s pydreamcoder/tests -p 'test_*.py'. No OCaml executable is required.
"""
import random
import time
import unittest

import dreamcoder.grammar as grammar_module
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.program import Primitive
from dreamcoder.type import Context, arrow, tint, tlist, t0
from pyec import enumerator as port


class EnumerationParity(unittest.TestCase):
    def setUp(self):
        # The OCaml implementation deliberately disables this Python-only filter.
        self.symmetry = grammar_module.violatesSymmetry
        grammar_module.violatesSymmetry = lambda *args: False
        self.primitives = [
            Primitive("parity-zero", tint, 0),
            Primitive("parity-one", tint, 1),
            Primitive("parity-add", arrow(tint, tint, tint), lambda a: lambda b: a+b),
            Primitive("parity-id", arrow(t0, t0), lambda x: x),
        ]

    def tearDown(self):
        grammar_module.violatesSymmetry = self.symmetry

    def check_band(self, grammar, request, lower, upper):
        reference = {str(p): ll for ll, _, p in grammar.enumeration(
            Context.EMPTY, [], request, upperBound=upper, lowerBound=lower,
            maximumDepth=99)}
        cg = port.CG(grammar)
        actual = {}
        en = port.Enumerator(cg, time.time()+30)
        en.programs(cg.no_context, port.EMPTY, port.from_official(request), [],
                    lower, upper, 99, lambda p, c, ll: actual.__setitem__(str(p), ll))
        self.assertTrue(reference, "a vacuous parity check is not evidence")
        self.assertEqual(set(reference), set(actual))
        for p, ll in reference.items():
            self.assertAlmostEqual(ll, actual[p], places=10, msg=p)
        return actual

    def test_uniform_polymorphic_and_higher_order_requests(self):
        grammar = Grammar.uniform(self.primitives)
        for request in (tint, arrow(tint, tint), arrow(arrow(tint, tint), tint),
                        arrow(tlist(tint), tlist(tint))):
            with self.subTest(request=str(request)):
                self.check_band(grammar, request, 0., 6.)

    def test_nonuniform_contextual_grammar(self):
        rng = random.Random(83)
        grammar = ContextualGrammar.fromGrammar(Grammar.uniform(self.primitives))
        grammar = grammar.randomWeights(lambda _: rng.uniform(-2., 0.))
        self.check_band(grammar, arrow(tint, tint), 0., 7.)

    def test_adjacent_bands_partition_programs(self):
        grammar = Grammar.uniform(self.primitives)
        whole = self.check_band(grammar, arrow(tint, tint), 0., 7.)
        left = self.check_band(grammar, arrow(tint, tint), 0., 3.5)
        right = self.check_band(grammar, arrow(tint, tint), 3.5, 7.)
        self.assertFalse(set(left) & set(right))
        self.assertEqual(set(whole), set(left) | set(right))


if __name__ == "__main__":
    unittest.main()
