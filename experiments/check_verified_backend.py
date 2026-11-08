"""Non-vacuous CPU-only checks of the validated backend's cube integration."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
backend = Path(os.environ['DREAMCODER_BACKEND_ROOT']).resolve()
sys.path[:0] = [str(root / 'src'), str(backend), str(backend / 'ec')]
os.environ['PYREPRO_PRIMITIVE_MODULES'] = 'rubik.ec_domain:register_all'
os.environ['PYREPRO_EXTRA_PATH'] = str(root / 'src')
from rubik import ec_domain as D, perm as P, cube2
from dreamcoder.grammar import Grammar, ContextualGrammar
from pyrepro.backend import solveForTask_pyfast, pyvsInduce
from pyrepro import helmholtz as H

D.register_all()
g = Grammar.uniform(D.primitives())
t = D.CubeTask('integration-undo-R', P.apply_to(cube2.SOLVED, P.inv(P.move('R'))), 4)
assert t.logLikelihood(next(p for p in D.primitives() if p.name == 'R')) == 0.
for grammar in [g, ContextualGrammar.fromGrammar(g)]:
    fs, _, _ = solveForTask_pyfast(g=grammar, tasks=[t], maximumFrontiers={t: 5},
                                 lowerBound=0., upperBound=5., budgetIncrement=5.,
                                 timeout=10., CPUs=1, evaluationTimeout=0.1)
    f = fs[t]
    assert not f.empty and any(str(e.program) == 'R' for e in f.entries)
    assert all(t.logLikelihood(e.program) == 0. for e in f.entries)
    print('PASS cube enumeration', type(grammar).__name__, len(f.entries), flush=True)
fs, _, _ = solveForTask_pyfast(g=g, tasks=[t], maximumFrontiers={t: 5}, lowerBound=0.,
                             upperBound=5., budgetIncrement=5., timeout=10., CPUs=1)
g2, rewritten = pyvsInduce(g, list(fs.values()), topK=2, pseudoCounts=30.,
                          aic=1., structurePenalty=1.5, a=3, CPUs=1)
assert len(rewritten) == 1 and not rewritten[0].empty
assert all(t.logLikelihood(e.program) == 0. for e in rewritten[0].entries)
print('PASS nonempty cube compression and executable rewrites', flush=True)
dreams = H.run_message(dict(request=D.talg.json(), DSL=g.json(), timeout=2.,
                           special='cube', extras=[], evaluationTimeout=0.1,
                           pyreproUpperBound=4.5))
assert len(dreams) > 1 and all(d['programs'] for d in dreams)
print('PASS cube Helmholtz', len(dreams), 'behaviours', flush=True)
print('PASS verified backend cube integration checks', flush=True)
