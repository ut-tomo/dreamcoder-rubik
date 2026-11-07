# pydreamcoder

A small, dependency-free Python reimplementation of DreamCoder
(Ellis et al., *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*, PLDI 2021).
The official implementation ([ellisk42/ec](https://github.com/ellisk42/ec)) needs OCaml and a Singularity image;
this package runs on any Python ≥ 3.9 (tested with the lab servers' Python 3.10/3.12).

It is domain-independent; the cube domain lives in the parent directory. Only the recognition model needs PyTorch.

## Layout
```
src/pydreamcoder/
  types.py       polymorphic types + unification
  program.py     lambda calculus with de Bruijn indices, evaluation, type inference
  grammar.py     probabilistic unigram grammar, top-down enumeration by description length
  library.py     sleep/abstraction: fragment mining, rewriting, MDL scoring, grammar refit
  recognition.py recognition model, program sampling, dreaming
  dreamcoder.py  Task, wake phase (shared or per-task), wake-sleep loop
repro/           reproduction of the paper's list-processing results
  data/list_tasks.json   official tasks copied from ec @ cb0e63f
```

## Reproducibility check
```
python repro/run_list.py --iterations 6 --timeout 600 --out results/list_full.json
python repro/run_list.py --iterations 6 --timeout 600 --no-compress --out results/list_baseline.json
```
Claim being checked: with the same enumeration budget, library learning solves more list tasks over iterations
than enumeration with the fixed base library, and the learned routines resemble the paper's (e.g. `map`/`fold` specialisations).
Results are recorded in `repro/RESULTS.md`.

## Differences from the paper (current status)
- **Recognition model is unigram** (`recognition.py`): an MLP from domain-supplied task features to one log-weight per
  library entry, trained on replays + fantasies (dreams) with the exact −log P(program | grammar) loss. The paper's best
  model is the bigram/contextual one, which is not implemented. Feature extractors and fantasy-task construction are domain hooks.
- **Simplified compressor**: candidate routines are closed subterms and one-hole fragments mined from frontiers,
  added greedily under the MDL score. The paper's version-space refactoring (which also finds multi-argument abstractions
  after η/β-refactoring) is not implemented.
- **Enumeration** is pure Python (≈10³–10⁵ programs/s), orders of magnitude slower than the OCaml solver,
  so absolute solve rates are lower at equal wall-clock time. Compare against the paper's ablations qualitatively.
