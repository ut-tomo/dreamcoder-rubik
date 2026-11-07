# dreamcoder-rubik

Can a program-synthesis system discover the kind of algorithm library that human speedcubers memorise?
This project applies DreamCoder-style wake-sleep learning to the 2×2×2 cube and measures what each of its two
learning mechanisms (**library learning** and **neural search guidance**) contributes.

**Interactive results:** <https://ut-tomo.github.io/dreamcoder-rubik/>

```
pydreamcoder/        the learning engine: a pure-Python DreamCoder (domain-independent)
  src/pydreamcoder/    types, programs, grammar + enumeration, compression, recognition model, wake-sleep loop
  repro/               list-processing tasks from the paper, used to sanity-check the engine
src/rubik/           the cube domain
  cube2.py             2×2×2 as a permutation of 24 stickers, built from 3D geometry; stage predicates
  perm.py              algs as group elements, whole-cube symmetries, orientation normalisation
  sample.py            uniform sampling of reachable states with chosen corners solved
  dsl.py               primitives
  tasks.py             task sets (stages, full solve) and the recognition-model hooks (features, fantasies)
experiments/
  run.py               one training run (any of the four arms)
  export_web.py        results/ → docs/data.json
results/             raw results of the runs reported below
docs/                GitHub Pages site (Three.js viewer)
```

## Reproduce the current experiment (v3)

```sh
python experiments/run.py --kind stages --recognition --seed 0 --iterations 20 --per-task 5 --out results/v3/stages_nn_lib_s0.json
# Repeat seeds 0/1/2 and kinds stages/full; drop --recognition and/or add --no-compress for other arms.
python experiments/export_web.py
```

## Setup

Twenty wake–abstraction–dreaming rounds, four arms (neural guidance × library learning), two training strategies, three seeds: 24 runs. Staged training uses 119 distinct subgoal tasks (24 each for stages 1–4, 23 for stage 5); full training uses 120 uniformly random unsolved cubes. Both exclude the fixed 100-cube end-to-end test set (seed 9999).

Stages: DRF corner → +DRB corner → D layer → +U face orientation → solved. Goals and stage transitions normalize whole-cube orientation with DLB home. Programs use R/U/F, seq/inv/setup/comm and x/y/z/mirror; no human algorithm library is provided. Up to three new MDL routines per round, structure penalty 1.0, frontier size 3. Recognition is a 153-input MLP with two 128-unit ReLU layers, 800 Adam steps per round on successful replays and 2000 fantasies.

NN wake search receives 5 seconds per task; plain search shares 5 seconds × task count across a batch. E2E staged evaluation chains stage solvers with reorientation after each stage, while full evaluation receives 25 seconds per cube (pooled for plain search). This matches nominal aggregate search wall time, not enumeration counts, total training cost or exact per-cube cost. NN models can train on fantasies even when there are no successful replays.

## Verified results — 8 November 2026

All 15 relaunched runs and nine earlier completed runs finished 20 rounds and E2E evaluation. Values below are seeds 0 / 1 / 2, each on the **same 100 unseen random cubes**.

| Strategy | Arm | E2E solved / 100 | Mean moves among solved cubes |
|---|---|---|---|
| Stages | NN + library | 100 / 100 / 100 | 465.2 / 997.8 / 192.6 |
| Stages | NN only | 16 / 18 / 25 | 34.7 / 36.7 / 43.4 |
| Stages | Library only | 100 / 100 / 100 | 240.7 / 120.1 / 137.3 |
| Stages | Neither | 17 / 17 / 17 | 19.4 / 19.4 / 19.4 |
| Full | NN + library | 0 / 0 / 0 | — |
| Full | NN only | 0 / 0 / 0 | — |
| Full | Library only | 1 / 1 / 1 | 9 / 9 / 9 |
| Full | Neither | 1 / 1 / 1 | 9 / 9 / 9 |

Library learning is effective with staged training but cannot bootstrap full random-cube training here: all full-task arms solve zero training tasks. Neural guidance shows no consistent E2E advantage at this budget. Long expanded move sequences reflect a program-probability objective, not move minimization; comparisons of lengths are conditional on solving different subsets. Reusing the test set across seeds does not yield 300 independent test cubes per arm.

The exporter requires all 24 final results and replays every exported training and E2E solution (2,023 total), including inter-stage normalization. Only small JSON results were fetched from shared NFS; no raw datasets or checkpoints were copied. Older v2 summaries remain as historical artifacts, not the site's current benchmark.

## Differences from the paper

This is a pure-Python DreamCoder-style reimplementation, not the official engine: unigram grammar and recognition, simplified MDL compression without version-space refactoring. See [pydreamcoder/README.md](pydreamcoder/README.md). Policy probabilities describe building-block preferences, not causal evidence that the neural network improved solve rate.
