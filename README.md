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
  export_web.py        results/ → docs/data.json (incl. per-stage program pools for the Try-it demo)
  verify_try.cjs       checks the Try-it solver on random scrambles
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

## v7 verified-backend ablation — 10 November 2026

Fresh 24-arm rerun (stages/full × NN/no-NN × library/no-library × seeds 0–2) through `experiments/launch_v6.sh` on the validated pyrepro backend. Regenerate with `python experiments/summarize_v7.py`; viewer: [docs/v7.html](https://ut-tomo.github.io/dreamcoder-rubik/v7.html).

21/24 arms complete. Pending: full_nn_lib_s1, full_nonn_lib_s2, full_nonn_nolib_s0.

Backend: dreamcoder-pyrepro fdf68ad (pyfast enumerator, pyvs compressor). This backend predates the list-domain parity fixes 0f7f479 and e473c9b; the cube results were not rerun on them.

Each cell is mean ± sd over the completed seeds (seed values in parentheses). E2E = solved out of the same 100 unseen random cubes (seed 9999); train = training tasks solved after 20 rounds; lib = learned library size at round 20. 20 rounds, -t 5, 4 CPUs; NN arms use contextual recognition + biasOptimal.

| Training | Arm | seeds | E2E solved / 100 | Train solved | Library size | Mean moves (solved cubes) |
|---|---|---|---|---|---|---|
| stages | NN + library | 3/3 | 100 ± 0 (100 / 100 / 100) | 119 ± 0 / 119 (119 / 119 / 119) | 35 ± 4.36 | 82.08 ± 14.56 |
| stages | NN, no library | 3/3 | 100 ± 0 (100 / 100 / 100) | 119 ± 0 / 119 (119 / 119 / 119) | 0 ± 0 | 51.72 ± 4.71 |
| stages | No NN + library | 3/3 | 94.67 ± 5.51 (95 / 89 / 100) | 108 ± 7.81 / 119 (112 / 113 / 99) | 27.67 ± 4.93 | 74.51 ± 15.19 |
| stages | No NN, no library | 3/3 | 6 ± 0 (6 / 6 / 6) | 55.33 ± 6.66 / 119 (61 / 57 / 48) | 0 ± 0 | 20.67 ± 0 |
| full | NN + library | 2/3 | 12.5 ± 2.12 (11 / 14) | 11 ± 5.66 / 120 (15 / 7) | 2.5 ± 2.12 | 21.81 ± 13.77 |
| full | NN, no library | 3/3 | 8.33 ± 0.58 (8 / 8 / 9) | 7.33 ± 2.08 / 120 (8 / 9 / 5) | 0 ± 0 | 12.44 ± 1.47 |
| full | No NN + library | 2/3 | 0 ± 0 (0 / 0) | 0 ± 0 / 120 (0 / 0) | 0 ± 0 | — |
| full | No NN, no library | 2/3 | 0 ± 0 (0 / 0) | 0 ± 0 / 120 (0 / 0) | 0 ± 0 | — |

With the curriculum, NN guidance reaches 100/100 E2E with or without the library, while plain search needs the library (95 vs 6). On full random cubes only NN arms solve anything (≈8–13/100); plain search solves no training task. The full-task NN library gain is within seed noise until the pending seed lands.

## Verified results — 8 November 2026 (v3, superseded by v7 above)

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
