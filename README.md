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

## Quick start

Python ≥ 3.9. The engine has no dependencies; the recognition model needs PyTorch.

```
python experiments/run.py --recognition --seed 0 --out results/v2_nn_lib_s0.json         # NN + library
python experiments/run.py --recognition --no-compress --seed 0 --out results/v2_nn_nolib_s0.json
python experiments/run.py --seed 0 --out results/v2_nonn_lib_s0.json                     # library only
python experiments/run.py --no-compress --seed 0 --out results/v2_nonn_nolib_s0.json     # neither
python experiments/export_web.py
```

## Method

**Tasks.** Five chained stages: (1) DRF corner, (2) + DRB corner, (3) whole D layer, (4) + U face one colour, (5) solved.
Each task is a uniformly random reachable state in which the earlier stages hold and the current one does not.
States are compared up to whole-cube orientation (the cube is turned so the DLB corner is home before every goal check).
Training: 24 tasks per stage (120). Held-out test: 12 per stage (60), drawn with a different seed.

**Programs.** One type, `alg` (a move sequence, i.e. a permutation). Primitives: `R U F`; `seq`, `inv`,
`setup a b = a b a⁻¹`, `comm a b = a b a⁻¹ b⁻¹`; the whole-cube symmetries `x y z mirror` (conjugation; 48 in total).
No colours, pieces or human solving methods are built in. The generative model is a unigram probabilistic grammar.

**Each of the 8 rounds:**

1. *Wake.* Best-first enumeration by decreasing prior (description-length window widened by 1.5 nats, depth ≤ 12);
   the best 3 solutions per task are kept. Without recognition: one shared enumeration checked against all tasks for
   1 s × #tasks. With recognition: per-task enumeration under the predicted grammar, 1 s per task (compute-matched).
2. *Abstraction* (library arms). Candidate routines (closed subterms and one-hole fragments occurring in ≥ 2 tasks) are
   scored by MDL: Σ_tasks max log P(program | grammar) − 1.0 · Σ routine size, after rewriting all solutions and refitting
   the grammar. The best candidate is added greedily while the score improves, ≤ 3 per round. Purely symbolic, as in the paper.
3. *Dreaming* (NN arms). An MLP (153 inputs = 24 sticker slots × 6 colours one-hot + 5-way stage id; 2 × 128 ReLU) outputs a
   log-weight per library entry, i.e. a task-specific unigram grammar. It is trained for 800 Adam steps (lr 1e-3, batch 64)
   on replays (solved tasks and their programs) plus 2000 fantasies: a program p is sampled from the current grammar, an end
   state G satisfying a stage goal is drawn, and the task starts at G·p⁻¹. Loss: −log P(program | predicted grammar).

## Results (seeds 0 / 1)

| arm | train solved | held-out solved | avg. moves (held-out) |
|---|---|---|---|
| NN + library | 117 / 120 | **60 / 59** of 60 | 29 / 208 |
| NN only | 93 / 87 | 39 / 38 | 6.7 / 7.2 |
| library only | 120 / 120 | **60 / 60** | 141 / 66 |
| neither | 92 / 87 | 42 / 32 | 5.0 / 3.6 |

(Move counts are averaged over the tasks each arm solved, so arms that only solve easy tasks look short.
Nothing in the objective prefers short move sequences, which is why library solutions can be very long.)

- Library learning is what makes the late stages solvable at this budget: without it, stage 5 is almost never solved.
- At this budget the recognition network does not increase the solve rate: the library arms are at ceiling,
  and NN only ≈ neither (39/38 vs 42/32). It does learn a task-dependent policy, whose preferred building blocks
  differ clearly between stages (see the site), but that has not yet translated into more solved tasks.
- A harder setting (smaller budget, full solves, 3×3×3) is needed to measure the NN's effect on solve rate.

## Differences from the paper

See [`pydreamcoder/README.md`](pydreamcoder/README.md). In short: unigram (not bigram) recognition model,
simplified compressor (no version-space refactoring), pure-Python enumeration.
The engine is being checked against the official implementation ([ellisk42/ec](https://github.com/ellisk42/ec)).
