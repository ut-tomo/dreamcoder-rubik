# Official-backend port: status

Updated 2026-11-08. This file overrides any stronger completion claims in
the recovered backend docstrings or previous chat reports.

## Source and scope

Official upstream: `ellisk42/ec`, commit
`cb0e63f5c33cd2de360b791038b0f5272750270e`.
`ec/dreamcoder/` reuses 52 upstream Python files. A byte comparison with
that checkout on 2026-11-08 found changes only in `enumeration.py`,
`compression.py`, and `dreamcoder.py`: backend registration and CLI choices.
`patch_ec.py` records those changes. `ec/LICENSE` is retained verbatim.
The ported backend lives in `pyec/`; cube integration is in
`src/rubik/ec_domain.py` and `experiments/run_ec.py`.

The existing simplified implementation in `src/pydreamcoder/` is preserved
for historical results. It is not the backend used by `run_ec.py`.

## Independently checked after recovery

- Official Python enumeration, port enumeration, and OCaml solver each
  enumerated 161 programs at upper bound 11 on the bootstrap list grammar.
  Python reference/port program sets and description lengths were identical.
  One loaded-server timing was 5.4s / 0.2s / 0.1s respectively; this is not
  a general performance claim (the OCaml path also evaluates candidates).
- Three differential unittest methods passed, covering polymorphic and
  higher-order requests, nonuniform contextual grammar, and adjacent
  search-band partitioning. Every comparison checks program sets and costs.
- The remote list benchmark processes were running when inspected.

Run the bounded differential suite in the configured server environment:

```
PYTHONPATH=pydreamcoder:pydreamcoder/ec python -m unittest discover \
  -s pydreamcoder/tests -p 'test_enumeration_parity.py' -v
```

## Verified on 2026-11-08 (second pass)

All tests below compare against the official OCaml binaries built from the same upstream commit
(`/mnt/home/unno/lab/rubiks_project/dreamcoder-ec/{solver,compression,helmholtz}`) and assert equality.

- `tests/test_ocaml_enumeration_parity.py` (4 tests, pass): a task with no examples makes the OCaml solver
  return the whole enumerated band with priors. Program sets and priors are identical for
  McCarthy+fix1 `list(int)->int` (88 programs at ub 9, 458 at ub 11; 39/196 recursive),
  `int->list(int)->list(int)` with fix2 (255 programs, 69 recursive), a non-recursive arity, and
  `maxParameters` 0/1/2 with `REAL`. The port now implements the enumeration.ml top-level fix point
  rewrite and the free-parameter bound. `REAL` programs are scored by the official Python task likelihood
  (DifferentiableTask fitting), not OCaml's optimiser: not bit-identical for real-valued domains.
- `pyec/helmholtz.py` ports Helmholtz.ml/Dreaming.ml (`--biasOptimal` dreams). It is opt-in via
  `PYEC_HELMHOLTZ=1` (4th patch to the vendored frontend, recorded in `patch_ec.py`).
  `tests/test_helmholtz_parity.py` (pass): identical 32 behaviours, priors and tied-program sets on a
  saturating `int->int->bool` space; remove_bad_dreams checked against its specification. Evaluation uses
  Python primitives (OCaml uses its own), with the same default 1 ms timeout.
- `tests/test_compression_parity.py` on the saved list-run messages (PyPy, as in training): inventions
  and rewritten frontiers identical on all completed messages (5 of 6 at the time of writing; one large
  rec_lib message still running). Production weights: identical on 3 messages, max |dlogP| 1.1e-3 and
  1.2e-3 on 2. Root cause, reproduced on a minimal 3-frontier message: at the `restrict` topK cut,
  several programs have mathematically equal posteriors. OCaml breaks the tie by float noise from summing
  a Core `Hashtbl.Poly` in hash order; choosing one of the other tied programs reproduces OCaml's initial
  score exactly (-74.586201). Bit-exact weights would require emulating OCaml's hash-table iteration order;
  not done. The test therefore asserts exact inventions/frontiers and weights within 1e-2.
- Cube pipeline smoke test with `--biasOptimal` (contextual + auxiliary + Helmholtz dreams via the cube
  special hash) runs end to end. `perm.Alg.__reduce__` (previously only on the server copy) is now in git.
- The PyPy worker crashed once with "Fatal RPython error: AssertionError" (a JIT fault, not a Python
  exception) during recognition-guided enumeration. `solveForTask_pyfast` now reruns such a job
  in-process with the remaining budget.

## List benchmark (bounded check, contextual=False, 4 iterations, -t 60, split 0.5, -R 240)

Held-out test hits by iteration (0..3), out of 78:

| arm | iter 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| no recognition, no library | 3 | 3 | 3 | 3 |
| no recognition, library | 3 | 7 | 7 | 8 |
| recognition, no library | 3 | 13 | 16 | 18 |
| recognition + library | 3 | 17 | 33 | 40 |

The ordering matches the paper's ablation direction (full > no-library > no-recognition > neither). It
is not a reproduction of the published curves: budgets are 12x smaller than the paper's list runs
(-t 720, 20 iterations, batch 10), non-contextual, and one seed. A bounded run with the paper's flags
(`--contextual --biasOptimal --auxiliary`, port Helmholtz, seed 1, -t 60, 4 iterations) is in
`runs/list_repro/paper_ctx_bo_s1` (first attempt crashed in PyPy, see above).

## Remaining gaps

- Paper-curve parity is not established (budgets, seeds, batch size, iterations differ).
- Compression weights differ from OCaml only through equal-posterior tie-breaks (above).
- Wall-clock budgets: the port evaluates and enumerates at Python speed, so it reaches fewer programs
  (and fewer Helmholtz dreams) per second than OCaml under the same timeout.
- The backend is not packaged by the legacy `pyproject.toml`.

## Cube runs

- v5 (`results/v5`): provisional, launched before verification, killed. Do not use.
- v6 (`results/v6`, logs in `logs/v6`): final seeded 8 arms x seeds 0-2, launched 2026-11-08 ~21:05
  JST from commit 224270a via `experiments/launch_v6.sh`. NN arms use contextual recognition +
  `--biasOptimal` + auxiliary loss. Every run writes `<name>.manifest.json`; Python/NumPy/PyTorch are
  seeded. Wall-clock enumeration budgets keep runs nondeterministic.

## Locations

- Main Git checkout on Mac:
  `/Users/tomoyuki/build/others/rubiks-proj/dreamcoder-rubik`
- Shared server working copy (not a Git checkout):
  `/mnt/home/unno/lab/rubiks_project/dreamcoder-rubik`
- List logs: `/mnt/home/unno/lab/rubiks_project/runs/list_repro/`
- Cube logs: `/mnt/home/unno/lab/rubiks_project/logs/v5/`
- Recovery snapshot on Mac:
  `/Users/tomoyuki/build/others/rubiks-proj/port-recovery-20261108/`

Do not overwrite the live server backend while its processes are running.
Keep final controlled retraining in a new result directory, separate from
the provisional v5 outputs. Commit as `ut-tomo`; no website results should
be replaced until the reproduction gate and model/export checks pass.
