# Official-backend port: recovered work, not a verified reproduction

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

## Open validation gaps

- The backend explicitly rejects `REAL` free parameters and `fix1`/`fix2`
  recursion. It is therefore not yet a complete replacement for OCaml.
- The recovered compressor's equality/performance claims need independent,
  assertion-based reruns on saved messages. Hash-order tie-breaking does
  not by itself prove score or subsequent-learning equivalence.
- The active four-arm list checks use `contextual=False` and four rounds.
  They compare an upstream configuration; they are not sufficient evidence
  of reproducing the paper's best contextual model or its published curves.
- Benchmark seeds, task splits, budgets, and evaluation need explicit
  manifests and matching against the official run and paper configuration.
- Cube jobs were already launched by the earlier worker before the
  reproduction gate was independently checked. Preserve their outputs as
  provisional; do not publish them as verified reproduction results.
- Seed handling in `run_ec.py` currently seeds Python but needs review for
  NumPy/PyTorch reproducibility before the final controlled retraining.
- The new backend is not yet packaged by the legacy `pyproject.toml`.
  Dependency setup and portable data paths still need documentation.
- Interactive browser search currently targets the old engine. It must be
  updated and checked against the final official contextual model, along
  with move-by-move animation verification.

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
