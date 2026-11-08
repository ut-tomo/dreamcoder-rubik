#!/bin/bash
# v6: final seeded 8-arm cube retraining on the official frontend + pyec backend (pyfast / pyvs / pyec.helmholtz).
#   arms: kind {stages, full} x recognition {nn: contextual + biasOptimal + auxiliary, nonn} x library {lib, nolib}
#   seeds 0 1 2; hyperparameters as v5 (paper full-system config, -t 5, 20 iterations), plus --biasOptimal for NN arms
# usage: experiments/launch_v6.sh HOST_SLOTS JOBFILE    (runs the jobs listed in JOBFILE, HOST_SLOTS at a time)
#        experiments/launch_v6.sh --list                (prints all 24 job specs)
set -u
ROOT=/mnt/home/unno/lab/rubiks_project/dreamcoder-rubik
OUT=results/v6
LOGS=/mnt/home/unno/lab/rubiks_project/logs/v6
PY=/mnt/home/unno/miniforge/envs/dcec/bin/python

if [ "${1:-}" = "--list" ]; then
  for seed in 0 1 2; do for kind in stages full; do for nn in nn nonn; do for lib in lib nolib; do
    echo "$kind $nn $lib $seed"
  done; done; done; done
  exit 0
fi

SLOTS=$1; JOBS=$2
mkdir -p "$ROOT/$OUT" "$LOGS"
run_one() {
  kind=$1; nn=$2; lib=$3; seed=$4
  name=${kind}_${nn}_${lib}_s${seed}
  [ -f "$ROOT/$OUT/$name.model.json" ] && { echo "skip $name (done)"; return; }
  flags=""
  [ "$nn" = nonn ] && flags="$flags --no-recognition"
  [ "$nn" = nn ] && flags="$flags --biasOptimal"
  [ "$lib" = nolib ] && flags="$flags --no-consolidation"
  cd "$ROOT" && SOURCE_COMMIT=${SOURCE_COMMIT:-} PYEC_HELMHOLTZ=1 PYEC_PYPY=/mnt/home/unno/miniforge/envs/pypy/bin/pypy3 OMP_NUM_THREADS=1 \
    nice -n 5 $PY experiments/run_ec.py --kind $kind --seed $seed --iterations 20 -t 5 -c 4 $flags \
    --out $OUT/$name.json > "$LOGS/$name.log" 2>&1
  echo "finished $name rc=$? $(date +%FT%T) on $(hostname)"
}
export -f run_one; export ROOT OUT LOGS PY
xargs -P "$SLOTS" -L 1 bash -c 'run_one $0 $1 $2 $3' < "$JOBS"
