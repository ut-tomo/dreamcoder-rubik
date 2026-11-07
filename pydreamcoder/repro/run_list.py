"""Reproduce the qualitative list-domain claim: library learning solves more tasks
than enumeration with the fixed base library (paper Fig. / ablations, no recognition model).

    python repro/run_list.py --iterations 6 --timeout 600 --out results/list_full.json
    python repro/run_list.py --iterations 6 --timeout 600 --no-compress --out results/list_baseline.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
sys.setrecursionlimit(10000)

from list_domain import primitives, tasks  # noqa: E402
from pydreamcoder.dreamcoder import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iterations", type=int, default=6)
ap.add_argument("--timeout", type=float, default=600, help="wake-phase seconds per iteration")
ap.add_argument("--no-compress", action="store_true")
ap.add_argument("--structure-penalty", type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=3)
ap.add_argument("--limit", type=int, default=0, help="use only the first N tasks (smoke test)")
ap.add_argument("--families", default="core",
                help="'all', 'core' (default subset, see SUBSETS) or comma-separated family names")
ap.add_argument("--out", default=None)
a = ap.parse_args()

# Subset: parametric families (k = 0..5) share structure, so a learned routine
# (e.g. "map (+ k)") should transfer across a family -- the paper's core claim.
SUBSETS = {"core": ["add-k", "mult-k", "append-k", "prepend-k", "drop-k", "take-k",
                    "count-k", "has-k", "keep-mod-k", "remove-mod-k",
                    "sum", "len", "max", "reverse"]}
ts = tasks()
if a.families != "all":
    fams = set(SUBSETS.get(a.families, a.families.split(",")))
    ts = [t for t in ts if t.name.split(" with")[0] in fams]
if a.limit:
    ts = ts[: a.limit]
print(f"{len(ts)} tasks, {len(primitives())} primitives", flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
run(primitives(), ts, iterations=a.iterations, timeout=a.timeout, compress_on=not a.no_compress,
    structure_penalty=a.structure_penalty, max_new=a.max_new, out=a.out,
    log=lambda s: print(s, flush=True))
