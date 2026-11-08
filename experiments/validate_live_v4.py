"""Reference (Python) side of the JS-vs-Python check of the in-browser search.

Runs the original pydreamcoder enumerator (Grammar.enumerate, same windows/depth as
wake_per_task) with the exported models in docs/models/, on the first N of the fixed
E2E test cubes, with a *deterministic* budget of --max-programs enumerated programs per
stage (wall-clock budgets differ between Python and JS). Writes, per config and cube,
the program found at each stage, so experiments/validate_live.cjs can demand identical
results.

    python3 experiments/validate_live.py --n 20 --max-programs 20000 --out /tmp/py_live.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "pydreamcoder/src")]
sys.setrecursionlimit(10000)
from pydreamcoder.grammar import Grammar  # noqa: E402
from pydreamcoder.program import Abstraction, Application, Index, Invented  # noqa: E402
from pydreamcoder.types import Context  # noqa: E402
from rubik import perm  # noqa: E402
from rubik.dsl import primitives, talg  # noqa: E402
from rubik.tasks import STAGES, features, random_states  # noqa: E402

PRIMS = {p.name: p for p in primitives("ruf")}


def parse(src):
    i = 0

    def ws():
        nonlocal i
        while i < len(src) and src[i] == " ":
            i += 1

    def atom():
        nonlocal i
        ws()
        c = src[i]
        if c == "#":
            i += 1
            return Invented(atom())
        if c == "$":
            j = i = i + 1
            while src[i].isdigit():
                i += 1
            return Index(int(src[j:i]))
        if c == "(":
            i += 1
            ws()
            if src.startswith("lambda ", i):
                i += 7
                b = atom()
                ws()
                i += 1
                return Abstraction(b)
            f = atom()
            ws()
            while src[i] != ")":
                f = Application(f, atom())
                ws()
            i += 1
            return f
        j = i
        while i < len(src) and src[i] not in " ()":
            i += 1
        return PRIMS[src[j:i]]

    p = atom()
    assert str(p) == src, (str(p), src)
    return p


def load(path):
    m = json.load(open(path))
    gr = m["grammar"]
    prods = []
    for l, t, s in gr["productions"]:
        p = parse(s)
        prods.append((l, p.tp if isinstance(p, Invented) else p.tp, p))
    return Grammar(gr["log_variable"], prods), m["recognition"]


def grammar_for(g, rec, start, k):
    if rec is None:
        return g
    h = features(start, k)
    for li, L in enumerate(rec["layers"]):
        out = []
        for w, b in zip(L["W"], L["b"]):
            s = b
            for wk, hk in zip(w, h):
                s += wk * hk
            out.append(max(0.0, s) if li < len(rec["layers"]) - 1 else s)
        h = out
    weights = {k_: h[i] for i, k_ in enumerate(rec["vocab"][:-1])}
    return g.with_weights(weights, h[-1])


class Stop(Exception):
    pass


def search(g, start, goal, max_programs):
    n = 0
    lower, upper = 0.0, 1.5
    try:
        while upper <= 200:
            for l, _, p in g.enumerate(Context(), [], talg, upper, lower, 12):
                if n >= max_programs:
                    raise Stop()
                n += 1
                try:
                    s = perm.apply_to(start, p.evaluate([]))
                except Exception:
                    continue
                if goal(s):
                    return str(p), s, n
            lower, upper = upper, upper + 1.5
    except Stop:
        pass
    return None, None, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-programs", type=int, default=20000)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cubes = random_states(100, 9999)[: a.n]
    files = sorted(p for p in (ROOT / "docs/models").glob("*_s*.json"))
    if a.configs != "all":
        files = [f for f in files if f.stem in a.configs.split(",")]
    res = {}
    for f in files:
        g, rec = load(f)
        kind = f.stem.split("_")[0]
        out = []
        for c in cubes:
            s = perm.normalize(c)
            parts = []
            for k in ([0, 1, 2, 3, 4] if kind == "stages" else [4]):
                goal = STAGES[k][2]
                if goal(s):
                    parts.append("skip")
                    continue
                prog, s2, n = search(grammar_for(g, rec, s, k), s, goal, a.max_programs)
                parts.append(prog)
                if prog is None:
                    break
                s = s2
            out.append(parts)
        res[f.stem] = out
        print(f.stem, sum(1 for p in out if None not in p and len(p) == (5 if kind == "stages" else 1)), "/", len(out), flush=True)
    json.dump({"n": a.n, "max_programs": a.max_programs, "results": res}, open(a.out, "w"), indent=0)


if __name__ == "__main__":
    main()
