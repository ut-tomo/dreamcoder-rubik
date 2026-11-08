"""Python reference for the in-browser search (docs/search.js), on the official DreamCoder stack.

Two checks, both on the exported web models (docs/models or --models):
  1. --check-model RESULTS_DIR: the float64 forward pass from the exported weights equals the official
     torch RecognitionModel.grammarOfTask (loaded from <cfg>.model.json) on the E2E cubes (max |dlog p|).
  2. search: per config and cube, the program found at each stage by pyec.enumerator (= OCaml solver order)
     with the contextual grammar of the exported model, under a deterministic budget of --max-programs
     enumerated programs per stage and the held-out testing semantics of maximumFrontier = 1 (best-prior
     hit of the 1.5-nat band containing the first hit). experiments/validate_live.cjs compare demands that
     search.js finds the identical programs.

    python experiments/validate_live.py --models docs/models --n 20 --max-programs 20000 --out /tmp/py_live.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "pydreamcoder"), str(ROOT / "pydreamcoder" / "ec")]
sys.setrecursionlimit(20000)
from dreamcoder.grammar import Grammar, ContextualGrammar  # noqa: E402
from dreamcoder.program import Program  # noqa: E402
from dreamcoder.type import Type  # noqa: E402
from pyec import enumerator as E  # noqa: E402
from rubik import ec_domain as D, perm as P  # noqa: E402

D.register_all()


def lse(xs):
    m = max(xs)
    return m if m == -math.inf else m + math.log(sum(math.exp(x - m) for x in xs))


def parse_type(s):
    return Type.fromjson(_tjson(s))


def _tjson(s):
    parts = [x.strip() for x in s.split("->")]
    t = {"constructor": parts[-1], "arguments": []}
    for a in reversed(parts[:-1]):
        t = {"constructor": "->", "arguments": [{"constructor": a, "arguments": []}, t]}
    return t


class WebModel(object):
    def __init__(self, path):
        m = json.load(open(path))
        self.raw = m
        self.prods = [(l, parse_type(t), Program.parse(s)) for l, t, s in m["grammar"]["productions"]]
        for _, _, p in self.prods:
            assert str(p) in {s for _, _, s in m["grammar"]["productions"]}
        self.g = Grammar(m["grammar"]["logVariable"], self.prods)
        self.rec = m["recognition"]
        if self.rec:
            types = {s: parse_type(t) for s, t in self.rec["productions"]}
            self.rec_prods = [(types[s], Program.parse(s)) for s, _ in self.rec["productions"]]

    def rows(self, start, stage_idx):
        """transition matrix rows (float64), as search.js transitionMatrix"""
        r = self.rec
        x = D.features(start, stage_idx)
        h = [math.tanh(b + sum(w * xi for w, xi in zip(W, x))) for W, b in zip(r["W1"], r["b1"])]
        m, n = r["n_grammars"], len(self.rec_prods) + 1
        if "M" in r:
            y = [b + sum(w * hi for w, hi in zip(W, h)) for W, b in zip(r["M"]["W"], r["M"]["b"])]
            return [y[i * n:(i + 1) * n] for i in range(m)]
        a = [b + sum(w * hi for w, hi in zip(W, h)) for W, b in zip(r["A"]["W"], r["A"]["b"])]
        bb = [b + sum(w * hi for w, hi in zip(W, h)) for W, b in zip(r["B"]["W"], r["B"]["b"])]
        k = r["rank"]
        return [[sum(a[i * k + q] * bb[q * n + j] for q in range(k)) for j in range(n)] for i in range(m)]

    def grammar(self, start, stage_idx):
        if not self.rec:
            return self.g
        rows = self.rows(start, stage_idx)

        def gv(row):  # grammarFromVector
            return Grammar(row[-1], [(row[k], t, p) for k, (t, p) in enumerate(self.rec_prods)])
        lib = {p: [gv(rows[j]) for j in self.rec["library_rows"][str(p)]] for _, p in self.rec_prods}
        return ContextualGrammar(gv(rows[-1]), gv(rows[-2]), lib)


class Stop(Exception):
    pass


def search_stage(model, start, stage_idx, goal, max_programs):
    cg = E.CG(model.grammar(start, stage_idx))
    en = E.Enumerator(cg, time.time() + 1e9)
    n = [0]
    best = [None]

    def k(p, _ctx, ll):
        if n[0] >= max_programs:
            raise Stop()
        n[0] += 1
        try:
            a = p.evaluate([])
            s = P.apply_to(start, a)
        except Exception:
            return
        if goal(s) and (best[0] is None or ll > best[0][1]):
            best[0] = (p, ll, s)
    lb = 0.
    try:
        while best[0] is None and lb + 1.5 <= 99:
            en.programs(cg.no_context, E.EMPTY, E.from_official(D.talg), [], lb, lb + 1.5, 9999, k)
            lb += 1.5
    except Stop:
        pass
    return best[0]


def solve(model, kind, start, max_programs):
    s = P.normalize(start)
    out = []
    for k in ([0, 1, 2, 3, 4] if kind == "stages" else [4]):
        goal = D.STAGES[k][2]
        if goal(s):
            out.append("skip")
            continue
        r = search_stage(model, s, k, goal, max_programs)
        if r is None:
            out.append(None)
            return out
        out.append(str(r[0]))
        s = r[2]
    return out


def check_model(models_dir, results_dir, cubes):
    """exported float64 forward == official torch grammarOfTask"""
    import torch
    from dreamcoder.recognition import RecognitionModel
    worst_all = 0.
    for f in sorted(Path(models_dir).glob("*_s*.json")):
        wm = WebModel(f)
        if not wm.rec:
            continue
        raw = json.load(open(Path(results_dir) / (f.stem + ".model.json")))
        g = Grammar(raw["grammar"]["logVariable"], wm.prods)
        rm = RecognitionModel(D.make_feature_extractor()([]), g, hidden=[64], activation="tanh", contextual=True)
        rm.load_state_dict({k: torch.tensor(v) for k, v in raw["recognition"]["state_dict"].items()})
        worst = 0.
        for i, s in enumerate(cubes):
            for k in range(5):
                t = D.CubeTask("c%d" % i, P.normalize(s), k)
                go = rm.grammarOfTask(t).untorch()
                gw = wm.grammar(P.normalize(s), k)
                for a, b in [(go.noParent, gw.noParent), (go.variableParent, gw.variableParent)] + \
                        [(x, y) for p in go.library for x, y in zip(go.library[p], gw.library[p])]:
                    la = {str(p): l for l, _, p in a.productions}
                    lb = {str(p): l for l, _, p in b.productions}
                    worst = max([worst, abs(a.logVariable - b.logVariable)] + [abs(la[q] - lb[q]) for q in la])
        print("%-24s max |dlogit| official torch vs exported float64: %.2e" % (f.stem, worst), flush=True)
        worst_all = max(worst_all, worst)
    return worst_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "docs" / "models"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-programs", type=int, default=20000)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-model", default=None, metavar="RESULTS_DIR")
    a = ap.parse_args()
    cubes = [tuple(c) for c in json.load(open(ROOT / "experiments" / "e2e_cubes.json"))][:a.n]
    if a.check_model:
        w = check_model(a.models, a.check_model, cubes)
        print("worst %.2e" % w)
        sys.exit(0 if w < 1e-4 else 1)
    res = {}
    only = a.configs.split(",") if a.configs else None
    for f in sorted(Path(a.models).glob("*_s*.json")):
        if only and f.stem not in only:
            continue
        m = WebModel(f)
        kind = f.stem.split("_")[0]
        t0 = time.time()
        res[f.stem] = [solve(m, kind, c, a.max_programs) for c in cubes]
        ok = sum(1 for r in res[f.stem] if r and r[-1] is not None)
        print("%-24s solved %d/%d  %.0fs" % (f.stem, ok, len(cubes), time.time() - t0), flush=True)
    if a.out:
        json.dump({"max_programs": a.max_programs, "n": a.n, "results": res}, open(a.out, "w"))
