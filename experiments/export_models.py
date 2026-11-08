"""Export trained models (run_ec.py <out>.model.json) and the cube tables for the in-browser search.

    python experiments/export_models.py RESULTS_DIR [--out docs/models]

    <out>/tables.json                 cube group tables (from src/rubik, so JS == Python)
    <out>/<kind>_<arm>_s<seed>.json   final grammar (+ official recognition network for NN arms)
    <out>/index.json                  list of exported configurations and their source files

Recognition model (official RecognitionModel, contextual): features -> _MLP (Linear+tanh) ->
ContextualGrammarNetwork_LowRank.transitionMatrix (LowRank: factored A/B, or full M). Weights are exported
at full float32 precision, so the browser and the Python reference compute the same grammars.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "pydreamcoder" / "ec")]
from rubik import cube2, perm  # noqa: E402
from rubik.ec_domain import DRB, DRF, STAGES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--out", default=str(ROOT / "docs" / "models"))
ap.add_argument("--seeds", default=None, help="comma-separated seeds to export (default: all)")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)


def corner_idx(pos):
    return [cube2.INDEX[(p, n)] for (p, n) in cube2.STICKERS if p == pos]


tables = {
    "face_turns": {m: list(p) for m, p in perm.FACE_TURNS.items()},
    "sym": {k: list(g) for k, g in perm.SYM_NAMES.items()},
    "rename": perm.RENAME,
    "rotations": [list(r) for r in perm.ROTATIONS],
    "dlb": perm._DLB_STICKERS,
    "face_of": cube2.FACE_OF,
    "stages": [n for n, _, _ in STAGES],
    # goal predicates on normalized states, as sticker sets that must be home / U stickers that must show U
    "goals": {
        "1-corner": {"home": corner_idx(DRF)},
        "2-corners": {"home": corner_idx(DRF) + corner_idx(DRB)},
        "3-Dlayer": {"home": [i for p in cube2.D_CORNERS for i in corner_idx(p)]},
        "4-Uorient": {"home": [i for p in cube2.D_CORNERS for i in corner_idx(p)],
                      "up": [i for i, (p, n) in enumerate(cube2.STICKERS) if n == (0, 1, 0)]},
        "5-solved": {"home": list(range(24))},
    },
}
json.dump(tables, open(OUT / "tables.json", "w"), separators=(",", ":"))


def check_first_order(productions):
    """search.js assumes every argument type is `alg` (no lambdas/variables are ever enumerated)."""
    for _, t, p in productions:
        args = t.split(" -> ")[:-1]
        if "(" in t or any(x.strip() != "alg" for x in args):
            raise SystemExit("higher-order production not supported by search.js: %s : %s" % (p, t))


def web_recognition(r):
    sd = r["state_dict"]
    out = {"activation": r["activation"], "n_features": r["n_features"], "n_grammars": r["n_grammars"],
           "library_rows": r["library_rows"], "W1": sd["_MLP.0.weight"], "b1": sd["_MLP.0.bias"]}
    assert "_MLP.2.weight" not in sd, "only one hidden layer is supported"
    pre = "grammarBuilder.transitionMatrix."
    if pre + "M.weight" in sd:
        out["M"] = {"W": sd[pre + "M.weight"], "b": sd[pre + "M.bias"]}
    else:
        out["A"] = {"W": sd[pre + "A.weight"], "b": sd[pre + "A.bias"]}
        out["B"] = {"W": sd[pre + "B.weight"], "b": sd[pre + "B.bias"]}
        out["rank"] = len(sd[pre + "A.bias"]) // r["n_grammars"]
    return out


seeds = None if a.seeds is None else {int(s) for s in a.seeds.split(",")}
index = []
for f in sorted(Path(a.results).glob("*.model.json")):
    cfg = f.name[:-len(".model.json")]
    if seeds is not None and int(cfg.rsplit("_s", 1)[1]) not in seeds:
        continue
    m = json.load(open(f))
    check_first_order(m["grammar"]["productions"])
    web = {"grammar": m["grammar"], "recognition": None}
    if m["recognition"]:
        r = m["recognition"]
        types = {p: t for _, t, p in m["grammar"]["productions"]}
        web["recognition"] = web_recognition(r)
        web["recognition"]["productions"] = [[p, types[p]] for p in r["productions"]]
    json.dump(web, open(OUT / (cfg + ".json"), "w"), separators=(",", ":"))
    index.append({"cfg": cfg, "source": str(f)})
json.dump(index, open(OUT / "index.json", "w"), indent=1)
print(f"wrote tables + {len(index)} models to {OUT}")

# the fixed 100 E2E test cubes (same as run_ec.py: random_states(100, 9999)), for validation
from rubik.ec_domain import random_states  # noqa: E402
json.dump([list(s) for s in random_states(100, 9999)], open(ROOT / "experiments" / "e2e_cubes.json", "w"))
