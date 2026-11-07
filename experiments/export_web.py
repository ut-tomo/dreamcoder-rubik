"""Export the v2 four-arm results (recognition x library learning) to docs/data.json for the web viewer."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "pydreamcoder", "src")]
from rubik import cube2, perm as P  # noqa: E402
from rubik.tasks import STAGES, stage_tasks  # noqa: E402

ARMS = [  # id, label, file stem
    ("nn_lib", "NN-guided search + library learning", "v2_nn_lib"),
    ("nn_nolib", "NN-guided search, no library learning", "v2_nn_nolib"),
    ("nonn_lib", "Plain search + library learning", "v2_nonn_lib"),
    ("nonn_nolib", "Plain search, no library learning", "v2_nonn_nolib"),
]
SEEDS = [0, 1]
GOALS = {n: g for n, _, g in STAGES}


def check(start, moves, group):
    st = start
    for m in moves:
        st = P.compose(st, P.FACE_TURNS[m])
    return GOALS[group](P.normalize(st))


out = {"stickers": [[list(p), list(n)] for p, n in cube2.STICKERS],
       "arms": [{"id": i, "label": l} for i, l, _ in ARMS], "runs": []}
for seed in SEEDS:
    ds = {i: json.load(open(os.path.join(ROOT, "results", f"{stem}_s{seed}.json"))) for i, _, stem in ARMS}
    a = ds["nn_lib"]["args"]
    train = stage_tasks(a["per_stage"], seed)
    test = stage_tasks(a["test_per_stage"], 1000 + seed)
    tasks = [{"name": t.name, "split": "train", "group": t.meta["stage"], "start": list(t.meta["start"])} for t in train]
    tasks += [{"name": "test:" + t.name, "split": "test", "group": t.meta["stage"], "start": list(t.meta["start"])} for t in test]
    arms = {}
    bad = 0
    for i, _, _ in ARMS:
        d = ds[i]
        sols = {}
        for k, mv in list(d["solutions_moves"].items()) + list(d.get("test_moves", {}).items()):
            moves = [] if mv == "(id)" else mv.split()
            prog = d["frontiers"][k][0] if k in d["frontiers"] else d["test"][k][0]
            sols[k] = {"moves": moves, "program": prog}
        for t in tasks:
            if t["name"] in sols and not check(tuple(t["start"]), sols[t["name"]]["moves"], t["group"]):
                bad += 1
        arms[i] = {"solutions": sols,
                   "history": [{"it": h["iteration"], "solved": h["solved_ever"], "lib": len(h["library"])} for h in d["history"]],
                   "library": d.get("library_readable", []),
                   "per_group": d["per_group"], "test_per_group": d["test_per_group"],
                   "test_solved": len(d.get("test", {})), "test_n": d.get("test_n", 0)}
    out["runs"].append({"id": f"stages_s{seed}", "label": f"Stages · R/U/F turns · seed {seed}", "args": a,
                        "tasks": tasks, "arms": arms})
    print(f"seed {seed}: {len(tasks)} tasks, unverified solutions: {bad}",
          {i: (sum(v[0] for v in arms[i]['per_group'].values()), arms[i]['test_solved']) for i in arms})
json.dump(out, open(os.path.join(ROOT, "docs", "data.json"), "w"), separators=(",", ":"))
