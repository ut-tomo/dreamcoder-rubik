"""Localise compressor score differences: per-frontier 'Initial score' of OCaml ./compression vs pyec.
usage: python debug_initial_score.py MESSAGE"""
import json, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyec.compress_cli import load_primitives
load_primitives()
from pyec import vs_compression as V
from dreamcoder.program import Program
from dreamcoder.type import Type

EC = os.environ.get("EC_ROOT", "/mnt/home/unno/lab/rubiks_project/dreamcoder-ec")
j = json.load(open(sys.argv[1]))
env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/miniforge/envs/dcec/lib"))


def port_grammar(j):
    lib = []
    for p in j["DSL"]["productions"]:
        e = Program.parse(p["expression"])
        t = e.tp if (e.isPrimitive or e.isInvented) else V.closed_inference(e)
        lib.append((e, t, p["logProbability"]))
    return V.CGrammar(j["DSL"]["logVariable"], lib, None)


g = port_grammar(j)
for i, f in enumerate(j["frontiers"]):
    m = dict(j, frontiers=[f], iterations=1, CPUs=1)
    p = subprocess.run([EC + "/compression"], input=json.dumps(m).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env)
    s = re.search(r"Initial score: (\S+)", p.stderr.decode())
    o = float(s.group(1)) if s else None
    cf = V.CFrontier(Type.fromjson(f["request"]), [(Program.parse(e["program"]), e["logLikelihood"])
                                                  for e in f["programs"]])
    _, ps = V.grammar_induction_score(j["aic"], j["structurePenalty"], j["pseudoCounts"],
                                      [V.restrict(j["topK"], g, cf)], g)
    flag = "" if o is not None and abs(o - ps) < 1e-5 else "   <-- DIFF"
    print(i, o, "%.6f" % ps, flag, [e["program"] for e in f["programs"]][:3] if flag else "", flush=True)
