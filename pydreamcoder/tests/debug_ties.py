"""Find frontiers whose topK restriction depends on (near-)ties of the posterior, and the initial score
obtained by breaking those ties the other way. usage: debug_ties.py MESSAGE [target_score]"""
import itertools, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyec.compress_cli import load_primitives
load_primitives()
from pyec import vs_compression as V
from dreamcoder.program import Program
from dreamcoder.type import Type

j = json.load(open(sys.argv[1]))
lib = []
for p in j["DSL"]["productions"]:
    e = Program.parse(p["expression"])
    lib.append((e, e.tp if (e.isPrimitive or e.isInvented) else V.closed_inference(e), p["logProbability"]))
g = V.CGrammar(j["DSL"]["logVariable"], lib, None)
K = j["topK"]
fs = [V.CFrontier(Type.fromjson(f["request"]), [(Program.parse(e["program"]), e["logLikelihood"])
                                               for e in f["programs"]]) for f in j["frontiers"]]
choices = []
for i, f in enumerate(fs):
    sc = [(ll + V.likelihood_under_grammar(g, f.request, p), k) for k, (p, ll) in enumerate(f.programs)]
    order = sorted(sc, key=lambda x: -x[0])
    alts = [tuple(k for _, k in order[:K])]
    if len(order) > K and abs(order[K - 1][0] - order[K][0]) < 1e-9:
        cut = order[K - 1][0]
        tied = [k for s, k in order if abs(s - cut) < 1e-9]
        above = [k for s, k in order[:K] if abs(s - cut) >= 1e-9]
        alts = [tuple(above) + c for c in itertools.combinations(tied, K - len(above))]
        print("frontier %d: %d tied at %.12f -> %d alternatives" % (i, len(tied), cut, len(alts)),
              [str(f.programs[k][0]) for k in tied][:4])
    choices.append(alts)


def score(sel):
    rs = [V.CFrontier(f.request, [f.programs[k] for k in ks]) for f, ks in zip(fs, sel)]
    return V.grammar_induction_score(j["aic"], j["structurePenalty"], j["pseudoCounts"], rs, g)[1]


base = [c[0] for c in choices]
print("port initial score %.6f" % score(base))
for i, c in enumerate(choices):
    for alt in c[1:]:
        sel = list(base)
        sel[i] = alt
        print("  frontier %d alt %s: %.6f" % (i, alt, score(sel)))
