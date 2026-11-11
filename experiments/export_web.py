"""Export the v5 runs (results/v5) for the web page: docs/data.json + docs/models/<run>.json.
Every displayed solution is replayed on the cube before publishing.
Run on the lab servers (needs the vendored official frontend for program parsing / type inference)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'pydreamcoder'), str(ROOT / 'pydreamcoder' / 'ec')]
from dreamcoder.program import Program  # noqa: E402
from rubik import cube2, perm  # noqa: E402
from rubik import ec_domain as D  # noqa: E402

D.register_all()
ARMS = [('nn_lib', 'NN-guided search + library learning'), ('nn_nolib', 'NN-guided search, no library learning'),
        ('nonn_lib', 'Plain search + library learning'), ('nonn_nolib', 'Plain search, no library learning')]
VERSION = sys.argv[1] if len(sys.argv) > 1 else 'v5'
SEEDS = [0, 1, 2]
out = dict(stickers=cube2.STICKERS, arms=[dict(id=i, label=l) for i, l in ARMS], runs=[], version=VERSION)
models = ROOT / 'docs' / 'models'
models.mkdir(exist_ok=True)
r5 = lambda xs: [r5(x) for x in xs] if isinstance(xs, list) else float('%.6g' % xs)


def web_model(mj):
    """run_ec.py model.json -> compact model for docs/search.js (grammar + official bigram network weights)."""
    web = {'grammar': mj['grammar'], 'recognition': None}
    r = mj['recognition']
    if r:
        sd = r['state_dict']
        prods = [[s, str(Program.parse(s).infer())] for s in r['productions']]
        rec = {'productions': prods, 'library_rows': r['library_rows'], 'n_grammars': r['n_grammars'],
               'activation': r['activation'], 'W1': r5(sd['_MLP.0.weight']), 'b1': r5(sd['_MLP.0.bias'])}
        if 'grammarBuilder.transitionMatrix.M.weight' in sd:
            rec['M'] = {'W': r5(sd['grammarBuilder.transitionMatrix.M.weight']),
                        'b': r5(sd['grammarBuilder.transitionMatrix.M.bias'])}
        else:
            rec['A'] = {'W': r5(sd['grammarBuilder.transitionMatrix.A.weight']),
                        'b': r5(sd['grammarBuilder.transitionMatrix.A.bias'])}
            rec['B'] = {'W': r5(sd['grammarBuilder.transitionMatrix.B.weight']),
                        'b': r5(sd['grammarBuilder.transitionMatrix.B.bias'])}
            rec['rank'] = len(sd['grammarBuilder.transitionMatrix.A.bias']) // r['n_grammars']
        web['recognition'] = rec
    return web


def main():
    verified = 0
    for kind in ['stages', 'full']:
        for seed in SEEDS:
            paths = {i: ROOT / f'results/{VERSION}/{kind}_{i}_s{seed}.json' for i, _ in ARMS}
            if not all(p.exists() for p in paths.values()):
                print('skip incomplete', kind, seed)
                continue
            ds = {i: json.loads(p.read_text()) for i, p in paths.items()}
            if not all('e2e' in d for d in ds.values()):
                print('skip unfinished', kind, seed)
                continue
            a = ds['nn_lib']['args']
            e2e = D.random_states(a['e2e_n'], a['e2e_seed'])
            train = D.stage_tasks(a['per_stage'], seed, exclude=e2e) if kind == 'stages' else \
                D.full_random_tasks(a['n_full'], seed, exclude=e2e)
            tasks = [dict(name=t.name, split='train', group=t.stage if kind == 'stages' else 'full', start=list(t.start))
                     for t in train]
            tasks += [dict(name=f'test:{j}', split='test', group='full', start=list(s)) for j, s in enumerate(e2e)]
            arms = {}
            for i, d in ds.items():
                assert d['args']['kind'] == kind and d['args']['seed'] == seed
                sols = {k: dict(moves=[] if v == '(id)' else v.split(), program=d['solutions'][k])
                        for k, v in d['solutions_moves'].items()}
                for j, chunks in d['e2e']['moves'].items():
                    sols['test:' + j] = dict(moves=[m for chunk in chunks for m in chunk.split() if m != '(id)'],
                                             program=' → '.join(chunks),
                                             boundaries=[sum(len([m for m in c.split() if m != '(id)']) for c in chunks[:k + 1])
                                                         for k in range(len(chunks))])
                goals = {n: g for n, _, g in D.STAGES}
                goals['full'] = cube2.solved
                for t in tasks:
                    if t['name'] not in sols:
                        continue
                    s = tuple(t['start'])
                    for idx, m in enumerate(sols[t['name']]['moves'], 1):
                        s = perm.compose(s, perm.FACE_TURNS[m])
                        if idx in sols[t['name']].get('boundaries', []):
                            s = perm.normalize(s)
                    assert goals[t['group']](perm.normalize(s)), (kind, seed, i, t['name'])
                    verified += 1
                assert len(d['e2e']['moves']) == d['e2e']['solved']
                test_pg = {k: [v, d['e2e']['n']] for k, v in (d['e2e']['reached'] or {'full': d['e2e']['solved']}).items()}
                arms[i] = dict(solutions=sols,
                               history=[dict(it=h['iteration'], solved=h['solved'], lib=len(h['library'])) for h in d['history']],
                               library=d['library_readable'], per_group=d['per_group'], test_per_group=test_pg,
                               test_solved=d['e2e']['solved'], test_n=d['e2e']['n'], policy=d.get('policy', {}))
                web = web_model(json.loads(paths[i].with_name(paths[i].stem + '.model.json').read_text()))
                (models / f'{kind}_{i}_s{seed}.json').write_text(json.dumps(web, separators=(',', ':')))
            out['runs'].append(dict(id=f'{kind}_s{seed}', label=f'{kind.title()} · seed {seed}', args=a, tasks=tasks,
                                    arms=arms))
    (ROOT / 'docs/data.json').write_text(json.dumps(out, separators=(',', ':')))
    print(f'Exported {len(out["runs"])} run groups; replay-verified {verified} solutions.')


if __name__ == '__main__':
    main()
