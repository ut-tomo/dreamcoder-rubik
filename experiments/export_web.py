"""Export completed v3 summaries only; replay every displayed solution before publishing."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'pydreamcoder/src')]
from rubik import cube2, perm
from rubik.tasks import STAGES, stage_tasks, full_random_tasks, random_states
ARMS = [('nn_lib','NN-guided search + library learning'),('nn_nolib','NN-guided search, no library learning'),('nonn_lib','Plain search + library learning'),('nonn_nolib','Plain search, no library learning')]
out = dict(stickers=cube2.STICKERS, arms=[dict(id=i,label=l) for i,l in ARMS], runs=[])
verified = 0
for kind in ['stages', 'full']:
    for seed in range(3):
        ds = {i:json.loads((ROOT / f'results/v3/{kind}_{i}_s{seed}.json').read_text()) for i,_ in ARMS}
        a = ds['nn_lib']['args']
        e2e = random_states(a['e2e_n'], a['e2e_seed'])
        train = stage_tasks(a['per_stage'], seed, exclude=e2e) if kind == 'stages' else full_random_tasks(a['n_full'], seed, exclude=e2e)
        tasks = [dict(name=t.name, split='train',group=t.meta['stage'],start=t.meta['start']) for t in train]
        tasks += [dict(name=f'test:{j}',split='test',group='full',start=s) for j,s in enumerate(e2e)]
        arms = {}
        for i,d in ds.items():
            assert len(d['history']) == a['iterations'] and 'e2e' in d, (kind,i,seed,'incomplete')
            assert d['args']['kind'] == kind and d['args']['seed'] == seed
            sols = {k:dict(moves=[] if v == '(id)' else v.split(),program=d['frontiers'][k][0]) for k,v in d['solutions_moves'].items()}
            for j,chunks in d['e2e']['moves'].items():
                sols['test:'+j] = dict(moves=[m for chunk in chunks for m in chunk.split() if m != '(id)'],program=' → '.join(chunks), boundaries=[sum(len(c.split()) for c in chunks[:k+1]) for k in range(len(chunks))])
            goals = {n:g for n,_,g in STAGES}; goals['full'] = cube2.solved
            for t in tasks:
                if t['name'] not in sols: continue
                s = tuple(t['start'])
                for idx,m in enumerate(sols[t['name']]['moves'],1):
                    s = perm.compose(s,perm.FACE_TURNS[m])
                    if idx in sols[t['name']].get('boundaries',[]): s = perm.normalize(s)
                assert goals[t['group']](perm.normalize(s)), (kind,seed,i,t['name'])
                verified += 1
            assert len(d['e2e']['moves']) == d['e2e']['solved']
            lengths = [len(v['moves']) for k,v in sols.items() if k.startswith('test:')]
            assert not lengths or abs(sum(lengths)/len(lengths)-d['e2e']['mean_len']) < 1e-6
            # pools: every distinct move sequence this run found, tagged by the stage it solves.
            # The web demo tries them (shortest first) on a user's cube and checks the stage goal.
            pool = {n: set() for n in goals} if kind == 'stages' else {'full': set()}
            for k, v in sols.items():
                if k.startswith('test:'):
                    s = perm.normalize(tuple(e2e[int(k[5:])]))
                    for chunk in d['e2e']['moves'][k[5:]]:
                        mv = [m for m in chunk.split() if m != '(id)']
                        g = 'full' if kind == 'full' else next(n for n, _, gl in STAGES if not gl(s))
                        pool[g].add(' '.join(mv))
                        for m in mv: s = perm.compose(s, perm.FACE_TURNS[m])
                        s = perm.normalize(s)
                elif v['moves']:
                    pool[k.split('#')[0] if kind == 'stages' else 'full'].add(' '.join(v['moves']))
            pool = {g: sorted(v, key=lambda x: (len(x.split()), x)) for g, v in pool.items() if g != 'full' or kind == 'full'}
            arms[i] = dict(pools=pool, solutions=sols, history=[dict(it=h['iteration'],solved=h['solved_ever'],lib=len(h['library'])) for h in d['history']],library=d['library_readable'],per_group=d['per_group'],test_per_group={k:[v,d['e2e']['n']] for k,v in (d['e2e']['reached'] or {'full':d['e2e']['solved']}).items()},test_solved=d['e2e']['solved'],test_n=d['e2e']['n'],policy=d.get('policy',{}))
        out['runs'].append(dict(id=f'{kind}_s{seed}',label=f'{kind.title()} · seed {seed}',args=a,tasks=tasks,arms=arms))
(ROOT/'docs/data.json').write_text(json.dumps(out,separators=(',',':')))
print(f'Exported 24 completed runs; replay-verified {verified} solutions.')
