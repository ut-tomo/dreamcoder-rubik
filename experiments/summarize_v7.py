"""Summarize results/v7_verified (24 arms: kind x nn x lib x seed) -> results/v7_verified/SUMMARY.md + docs/v7.json.
Arms whose final result (20 rounds + E2E) is missing are reported as pending, never imputed."""
import json, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results/v7_verified'
ARMS = [('nn_lib', 'NN + library'), ('nn_nolib', 'NN, no library'), ('nonn_lib', 'No NN + library'), ('nonn_nolib', 'No NN, no library')]
KINDS = [('stages', 'Stages (subtask curriculum)'), ('full', 'Full task only')]

def load(kind, arm, seed):
    p = R / f'{kind}_{arm}_s{seed}.json'
    if not p.exists(): return None
    d = json.loads(p.read_text())
    if 'e2e' not in d or len(d['history']) != d['args']['iterations']: return None
    m = json.loads((R / f'{kind}_{arm}_s{seed}.manifest.json').read_text())
    h = d['history']
    return dict(seed=seed, e2e=d['e2e']['solved'], e2e_n=d['e2e']['n'], mean_len=d['e2e']['mean_len'],
                train=h[-1]['solved'], n_tasks=h[-1]['n_tasks'], lib=len(h[-1]['library']),
                hours=round(h[-1]['elapsed'] / 3600, 2), curve=[x['solved'] for x in h],
                backend=m.get('backend_commit', '')[:7], host=m.get('host'))

def ms(xs):
    if not xs: return None
    return [round(st.mean(xs), 2), round(st.stdev(xs), 2) if len(xs) > 1 else None]

def fmt(v):
    if v is None: return '—'
    return f'{v[0]:g}' if v[1] is None else f'{v[0]:g} ± {v[1]:g}'

rows, pending, backends = [], [], set()
for kind, klabel in KINDS:
    for arm, alabel in ARMS:
        runs = []
        for s in range(3):
            r = load(kind, arm, s)
            if r: runs.append(r); backends.add(r['backend'])
            else: pending.append(f'{kind}_{arm}_s{s}')
        rows.append(dict(kind=kind, kind_label=klabel, arm=arm, arm_label=alabel, runs=runs,
                         e2e=ms([r['e2e'] for r in runs]), train=ms([r['train'] for r in runs]),
                         lib=ms([r['lib'] for r in runs]),
                         mean_len=ms([r['mean_len'] for r in runs if r['e2e']]),
                         n_tasks=runs[0]['n_tasks'] if runs else None))

done = 24 - len(pending)
note = ('Backend: dreamcoder-pyrepro ' + ', '.join(sorted(backends)) + ' (pyfast enumerator, pyvs compressor). '
        'This backend predates the list-domain parity fixes 0f7f479 and e473c9b; the cube results were not rerun on them.')
(ROOT / 'docs/v7.json').write_text(json.dumps(dict(done=done, pending=pending, note=note, rows=rows), separators=(',', ':')))

L = ['# v7_verified results (verified pyrepro backend)', '',
     f'{done}/24 arms complete. Pending: ' + (', '.join(pending) if pending else 'none') + '.', '', note, '',
     'Each cell is mean ± sd over the completed seeds (seed values in parentheses). E2E = solved out of the same 100 unseen random cubes (seed 9999); '
     'train = training tasks solved after 20 rounds; lib = learned library size at round 20. 20 rounds, -t 5, 4 CPUs; NN arms use contextual recognition + biasOptimal.', '',
     '| Training | Arm | seeds | E2E solved / 100 | Train solved | Library size | Mean moves (solved cubes) |', '|---|---|---|---|---|---|---|']
for r in rows:
    sv = lambda k: ' (' + ' / '.join(f"{x[k]:g}" for x in r['runs']) + ')' if r['runs'] else ''
    L.append(f"| {r['kind']} | {r['arm_label']} | {len(r['runs'])}/3 | {fmt(r['e2e'])}{sv('e2e')} | {fmt(r['train'])} / {r['n_tasks'] or '—'}{sv('train')} | {fmt(r['lib'])} | {fmt(r['mean_len'])} |")
(R / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
