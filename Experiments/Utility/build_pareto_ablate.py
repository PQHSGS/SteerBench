#!/usr/bin/env python3
"""Pareto plot — ACT/CHARS normal vs ablate, for deception/evil/toxic."""

import json, glob, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_DIRS = {
    'ACT': ('linearact', '#1f77b4', 'o'),
    'ACT-ABLATE': ('linearact_ablate', '#e6194b', 's'),
    'CHARS': ('chars', '#3cb44b', 'D'),
    'CHARS-ABLATE': ('chars_ablate', '#4363d8', '^'),
}
VALID_COEFFS = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}
MIN_DATE = '20260615'

# Config-based filters: expected (position, train_dataset) per method×task
CFG_EXPECTED = {}
for mname, cfg_dir in [('ACT', 'Configs/Eval/LinearAcT/Gemma'), ('CHARS', 'Configs/Eval/CHARS/Gemma')]:
    for task_key in ['deception', 'evil', 'toxic']:
        p = f'{cfg_dir}/gemma_{task_key}.json'
        if not os.path.exists(p):
            CFG_EXPECTED[(mname, task_key)] = (None, None)
            continue
        with open(p) as fh:
            c = json.load(fh)
        pos = c.get('extractor', {}).get('position')
        train = c.get('train_dataset')
        CFG_EXPECTED[(mname, task_key)] = (pos, train)

DATASETS = {
    'Deception': {
        'task_key': 'deception',
        'keywords': ['deception'],
        'ylabel': 'Deceptiveness (100 - honesty %)',
        'transform': lambda acc: acc,   # accuracy is now deceptiveness directly
        'ood_threshold': 25.0,
        'xmax': 25,
    },
    'Evil': {
        'task_key': 'evil',
        'keywords': ['evil'],
        'ylabel': 'Evil Success Rate (%)',
        'transform': lambda acc: acc,
        'ood_threshold': 10.0,
        'xmax': 10,
    },
    'Toxic': {
        'task_key': 'toxic',
        'keywords': ['toxic'],
        'ylabel': 'Toxic Success Rate (%)',
        'transform': lambda acc: acc,
        'ood_threshold': 20.0,
        'xmax': 20,
    },
}

def load_results(keywords, task_key, ood_thresh=25.0):
    results = []
    for mname, (mdir, color, marker) in METHOD_DIRS.items():
        # Map mname (e.g. ACT, ACT-ABLATE) to config key (ACT, CHARS)
        cfg_key = 'ACT' if mname.startswith('ACT') else 'CHARS'
        exp_pos, exp_train = CFG_EXPECTED.get((cfg_key, task_key), (None, None))

        for f in sorted(glob.glob(f'Results/{mdir}/*.json')):
            fname = f.split('/')[-1].lower()
            if not any(kw in fname for kw in keywords):
                continue
            try:
                d = json.load(open(f))
            except:
                continue
            r = d.get('result', {})
            if not r or not isinstance(r, dict):
                continue
            cfg = d.get('config', {})
            steer = cfg.get('steer', {})

            # Config-based filter
            pos = cfg.get('extractor', {}).get('position')
            train = cfg.get('train_dataset')
            if exp_pos is not None and pos != exp_pos:
                continue
            if exp_train is not None and train != exp_train:
                continue
            if steer.get('ot_steer') in (True, 'True', 'true'):
                continue
            coeff = steer.get('coeff', 0)
            if isinstance(coeff, dict):
                coeff = list(coeff.values())[0]
            coeff = float(coeff)
            if coeff < 0:
                continue
            acc = r.get('accuracy')
            ppl = r.get('perplexity')
            rep = r.get('repetition_rate')
            lsp = r.get('lsp_score')
            if lsp is None and ppl is not None and rep is not None:
                lsp = ppl / (1.0 - rep + 1e-8)

            if acc is None or ppl is None or ppl <= 0 or ppl != ppl:
                continue
            if coeff <= 0 or coeff not in VALID_COEFFS:
                continue
            date = re.search(r'(\d{8})', f).group(1)
            if date < MIN_DATE:
                continue
            # Evaluator bug fix: Evil results before Jun 20 used old evaluator (~60% FP)
            if any(kw in fname for kw in ['evil']) and date < '20260620':
                continue

            ood = (lsp is not None and lsp > ood_thresh)
            results.append({
                'method': mname,
                'coeff': coeff,
                'acc': acc * 100,
                'lsp': lsp,
                'date': date,
                'ood': ood,
                'color': color,
                'marker': marker,
            })
    return results

def pick_best(results):
    best = {}
    for r in results:
        key = (r['method'], r['coeff'])
        if key not in best:
            best[key] = r
        else:
            cur = best[key]
            if r['ood'] and not cur['ood']:
                continue
            if not r['ood'] and cur['ood']:
                best[key] = r
            elif r['yval'] > cur['yval']:
                best[key] = r
    return list(best.values())

def pareto_frontier(points):
    sorted_pts = sorted(points, key=lambda r: r['lsp'])
    frontier = []
    for r in sorted_pts:
        if not frontier or r['yval'] > frontier[-1]['yval']:
            frontier.append(r)
    return frontier

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for idx, (ds_name, ds_info) in enumerate(DATASETS.items()):
    ax = axes[idx]
    transform = ds_info['transform']
    raw = load_results(ds_info['keywords'], ds_info['task_key'])
    for r in raw:
        r['yval'] = transform(r['acc'])
    best = pick_best(raw)

    clean_pts = [r for r in best if not r['ood']]
    ood_pts = [r for r in best if r['ood']]

    for mname, (mdir, color, marker) in METHOD_DIRS.items():
        cpts = [r for r in clean_pts if r['method'] == mname]
        opts = [r for r in ood_pts if r['method'] == mname]

        if cpts:
            xs = [r['lsp'] for r in cpts]
            ys = [r['yval'] for r in cpts]
            labels = [f"c={r['coeff']:.0f}" for r in cpts]
            ax.scatter(xs, ys, c=color, marker=marker, s=90, label=mname, zorder=5)
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=6,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

        if opts:
            xs = [25.0 for _ in opts]
            ys = [r['yval'] for r in opts]
            labels = [f"c={r['coeff']:.0f}" for r in opts]
            ax.scatter(xs, ys, facecolors='none', edgecolors=color, s=80, alpha=0.25, linewidths=0.8, zorder=3)
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, -10), fontsize=6,
                           alpha=0.5, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.5))

    full_frontier = pareto_frontier(best)
    if len(full_frontier) > 1:
        px = [25.0 if r['ood'] else r['lsp'] for r in full_frontier]
        py = [r['yval'] for r in full_frontier]
        ax.plot(px, py, '--', color='gray', alpha=0.5, linewidth=1.5, label='Pareto (all)')

    if len(clean_pts) > 1:
        clean_frontier = pareto_frontier(clean_pts)
        if len(clean_frontier) > 1:
            px = [r['lsp'] for r in clean_frontier]
            py = [r['yval'] for r in clean_frontier]
            ax.plot(px, py, '-', color='black', alpha=0.7, linewidth=1.5, label='Pareto (clean)')

    ax.set_xlabel('OOD Score (LSP)')
    ax.set_ylabel(ds_info['ylabel'])
    ax.set_title(f'{ds_name} — ACT/CHARS Normal vs Ablate', fontsize=14, fontweight='bold')
    ax.set_xlim(0, ds_info['xmax'])
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'Pareto/pareto_act_chars_ablate.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

print('\nClean-only Pareto frontier:')
for ds_name, ds_info in DATASETS.items():
    transform = ds_info['transform']
    raw = load_results(ds_info['keywords'], ds_info['task_key'], ds_info['ood_threshold'])
    for r in raw:
        r['yval'] = transform(r['acc'])
    best = pick_best(raw)
    clean_pts = [r for r in best if not r['ood']]
    cf = pareto_frontier(clean_pts)
    print(f'\n{ds_name}:')
    for r in cf:
        print(f'  {r["method"]:<14} c={r["coeff"]:<4.0f}  y={r["yval"]:<6.2f}%  lsp={r["lsp"]:<8.4f}  {r["date"]}')
