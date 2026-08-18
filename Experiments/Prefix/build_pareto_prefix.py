#!/usr/bin/env python3
"""Pareto plot for prefix-split experiments — 4 panels: Toxic prefix_pos/neg, Evil prefix_pos/neg."""

import json, glob, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_DIRS = {
    'CAA': 'caa',
    'ACT': 'linearact',
    'CHARS': 'chars',
    'PCA-OT': 'pcaot',
    'WEIGHTSTEER': 'weightsteer',
    'REPS': 'reps',
}
COLORS = ['#1f77b4','#e6194b','#3cb44b','#4363d8','#f58231','#911eb4']
MARKERS = ['o','s','D','^','v','p']
FINE_TUNE = {'WEIGHTSTEER', 'REPS'}
VALID_COEFFS = {1.0, 1.5, 2.0, 3.0, 5.0, 7.0}
MIN_DATE = '20260615'

def load_results(split, task):
    """Load prefix-split results for a given train_dataset split and eval task."""
    results = []
    for mname, mdir in METHOD_DIRS.items():
        for f in sorted(glob.glob(f'Results/{mdir}/*.json')):
            fname = f.split('/')[-1].lower()
            # Must match task and split from filename
            # Filenames: eval_gemma_{task}_prefix_{split}_coeff*.json
            task_kw = 'evil' if task == 'Evil' else 'toxic'
            if task_kw not in fname:
                continue
            if 'prefix_' + split not in fname:
                continue
            try:
                d = json.load(open(f))
            except:
                continue
            r = d.get('result', {})
            if not r:
                continue
            cfg = d.get('config', {})
            train = cfg.get('train_dataset', '')
            # Verify train_dataset matches split
            if task == 'Toxic':
                expected_train = f'toxic_jigsaw_prefix_{split}'
            else:
                expected_train = f'evil_prefix_{split}'
            if train and train != expected_train:
                continue
            steer = cfg.get('steer', {})
            coeff = steer.get('coeff', 0)
            if isinstance(coeff, dict):
                coeff = list(coeff.values())[0]
            coeff = float(coeff)
            if coeff < 0 or coeff not in VALID_COEFFS:
                continue
            acc = r.get('accuracy')
            ppl = r.get('perplexity')
            rep = r.get('repetition_rate')
            lsp = r.get('lsp_score') or d.get('lsp_score')
            if lsp is None and ppl is not None and rep is not None:
                lsp = ppl / (1.0 - rep + 1e-8)
            if acc is None or ppl is None or ppl <= 0:
                continue
            if ppl > 10:
                continue
            date = re.search(r'(\d{8})', f).group(1)
            if date < MIN_DATE:
                continue
            ood = (lsp is not None and lsp > 25.0) or (rep is not None and rep > 0.2) or (ppl is not None and ppl > 8.0)
            results.append({
                'method': mname,
                'coeff': coeff,
                'acc': acc * 100,
                'yval': acc * 100,  # for Toxic/Evil, higher = better
                'lsp': lsp,
                'date': date,
                'ood': ood,
                'file': os.path.basename(f),
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

# Build: 4 panels (2x2)
tasks = ['Toxic', 'Evil']
splits = ['pos', 'neg']
split_labels = {'pos': 'Prefix-Positive', 'neg': 'Prefix-Negative'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Prefix-Split Pareto: Steering Success Rate vs OOD Score (LCS PPL_lsp)', fontsize=16, fontweight='bold')

for row, task in enumerate(tasks):
    for col, split in enumerate(splits):
        ax = axes[row][col]
        raw = load_results(split, task)
        best = pick_best(raw)
        clean_pts = [r for r in best if not r['ood']]
        ood_pts = [r for r in best if r['ood']]

        # Per-method plot
        for i, mname in enumerate(METHOD_DIRS.keys()):
            cpts = [r for r in clean_pts if r['method'] == mname]
            opts = [r for r in ood_pts if r['method'] == mname]

            if cpts:
                xs = [r['lsp'] for r in cpts]
                ys = [r['yval'] for r in cpts]
                ax.scatter(xs, ys, c=COLORS[i], marker=MARKERS[i], s=100, label=mname, zorder=5, edgecolors='black', linewidths=0.5)
                for r in cpts:
                    offset = (8, 5)
                    ax.annotate(f"c={r['coeff']:.0f}", (min(r['lsp'], 100.0), r['yval']),
                                textcoords="offset points", xytext=offset, fontsize=6,
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

            if opts:
                xs = [100.0 for r in opts]
                ys = [r['yval'] for r in opts]
                ax.scatter(xs, ys, facecolors='none', edgecolors=COLORS[i], s=80, alpha=0.15, linewidths=0.8, zorder=3)
                for r in opts:
                    ax.annotate(f"c={r['coeff']:.0f}", (min(r['lsp'], 100.0), r['yval']),
                                textcoords="offset points", xytext=(8, -10), fontsize=5,
                                alpha=0.5, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.5))

        # Pareto frontiers
        if len(best) > 1:
            full_frontier = pareto_frontier(best)
            if len(full_frontier) > 1:
                px = [100.0 if r['ood'] else r['lsp'] for r in full_frontier]
                py = [r['yval'] for r in full_frontier]
                ax.plot(px, py, '--', color='gray', alpha=0.5, linewidth=1.5, label='Pareto (all)')

        if len(clean_pts) > 1:
            clean_frontier = pareto_frontier(clean_pts)
            if len(clean_frontier) > 1:
                px = [r['lsp'] for r in clean_frontier]
                py = [r['yval'] for r in clean_frontier]
                ax.plot(px, py, '-', color='black', alpha=0.7, linewidth=1.5, label='Pareto (clean)')

        title = f'{task} — {split_labels[split]}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('OOD Score (LSP PPL_lsp)')
        ax.set_ylabel(f'{task} Success Rate ↑ (%)' if task != 'Deception' else 'Deceptiveness ↑ (%)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # Print detail
        print(f'\n=== {task} {split_labels[split]} ===')
        for r in best:
            marker = '' if not r['ood'] else ' [OOD]'
            print(f"  {r['method']:<12} c={r['coeff']:<4.0f}  {r['yval']:>5.1f}%  lsp={r['lsp']:<5.2f}  rep={r.get('repetition_rate',0):.4f}{marker}")
        cf = pareto_frontier(clean_pts)
        if cf:
            print(f"  --- Clean Pareto ---")
            for r in cf:
                print(f"  {r['method']:<12} c={r['coeff']:<4.0f}  {r['yval']:>5.1f}%  lsp={r['lsp']:<5.2f}")

plt.tight_layout()
plt.savefig('Pareto/pareto_prefix.png', dpi=150, bbox_inches='tight')
print('\nSaved Pareto/pareto_prefix.png')
