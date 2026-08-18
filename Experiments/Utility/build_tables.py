#!/usr/bin/env python3
"""Full tables for deception/evil/toxic with date + OOD tags."""

import json, glob, re

METHOD_MAP = {
    'linearact': 'ACT',
    'curve': 'CURVE',
    'flas': 'FLAS',
    'chars': 'CHARS',
    'reps': 'REPS',
    'weightsteer': 'WEIGHTSTEER',
}

METHOD_ORDER = ['ACT', 'CURVE', 'FLAS', 'CHARS', 'REPS', 'WEIGHTSTEER']

DATASETS = {
    'deception': ['deception', 'liarbench'],
    'evil': ['evil'],
    'toxic': ['toxic'],
}

def extract_date(fname):
    m = re.search(r'(\d{8})', fname)
    return m.group(1) if m else '????????'

def has_ot_steer(d):
    steer = d.get('config', {}).get('steer', {})
    return steer.get('ot_steer') in (True, 'True', 'true', 1, '1')

all_results = []

for f in sorted(glob.glob('Results/*/*.json')):
    dirname = f.split('/')[1]
    method = METHOD_MAP.get(dirname)
    if not method:
        # skip subdirectories
        continue
    fname = f.split('/')[-1]
    
    # Determine which dataset this belongs to
    dataset = None
    for ds_name, keywords in DATASETS.items():
        if any(kw in fname.lower() for kw in keywords):
            dataset = ds_name
            break
    if dataset is None:
        continue
    
    try:
        data = json.load(open(f))
    except:
        continue
    
    if has_ot_steer(data):
        continue
    
    r = data.get('result', {})
    if not r:
        continue
    
    coeff_d = r.get('coeff', {})
    if isinstance(coeff_d, dict):
        coeff = list(coeff_d.values())[0]
    else:
        coeff = coeff_d
    if coeff is None:
        continue
    coeff = float(coeff)
    
    acc = r.get('accuracy')
    ppl = r.get('perplexity')
    rep = r.get('repetition_rate') or r.get('repetition_3gram') or r.get('repetition')
    
    if acc is None:
        continue
    if ppl is None or ppl <= 0 or (ppl != ppl):
        continue
    
    date = extract_date(fname)
    ood = ''
    if rep is not None and rep > 0.2:
        ood = 'OOD'
    
    all_results.append({
        'dataset': dataset,
        'method': method,
        'coeff': coeff,
        'acc': acc * 100,
        'ppl': ppl,
        'rep': rep,
        'date': date,
        'ood': ood,
        'fname': fname,
    })

# Print tables
for ds_name in ['deception', 'evil', 'toxic']:
    entries = [r for r in all_results if r['dataset'] == ds_name]
    # Filter ppl <= 8
    entries = [r for r in entries if r['ppl'] <= 8]
    # Sort
    entries.sort(key=lambda x: (METHOD_ORDER.index(x['method']) if x['method'] in METHOD_ORDER else 99, x['coeff'], x['date']))
    
    print(f"\n{'='*110}")
    print(f"  {ds_name.upper()}  (ppl ≤ 8 filter, ot_steer excluded)")
    print(f"{'='*110}")
    print(f"{'Method':<12} {'Coeff':<8} {'Acc%':<8} {'PPL':<10} {'Rep':<10} {'Date':<10} {'OOD':<6} {'File':<40}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*40}")
    
    for r in entries:
        rep_str = f"{r['rep']:.4f}" if r['rep'] is not None else '-'
        fname_short = r['fname'][:40]
        print(f"{r['method']:<12} {r['coeff']:<8.1f} {r['acc']:<8.2f} {r['ppl']:<10.4f} {rep_str:<10} {r['date']:<10} {r['ood']:<6} {fname_short}")
    
    # Summary: best per method
    best = {}
    for r in entries:
        if r['method'] not in best or r['acc'] < best[r['method']]['acc']:
            best[r['method']] = r
    print(f"\n  Best per method (lowest acc):")
    for m in METHOD_ORDER:
        if m in best:
            r = best[m]
            rep_str = f"{r['rep']:.4f}" if r['rep'] is not None else '-'
            print(f"    {m:<12} coeff={r['coeff']:<4.1f}  acc={r['acc']:<6.2f}%  ppl={r['ppl']:<8.4f}  rep={rep_str}  {r['date']}  {r['ood']}")
