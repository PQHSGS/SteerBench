"""
Analyze Experiment F results: per-token KL curves.
Generates: text summary, correlation table, optional plots.

Usage:
    python Experiments/ExpF/analyze_kl.py [--path Experiments/ExpF/expF_results.json]
"""

import json, sys, numpy as np
from pathlib import Path

def load(path):
    with open(path) as f:
        return json.load(f)

def analyze(data):
    meta = data.get("metadata", {})
    print(f"Model: {meta.get('model', '?')} | Prompts: {meta.get('n_prompts', '?')} | Max tokens: {meta.get('max_new_tokens', '?')}")
    print()

    # 1. Table
    print(f"{'Task':<12} {'c':>4} {'Acc':>7} {'KL[0-5]':>10} {'KL[5+]':>10} {'Ratio':>8}  {'KL_cor[0-5]':>12} {'KL_inc[0-5]':>12}")
    print(f"{'-'*80}")
    for task in ["toxic", "evil", "deception", "refusal"]:
        if task not in data:
            continue
        for c_str in ["1", "3", "5"]:
            if c_str not in data[task]:
                continue
            d = data[task][c_str]
            e = d.get('kl_0_5')
            l = d.get('kl_5plus')
            r = d.get('kl_ratio_early_late')
            r_str = f"{r:.2f}x" if r else "N/A"
            e_str = f"{e:.6f}" if e else "N/A"
            l_str = f"{l:.6f}" if l else "N/A"
            ck = f"{np.mean(d['kl_correct'][:6]):.6f}" if d.get('kl_correct') and len(d['kl_correct']) >= 6 else "N/A"
            ik = f"{np.mean(d['kl_incorrect'][:6]):.6f}" if d.get('kl_incorrect') and len(d['kl_incorrect']) >= 6 else "N/A"
            print(f"{task:<12} {c_str:>4} {d['accuracy']:>6.1%} {e_str:>10} {l_str:>10} {r_str:>8}  {ck:>12} {ik:>12}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # 2. Check correlation: does KL ratio predict accuracy?
    tasks = ["toxic", "evil", "deception"]
    coeffs = ["1", "3", "5"]
    ratios, accs = [], []
    for task in tasks:
        if task not in data:
            continue
        for c in coeffs:
            if c in data[task]:
                d = data[task][c]
                r = d.get('kl_ratio_early_late')
                a = d.get('accuracy')
                if r and a is not None:
                    ratios.append(r)
                    accs.append(a)

    if len(ratios) >= 3:
        corr = np.corrcoef(ratios, accs)[0, 1]
        print(f"\nRank correlation (KL ratio vs accuracy): {corr:+.3f}")
        if abs(corr) > 0.5:
            print(f"  STRONG: KL concentration {'negatively' if corr < 0 else 'positively'} predicts accuracy")
            if corr < 0:
                print(f"  → Higher KL ratio (more early-concentrated) → LOWER accuracy")
                print(f"  → Consistent with: cancellation filter blocks early-token changes")
            else:
                print(f"  → Higher KL ratio → HIGHER accuracy")
                print(f"  → Consistent with: early-token changes drive steering success")
        else:
            print(f"  WEAK: KL concentration does not strongly predict accuracy")
            print(f"  → Suggests cancellation is a separate mechanism from shallow KL")

    # 3. Task comparison
    print(f"\n--- Task Comparison ---")
    for c in coeffs:
        print(f"\n  Coeff={c}:")
        for task in tasks:
            if task in data and c in data[task]:
                d = data[task][c]
                r = d.get('kl_ratio_early_late')
                a = d.get('accuracy')
                print(f"    {task:<10} Acc={a:.1%}  Ratio={r:.2f}x  KL[0-5]={d.get('kl_0_5', 'N/A'):>8}  KL[5+]={d.get('kl_5plus', 'N/A'):>8}")

    # 4. Correct vs incorrect early KL
    print(f"\n--- Correct vs Incorrect Early KL ---")
    for task in tasks:
        if task not in data:
            continue
        print(f"\n  {task}:")
        for c in coeffs:
            if c in data[task]:
                d = data[task][c]
                ck = np.mean(d['kl_correct'][:6]) if d.get('kl_correct') and len(d['kl_correct']) >= 6 else None
                ik = np.mean(d['kl_incorrect'][:6]) if d.get('kl_incorrect') and len(d['kl_incorrect']) >= 6 else None
                if ck is not None and ik is not None:
                    diff = ck - ik
                    print(f"    c={c}: Correct={ck:.6f}  Incorrect={ik:.6f}  Diff={diff:+.6f}")
                    if diff > 0:
                        print(f"      → Correct samples have HIGHER early KL (more logit change = success)")
                    else:
                        print(f"      → Correct samples have LOWER early KL (less logit change = success??)")

    # 5. Check if KL ratio matches shallow alignment theory
    print(f"\n--- Theory Check ---")
    print(f"  Shallow alignment predicts: KL concentrated in first ~5 tokens (ratio >> 5x)")
    print(f"  If true for ALL tasks: KL concentration is universal, NOT the ceiling mechanism")
    print(f"  If true only for Toxic/Evil: KL concentration IS the ceiling mechanism")
    print(f"")
    for task in tasks:
        if task in data and "3" in data[task]:
            r = data[task]["3"].get('kl_ratio_early_late')
            a = data[task]["3"].get('accuracy')
            if r:
                tag = "SHALLOW" if r > 5 else "DEEP"
                ceiling = "CEILING" if a is not None and a < 0.5 else "OK"
                print(f"  {task}: ratio={r:.2f}x → {tag} | acc={a:.1%} → {ceiling}")


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--path" else "Experiments/ExpF/expF_results.json"
    if Path(path).exists():
        data = load(path)
        analyze(data)
    else:
        print(f"Results not found: {path}")
        print("Run Experiments/ExpF/expF_per_token_kl.py first.")
