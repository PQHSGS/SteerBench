#!/usr/bin/env python3
"""
Map activation geometry diagnostics to method success/failure predictions.

For each of the 6 methods (ACT, CHaRS, CurveBall, FLAS, LinEAS, CAA),
explains WHY they succeed/fail on each task (toxic/deception/evil)
using the diagnostic measurements.

Reads: Analysis/results/activation_geometry_diagnostics.json
Writes: Analysis/results/method_analysis.md
"""

import json
import numpy as np
from pathlib import Path

RESULTS_PATH = Path("Analysis/results/activation_geometry_diagnostics.json")
OUTPUT_PATH = Path("Analysis/results/method_analysis.md")

def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def get_layer_data(data, task: str, experiment: str, layer_idx: int = 2):
    """Get a specific measurement for layer_idx across all tasks."""
    layers = ["6", "10", "14", "18", "22", "25"]
    layer = layers[layer_idx]
    return data[task][experiment].get(layer, {})

def pretty(v, fmt=".3f"):
    if isinstance(v, (int, float)):
        return format(v, fmt)
    return str(v)

def make_table(data, metric_path, header, row_label="Task", fmt=".3f"):
    """Create a markdown table from a metric path like exp1/w2 at each layer."""
    tasks = list(data.keys())
    layers = ["6", "10", "14", "18", "22", "25"]
    
    lines = [f"| {row_label} | " + " | ".join(f"L{l}" for l in layers) + " |"]
    lines.append("|" + "---|" * (len(layers) + 1))
    
    for task in tasks:
        vals = []
        for l in layers:
            d = data[task]
            for key in metric_path.split("/"):
                d = d.get(key, {})
            if isinstance(d, dict):
                vals.append(pretty(d.get(l, "?"), fmt))
            else:
                vals.append(pretty(d, fmt) if not isinstance(d, dict) else "?")
        lines.append(f"| {task} | " + " | ".join(vals) + " |")
    
    return "\n".join(lines) + "\n"

def avg_across_layers(data, task, experiment, metric):
    """Average a metric across layers. experiment like 'exp1_distribution_overlap', metric like 'w2'."""
    vals = []
    layers = ["6", "10", "14", "18", "22", "25"]
    for l in layers:
        v = data[task].get(experiment, {}).get(l, {}).get(metric)
        if v is not None and isinstance(v, (int, float)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else 0.0

def main():
    data = load_results()
    tasks = list(data.keys())
    layers = ["6", "10", "14", "18", "22", "25"]
    
    lines = []
    lines.append("# Method Success/Failure Analysis from Activation Geometry\n")
    lines.append(f"*Generated from 7 experiments across {len(tasks)} tasks × 6 layers*\n")
    
    # ── Summary Table ─────────────────────────────────────────────────────
    lines.append("## 1. Summary of Known Results\n")
    lines.append("| Task | LinNEAS 26L-c1 | LinNEAS 26L-c2 | LinNEAS L14-c2 | Best config |")
    lines.append("|------|---------------|---------------|---------------|-------------|")
    lines.append("| Toxic (↑) | 0.00/1.78 | 0.06/8.56 | 0.00/1.70 | 26L-c2=0.06/8.56 (broken) |")
    lines.append("| Deception (↓) | 0.76/1.56 | 0.68/14.07 | 0.74/1.41 | 26L-c2=0.68/14.07 (high ppl) |")
    lines.append("| Evil (↑) | 0.11/1.78 | 0.46/4.00 | 0.26/2.07 | 26L-c2=0.46/4.00 |")
    lines.append("")
    
    # ── Experiment 1: Distribution Overlap ─────────────────────────────────
    lines.append("## 2. Experiment 1: Distribution Overlap (Lin-ACT Fig 10)\n")
    lines.append("**What it measures**: W2 distance between source and target activation distributions per layer. Lower W2 = distributions are more similar = easier to transport between them.\n")
    
    lines.append("### W2 Distance (lower = easier transport)\n")
    for task in tasks:
        w2_vals = [get_layer_data(data, task, "exp1_distribution_overlap", i).get("w2", 0) for i in range(6)]
        lines.append(f"- **{task}**: W2={[pretty(v) for v in w2_vals]} at L{layers}; "
                     f"mean={pretty(np.mean(w2_vals))}")
    lines.append("")
    
    lines.append("### Cosine Similarity (higher = more aligned direction)\n")
    for task in tasks:
        cos_vals = [get_layer_data(data, task, "exp1_distribution_overlap", i).get("cosine_similarity", 0) for i in range(6)]
        lines.append(f"- **{task}**: cos={[pretty(v) for v in cos_vals]}; "
                     f"mean={pretty(np.mean(cos_vals))}")
    lines.append("")
    
    # Interpretation
    lines.append("### Method Predictions\n")
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    
    # ACT: linear OT works if W2 is low and cos sim is high
    w2_toxic = avg_across_layers(data, "toxic", "exp1_distribution_overlap", "w2")
    w2_dec = avg_across_layers(data, "deception", "exp1_distribution_overlap", "w2")
    w2_evil = avg_across_layers(data, "evil", "exp1_distribution_overlap", "w2")
    
    cos_toxic = avg_across_layers(data, "toxic", "exp1_distribution_overlap", "cosine_similarity")
    cos_dec = avg_across_layers(data, "deception", "exp1_distribution_overlap", "cosine_similarity")
    cos_evil = avg_across_layers(data, "evil", "exp1_distribution_overlap", "cosine_similarity")
    
    lines.append(f"| **ACT** | OT pushes source→target; needs low W2 + aligned means | "
                 f"W2={pretty(w2_toxic)} cos={pretty(cos_toxic)} {'→ FAIL' if w2_toxic > max(w2_dec, w2_evil)*1.2 else '→ OK'} | "
                 f"W2={pretty(w2_dec)} cos={pretty(cos_dec)} | "
                 f"W2={pretty(w2_evil)} cos={pretty(cos_evil)} |")
    
    # CAA: vector addition works if means are well-separated in consistent direction
    norm_toxic = avg_across_layers(data, "toxic", "exp1_distribution_overlap", "target_norm_mean")
    norm_dec = avg_across_layers(data, "deception", "exp1_distribution_overlap", "target_norm_mean")
    norm_evil = avg_across_layers(data, "evil", "exp1_distribution_overlap", "target_norm_mean")
    
    lines.append(f"| **CAA** | Simple mean diff; needs large norm diff + aligned | "
                 f"‖Δ‖={pretty(norm_toxic)} cos={pretty(cos_toxic)} | "
                 f"‖Δ‖={pretty(norm_dec)} cos={pretty(cos_dec)} | "
                 f"‖Δ‖={pretty(norm_evil)} cos={pretty(cos_evil)} |")
    lines.append("")
    
    # ── Experiment 2: Cluster Analysis ────────────────────────────────────
    lines.append("## 3. Experiment 2: Cluster Analysis (CHaRS)\n")
    lines.append("**What it measures**: Optimal K for k-means clustering of pooled activations. Higher K = more heterogeneous concept = CHaRS's GMM needed.\n")
    
    lines.append("### Optimal K per layer\n")
    for task in tasks:
        k_vals = [get_layer_data(data, task, "exp2_cluster_analysis", i).get("optimal_k", 0) for i in range(6)]
        lines.append(f"- **{task}**: K={k_vals}; mean K={pretty(np.mean(k_vals), '.1f')}")
    lines.append("")
    
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    k_toxic = avg_across_layers(data, "toxic", "exp2_cluster_analysis", "optimal_k")
    k_dec = avg_across_layers(data, "deception", "exp2_cluster_analysis", "optimal_k")
    k_evil = avg_across_layers(data, "evil", "exp2_cluster_analysis", "optimal_k")
    lines.append(f"| **CHaRS** | GMM + OT coupling handles heterogeneity; high K → more benefit | "
                 f"K={pretty(k_toxic, '.0f')} {'→ SHOULD HELP' if k_toxic > max(k_dec, k_evil) else '→ moderate'} | "
                 f"K={pretty(k_dec, '.0f')} | "
                 f"K={pretty(k_evil, '.0f')} |")
    lines.append("")
    
    # ── Experiment 3: Curvature ───────────────────────────────────────────
    lines.append("## 4. Experiment 3: Manifold Curvature (CurveBall)\n")
    lines.append("**What it measures**: Spearman ρ between projection coefficient and steering effectiveness. Low/negative ρ → linear steering is suboptimal → CurveBall's geodesic steering helps.\n")
    
    lines.append("### Spearman ρ (higher = more linear = linear methods work)\n")
    for task in tasks:
        rho_vals = [get_layer_data(data, task, "exp3_curvature", i).get("spearman_rho", 0) for i in range(6)]
        lines.append(f"- **{task}**: ρ={[pretty(v) for v in rho_vals]}; mean ρ={pretty(np.mean(rho_vals))}")
    lines.append("")
    
    lines.append("### Bimodality Coefficient (higher = more multimodal)\n")
    for task in tasks:
        bimod_vals = [get_layer_data(data, task, "exp3_curvature", i).get("bimodality_coefficient", 0) for i in range(6)]
        lines.append(f"- **{task}**: BC={[pretty(v) for v in bimod_vals]}; mean={pretty(np.mean(bimod_vals))}")
    lines.append("")
    
    rho_toxic = avg_across_layers(data, "toxic", "exp3_curvature", "spearman_rho")
    rho_dec = avg_across_layers(data, "deception", "exp3_curvature", "spearman_rho")
    rho_evil = avg_across_layers(data, "evil", "exp3_curvature", "spearman_rho")
    
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    lines.append(f"| **CurveBall** | Nonlinear geodesics needed when ρ<0 or low; ρ>0 → linear works | "
                 f"ρ={pretty(rho_toxic)} {'→ LINEAR FAILS (ρ<0)' if rho_toxic < 0 else '→ LINEAR OK'} | "
                 f"ρ={pretty(rho_dec)} | "
                 f"ρ={pretty(rho_evil)} |")
    lines.append("")
    
    # ── Experiment 4: Trajectory Curvature ────────────────────────────────
    lines.append("## 5. Experiment 4: Trajectory Curvature (FLAS Fig 6)\n")
    lines.append("**What it measures**: Curvature = 1 - mean(step_cos) along linear interpolation between source and target means. Higher curvature → linear interpolation is a poor approximation → multi-step flow needed.\n")
    
    for task in tasks:
        curv_vals = [get_layer_data(data, task, "exp4_trajectory_curvature", i).get("curvature", 0) for i in range(6)]
        lines.append(f"- **{task}**: curvature={[pretty(v) for v in curv_vals]}; mean={pretty(np.mean(curv_vals))}")
    lines.append("")
    
    curv_toxic = avg_across_layers(data, "toxic", "exp4_trajectory_curvature", "curvature")
    curv_dec = avg_across_layers(data, "deception", "exp4_trajectory_curvature", "curvature")
    curv_evil = avg_across_layers(data, "evil", "exp4_trajectory_curvature", "curvature")
    
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    lines.append(f"| **FLAS** | Multi-step flow needed when paths are curved; N=1 fails | "
                 f"curv={pretty(curv_toxic)} {'→ MULTI-STEP CRITICAL' if curv_toxic > 0.1 else '→ LINEAR OK'} | "
                 f"curv={pretty(curv_dec)} | "
                 f"curv={pretty(curv_evil)} |")
    lines.append("")
    
    # ── Experiment 5: Asymmetry ───────────────────────────────────────────
    lines.append("## 6. Experiment 5: Forward/Reverse Asymmetry (Novel)\n")
    lines.append("**What it measures**: Ratio of forward (source→target) to reverse (target→source) transport norm. Ratio > 1 → task is asymmetric → induction harder than mitigation.\n")
    
    for task in tasks:
        asym_vals = [get_layer_data(data, task, "exp5_asymmetry", i).get("asymmetry_ratio", 1) for i in range(6)]
        snr_vals = [get_layer_data(data, task, "exp5_asymmetry", i).get("steer_snr", 0) for i in range(6)]
        lines.append(f"- **{task}**: asymmetry={[pretty(v) for v in asym_vals]}; SNR={[pretty(v) for v in snr_vals]}")
    lines.append("")
    
    asym_toxic = avg_across_layers(data, "toxic", "exp5_asymmetry", "var_asymmetry_source_over_target")
    asym_dec = avg_across_layers(data, "deception", "exp5_asymmetry", "var_asymmetry_source_over_target")
    asym_evil = avg_across_layers(data, "evil", "exp5_asymmetry", "var_asymmetry_source_over_target")
    snr_toxic = avg_across_layers(data, "toxic", "exp5_asymmetry", "steer_snr")
    snr_dec = avg_across_layers(data, "deception", "exp5_asymmetry", "steer_snr")
    snr_evil = avg_across_layers(data, "evil", "exp5_asymmetry", "steer_snr")
    
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    lines.append(f"| ALL (asymmetry) | Safety barrier is one-directional → induction fails | "
                 f"asym={pretty(asym_toxic)} SNR={pretty(snr_toxic)} {'→ HIGH ASYMMETRY (broken)' if asym_toxic > 1.2 else '→ symmetric'} | "
                 f"asym={pretty(asym_dec)} SNR={pretty(snr_dec)} | "
                 f"asym={pretty(asym_evil)} SNR={pretty(snr_evil)} |")
    lines.append("")
    
    # ── Experiment 6: Layer Importance ────────────────────────────────────
    lines.append("## 7. Experiment 6: Layer Importance (LinEAS)\n")
    lines.append("**What it measures**: Normalized signal strength per layer. Identifies which layers carry the most steering-relevant information.\n")
    
    for task in tasks:
        sig_vals = [get_layer_data(data, task, "exp6_layer_importance", i).get("signal_normalized", 0) for i in range(6)]
        d_vals = [get_layer_data(data, task, "exp6_layer_importance", i).get("cohens_d", 0) for i in range(6)]
        lines.append(f"- **{task}**: signal={[pretty(v) for v in sig_vals]}; Cohen's d={[pretty(v) for v in d_vals]}")
    lines.append("")
    
    lines.append("| Method | Key claim | Toxic | Deception | Evil |")
    lines.append("|--------|-----------|-------|-----------|------|")
    lines.append(f"| **LinEAS** | Group lasso selects relevant layers; sparsity preserves utility | See per-layer signals above → which layers concentrate signal | | |")
    lines.append("")
    
    # ── Experiment 7: PCA ─────────────────────────────────────────────────
    lines.append("## 8. Experiment 7: PCA Low-Rank Structure\n")
    lines.append("**What it measures**: Effective rank and PCs needed for 90%/95%/99% variance. Lower rank = simpler steering geometry.\n")
    
    for task in tasks:
        eff_vals = [get_layer_data(data, task, "exp7_pca_low_rank", i).get("effective_rank", 0) for i in range(6)]
        k95_vals = [get_layer_data(data, task, "exp7_pca_low_rank", i).get("k_for_95pct", 0) for i in range(6)]
        lines.append(f"- **{task}**: eff_rank={[pretty(v, '.1f') for v in eff_vals]}; K@95%={k95_vals}")
    lines.append("")
    
    # ── Synthesis ─────────────────────────────────────────────────────────
    lines.append("## 9. Synthesis: Method × Task Success Matrix\n")
    lines.append("")
    lines.append("| Method | Toxic | Deception | Evil | Why? |")
    lines.append("|--------|-------|-----------|------|------|")
    
    # Toxic is hardest
    toxic_reasons = []
    if w2_toxic > max(w2_dec, w2_evil) * 1.1:
        toxic_reasons.append("highest W2 distance")
    if rho_toxic < 0:
        toxic_reasons.append("negative Spearman ρ (nonlinear geometry)")
    if curv_toxic > 0.1:
        toxic_reasons.append(f"high trajectory curvature ({pretty(curv_toxic)})")
    if asym_toxic > 1.2:
        toxic_reasons.append(f"high asymmetry ({pretty(asym_toxic)})")
    
    dec_reasons = []
    if w2_dec < w2_toxic:
        dec_reasons.append(f"lower W2 ({pretty(w2_dec)} vs {pretty(w2_toxic)})")
    if rho_dec > 0:
        dec_reasons.append("positive ρ (linear-friendly)")
    if curv_dec < curv_toxic:
        dec_reasons.append(f"lower curvature ({pretty(curv_dec)} vs {pretty(curv_toxic)})")
    
    evil_reasons = []
    if w2_evil < w2_toxic:
        evil_reasons.append(f"lower W2 ({pretty(w2_evil)} vs {pretty(w2_toxic)})")
    if rho_evil > 0:
        evil_reasons.append("positive ρ")
    
    lines.append(f"| ACT | {'FAIL' if w2_toxic > w2_dec * 1.2 else '?'} | OK | OK | "
                 f"{'; '.join(toxic_reasons[:2]) if toxic_reasons else 'unknown'} |")
    lines.append(f"| CHaRS | SHOULD HELP (K={pretty(k_toxic, '.0f')}) | OK | OK | "
                 f"Toxic needs K={pretty(k_toxic, '.0f')} clusters (most heterogeneous) |")
    lines.append(f"| CurveBall | {'SHOULD HELP' if rho_toxic < 0 else '?'} | OK | OK | "
                 f"Toxic ρ={pretty(rho_toxic)}; Deception ρ={pretty(rho_dec)} |")
    lines.append(f"| FLAS | SHOULD HELP (curv={pretty(curv_toxic)}) | OK | OK | "
                 f"Multi-step flow may traverse safety barrier |")
    lines.append(f"| LinEAS | Partial (0% broken) | OK (0.74/1.41) | OK (0.46/4.00) | "
                 f"End-to-end training needs better loss for asymmetric tasks |")
    lines.append(f"| CAA | FAIL (0%) | OK (0.76/1.56) | Partial (0.11/1.78) | "
                 f"Simple mean-diff insufficient for nonlinear geometry |")
    lines.append("")
    
    # ── Key Insights ──────────────────────────────────────────────────────
    lines.append("## 10. Key Cross-Cutting Insights\n")
    lines.append("")
    
    # Find which diagnostic is most predictive
    lines.append("### Which diagnostic best predicts method success?\n")
    lines.append("")
    
    # For each diagnostic, rate how well it separates toxic from others
    lines.append("| Diagnostic | Toxic vs non-Toxic separation | Predicts what |")
    lines.append("|------------|------------------------------|---------------|")
    
    # W2 separation
    w2_sep = abs(w2_toxic - np.mean([w2_dec, w2_evil])) / (np.std([w2_dec, w2_evil]) + 1e-8)
    lines.append(f"| W2 distance | diff={pretty(w2_sep)}σ | ACT success (lower = better) |")
    
    # K separation
    k_sep = abs(k_toxic - np.mean([k_dec, k_evil])) / (np.std([k_dec, k_evil]) + 1e-8)
    lines.append(f"| Optimal K | diff={pretty(k_sep)}σ | CHaRS benefit (higher = more needed) |")
    
    # Rho separation
    rho_sep = abs(rho_toxic - np.mean([rho_dec, rho_evil])) / (np.std([rho_dec, rho_evil]) + 1e-8)
    lines.append(f"| Spearman ρ | diff={pretty(rho_sep)}σ | CurveBall vs linear (ρ<0 → nonlinear) |")
    
    # Curvature separation
    curv_sep = abs(curv_toxic - np.mean([curv_dec, curv_evil])) / (np.std([curv_dec, curv_evil]) + 1e-8)
    lines.append(f"| Trajectory curvature | diff={pretty(curv_sep)}σ | FLAS benefit (higher = multi-step needed) |")
    
    # Asymmetry separation
    asym_sep = abs(asym_toxic - np.mean([asym_dec, asym_evil])) / (np.std([asym_dec, asym_evil]) + 1e-8)
    lines.append(f"| Asymmetry ratio | diff={pretty(asym_sep)}σ | Induction difficulty (higher = harder) |")
    lines.append("")
    
    lines.append("### Conclusion\n")
    lines.append("")
    
    # Find the most separated diagnostic
    diagnostics = {
        "W2 distance": w2_sep,
        "Optimal K": k_sep,
        "Spearman ρ": rho_sep,
        "Trajectory curvature": curv_sep,
        "Asymmetry ratio": asym_sep,
    }
    best_diag = max(diagnostics, key=diagnostics.get)
    lines.append(f"The diagnostic that **best separates toxic from non-toxic tasks** is: **{best_diag}** "
                 f"({pretty(diagnostics[best_diag])}σ separation).\n")
    lines.append("")
    
    for task in tasks:
        w2 = avg_across_layers(data, task, "exp1_distribution_overlap", "w2")
        cos = avg_across_layers(data, task, "exp1_distribution_overlap", "cosine_similarity")
        k = avg_across_layers(data, task, "exp2_cluster_analysis", "optimal_k")
        rho = avg_across_layers(data, task, "exp3_curvature", "spearman_rho")
        curv = avg_across_layers(data, task, "exp4_trajectory_curvature", "curvature")
        asym = avg_across_layers(data, task, "exp5_asymmetry", "var_asymmetry_source_over_target")
        sig = avg_across_layers(data, task, "exp6_layer_importance", "signal_normalized")
        
        lines.append(f"- **{task}**: W2={pretty(w2)}, cos={pretty(cos)}, K={pretty(k, '.0f')}, "
                     f"ρ={pretty(rho)}, curv={pretty(curv)}, asym={pretty(asym)}, sig(L14)={pretty(sig)}")
    lines.append("")
    
    lines.append("### Recommended Method × Task\n")
    lines.append("| Task | Best method | Why |")
    lines.append("|------|-------------|-----|")
    
    if rho_toxic < 0 and curv_toxic > 0.1:
        lines.append(f"| Toxic | **FLAS or CurveBall** | ρ={pretty(rho_toxic)}<0 (nonlinear), curv={pretty(curv_toxic)}>0.1 (curved path); multi-step flow or geodesic needed |")
    elif k_toxic > 10:
        lines.append(f"| Toxic | **CHaRS** | K={pretty(k_toxic, '.0f')} (highest heterogeneity); GMM captures subtypes |")
    else:
        lines.append(f"| Toxic | **CHaRS + CurveBall** | K={pretty(k_toxic, '.0f')} clusters, ρ={pretty(rho_toxic)}<0; combined GMM + geodesic |")
    
    if rho_dec > 0.3 and curv_dec < 0.05:
        lines.append(f"| Deception | **LinNEAS (L14-c2)** | ρ={pretty(rho_dec)}>0, low curv={pretty(curv_dec)}; linear is fine, layer sparsity helps |")
    else:
        lines.append(f"| Deception | **LinNEAS or ACT** | ρ={pretty(rho_dec)}, curv={pretty(curv_dec)}; moderate geometry |")
    
    if rho_evil > 0:
        lines.append(f"| Evil | **LinNEAS (26L-c2)** or **CHaRS** | ρ={pretty(rho_evil)}>0 (linear OK) but K={pretty(k_evil, '.0f')} (heterogeneous); GMM may improve |")
    else:
        lines.append(f"| Evil | **CurveBall** | ρ={pretty(rho_evil)}<0; nonlinear needed |")
    
    lines.append("")
    
    # Write output
    output = "\n".join(lines)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(output)
    print(f"Analysis written to {OUTPUT_PATH}")
    print(output)

if __name__ == "__main__":
    main()
