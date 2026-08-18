"""
Coupling mass analysis for a single (task, K) using pre-extracted source activations.
Usage: python analyze_coupling.py <task_name> <K>
  task_name: toxic | evil
  K: 3 | 5 | 10 | 20

Saves results to Experiments/coupling_results/{task}_K{K}.json
"""
import sys, os, json, torch
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LAYER = 14

def main():
    task_name = sys.argv[1]
    K = int(sys.argv[2])

    # Paths
    acts_path = Path(f"Vector/CHARS/Gemma/{task_name}_source_acts.pt")
    vector_dir = Path(f"Vector/CHARS/Gemma/{task_name}_K{K}")
    md_path = vector_dir / "metadata.pt"

    assert acts_path.exists(), f"Source activations not found: {acts_path}"
    assert md_path.exists(), f"CHARS metadata not found: {md_path}"

    # Load source activations
    source_acts = torch.load(acts_path, map_location="cpu", weights_only=True).float()
    print(f"Source activations: {source_acts.shape}")

    # Load CHARS metadata
    md = torch.load(str(md_path), map_location="cpu", weights_only=True)
    layer_key = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    centroids_A = md["chars_centroids_A"][layer_key].float()
    P_star = md["chars_coupling"][layer_key].float()
    actual_K = md["chars_k"][layer_key]
    print(f"  CHARS centroids: {centroids_A.shape}, coupling: {P_star.shape}, K={actual_K}")

    # Assign samples to nearest centroid
    dists = torch.cdist(source_acts, centroids_A, p=2.0)
    assigned_centroid = dists.argmin(dim=1)
    n_per_centroid = torch.bincount(assigned_centroid, minlength=actual_K)
    n_active = (n_per_centroid > 0).sum().item()
    print(f"  Samples per centroid: {n_per_centroid.tolist()}")
    print(f"  Active centroids: {n_active} / {actual_K}")

    # Coupling mass per centroid
    coupling_mass_per_centroid = P_star.sum(dim=1).numpy()  # [K]

    # Sample norms
    source_norms = source_acts.norm(dim=1).numpy()

    # Per-decile breakdown
    percentiles = np.arange(0, 101, 10)
    norm_percentiles = np.percentile(source_norms, percentiles)
    norm_bin_idx = np.digitize(source_norms, norm_percentiles[1:-1])

    decile_data = []
    for i in range(10):
        mask = norm_bin_idx == i
        n = mask.sum()
        if n == 0:
            continue
        centroid_ids = assigned_centroid[mask]
        avg_mass = coupling_mass_per_centroid[centroid_ids].mean()
        norm_range = f"{norm_percentiles[i]:.0f}-{norm_percentiles[i+1]:.0f}"
        decile_data.append({"decile": f"{i*10}-{(i+1)*10}%", "n": int(n),
                            "norm_range": norm_range, "avg_coupling_mass": float(avg_mass)})
        print(f"  {i*10:>3d}-{(i+1)*10:>3d}%  n={n:>4d}  norm={norm_range:>10s}  mass={avg_mass:.4f}")

    # Spearman: sample norm vs coupling mass
    sample_mass = coupling_mass_per_centroid[assigned_centroid.numpy()]
    rho, p = spearmanr(source_norms, sample_mass)
    r_pearson, p_pearson = pearsonr(source_norms, sample_mass)
    print(f"\n  Spearman ρ(norm, mass) = {rho:.4f}, p = {p:.6f}")
    print(f"  Pearson r(norm, mass) = {r_pearson:.4f}, p = {p_pearson:.6f}")

    # Tail analysis: top 5% vs body
    tail_threshold = np.percentile(source_norms, 95)
    tail_mask = source_norms >= tail_threshold
    body_mask = source_norms < tail_threshold
    tail_mass = float(sample_mass[tail_mask].mean())
    body_mass = float(sample_mass[body_mask].mean())
    tail_body_ratio = tail_mass / body_mass if body_mass > 0 else float('nan')
    print(f"  Top 5% tail ({tail_mask.sum()}): avg mass = {tail_mass:.4f}")
    print(f"  Bottom 95% ({body_mask.sum()}): avg mass = {body_mass:.4f}")
    print(f"  Tail/Body ratio = {tail_body_ratio:.4f}")

    # Centroid-level stats
    centroid_norms = centroids_A.norm(dim=1).numpy()
    centroid_cv = float(centroid_norms.std() / centroid_norms.mean())
    rho_c, p_c = spearmanr(centroid_norms, coupling_mass_per_centroid)
    print(f"  Centroid CV = {centroid_cv:.4f}")
    print(f"  Centroid ρ(norm, mass) = {rho_c:.4f}, p = {p_c:.6f}")

    # Result
    result = {
        "task": task_name,
        "K": actual_K,
        "K_requested": K,
        "n_active_centroids": n_active,
        "centroid_norms": [float(v) for v in centroid_norms],
        "samples_per_centroid": [int(v) for v in n_per_centroid.tolist()],
        "coupling_mass_per_centroid": [float(v) for v in coupling_mass_per_centroid],
        "centroid_norm_cv": centroid_cv,
        "spearman_norm_vs_mass_sample": float(rho) if not np.isnan(rho) else None,
        "spearman_p_sample": float(p) if not np.isnan(p) else None,
        "pearson_norm_vs_mass_sample": float(r_pearson) if not np.isnan(r_pearson) else None,
        "pearson_p_sample": float(p_pearson) if not np.isnan(p_pearson) else None,
        "spearman_norm_vs_mass_centroid": float(rho_c) if not np.isnan(rho_c) else None,
        "spearman_p_centroid": float(p_c) if not np.isnan(p_c) else None,
        "tail_5pct_avg_mass": tail_mass,
        "body_95pct_avg_mass": body_mass,
        "tail_body_ratio": tail_body_ratio,
        "decile_data": decile_data,
        "n_samples": source_acts.shape[0],
    }

    # Save
    out_dir = Path("Experiments/coupling_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_K{K}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
