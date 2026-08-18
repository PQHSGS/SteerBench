"""
Deep P* coupling inspection: where does mass go, per PCA components.
Three definitions of "tail":
  (a) L2 norm (canonical)
  (b) In-manifold L2 (top-k PCs, 90% var)
  (c) Off-manifold L2 (residual)
Plus per-centroid mass destination: tail-to-tail, tail-to-common, or uniform?
"""
import json, torch
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

LAYER = 14
PC_VAR = 0.90  # keep 90% variance

def pca_decompose(X, var_ratio=PC_VAR):
    """Return in-manifold and off-manifold components."""
    pca = PCA(n_components=min(X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, var_ratio) + 1)
    # In-manifold: project back from top-k PCs
    X_in = pca.inverse_transform(np.column_stack([
        X_pca[:, :n_keep],
        np.zeros((X_pca.shape[0], X_pca.shape[1] - n_keep))
    ]))
    X_off = X - X_in
    return X_in, X_off, n_keep, pca

def inspect_P_structure(task, K):
    # Load metadata
    if task == "toxic" and K == 10:
        path = Path("Vector/CHARS/Gemma/toxic/metadata.pt")
    elif task == "evil" and K == 20:
        path = Path("Vector/CHARS/Gemma/evil/metadata.pt")
    else:
        path = Path(f"Vector/CHARS/Gemma/{task}_K{K}/metadata.pt")
    if not path.exists():
        return
    md = torch.load(str(path), map_location="cpu", weights_only=True)
    lk = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    cA = md["chars_centroids_A"][lk].float().numpy()  # [K, d]
    cB = md["chars_centroids_B"][lk].float().numpy()
    P = md["chars_coupling"][lk].float().numpy()       # [K, K]
    KA = P.shape[0]
    
    print(f"\n{'='*70}")
    print(f"  {task} K={K}")
    print(f"{'='*70}")
    
    # === 1. PCA decomposition on pooled centroids ===
    pooled = np.vstack([cA, cB])
    cA_in, cA_off, n_pc, pca_model = pca_decompose(cA)
    cB_in, cB_off, _, _ = pca_decompose(cB)
    
    # Compute three metrics per centroid
    norms_l2 = np.linalg.norm(cA, axis=1)
    norms_in = np.linalg.norm(cA_in, axis=1)
    norms_off = np.linalg.norm(cA_off, axis=1)
    
    print(f"\n  PCA: {n_pc} PCs explain {PC_VAR*100:.0f}% variance (d={cA.shape[1]})")
    
    # === 2. Mass distribution per source centroid ===
    mass_from = P.sum(axis=1)  # [K]
    
    # For each source centroid, where does mass go? Measure entropy of destination
    dest_entropy = []
    for i in range(KA):
        p_dest = P[i, :] / mass_from[i] if mass_from[i] > 0 else np.ones(KA)/KA
        ent = -np.sum(p_dest * np.log(p_dest + 1e-10)) / np.log(KA)
        dest_entropy.append(ent)
    
    # === 3. Tail-to-tail vs tail-to-common analysis ===
    # Define "tail" source centroids: top 50% by norm (for K small, use top 1/3)
    tail_idx_l2 = np.argsort(norms_l2)[-KA//3:]  # top 33% by L2
    tail_idx_in = np.argsort(norms_in)[-KA//3:]
    tail_idx_off = np.argsort(norms_off)[-KA//3:]
    
    # For tail sources, where does mass go? Is it concentrated on tail targets?
    target_norms_l2 = np.linalg.norm(cB, axis=1)
    
    def mass_destination(P_slice, target_norms):
        """Given source rows of P, where does their mass go? Return (tail_mass, common_mass, self_mass, diag_mass)"""
        total = P_slice.sum()
        if total == 0:
            return 0, 0, 0, 0
        # Tail targets: top 33%
        tgt_thresh = np.percentile(target_norms, 66.7)
        is_tail_tgt = target_norms >= tgt_thresh
        mass_to_tail = P_slice[:, is_tail_tgt].sum() / total
        
        # Self-transport: mass to same index
        mass_self = np.diag(P_slice).sum() / total
        
        # Diagonal concentration: fraction of mass on self + immediate neighbors
        return float(mass_to_tail), float(mass_self), float(mass_self)
    
    # Print table
    print(f"\n  {'centroid':>9s} {'L2_norm':>8s} {'in_manif':>8s} {'off_manif':>8s} {'mass_from':>10s} {'dest_entropy':>12s} {'self_mass':>10s}")
    print(f"  {'-'*65}")
    
    self_transport = []
    sorted_idx = np.argsort(norms_l2)
    for i in sorted_idx:
        mass_self = P[i, i] / mass_from[i] if mass_from[i] > 0 else 0
        self_transport.append(mass_self)
        print(f"  {i:>9d} {norms_l2[i]:>8.1f} {norms_in[i]:>8.1f} {norms_off[i]:>8.1f} {mass_from[i]:>10.6f} {dest_entropy[i]:>12.3f} {mass_self:>10.3f}")
    
    # === 4. Per-centroid mass destination profile ===
    print(f"\n  --- Full P* matrix (source→target mass, normalized per row) ---")
    print(f"  (Rows=source sorted by L2 norm ascending, Cols=target)")
    sorted_idx2 = np.argsort(norms_l2)
    P_sorted = P[sorted_idx2, :]
    # Normalize rows
    P_row = P_sorted / (P_sorted.sum(axis=1, keepdims=True) + 1e-10)
    print(f"  Row-normalized P*:")
    for i, orig_i in enumerate(sorted_idx2):
        row_str = " ".join(f"{v:.2f}" for v in P_row[i])
        print(f"  src[{orig_i}] norm={norms_l2[orig_i]:.0f}: {row_str}")
    
    # === 5. Summary stats ===
    print(f"\n  --- Summary ---")
    print(f"  L2:   ρ(norm, mass_from)={spearmanr(norms_l2, mass_from).statistic:.4f}")
    print(f"  In:   ρ(in_norm, mass_from)={spearmanr(norms_in, mass_from).statistic:.4f}")
    print(f"  Off:  ρ(off_norm, mass_from)={spearmanr(norms_off, mass_from).statistic:.4f}")
    
    # Tail-to-tail analysis
    print(f"\n  Tail sources (top 33% by L2 norm):")
    mass_to_tail_l2, self_mass_l2, _ = mass_destination(P[tail_idx_l2, :], target_norms_l2)
    print(f"    Mass to tail targets: {mass_to_tail_l2:.3f} (uniform would be {1/3:.3f})")
    print(f"    Self-transport: {self_mass_l2:.3f}")
    
    mass_to_tail_in, self_mass_in, _ = mass_destination(P[tail_idx_in, :], 
        np.linalg.norm(cB, axis=1))
    print(f"  Tail sources (top 33% by in-manifold norm):")
    print(f"    Mass to tail targets: {mass_to_tail_in:.3f}")
    
    mass_to_tail_off, self_mass_off, _ = mass_destination(P[tail_idx_off, :],
        np.linalg.norm(cB, axis=1))
    print(f"  Tail sources (top 33% by off-manifold norm):")
    print(f"    Mass to tail targets: {mass_to_tail_off:.3f}")
    
    # === 6. RBF kernel matrix ===
    print(f"\n  --- RBF kernel matrix (sorted by L2 norm) ---")
    dists = np.linalg.norm(cA[sorted_idx2, None, :] - cB[None, sorted_idx2, :], axis=2)
    # Estimate tau as median pairwise distance
    tau = np.median(dists)
    rbf = np.exp(-dists**2 / (2 * tau**2))
    print(f"  RBF tau={tau:.2f}")
    for i, orig_i in enumerate(sorted_idx2):
        row_str = " ".join(f"{v:.4f}" for v in rbf[i])
        if rbf[i].max() < 0.1:  # All near-zero
            print(f"  src[{orig_i}] norm={norms_l2[orig_i]:.0f}: ALL RBF ≈ 0 (max={rbf[i].max():.6f})")
        else:
            print(f"  src[{orig_i}] norm={norms_l2[orig_i]:.0f}: {row_str}")

# Run
inspect_P_structure("toxic", 10)
inspect_P_structure("evil", 20)
print("\n\nDone.")
