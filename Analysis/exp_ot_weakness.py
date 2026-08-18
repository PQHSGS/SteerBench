"""
OT Method Weakness Diagnostics
Exp 1: ACT — effective rank of source covariance vs clean accuracy
Exp 2: CHARS — barycentric cancellation score (cosine between transport directions)
Exp 3: CHARS — centroid norm spread (isotropic assumption violation)

All CPU-only. Uses existing Vector/ files.
"""

import torch
import numpy as np
import json
import os
import glob
from pathlib import Path

RESULTS_DIR = "/home/caotue/SAESteeringBench/Results"
VECTOR_DIR  = "/home/caotue/SAESteeringBench/Vector"
OUT_FILE    = "/home/caotue/SAESteeringBench/Analysis/results/ot_weakness_diagnostics.json"

LAYER = 14

# Known clean-best accuracies from benchmark tables
ACT_CLEAN = {
    "deception": 0.90,
    "evil":      0.50,
    "toxic":     0.21,
    "refusal":   None,
}
CHARS_CLEAN = {
    "deception":        0.85,
    "evil":             0.87,
    "toxic":            0.02,
    "refusal_response": None,
}

CHARS_TASKS = ["evil", "deception", "toxic", "refusal_response"]
ACT_TASKS   = ["evil", "deception", "toxic"]

# ─────────────────────────────────────────────────────────────────────────────
def load_chars_metadata(task):
    path = f"{VECTOR_DIR}/CHARS/Gemma/{task}/metadata.pt"
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)

def effective_rank(matrix):
    """Effective rank = exp(Shannon entropy of normalized singular values)."""
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    S = S[S > 1e-8]
    p = S / S.sum()
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H))

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: ACT — Effective Rank
# Use CHARS source centroids as representative samples of each task's
# activation manifold at layer 14.
# ─────────────────────────────────────────────────────────────────────────────
def exp1_act_effrank():
    print("\n" + "="*60)
    print("EXP 1: ACT — Effective Rank of Source Covariance")
    print("="*60)
    results = {}
    for task in CHARS_TASKS:
        meta = load_chars_metadata(task)
        if meta is None:
            print(f"  {task}: no metadata, skip")
            continue
        A = meta["chars_centroids_A"][LAYER].float()   # (K, d)
        eff_r = effective_rank(A)

        norms = A.norm(dim=1)
        norm_max, norm_min = float(norms.max()), float(norms.min())
        norm_spread = norm_max / norm_min if norm_min > 0 else float("inf")

        task_key = task.replace("_response", "")
        clean_acc = ACT_CLEAN.get(task_key, ACT_CLEAN.get(task, None))
        print(f"  {task:22s}  eff_rank={eff_r:6.1f}  "
              f"norm_spread={norm_spread:5.2f}x  "
              f"ACT_best_clean={str(clean_acc)}")
        results[task] = {
            "eff_rank": round(eff_r, 2),
            "centroid_norm_max": round(norm_max, 2),
            "centroid_norm_min": round(norm_min, 2),
            "centroid_norm_spread_ratio": round(norm_spread, 2),
            "act_best_clean_acc": clean_acc,
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2: CHARS — Barycentric Cancellation Score
# ─────────────────────────────────────────────────────────────────────────────
def exp2_chars_cancellation():
    print("\n" + "="*60)
    print("EXP 2: CHARS — Barycentric Cancellation Score")
    print("="*60)
    results = {}
    for task in CHARS_TASKS:
        meta = load_chars_metadata(task)
        if meta is None:
            print(f"  {task}: no metadata"); continue

        A  = meta["chars_centroids_A"][LAYER].float()   # (K, d)
        B  = meta["chars_centroids_B"][LAYER].float()   # (K, d)
        Pi = meta["chars_coupling"][LAYER].float()      # (K, K)
        K  = A.shape[0]

        # Build weighted transport direction vectors d_{ij} = b_j - a_i
        directions, weights = [], []
        for i in range(K):
            for j in range(K):
                w = float(Pi[i, j])
                if w < 1e-6:
                    continue
                d = B[j] - A[i]
                if d.norm() < 1e-8:
                    continue
                directions.append(d / d.norm())
                weights.append(w)

        n = len(directions)
        w_t    = torch.tensor(weights)
        dir_mat = torch.stack(directions)   # (n, d)

        # Pairwise cosine similarity
        cos_mat = dir_mat @ dir_mat.T       # (n, n)

        W = w_t.unsqueeze(1) * w_t.unsqueeze(0)
        negative_cos = torch.clamp(cos_mat, max=0.0)
        weighted_cancellation = float((W * negative_cos).sum())

        off_diag_mask = ~torch.eye(n, dtype=torch.bool)
        off_diag_cos  = cos_mat[off_diag_mask]
        frac_opposing = float((off_diag_cos < -0.3).float().mean())
        mean_cos      = float(off_diag_cos.mean())

        # Net transport vector (uniform source; weighted by Pi marginal)
        net_vec = torch.zeros(A.shape[1])
        for i in range(K):
            for j in range(K):
                net_vec += float(Pi[i, j]) * (B[j] - A[i])
        net_norm = float(net_vec.norm())
        # Average magnitude of individual directions (unweighted)
        avg_raw_norm = float((B - A).norm(dim=1).mean())
        cancellation_ratio = 1.0 - (net_norm / avg_raw_norm) if avg_raw_norm > 0 else 1.0

        task_key  = task.replace("_response", "")
        clean_acc = CHARS_CLEAN.get(task, CHARS_CLEAN.get(task_key, None))
        print(f"\n  {task}")
        print(f"    Active direction pairs  : {n}")
        print(f"    Mean pairwise cos       : {mean_cos:+.4f}")
        print(f"    Fraction opposing pairs : {frac_opposing:.1%}  (cos < -0.3)")
        print(f"    Weighted cancellation   : {weighted_cancellation:+.6f}")
        print(f"    Net transport norm      : {net_norm:.4f}")
        print(f"    Avg individual dir norm : {avg_raw_norm:.4f}")
        print(f"    Cancellation ratio      : {cancellation_ratio:.4f}  (0=perfect alignment, 1=total cancel)")
        print(f"    CHARS best clean acc    : {clean_acc}")

        results[task] = {
            "n_active_direction_pairs": n,
            "mean_pairwise_cos": round(mean_cos, 4),
            "fraction_opposing_pairs": round(frac_opposing, 4),
            "weighted_cancellation_score": round(weighted_cancellation, 6),
            "net_transport_norm": round(net_norm, 4),
            "avg_individual_dir_norm": round(avg_raw_norm, 4),
            "cancellation_ratio": round(max(cancellation_ratio, 0.0), 4),
            "chars_best_clean_acc": clean_acc,
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: CHARS — Centroid Norm Spread (isotropic assumption violation)
# ─────────────────────────────────────────────────────────────────────────────
def exp3_chars_covariance_violation():
    print("\n" + "="*60)
    print("EXP 3: CHARS — Centroid Norm Spread (Equal-Cov Assumption Violation)")
    print("="*60)
    results = {}
    for task in CHARS_TASKS:
        meta = load_chars_metadata(task)
        if meta is None:
            print(f"  {task}: no metadata"); continue

        A  = meta["chars_centroids_A"][LAYER].float()
        B  = meta["chars_centroids_B"][LAYER].float()
        Pi = meta["chars_coupling"][LAYER].float()

        a_norms = A.norm(dim=1)
        b_norms = B.norm(dim=1)

        a_spread = float(a_norms.max() / a_norms.min())
        b_spread = float(b_norms.max() / b_norms.min())
        a_cv     = float(a_norms.std() / a_norms.mean())
        b_cv     = float(b_norms.std() / b_norms.mean())

        # Inter-centroid pairwise distance spread
        diffs     = A.unsqueeze(0) - A.unsqueeze(1)   # (K,K,d)
        pdist     = diffs.norm(dim=2)                  # (K,K)
        off_d     = pdist[~torch.eye(A.shape[0], dtype=torch.bool)]
        dist_cv   = float(off_d.std() / off_d.mean())

        # Effective cluster count from Pi entropy
        row_sums  = Pi.sum(dim=1)
        row_sums  = row_sums[row_sums > 1e-8]
        p         = row_sums / row_sums.sum()
        H         = -(p * torch.log(p + 1e-12)).sum()
        eff_k     = float(torch.exp(H))

        task_key  = task.replace("_response", "")
        clean_acc = CHARS_CLEAN.get(task, CHARS_CLEAN.get(task_key, None))
        print(f"\n  {task}")
        print(f"    K (stored)              : {int(meta['chars_k'][LAYER])}")
        print(f"    Eff K (Pi entropy)      : {eff_k:.1f}")
        print(f"    Source norm range       : {a_norms.min():.1f} – {a_norms.max():.1f}  spread={a_spread:.1f}x  CV={a_cv:.3f}")
        print(f"    Target norm range       : {b_norms.min():.1f} – {b_norms.max():.1f}  spread={b_spread:.1f}x  CV={b_cv:.3f}")
        print(f"    Inter-centroid dist CV  : {dist_cv:.3f}")
        print(f"    CHARS best clean acc    : {clean_acc}")

        results[task] = {
            "K_stored": int(meta["chars_k"][LAYER]),
            "eff_K": round(eff_k, 2),
            "source_norm_min": round(float(a_norms.min()), 2),
            "source_norm_max": round(float(a_norms.max()), 2),
            "source_norm_spread_ratio": round(a_spread, 2),
            "source_norm_cv": round(a_cv, 4),
            "target_norm_spread_ratio": round(b_spread, 2),
            "target_norm_cv": round(b_cv, 4),
            "inter_centroid_dist_cv": round(dist_cv, 4),
            "chars_best_clean_acc": clean_acc,
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    exp1 = exp1_act_effrank()
    exp2 = exp2_chars_cancellation()
    exp3 = exp3_chars_covariance_violation()

    output = {
        "exp1_act_effrank": exp1,
        "exp2_chars_cancellation": exp2,
        "exp3_chars_covariance_violation": exp3,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved → {OUT_FILE}")

    print("\n" + "="*70)
    print("SUMMARY — ACT eff_rank vs Clean Accuracy")
    print("="*70)
    print(f"{'Task':<22} {'eff_rank':>10} {'ACT clean':>12}")
    for t, r in exp1.items():
        print(f"{t:<22} {r['eff_rank']:>10.1f} {str(r['act_best_clean_acc']):>12}")

    print("\n" + "="*70)
    print("SUMMARY — CHARS Cancellation Score vs Clean Accuracy")
    print("="*70)
    print(f"{'Task':<22} {'mean_cos':>10} {'frac_opp':>10} {'cancel_ratio':>14} {'CHARS clean':>12}")
    for t, r in exp2.items():
        print(f"{t:<22} {r['mean_pairwise_cos']:>+10.4f} "
              f"{r['fraction_opposing_pairs']:>10.1%} "
              f"{r['cancellation_ratio']:>14.4f} "
              f"{str(r['chars_best_clean_acc']):>12}")

    print("\n" + "="*70)
    print("SUMMARY — CHARS Centroid Norm Spread vs Clean Accuracy")
    print("="*70)
    print(f"{'Task':<22} {'src_spread':>11} {'src_CV':>8} {'eff_K':>7} {'CHARS clean':>12}")
    for t, r in exp3.items():
        print(f"{t:<22} {r['source_norm_spread_ratio']:>10.1f}x "
              f"{r['source_norm_cv']:>8.3f} "
              f"{r['eff_K']:>7.1f} "
              f"{str(r['chars_best_clean_acc']):>12}")
