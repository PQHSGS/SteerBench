"""
CHARS Tail-Cluster Transport Analysis — Hill estimator version.

Tail definition: per-dim Hill tail index α < α_thresh on pooled
target+contrast activations (absolute values). Small α = heavier tail.

Ask: Do centroids with large projection into heavy-tail dims transport
differently? Are they isolated (low RBF, high self-mass)?

For every existing CHARS metadata file across all tasks × K:
1. Load activations from ExpG .pt file (pooled target+contrast)
2. Per-dim Hill α → tail-dim mask (α < TAIL_ALPHA_THRESH)
3. Load CHARS centroids cA, cB, coupling P*
4. Per-centroid: norm in tail dims only
5. Tail centroids = top-quartile by tail-projection norm
6. Same downstream: mass-to-tail, self-mass, enrichment, RBF, dead

Usage:
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python -m Experiments.TailAnalysis.chars_tail_clusters
"""

import json, torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("Experiments/TailAnalysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 14
TAIL_QUARTILE = 0.75  # top 25% of centroids by tail-norm = "tail"
TAIL_ALPHA_THRESH = 3.5  # Hill α < 3.5 → heavy tail (bottom ~1% of activation dims)
HILL_K_FRAC = 0.1        # fraction of samples for Hill upper order statistics

# --- sample-level tail-cluster definition (faithful to OT sample view) ---
TAIL_SAMPLE_FRAC = 0.10                # top-10% of K as tail clusters
TAIL_SAMPLE_MIN_CLUSTERS = 1           # floor on #tail clusters
TAIL_SAMPLE_PCTL = 95                  # tail sample = norm >= 95th pct
TAIL_SAMPLE_HAS_TAIL_ALPHA = 3.0       # Hill α(norm) < 3 → distribution "has a tail"
TAIL_SAMPLE_MIN_ABS_FLOOR = 3          # abs floor on tail-sample count per cluster

# === Constants for combined-rule tail_sample definition (faithful) ===
# Sample-level: per-sample N-dim norm >= 95th pct = "tail sample"
# Existence: Hill α of N-dim norm dist < 3 = "distribution has tail"
# Cluster-level: top max(1, ceil(K*0.10)) clusters by tail-sample count
# Absolute noise floor: cluster must own >= max(3, ceil(|tail|*0.10)) tail samples
NORM_PCTL_THRESHOLD    = 95     # sample-level tail percentile
HILL_ALPHA_NORM_HAS_TAIL = 3.0  # per-distribution existence check (Hill α on norms)
TAIL_CLUSTER_FRACTION  = 0.10   # top 10% of K clusters by tail-sample count
MIN_TAIL_CLUSTERS      = 1      # floor on number of tail-clusters
MIN_TAIL_SAMPLES_FLOOR = 0.10   # absolute floor = 10% of |tail_samples|, max with 3
MIN_TAIL_ABS_MINIMUM   = 3      # never less than 3 tail samples per cluster
PCTL = 99                       # percentile threshold for tail-sample definition

# All CHARS variant paths with known K and task name
def hill_estimator(x, k=None):
    """Hill estimator for tail index α on absolute values.
    Returns (gamma, alpha) where gamma = extreme-value index, α = 1/γ.
    Larger γ → heavier tail. α < TAIL_ALPHA_THRESH → heavy-tailed dim.
    """
    n = len(x)
    if k is None:
        k = max(2, int(n * HILL_K_FRAC))
    k = min(k, n // 4)
    x_abs = np.abs(x)
    x_sorted = np.sort(x_abs)
    if k < 2 or len(x_sorted) <= k or x_sorted[-k-1] <= 0:
        return np.nan, np.nan
    log_data = np.log(x_sorted[-k:])
    log_k = np.log(max(x_sorted[-k-1], 1e-12))
    gamma = (log_data - log_k).mean()
    if gamma <= 1e-12:
        gamma = np.nan
        alpha = np.inf
    else:
        alpha = 1.0 / gamma
    return gamma, alpha


CHARS_ENTRIES = [
    {"task": "toxic", "K": 3,   "path": "Vector/CHARS/Gemma/toxic_K3/metadata.pt"},
    {"task": "toxic", "K": 5,   "path": "Vector/CHARS/Gemma/toxic_K5/metadata.pt"},
    {"task": "toxic", "K": 10,  "path": "Vector/CHARS/Gemma/toxic/metadata.pt"},
    {"task": "toxic", "K": 20,  "path": "Vector/CHARS/Gemma/toxic_K20/metadata.pt"},
    {"task": "evil",  "K": 3,   "path": "Vector/CHARS/Gemma/evil_K3/metadata.pt"},
    {"task": "evil",  "K": 5,   "path": "Vector/CHARS/Gemma/evil_K5/metadata.pt"},
    {"task": "evil",  "K": 10,  "path": "Vector/CHARS/Gemma/evil_K10/metadata.pt"},
    {"task": "deception", "K": 5,   "path": "Vector/CHARS/Gemma/deception_K5/metadata.pt"},
    {"task": "deception", "K": 10,  "path": "Vector/CHARS/Gemma/deception_k10/metadata.pt"},
    {"task": "deception", "K": 20,  "path": "Vector/CHARS/Gemma/deception_K20/metadata.pt"},
    {"task": "deception", "K": 50,  "path": "Vector/CHARS/Gemma/deception_K50/metadata.pt"},
    {"task": "refusal",  "K": 5,   "path": "Vector/CHARS/Gemma/refusal_K5/metadata.pt"},
    {"task": "refusal",  "K": 10,  "path": "Vector/CHARS/Gemma/refusal_K10/metadata.pt"},
    {"task": "refusal",  "K": 20,  "path": "Vector/CHARS/Gemma/refusal_K20/metadata.pt"},
    {"task": "refusal",  "K": 50,  "path": "Vector/CHARS/Gemma/refusal_K50/metadata.pt"},
]

ACTIVATION_FILES = {
    "toxic": "toxic_activations.pt",
    "evil": "evil_activations.pt",
    "deception": "deception_activations.pt",
    "refusal": "refusal_activations.pt",
}


def compute_sample_level_tail(cA, cB, P, target_acts, contrast_acts, K_act):
    """
    Combined-rule tail_sample block (faithful to OT + CHaRS structure).

    Branches (faithful-condition design):
      1. Per-distribution existence: Hill α of per-sample
          N-dim norm distribution. Threshold HILL_ALPHA_NORM_HAS_TAIL. If False -> empty
         I_tail on that side; downstream metrics deferred (enrichment=0, no I_tail).
      2. Per-sample tail: per-sample N-dim norm >= NORM_PCTL_THRESHOLD (95th pct),
         ranked within source / target separately. Last-token pooling, matching
         inference (per §9.15).
      3. Per-cluster count: nearest-centroid (L2) assignment of tail samples to
         their own side's centroids. contrast_acts -> cA (source), target_acts -> cB.
      4. Selection rule: cluster i qualifies iff BOTH
            (a) count[i] >= max(3, ceil(|tail_samples| * 0.10))   -- absolute noise floor
            (b) ranked in top max(1, ceil(K * 0.10))             -- K-adaptive fraction
         Tie-break by smaller centroid index (stable, deterministic).
      5. OT-flow metrics: same formulas as tail_hill block; new I_tail sets.
    """
    n_target, d_target = target_acts.shape
    n_contrast, d_contrast = contrast_acts.shape

    norm_target = np.linalg.norm(target_acts, axis=1)
    norm_contrast = np.linalg.norm(contrast_acts, axis=1)

    _, alpha_target = hill_estimator(norm_target)
    _, alpha_contrast = hill_estimator(norm_contrast)
    HAS_TAIL_T = bool(not np.isnan(alpha_target) and alpha_target < HILL_ALPHA_NORM_HAS_TAIL)
    HAS_TAIL_S = bool(not np.isnan(alpha_contrast) and alpha_contrast < HILL_ALPHA_NORM_HAS_TAIL)

    if HAS_TAIL_T:
        thr_t = float(np.percentile(norm_target, NORM_PCTL_THRESHOLD))
        t_tail_idx = np.where(norm_target >= thr_t)[0]
    else:
        thr_t = float("inf")
        t_tail_idx = np.array([], dtype=np.int64)
    if HAS_TAIL_S:
        thr_s = float(np.percentile(norm_contrast, NORM_PCTL_THRESHOLD))
        s_tail_idx = np.where(norm_contrast >= thr_s)[0]
    else:
        thr_s = float("inf")
        s_tail_idx = np.array([], dtype=np.int64)

    n_tail_s = int(len(s_tail_idx))
    n_tail_t = int(len(t_tail_idx))

    MIN_TAIL_ABS = max(MIN_TAIL_ABS_MINIMUM, int(np.ceil(n_tail_s * MIN_TAIL_SAMPLES_FLOOR)))
    m_frac = max(MIN_TAIL_CLUSTERS, int(np.ceil(K_act * TAIL_CLUSTER_FRACTION)))

    count_s = np.zeros(K_act, dtype=np.int64)
    count_t = np.zeros(K_act, dtype=np.int64)
    if HAS_TAIL_S and n_tail_s > 0 and d_contrast == cA.shape[1]:
        D_src = np.linalg.norm(contrast_acts[s_tail_idx, None, :] - cA[None, :, :], axis=2)
        src_assign = np.argmin(D_src, axis=1)
        count_s = np.bincount(src_assign, minlength=K_act).astype(np.int64)
    if HAS_TAIL_T and n_tail_t > 0 and d_target == cB.shape[1]:
        D_tgt = np.linalg.norm(target_acts[t_tail_idx, None, :] - cB[None, :, :], axis=2)
        tgt_assign = np.argmin(D_tgt, axis=1)
        count_t = np.bincount(tgt_assign, minlength=K_act).astype(np.int64)

    elig_s = sorted([int(i) for i in range(K_act) if count_s[i] >= MIN_TAIL_ABS],
                    key=lambda i: (-count_s[i], i))
    elig_t = sorted([int(j) for j in range(K_act) if count_t[j] >= MIN_TAIL_ABS],
                    key=lambda j: (-count_t[j], j))

    I_tail_s = set(elig_s[:m_frac])
    I_tail_t = set(elig_t[:m_frac])

    n_sel_s = len(I_tail_s)
    n_sel_t = len(I_tail_t)

    mass_to_tail = 0.0
    mass_self = 0.0
    mass_to_common = 0.0
    uniform_baseline = float(n_sel_s / max(K_act, 1))
    enrichment = 0.0
    tot_mass_to_tail = 0.0
    tot_mass_from_tail = 0.0

    if n_sel_s > 0 and n_sel_t > 0:
        s_idx_list = sorted(I_tail_s)
        t_idx_list = sorted(I_tail_t)
        P_tail = P[np.ix_(s_idx_list, t_idx_list)]
        tot_mass_from_tail = float(P[s_idx_list].sum())
        tot_mass_to_tail = float(P_tail.sum())
        mass_to_tail = tot_mass_to_tail / max(tot_mass_from_tail, 1e-12)
        norm_t_list = [j for j in range(K_act) if j not in I_tail_t]
        if norm_t_list:
            mass_to_common = float(P[s_idx_list][:, norm_t_list].sum()) / max(tot_mass_from_tail, 1e-12)
        shared = sorted(I_tail_s & I_tail_t)
        if shared:
            mass_self = float(sum(P[i, i] for i in shared)) / max(tot_mass_from_tail, 1e-12)
        enrichment = float(mass_to_tail / max(uniform_baseline, 1e-12))

    # 2x2 flow matrix (fraction, sums to 1)
    tt = tn = nt = nn = 0.0
    if K_act > 0:
        s_list_full = sorted(I_tail_s)
        t_list_full = sorted(I_tail_t)
        s_norm_list = [i for i in range(K_act) if i not in I_tail_s]
        t_norm_list = [j for j in range(K_act) if j not in I_tail_t]
        if s_list_full and t_list_full:
            tt = float(P[np.ix_(s_list_full, t_list_full)].sum())
        if s_list_full and t_norm_list:
            tn = float(P[np.ix_(s_list_full, t_norm_list)].sum())
        if s_norm_list and t_list_full:
            nt = float(P[np.ix_(s_norm_list, t_list_full)].sum())
        if s_norm_list and t_norm_list:
            nn = float(P[np.ix_(s_norm_list, t_norm_list)].sum())
        total = tt + tn + nt + nn
        if total > 0:
            tt /= total; tn /= total; nt /= total; nn /= total

    flow_matrix = {
        "tail_to_tail": tt,
        "tail_to_normal": tn,
        "normal_to_tail": nt,
        "normal_to_normal": nn,
    }

    # Concentration: per-row max-fraction (mean across tail vs non-tail clusters)
    conc = {}
    if K_act > 0:
        tail_max = []
        norm_max = []
        for i in range(K_act):
            denom = max(P[i].sum(), 1e-12)
            row_max = float(P[i].max() / denom)
            if i in I_tail_s:
                tail_max.append(row_max)
            else:
                norm_max.append(row_max)
        conc["tail_mean_max_frac"] = float(np.mean(tail_max)) if tail_max else None
        conc["normal_mean_max_frac"] = float(np.mean(norm_max)) if norm_max else None

    return {
        "has_tail_source": HAS_TAIL_S,
        "has_tail_target": HAS_TAIL_T,
        "hill_alpha_norm_source": alpha_contrast,
        "hill_alpha_norm_target": alpha_target,
        "norm_percentile_threshold_source": thr_s,
        "norm_percentile_threshold_target": thr_t,
        "abs_floor_used": MIN_TAIL_ABS,
        "frac_floor_used": m_frac,
        "n_tail_source_samples": n_tail_s,
        "n_tail_target_samples": n_tail_t,
        "per_cluster_tail_count_source": [int(c) for c in count_s.tolist()],
        "per_cluster_tail_count_target": [int(c) for c in count_t.tolist()],
        "n_tail": n_sel_s,
        "n_tail_targets": n_sel_t,
        "selected_cluster_ids_source": sorted(I_tail_s),
        "selected_cluster_ids_target": sorted(I_tail_t),
        "mass_to_tail": mass_to_tail,
        "mass_self": mass_self,
        "mass_to_common": mass_to_common,
        "uniform_baseline": uniform_baseline,
        "enrichment_over_uniform": enrichment,
        "tot_mass_to_tail": tot_mass_to_tail,
        "tot_mass_from_tail": tot_mass_from_tail,
        "flow_matrix": flow_matrix,
        "concentration": conc,
    }


def get_tail_dim_mask(task_name):
    """Return boolean mask of heavy-tail dims (Hill α < TAIL_ALPHA_THRESH) per Tail Annealing paper §4.2.
    Fixed threshold α_max=4: dim is heavy-tailed if Hill tail index α < 4.
    """
    fname = ACTIVATION_FILES[task_name]
    path = OUTPUT_DIR / fname
    if not path.exists():
        print(f"  WARN: activations not found at {path}, no tail dims")
        return None

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    t = d["target_acts"].float().numpy()
    c = d["contrast_acts"].float().numpy()
    pooled = np.concatenate([t, c], axis=0)
    n, d_model = pooled.shape
    alphas = np.array([hill_estimator(pooled[:, j])[1] for j in range(d_model)])
    valid = ~np.isnan(alphas)
    if valid.sum() < 2:
        print(f"  WARN: too few valid Hill estimates for {task_name}")
        return None
    mask = np.zeros(d_model, dtype=bool)
    mask[valid] = alphas[valid] < TAIL_ALPHA_THRESH
    n_tail = mask.sum()
    n_total = d_model
    hill_mean = np.nanmean(alphas)
    hill_min = np.nanmin(alphas)
    print(f"  Hill tail dims (α < {TAIL_ALPHA_THRESH}): {n_tail}/{n_total} "
          f"(min α={hill_min:.2f}, mean α={hill_mean:.3f})")
    return mask


def get_pca_tail_info(task_name):
    """Return fitted PCA model and n_keep components for target+contrast activations."""
    fname = ACTIVATION_FILES[task_name]
    path = OUTPUT_DIR / fname
    if not path.exists():
        print(f"  WARN: activations not found at {path}, no PCA tail info")
        return None, 0

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    t = d["target_acts"].float().numpy()
    c = d["contrast_acts"].float().numpy()
    pooled = np.concatenate([t, c], axis=0)
    
    pca = PCA(n_components=min(pooled.shape[0], pooled.shape[1]))
    pca.fit(pooled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # Define "tail" or "off-manifold" subspace as components starting past 90% cumulative variance
    n_keep = int(np.searchsorted(cumvar, 0.90) + 1)
    print(f"  PCA tail for {task_name}: keeping top {n_keep}/{pooled.shape[1]} PCs (covering {cumvar[n_keep-1]:.4f} var). Tail PCs: {pooled.shape[1] - n_keep}")
    return pca, n_keep


def get_variance_tail_dim_mask(task_name):
    """Return boolean mask of dims where variance is highest on pooled activations."""
    fname = ACTIVATION_FILES[task_name]
    path = OUTPUT_DIR / fname
    if not path.exists():
        print(f"  WARN: activations not found at {path}, no variance dims")
        return None

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    t = d["target_acts"].float().numpy()
    c = d["contrast_acts"].float().numpy()
    pooled = np.concatenate([t, c], axis=0)
    
    # Compute variance per dimension
    var = np.var(pooled, axis=0)
    
    # Use top 5% of dims by variance
    thresh = np.percentile(var, 95)
    mask = var >= thresh
    n_tail = mask.sum()
    n_total = len(var)
    print(f"  Variance tail dims (top 5% by var): {n_tail}/{n_total} "
          f"(thresh={thresh:.3f}, max var={np.max(var):.2f}, mean var={np.mean(var):.3f})")
    return mask


def analyze_one(entry, tail_mask, var_mask=None, pca_info=None, target_acts=None, contrast_acts=None):
    path = Path(entry["path"])
    if not path.exists():
        return None

    md = torch.load(str(path), map_location="cpu", weights_only=True)
    lk = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    cA = md["chars_centroids_A"][lk].float().numpy()
    cB = md["chars_centroids_B"][lk].float().numpy()
    P = md["chars_coupling"][lk].float().numpy()
    K_actual = P.shape[0]

    d = cA.shape[1]
    # Skip if centroid dimension doesn't match tail mask (e.g. SAE-space CHARS)
    if tail_mask is not None and d != len(tail_mask):
        print(f"  SKIP: centroid dim {d} != tail mask dim {len(tail_mask)}")
        return None

    result = {
        "task": entry["task"],
        "K": entry["K"],
        "K_actual": K_actual,
        "path": entry["path"],
        "d_model": d,
        "n_tail_dims_total": int(tail_mask.sum()) if tail_mask is not None else 0,
    }

    # ── Norm in tail dims only ──
    if tail_mask is not None and tail_mask.sum() > 0:
        cA_tail_norm = np.linalg.norm(cA[:, tail_mask], axis=1)
        cB_tail_norm = np.linalg.norm(cB[:, tail_mask], axis=1)
    else:
        cA_tail_norm = np.zeros(K_actual)
        cB_tail_norm = np.zeros(K_actual)

    norms_l2 = np.linalg.norm(cA, axis=1)
    mass_from = P.sum(axis=1)

    result["centroid_norm_cv_l2"] = float(norms_l2.std() / max(norms_l2.mean(), 1e-12))
    result["centroid_norm_cv_tail"] = float(cA_tail_norm.std() / max(cA_tail_norm.mean(), 1e-12)) if cA_tail_norm.mean() > 1e-12 else 0.0
    result["active_centroids"] = int((mass_from > 0).sum())

    # ── Per-centroid ──
    max_entropy = np.log(max(K_actual, 2))
    per_centroid = []
    for i in range(K_actual):
        p_row = P[i] / max(mass_from[i], 1e-12)
        ent = float(-np.sum(p_row * np.log(p_row + 1e-12)))
        ent_norm = ent / max_entropy
        self_mass = float(P[i, i] / max(mass_from[i], 1e-12))
        is_tail_tail = cA_tail_norm[i] >= np.percentile(cA_tail_norm, TAIL_QUARTILE * 100) if tail_mask.sum() > 0 else False
        is_tail_l2 = norms_l2[i] >= np.percentile(norms_l2, TAIL_QUARTILE * 100)
        per_centroid.append({
            "idx": int(i),
            "norm_l2": float(norms_l2[i]),
            "norm_tail": float(cA_tail_norm[i]),
            "mass_from": float(mass_from[i]),
            "dest_entropy_norm": ent_norm,
            "self_mass": self_mass,
            "is_tail_tail": bool(is_tail_tail),
            "is_tail_l2": bool(is_tail_l2),
            "source_idx": i,
        })
    result["per_centroid"] = per_centroid

    # ── HILL tail definition ──
    c_tgt_norm = cB_tail_norm
    c_src_norm = cA_tail_norm
    r_src = c_src_norm
    r_tgt = c_tgt_norm

    tail_thresh_src = float(np.percentile(r_src, TAIL_QUARTILE * 100)) if r_src.max() > 0 else float("inf")
    tail_thresh_tgt = float(np.percentile(r_tgt, TAIL_QUARTILE * 100)) if r_tgt.max() > 0 else float("inf")

    tail_mask_src = r_src >= tail_thresh_src
    tail_mask_tgt = r_tgt >= tail_thresh_tgt
    tail_idx = np.where(tail_mask_src)[0]
    n_tail = len(tail_idx)
    n_tgt_tail = int(tail_mask_tgt.sum())

    if n_tail > 0 and r_src.max() > 0:
        P_tail = P[tail_idx]
        mass_to_tail = float(P_tail[:, tail_mask_tgt].sum() / max(P_tail.sum(), 1e-12))
        mass_self = float(np.diag(P_tail).sum() / max(P_tail.sum(), 1e-12))
        uniform_baseline = n_tail / max(K_actual, 1)
        enrichment = mass_to_tail / max(uniform_baseline, 1e-12)
        mass_to_common = float(P_tail[:, ~tail_mask_tgt].sum() / max(P_tail.sum(), 1e-12))
        partner_count = int(((P_tail[:, tail_mask_src] > 0).sum(axis=1) > 1).sum())
        tail_self = np.array([per_centroid[j]["self_mass"] for j in tail_idx])
        tail_norms = r_src[tail_idx]
        rho_norm_self = float(spearmanr(tail_norms, tail_self).statistic) if n_tail > 2 else 0.0
    else:
        mass_to_tail = mass_self = uniform_baseline = enrichment = mass_to_common = 0.0
        partner_count = 0
        rho_norm_self = 0.0

    result["tail_hill"] = {
        "n_tail": int(n_tail),
        "n_tail_targets": n_tgt_tail,
        "mass_to_tail": mass_to_tail,
        "mass_self": mass_self,
        "mass_to_common": mass_to_common,
        "uniform_baseline": float(uniform_baseline),
        "enrichment_over_uniform": float(enrichment),
        "spearman_norm_vs_self_mass": rho_norm_self,
        "n_tail_with_inter_tail_partner": partner_count,
        "tot_mass_to_tail": float(P[tail_idx][:, tail_mask_tgt].sum()) if n_tail > 0 and r_src.max() > 0 else 0.0,
        "tot_mass_from_tail": float(P[tail_idx].sum()) if n_tail > 0 and r_src.max() > 0 else 0.0,
    }

    # ── Full flow matrix: tail/normal × tail/normal ──
    # Partition P* into 4 quadrants
    norm_idx = np.where(~tail_mask_src)[0]
    norm_tgt_mask = ~tail_mask_tgt
    tt = float(P[tail_idx][:, tail_mask_tgt].sum()) if n_tail > 0 and tail_mask_tgt.sum() > 0 else 0.0
    tn = float(P[tail_idx][:, norm_tgt_mask].sum()) if n_tail > 0 and norm_tgt_mask.sum() > 0 else 0.0
    nt = float(P[norm_idx][:, tail_mask_tgt].sum()) if len(norm_idx) > 0 and tail_mask_tgt.sum() > 0 else 0.0
    nn = float(P[norm_idx][:, norm_tgt_mask].sum()) if len(norm_idx) > 0 and norm_tgt_mask.sum() > 0 else 0.0
    total = tt + tn + nt + nn
    flow_matrix = {
        "tail_to_tail": float(tt / max(total, 1e-12)),
        "tail_to_normal": float(tn / max(total, 1e-12)),
        "normal_to_tail": float(nt / max(total, 1e-12)),
        "normal_to_normal": float(nn / max(total, 1e-12)),
    }
    result["tail_hill"]["flow_matrix"] = flow_matrix

    # ── Target concentration: do tail sources dump into fewer targets? ──
    if n_tail > 0:
        # For each tail source, fraction of mass going to its top-1 target
        tail_max_fracs = [(P[i] / max(P[i].sum(), 1e-12)).max() for i in tail_idx]
        # For each normal source
        norm_max_fracs = [(P[i] / max(P[i].sum(), 1e-12)).max() for i in norm_idx] if len(norm_idx) > 0 else []
        # Entropy of destination distribution
        tail_entropies = [per_centroid[i]["dest_entropy_norm"] for i in tail_idx]
        norm_entropies = [per_centroid[i]["dest_entropy_norm"] for i in norm_idx] if len(norm_idx) > 0 else []
        result["tail_hill"]["concentration"] = {
            "tail_mean_max_frac": float(np.mean(tail_max_fracs)),
            "normal_mean_max_frac": float(np.mean(norm_max_fracs)) if norm_max_fracs else None,
            "tail_mean_entropy_norm": float(np.mean(tail_entropies)),
            "normal_mean_entropy_norm": float(np.mean(norm_entropies)) if norm_entropies else None,
        }
    else:
        result["tail_hill"]["concentration"] = {}

    # ── L2 tail definition (for comparison) ──
    l2_thresh_src = float(np.percentile(norms_l2, TAIL_QUARTILE * 100))
    l2_thresh_tgt = float(np.percentile(np.linalg.norm(cB, axis=1), TAIL_QUARTILE * 100))
    l2_mask = norms_l2 >= l2_thresh_src
    l2_idx = np.where(l2_mask)[0]
    n_l2 = len(l2_idx)
    tgt_l2_mask = np.linalg.norm(cB, axis=1) >= l2_thresh_tgt
    if n_l2 > 0:
        P_l2 = P[l2_idx]
        l2_mass_to_tail = float(P_l2[:, tgt_l2_mask].sum() / max(P_l2.sum(), 1e-12))
        l2_enrich = l2_mass_to_tail / max(n_l2 / max(K_actual, 1), 1e-12)
    else:
        l2_mass_to_tail = 0.0
        l2_enrich = 0.0

    result["tail_L2_comparison"] = {
        "n_tail": int(n_l2),
        "mass_to_tail": l2_mass_to_tail,
        "enrichment_over_uniform": float(l2_enrich),
    }

    # L2 flow matrix
    norm_l2_idx = np.where(~l2_mask)[0]
    tgt_norm_l2_mask = ~tgt_l2_mask
    tt_l2 = float(P[l2_idx][:, tgt_l2_mask].sum()) if n_l2 > 0 and tgt_l2_mask.sum() > 0 else 0.0
    tn_l2 = float(P[l2_idx][:, tgt_norm_l2_mask].sum()) if n_l2 > 0 and tgt_norm_l2_mask.sum() > 0 else 0.0
    nt_l2 = float(P[norm_l2_idx][:, tgt_l2_mask].sum()) if len(norm_l2_idx) > 0 and tgt_l2_mask.sum() > 0 else 0.0
    nn_l2 = float(P[norm_l2_idx][:, tgt_norm_l2_mask].sum()) if len(norm_l2_idx) > 0 and tgt_norm_l2_mask.sum() > 0 else 0.0
    total_l2 = tt_l2 + tn_l2 + nt_l2 + nn_l2
    result["tail_L2_comparison"]["flow_matrix"] = {
        "tail_to_tail": float(tt_l2 / max(total_l2, 1e-12)),
        "tail_to_normal": float(tn_l2 / max(total_l2, 1e-12)),
        "normal_to_tail": float(nt_l2 / max(total_l2, 1e-12)),
        "normal_to_normal": float(nn_l2 / max(total_l2, 1e-12)),
    }
    # L2 concentration
    if n_l2 > 0:
        l2_max_fracs = [(P[i] / max(P[i].sum(), 1e-12)).max() for i in l2_idx]
        norm_l2_max_fracs = [(P[i] / max(P[i].sum(), 1e-12)).max() for i in norm_l2_idx] if len(norm_l2_idx) > 0 else []
        l2_ents = [per_centroid[i]["dest_entropy_norm"] for i in l2_idx]
        norm_l2_ents = [per_centroid[i]["dest_entropy_norm"] for i in norm_l2_idx] if len(norm_l2_idx) > 0 else []
        result["tail_L2_comparison"]["concentration"] = {
            "tail_mean_max_frac": float(np.mean(l2_max_fracs)),
            "normal_mean_max_frac": float(np.mean(norm_l2_max_fracs)) if norm_l2_max_fracs else None,
            "tail_mean_entropy_norm": float(np.mean(l2_ents)),
            "normal_mean_entropy_norm": float(np.mean(norm_l2_ents)) if norm_l2_ents else None,
        }

    # ── VARIANCE-based tail definition ──
    if var_mask is not None and var_mask.sum() > 0:
        cA_var_norm = np.linalg.norm(cA[:, var_mask], axis=1)
        cB_var_norm = np.linalg.norm(cB[:, var_mask], axis=1)
        
        var_thresh_src = float(np.percentile(cA_var_norm, TAIL_QUARTILE * 100))
        var_thresh_tgt = float(np.percentile(cB_var_norm, TAIL_QUARTILE * 100))
        
        var_mask_src = cA_var_norm >= var_thresh_src
        var_mask_tgt = cB_var_norm >= var_thresh_tgt
        var_tail_idx = np.where(var_mask_src)[0]
        n_var_tail = len(var_tail_idx)
        n_var_tgt_tail = int(var_mask_tgt.sum())
        
        if n_var_tail > 0:
            P_var_tail = P[var_tail_idx]
            var_mass_to_tail = float(P_var_tail[:, var_mask_tgt].sum() / max(P_var_tail.sum(), 1e-12))
            var_enrich = var_mass_to_tail / max(n_var_tail / max(K_actual, 1), 1e-12)
        else:
            var_mass_to_tail = var_enrich = 0.0
            
        result["tail_variance_comparison"] = {
            "n_tail": int(n_var_tail),
            "mass_to_tail": var_mass_to_tail,
            "enrichment_over_uniform": float(var_enrich),
        }
        
        # Variance tail flow matrix
        norm_var_idx = np.where(~var_mask_src)[0]
        tgt_norm_var_mask = ~var_mask_tgt
        tt_var = float(P[var_tail_idx][:, var_mask_tgt].sum()) if n_var_tail > 0 and var_mask_tgt.sum() > 0 else 0.0
        tn_var = float(P[var_tail_idx][:, tgt_norm_var_mask].sum()) if n_var_tail > 0 and tgt_norm_var_mask.sum() > 0 else 0.0
        nt_var = float(P[norm_var_idx][:, var_mask_tgt].sum()) if len(norm_var_idx) > 0 and var_mask_tgt.sum() > 0 else 0.0
        nn_var = float(P[norm_var_idx][:, tgt_norm_var_mask].sum()) if len(norm_var_idx) > 0 and tgt_norm_var_mask.sum() > 0 else 0.0
        total_var = tt_var + tn_var + nt_var + nn_var
        result["tail_variance_comparison"]["flow_matrix"] = {
            "tail_to_tail": float(tt_var / max(total_var, 1e-12)),
            "tail_to_normal": float(tn_var / max(total_var, 1e-12)),
            "normal_to_tail": float(nt_var / max(total_var, 1e-12)),
            "normal_to_normal": float(nn_var / max(total_var, 1e-12)),
        }

    # ── PCA-based (Spectral Variance) tail definition ──
    if pca_info is not None:
        pca, n_keep = pca_info
        cA_pc = pca.transform(cA)
        cB_pc = pca.transform(cB)
        cA_pca_tail_norm = np.linalg.norm(cA_pc[:, n_keep:], axis=1)
        cB_pca_tail_norm = np.linalg.norm(cB_pc[:, n_keep:], axis=1)
        
        pca_thresh_src = float(np.percentile(cA_pca_tail_norm, TAIL_QUARTILE * 100))
        pca_thresh_tgt = float(np.percentile(cB_pca_tail_norm, TAIL_QUARTILE * 100))
        
        pca_mask_src = cA_pca_tail_norm >= pca_thresh_src
        pca_mask_tgt = cB_pca_tail_norm >= pca_thresh_tgt
        pca_tail_idx = np.where(pca_mask_src)[0]
        n_pca_tail = len(pca_tail_idx)
        n_pca_tgt_tail = int(pca_mask_tgt.sum())
        
        if n_pca_tail > 0:
            P_pca_tail = P[pca_tail_idx]
            pca_mass_to_tail = float(P_pca_tail[:, pca_mask_tgt].sum() / max(P_pca_tail.sum(), 1e-12))
            pca_enrich = pca_mass_to_tail / max(n_pca_tail / max(K_actual, 1), 1e-12)
        else:
            pca_mass_to_tail = pca_enrich = 0.0
            
        result["tail_PCA_variance_comparison"] = {
            "n_tail": int(n_pca_tail),
            "mass_to_tail": pca_mass_to_tail,
            "enrichment_over_uniform": float(pca_enrich),
        }
        
        # PCA tail flow matrix
        norm_pca_idx = np.where(~pca_mask_src)[0]
        tgt_norm_pca_mask = ~pca_mask_tgt
        tt_pca = float(P[pca_tail_idx][:, pca_mask_tgt].sum()) if n_pca_tail > 0 and pca_mask_tgt.sum() > 0 else 0.0
        tn_pca = float(P[pca_tail_idx][:, tgt_norm_pca_mask].sum()) if n_pca_tail > 0 and tgt_norm_pca_mask.sum() > 0 else 0.0
        nt_pca = float(P[norm_pca_idx][:, pca_mask_tgt].sum()) if len(norm_pca_idx) > 0 and pca_mask_tgt.sum() > 0 else 0.0
        nn_pca = float(P[norm_pca_idx][:, tgt_norm_pca_mask].sum()) if len(norm_pca_idx) > 0 and tgt_norm_pca_mask.sum() > 0 else 0.0
        total_pca = tt_pca + tn_pca + nt_pca + nn_pca
        result["tail_PCA_variance_comparison"]["flow_matrix"] = {
            "tail_to_tail": float(tt_pca / max(total_pca, 1e-12)),
            "tail_to_normal": float(tn_pca / max(total_pca, 1e-12)),
            "normal_to_tail": float(nt_pca / max(total_pca, 1e-12)),
            "normal_to_normal": float(nn_pca / max(total_pca, 1e-12)),
        }

    # ── Overlap between Hill-tail and L2-tail ──
    tail_set = set(tail_idx)
    l2_set = set(l2_idx)
    overlap = tail_set & l2_set
    result["tail_overlap_hill_vs_l2"] = {
        "n_hill_tail": int(len(tail_set)),
        "n_l2_tail": int(len(l2_set)),
        "intersection": int(len(overlap)),
        "iou": float(len(overlap) / max(len(tail_set | l2_set), 1)),
    }

    # ── Spearman ──
    result["spearman"] = {
        "norm_tail_vs_mass_from": float(spearmanr(cA_tail_norm, mass_from).statistic) if cA_tail_norm.std() > 0 else 0.0,
        "norm_l2_vs_mass_from": float(spearmanr(norms_l2, mass_from).statistic),
        "norm_tail_vs_self_mass": float(spearmanr(cA_tail_norm, [pc["self_mass"] for pc in per_centroid]).statistic) if cA_tail_norm.std() > 0 else 0.0,
    }

    # ── Dead centroids ──
    result["dead_centroids"] = {
        "exactly_zero": int((mass_from == 0).sum()),
        "near_zero": int((mass_from < mass_from.mean() * 0.01).sum()),
        "fraction_zero": float((mass_from == 0).sum() / max(K_actual, 1)),
    }

    # ── RBF isolation ──
    sorted_idx = np.argsort(norms_l2)
    cA_sorted = cA[sorted_idx]
    cB_sorted = cB[sorted_idx]
    dists = np.linalg.norm(cA_sorted[:, None, :] - cB_sorted[None, :, :], axis=2)
    tau = float(np.median(dists))
    rbf = np.exp(-dists ** 2 / (2 * max(tau, 1e-12) ** 2))
    result["rbf"] = {"tau": tau}
    if tail_mask_src.sum() > 0:
        tail_pos = [int(np.where(sorted_idx == j)[0][0]) for j in tail_idx if np.any(sorted_idx == j)]
        if tail_pos:
            rbf_tail = rbf[tail_pos]
            result["rbf"]["tail_min"] = float(rbf_tail.min())
            result["rbf"]["tail_max"] = float(rbf_tail.max())
            result["rbf"]["tail_mean"] = float(rbf_tail.mean())
            common_mask = np.ones(K_actual, dtype=bool)
            common_mask[tail_pos] = False
            if common_mask.sum() > 0:
                result["rbf"]["common_min"] = float(rbf[common_mask].min())
                result["rbf"]["common_mean"] = float(rbf[common_mask].mean())

    print(f"         n_tail(hill)={n_tail:>3d}/{n_tgt_tail:>3d} targets, "
          f"mass→tail={mass_to_tail:.3f}, enrich={enrichment:.2f}x, "
          f"overlap_with_L2={len(overlap)}/{n_l2}")

    # --- Faithful-condition tail_sample block (combined rule, unfalsified) ---
    if target_acts is not None and contrast_acts is not None:
        try:
            ts_block = compute_sample_level_tail(
                cA=cA, cB=cB, P=P,
                target_acts=target_acts, contrast_acts=contrast_acts,
                K_act=K_actual,
            )
            result["tail_sample"] = ts_block
            print(f"         tail_sample: has_src={ts_block['has_tail_source']} "
                  f"has_tgt={ts_block['has_tail_target']} "
                  f"α_src={ts_block['hill_alpha_norm_source']:.2f} "
                  f"α_tgt={ts_block['hill_alpha_norm_target']:.2f} "
                  f"n_sel={ts_block['n_tail']}/{ts_block['n_tail_targets']} "
                  f"enrich={ts_block['enrichment_over_uniform']:.2f}x "
                  f"counts_src={ts_block['per_cluster_tail_count_source']}")
        except Exception as e:
            print(f"  WARN: tail_sample failed: {e}")

    return result


def print_cluster_table(entry, task_acts_tuple, tail_mask):
    """
    Per-cluster table: cluster_id, n, act_frac, tail_frac.
    Uses Hill-based tail dims + nearest-centroid assignment of pooled activations.
    Same format as old print_cluster_tables.py, works for any K variant.
    """
    if task_acts_tuple[0] is None:
        return
    target_acts, contrast_acts = task_acts_tuple
    pooled = np.concatenate([target_acts, contrast_acts], axis=0)

    path = Path(entry["path"])
    if not path.exists():
        return
    md = torch.load(str(path), map_location="cpu", weights_only=True)
    lk = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    cA = md["chars_centroids_A"][lk].float().numpy()

    # Check dim match with activations
    if cA.shape[1] != pooled.shape[1]:
        print(f"    SKIP cluster table: centroid dim {cA.shape[1]} != act dim {pooled.shape[1]}")
        return

    # Tail dim mask from Hill α
    if tail_mask is None or tail_mask.sum() < 2:
        print(f"    SKIP cluster table: no tail dims")
        return

    # Tail samples
    tail_vals = np.abs(pooled[:, tail_mask])
    thresh = np.percentile(tail_vals, PCTL, axis=0)
    is_tail = (tail_vals >= thresh[None, :]).any(axis=1)

    # Nearest-centroid assignment
    d2 = ((pooled[:, None, :] - cA[None, :, :]) ** 2).sum(axis=-1)
    assign = d2.argmin(axis=1)
    K = cA.shape[0]
    n_total = len(pooled)

    rows = []
    for cid in range(K):
        mask_c = assign == cid
        n_c = mask_c.sum()
        act_frac = n_c / n_total if n_total > 0 else 0.0
        tail_frac = float(is_tail[mask_c].mean()) if n_c > 0 else 0.0
        rows.append((cid, n_c, act_frac, tail_frac))

    rows.sort(key=lambda r: -r[3])

    print(f"\n    Cluster table K={K} (Hill tail dims={tail_mask.sum()}):")
    print(f"    {'C':>4} {'n':>5} {'act_frac':>9} {'tail_frac':>10}  note")
    print(f"    {'-'*4} {'-'*5} {'-'*9} {'-'*10}  {'-'*10}")
    for cid, n_c, af, tf in rows:
        note = ""
        if tf >= 0.5:
            note = "<-- tail-heavy"
        if n_c == 0:
            note = "EMPTY"
        print(f"    C{cid:>2} {n_c:>5} {af:>9.4f} {tf:>10.4f}  {note}")
    print(f"    {'--':>4} {'--':>5} {'--':>9} {'--':>10}")
    print(f"    {'sum':>4} {n_total:>5} {'1.0000':>9} {float(is_tail.mean()):>10.4f}")


def main():
    all_results = {}
    # Pre-compute tail masks per task
    tail_masks = {}
    variance_masks = {}
    # Pre-load activations per task for sample-level tail analysis (last-token pooled)
    task_activations = {}
    pca_infos = {}
    for task in ["toxic", "evil", "deception", "refusal"]:
        print(f"\n=== Computing tail dims for {task} ===")
        mask = get_tail_dim_mask(task)
        tail_masks[task] = mask
        var_mask = get_variance_tail_dim_mask(task)
        variance_masks[task] = var_mask
        pca_model, n_keep = get_pca_tail_info(task)
        pca_infos[task] = (pca_model, n_keep) if pca_model is not None else None
        
        act_path = OUTPUT_DIR / ACTIVATION_FILES[task]
        if act_path.exists():
            d = torch.load(str(act_path), map_location="cpu", weights_only=False)
            task_activations[task] = (
                d["target_acts"].float().numpy(),
                d["contrast_acts"].float().numpy(),
            )
        else:
            task_activations[task] = (None, None)

    for entry in CHARS_ENTRIES:
        if not Path(entry["path"]).exists():
            print(f"SKIP: {entry['path']} (not found)")
            continue
        print(f"\nAnalyzing {entry['task']} K={entry['K']} ...")
        try:
            target_acts, contrast_acts = task_activations.get(entry["task"], (None, None))
            r = analyze_one(
                entry, tail_masks[entry["task"]],
                var_mask=variance_masks[entry["task"]],
                pca_info=pca_infos[entry["task"]],
                target_acts=target_acts,
                contrast_acts=contrast_acts,
            )
            if r is not None:
                key = f"{r['task']}_K{r['K']}"
                all_results[key] = r
            # Print per-cluster table
            print_cluster_table(entry, (target_acts, contrast_acts), tail_masks[entry["task"]])
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save full results
    out_path = OUTPUT_DIR / "expI_hill_results_full.pt"
    torch.save(all_results, out_path)
    print(f"\nFull results saved to {out_path}")

    # Save summary (without per-centroid)
    summary = {}
    for key, r in all_results.items():
        s = {k: v for k, v in r.items() if k != "per_centroid"}
        summary[key] = s
    json_path = OUTPUT_DIR / "expI_hill_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")

    # Summary table
    print("\n\n" + "=" * 120)
    hdr = (f"{'Key':>20s}  {'K':>4s}  {'nTail':>5s}  {'nTgt':>5s}  "
           f"{'T→T':>8s}  {'Self':>8s}  {'Enr':>7s}  "
           f"{'nL2':>4s}  {'L2_Enr':>7s}  {'OvIp':>5s}  "
           f"{'Dead':>5s}  {'ρ(t,m)':>7s}  {'ρ(L2,m)':>8s}")
    print(hdr)
    print("-" * 120)
    for key in sorted(all_results.keys()):
        r = summary[key]
        tk = r.get("tail_hill", {})
        tl = r.get("tail_L2_comparison", {})
        ov = r.get("tail_overlap_hill_vs_l2", {})
        d = r.get("dead_centroids", {})
        sp = r.get("spearman", {})
        print(f"  {key:>20s}  {r['K']:>4d}  {tk.get('n_tail', 0):>5d}  {tk.get('n_tail_targets', 0):>5d}  "
              f"{tk.get('mass_to_tail', 0):>8.4f}  {tk.get('mass_self', 0):>8.4f}  "
              f"{tk.get('enrichment_over_uniform', 0):>7.2f}x  "
              f"{tl.get('n_tail', 0):>4d}  {tl.get('enrichment_over_uniform', 0):>7.2f}x  "
              f"{ov.get('iou', 0):>5.3f}  "
              f"{d.get('exactly_zero', 0):>5d}  {sp.get('norm_tail_vs_mass_from', 0):>7.4f}  "
              f"{sp.get('norm_l2_vs_mass_from', 0):>8.4f}")

    # Cross-K analysis per task
    print("\n\n" + "=" * 80)
    print("Cross-K (Hill tail):")
    for task in ["toxic", "evil", "deception", "refusal"]:
        task_keys = [k for k in sorted(all_results.keys()) if k.startswith(task)]
        if len(task_keys) < 2:
            continue
        print(f"\n  {task}:")
        hdr = (f"    {'K':>4s}  {'nTail':>5s}  {'T→T':>8s}  {'Enr':>7s}  "
               f"{'Self':>7s}  {'nL2':>4s}  {'OvIp':>5s}  {'Dead':>5s}")
        print(f"    {'-'*len(hdr)}")
        for k in task_keys:
            r = summary[k]
            tk = r.get("tail_hill", {})
            tl = r.get("tail_L2_comparison", {})
            ov = r.get("tail_overlap_hill_vs_l2", {})
            d = r.get("dead_centroids", {})
            print(f"    {r['K']:>4d}  {tk.get('n_tail', 0):>5d}  {tk.get('mass_to_tail', 0):>8.4f}  "
                  f"{tk.get('enrichment_over_uniform', 0):>7.2f}x  {tk.get('mass_self', 0):>7.4f}  "
                  f"{tl.get('n_tail', 0):>4d}  {ov.get('iou', 0):>5.3f}  {d.get('exactly_zero', 0):>5d}")

    # === Tail-sample (faithful-condition, combined rule) summary ===
    print("\n\n" + "=" * 110)
    print("Tail-sample (faithful-condition, combined rule):")
    hdr_ts = (f"  {'Key':>20s}  {'K':>4s}  {'hasS':>5s}  {'hasT':>5s}  "
              f"{'α_src':>6s}  {'α_tgt':>6s}  {'nTS':>4s}  {'nTT':>4s}  "
              f"{'nSelS':>5s}  {'nSelT':>5s}  {'Enr':>8s}  {'T->T':>7s}  {'T->N':>7s}")
    print(hdr_ts)
    print("  " + "-" * len(hdr_ts))
    for key in sorted(all_results.keys()):
        r = summary[key]
        ts = r.get("tail_sample", {})
        fm = ts.get("flow_matrix", {})
        print(f"  {key:>20s}  {r['K']:>4d}  "
              f"{str(ts.get('has_tail_source', '?')):>5s}  "
              f"{str(ts.get('has_tail_target', '?')):>5s}  "
              f"{ts.get('hill_alpha_norm_source', 0):>6.2f}  "
              f"{ts.get('hill_alpha_norm_target', 0):>6.2f}  "
              f"{ts.get('n_tail_source_samples', 0):>4d}  "
              f"{ts.get('n_tail_target_samples', 0):>4d}  "
              f"{ts.get('n_tail', 0):>5d}  "
              f"{ts.get('n_tail_targets', 0):>5d}  "
              f"{ts.get('enrichment_over_uniform', 0):>8.2f}x  "
              f"{fm.get('tail_to_tail', 0):>7.3f}  "
              f"{fm.get('tail_to_normal', 0):>7.3f}")


    # === PCA-based (Spectral Variance) tail summary ===
    print("\n\n" + "=" * 80)
    print("PCA-based (Spectral Variance) tail summary:")
    hdr_pca = f"  {'Key':>20s}  {'K':>4s}  {'nTail':>5s}  {'T→T':>8s}  {'Enr':>7s}"
    print(hdr_pca)
    print("  " + "-" * len(hdr_pca))
    for key in sorted(all_results.keys()):
        r = summary[key]
        tpca = r.get("tail_PCA_variance_comparison", {})
        print(f"  {key:>20s}  {r['K']:>4d}  "
              f"{tpca.get('n_tail', 0):>5d}  "
              f"{tpca.get('mass_to_tail', 0):>8.4f}  "
              f"{tpca.get('enrichment_over_uniform', 0):>7.2f}x")

    # === Variance-based tail summary ===
    print("\n\n" + "=" * 80)
    print("Variance-based tail summary:")
    hdr_var = f"  {'Key':>20s}  {'K':>4s}  {'nTail':>5s}  {'T→T':>8s}  {'Enr':>7s}"
    print(hdr_var)
    print("  " + "-" * len(hdr_var))
    for key in sorted(all_results.keys()):
        r = summary[key]
        tvar = r.get("tail_variance_comparison", {})
        print(f"  {key:>20s}  {r['K']:>4d}  "
              f"{tvar.get('n_tail', 0):>5d}  "
              f"{tvar.get('mass_to_tail', 0):>8.4f}  "
              f"{tvar.get('enrichment_over_uniform', 0):>7.2f}x")


if __name__ == "__main__":
    main()
