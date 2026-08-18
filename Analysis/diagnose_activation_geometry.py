"""
Activation Geometry Diagnostics — adapted from 7 papers' internal experiments.

Implements and applies each paper's most persuasive internal experiment to
explain method success/failure patterns across toxic/deception/evil/refusal.

Experiments:
  E1: Distribution Overlap (Lin-ACT Fig 10) — W2 + marginal overlap
  E2: Cluster Analysis (CHaRS) — optimal K, silhouette, cluster W2 cost
  E3: Manifold Curvature (CurveBall) — Spearman rho, reconstruction error
  E4: Trajectory Curvature (FLAS Fig 6) — path bending, per-step cos
  E5: Asymmetry (novel) — forward vs reverse transport cost
  E6: Layer Importance (LinEAS Fig 4/10) — per-layer signal strength
  E7: PCA Low-Rank Structure (Lin-ACT + CHaRS) — variance explained

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m Analysis.diagnose_activation_geometry
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer_lens import HookedTransformer

from Steering.data.loader import DataLoader, EvalDataLoader
from Steering.utils import collect_dense_activations, get_resid_acts

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [6, 10, 14, 18, 22, 25]  # cover early/mid/late
HOOK_POINT = "pre"
POSITION = "last"
BATCH_SIZE = 8
N_SAMPLES = 200  # per source/target per task
SEED = 42
SAVE_DIR = Path("Analysis/results")

np.random.seed(SEED)
torch.manual_seed(SEED)

TASKS = {
    "toxic": {
        "train_dataset": "toxic_jigsaw",
        "source_key": "false_prompt",      # non-toxic
        "target_key": "correct_prompt",    # toxic
    },
    "deception": {
        "train_dataset": "liarbench",
        "source_key": "correct_prompt",    # truthful
        "target_key": "false_prompt",      # deceptive
    },
    "evil": {
        "train_dataset": "evil",
        "source_key": "false_prompt",      # normal (non-evil)
        "target_key": "correct_prompt",    # evil
    },
    "refusal": {
        "train_dataset": "refusal_caa",
        "source_key": "false_prompt",      # non-refusal
        "target_key": "correct_prompt",    # refusal
    },
}

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
        default_padding_side="left",
    )
    return model

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_task_data(task_name: str, n_samples: int = N_SAMPLES):
    cfg = TASKS[task_name]
    dataloader = DataLoader()
    samples = dataloader.load(
        cfg["train_dataset"],
        n_samples=n_samples,
        format=True,
        apply_chat_template=True,
        tokenizer=None,
    )
    source_texts = [s[cfg["source_key"]] for s in samples]
    target_texts = [s[cfg["target_key"]] for s in samples]
    return source_texts, target_texts

# ─── Activation Collection ────────────────────────────────────────────────────

def collect_activations(model, texts: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
    return collect_dense_activations(
        model=model,
        texts=texts,
        layers=layers,
        hook_point=HOOK_POINT,
        batch_size=BATCH_SIZE,
        pooling=POSITION,
        device=model.cfg.device,
        tokenizer=model.tokenizer,
        reduce="none",
        return_key_format="layer",
    )

# ─── Experiment 1: Distribution Overlap (Lin-ACT Fig 10) ──────────────────────

def exp1_distribution_overlap(source_acts: Dict[int, np.ndarray],
                              target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        # W2 distance (approx via 1D sliced Wasserstein over random directions)
        n_dirs = 100
        np.random.seed(SEED + layer)
        dirs = np.random.randn(s.shape[1], n_dirs)
        dirs = dirs / np.linalg.norm(dirs, axis=0, keepdims=True)
        s_proj = s @ dirs
        t_proj = t @ dirs
        # Sort projections and compute 1D W2 = mean(abs(sort diff))
        s_sorted = np.sort(s_proj, axis=0)
        t_sorted = np.sort(t_proj, axis=0)
        w2_1d = np.mean((s_sorted - t_sorted) ** 2)
        w2 = np.sqrt(w2_1d)  # approximate W2

        # Cosine similarity between means
        s_mean = s.mean(axis=0)
        t_mean = t.mean(axis=0)
        cos_sim = np.dot(s_mean, t_mean) / (np.linalg.norm(s_mean) * np.linalg.norm(t_mean) + 1e-8)

        # Norm ratio
        norm_ratio = np.linalg.norm(t_mean) / (np.linalg.norm(s_mean) + 1e-8)

        # Variance ratio
        var_ratio = t.var() / (s.var() + 1e-8)

        results[layer] = {
            "w2": float(w2),
            "cosine_similarity": float(cos_sim),
            "norm_ratio": float(norm_ratio),
            "variance_ratio": float(var_ratio),
            "source_norm_mean": float(np.linalg.norm(s_mean)),
            "target_norm_mean": float(np.linalg.norm(t_mean)),
        }
    return results

# ─── Experiment 2: Cluster Analysis (CHaRS) ──────────────────────────────────

def exp2_cluster_analysis(source_acts: Dict[int, np.ndarray],
                          target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        pooled = np.vstack([s, t])

        # Silhouette for K=2..15
        k_range = range(2, 16)
        sil_scores = []
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=5)
            labels = km.fit_predict(pooled)
            if len(set(labels)) > 1:
                sil = silhouette_score(pooled, labels)
            else:
                sil = 0.0
            sil_scores.append(float(sil))
            inertias.append(float(km.inertia_))

        optimal_k = int(k_range[np.argmax(sil_scores)])

        # W2 distance per cluster
        km_opt = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=5)
        labels = km_opt.fit_predict(pooled)
        cluster_w2 = {}
        for c in range(optimal_k):
            mask = labels == c
            cluster_s = s[mask[:len(s)]]
            cluster_t = t[mask[len(s):]]
            if len(cluster_s) < 2 or len(cluster_t) < 2:
                continue
            s_mean = cluster_s.mean(axis=0)
            t_mean = cluster_t.mean(axis=0)
            w2 = np.linalg.norm(s_mean - t_mean)
            cluster_w2[int(c)] = {
                "w2": float(w2),
                "size_source": int(len(cluster_s)),
                "size_target": int(len(cluster_t)),
            }

        results[layer] = {
            "optimal_k": optimal_k,
            "silhouette_scores": {int(k): float(s) for k, s in zip(k_range, sil_scores)},
            "inertias": {int(k): float(i) for k, i in zip(k_range, inertias)},
            "cluster_w2": cluster_w2,
            "pooled_variance_explained_kmeans": float(1.0 - km_opt.inertia_ / pooled.var()),
        }
    return results

# ─── Experiment 3: Manifold Curvature & Spearman Rho (CurveBall) ────────────

def exp3_curvature_diagnostics(source_acts: Dict[int, np.ndarray],
                               target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        pooled = np.vstack([s, t])

        # PCA for manifold approximation
        pca = PCA(n_components=min(32, s.shape[1]))
        pca.fit(pooled)

        # Reconstruction error along linear interpolation
        n_steps = 20
        s_mean = s.mean(axis=0)
        t_mean = t.mean(axis=0)
        interp_points = np.linspace(0, 1, n_steps)[:, None] * (t_mean - s_mean) + s_mean
        recon_errors = []
        for pt in interp_points:
            proj = pca.transform(pt.reshape(1, -1))
            recon = pca.inverse_transform(proj)
            err = float(np.linalg.norm(pt - recon))
            recon_errors.append(err)

        # Spearman rho: correlation between projection onto source→target direction
        # and actual steering effectiveness (approximated by individual sample displacement)
        steer_dir = (t_mean - s_mean) / (np.linalg.norm(t_mean - s_mean) + 1e-8)
        s_proj = s @ steer_dir
        # "steering effectiveness" = how far each source sample would move toward target
        # measured as cosine similarity to target mean after adding steer_dir
        # We use: effect = s_proj (projection onto steering direction)
        # as a proxy for individual steering response
        t_proj = t @ steer_dir
        # Higher projection onto steer_dir means sample is easier to steer

        # Spearman between source projection coeff and target proximity
        s_proj_flat = s_proj.flatten()
        t_proj_flat = t_proj.flatten()
        # If Rho is high → high projection = closer to target → linear steering works
        # If Rho is low/negative → projection doesn't predict target proximity → nonlinear needed
        rho_s, p_s = spearmanr(s_proj_flat, t_proj_flat[:len(s_proj_flat)])

        # Bimodality of cosine sim distribution (CurveBall diagnostic)
        cos_to_mean_s = s @ s_mean / (np.linalg.norm(s, axis=1, keepdims=True) * np.linalg.norm(s_mean) + 1e-8)
        cos_to_mean_t = t @ t_mean / (np.linalg.norm(t, axis=1, keepdims=True) * np.linalg.norm(t_mean) + 1e-8)
        all_cos = np.concatenate([cos_to_mean_s.flatten(), cos_to_mean_t.flatten()])

        # Simple bimodality coefficient (SAS) — high = multimodal
        from scipy.stats import moment
        m2 = moment(all_cos, 2)
        m3 = moment(all_cos, 3)
        m4 = moment(all_cos, 4)
        if m2 > 1e-10:
            bimodality = (m4 + 3 * m2**2) / (m2**2 + 1e-10)
        else:
            bimodality = 0.0

        results[layer] = {
            "spearman_rho": float(rho_s),
            "spearman_p": float(p_s),
            "pca_recon_error_interp": recon_errors,
            "pca_variance_explained_32": float(pca.explained_variance_ratio_.sum()),
            "bimodality_coefficient": float(bimodality),
            "steer_dir_norm": float(np.linalg.norm(t_mean - s_mean)),
        }
    return results

# ─── Experiment 4: Trajectory Curvature (FLAS Fig 6) ─────────────────────────

def exp4_trajectory_curvature(source_acts: Dict[int, np.ndarray],
                              target_acts: Dict[int, np.ndarray]) -> Dict:
    from sklearn.decomposition import KernelPCA
    from sklearn.decomposition import PCA
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        s_mean = s.mean(axis=0)
        t_mean = t.mean(axis=0)

        pooled = np.vstack([s, t])
        n_components = min(16, s.shape[0])
        # RBF Kernel PCA with inverse transform enabled to compute curved geodesics
        # gamma='scale' uses 1/(n_features * X.var()) which is more robust than None (1/n_features)
        kpca = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=True, gamma='scale')
        try:
            kpca.fit(pooled)
        except ValueError:
            # Fallback if KPCA fails (e.g., kernel matrix rank deficiency):
            # use a larger gamma or fewer components
            kpca = KernelPCA(
                n_components=min(8, s.shape[0]),
                kernel="rbf",
                fit_inverse_transform=True,
                gamma=1.0 / pooled.shape[-1],
            )
            kpca.fit(pooled)

        # Ambient linear interpolation path from s_mean to t_mean
        n_steps = 20
        interp_ambient = np.linspace(0, 1, n_steps)[:, None] * (t_mean - s_mean) + s_mean

        # Project ambient path to KPCA space
        interp_kpca = kpca.transform(interp_ambient)

        # Step vectors in KPCA space
        step_vecs_kpca = np.diff(interp_kpca, axis=0)
        step_cos_kpca = []
        for i in range(len(step_vecs_kpca) - 1):
            v1 = step_vecs_kpca[i]
            v2 = step_vecs_kpca[i + 1]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            step_cos_kpca.append(float(cos))

        # Trajectory Bending in KPCA space: 1 - mean cosine
        mean_step_cos = np.mean(step_cos_kpca) if step_cos_kpca else 1.0
        trajectory_bending = float(1.0 - mean_step_cos)

        # KPCA path length and straight line distance
        path_len_kpca = float(np.sum(np.linalg.norm(step_vecs_kpca, axis=1)))
        straight_line_kpca = float(np.linalg.norm(interp_kpca[-1] - interp_kpca[0]))
        curvature_ratio = float(path_len_kpca / (straight_line_kpca + 1e-8))

        # Pullback deviation: how far does the ambient linear path deviate from the KPCA manifold?
        # Measured as the reconstruction error of the ambient path via KPCA pullback
        interp_recon_kpca = kpca.inverse_transform(interp_kpca)
        pullback_errors = np.linalg.norm(interp_ambient - interp_recon_kpca, axis=1)
        pullback_dev = float(np.max(pullback_errors))

        # PCA Tangent Space Reconstruction Error along the path (ambient to linear PCA)
        pca_temp = PCA(n_components=n_components)
        pca_temp.fit(pooled)
        interp_proj = pca_temp.transform(interp_ambient)
        interp_recon = pca_temp.inverse_transform(interp_proj)
        pca_recon_err = float(np.mean(np.linalg.norm(interp_ambient - interp_recon, axis=1)))

        # Also get direct ambient line norm for reference
        line_vec = t_mean - s_mean
        line_norm = np.linalg.norm(line_vec)

        results[layer] = {
            "curvature": trajectory_bending,
            "trajectory_bending": trajectory_bending,
            "mean_step_cosine": float(mean_step_cos),
            "step_cosines": step_cos_kpca,
            "deviations_from_linear": pullback_errors.tolist(),
            "pullback_deviation": pullback_dev,
            "path_length": path_len_kpca,
            "straight_line_distance": straight_line_kpca,
            "path_efficiency": float(straight_line_kpca / (path_len_kpca + 1e-8)),
            "manifold_curvature_ratio": curvature_ratio,
            "tangent_space_reconstruction_error": pca_recon_err,
            "ambient_line_norm": float(line_norm),
        }
    return results

# ─── Experiment 5: Forward vs Reverse Asymmetry (Novel) ─────────────────────

def exp5_asymmetry(source_acts: Dict[int, np.ndarray],
                   target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        s_mean = s.mean(axis=0)
        t_mean = t.mean(axis=0)

        # Asymmetry: variance asymmetry along steering direction
        # If source has much higher variance than target along the steer direction,
        # steering from source→target is harder because there's more "noise" to overcome
        steer_dir = (t_mean - s_mean)
        steer_norm = float(np.linalg.norm(steer_dir))
        if steer_norm > 1e-8:
            steer_dir = steer_dir / steer_norm

        s_proj_vals = s @ steer_dir
        t_proj_vals = t @ steer_dir
        s_var_along = float(np.var(s_proj_vals))
        t_var_along = float(np.var(t_proj_vals))

        # Variance asymmetry: >1 means source is more spread out (harder to steer from)
        var_asymmetry = float(s_var_along / (t_var_along + 1e-8))

        # Steer SNR: how many std devs apart are the means?
        steer_snr = float(steer_norm / (np.sqrt(np.mean([s_var_along, t_var_along])) + 1e-8))

        # Overlap coefficient: fraction of source points on the "wrong" side of the target mean
        # High overlap = poor separation = hard to steer
        s_wrong_side = float((s_proj_vals > np.mean(t_proj_vals)).mean())
        t_wrong_side = float((t_proj_vals < np.mean(s_proj_vals)).mean())
        overlap = min(s_wrong_side, t_wrong_side)  # 0 = perfect separation, 0.5 = random

        results[layer] = {
            "steer_norm": steer_norm,
            "source_var_along_steer": s_var_along,
            "target_var_along_steer": t_var_along,
            "var_asymmetry_source_over_target": var_asymmetry,
            "steer_snr": steer_snr,
            "source_fraction_on_target_side": float((s_proj_vals <= np.mean(t_proj_vals)).mean()),
            "overlap_coefficient": overlap,
        }
    return results

# ─── Experiment 6: Layer Importance (LinEAS Fig 4/10) ────────────────────────

def exp6_layer_importance(source_acts: Dict[int, np.ndarray],
                          target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    all_signal_strengths = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        s_mean = s.mean(axis=0)
        t_mean = t.mean(axis=0)
        steer_dir = t_mean - s_mean
        signal = float(np.linalg.norm(steer_dir))
        all_signal_strengths[layer] = signal

        # Cohen's d effect size
        s_proj = s @ steer_dir / (np.linalg.norm(steer_dir) + 1e-8)
        t_proj = t @ steer_dir / (np.linalg.norm(steer_dir) + 1e-8)
        pooled_std = np.sqrt((np.var(s_proj) + np.var(t_proj)) / 2)
        cohens_d = float((np.mean(t_proj) - np.mean(s_proj)) / (pooled_std + 1e-8))

        # KL divergence approx between source/target (Gaussian approximation)
        s_var = np.var(s_proj) + 1e-8
        t_var = np.var(t_proj) + 1e-8
        kl_div = float(0.5 * (np.log(t_var / s_var) + (s_var + (np.mean(s_proj) - np.mean(t_proj))**2) / t_var - 1))

        results[layer] = {
            "signal_strength": signal,
            "cohens_d": cohens_d,
            "kl_divergence": kl_div,
            "mean_source_proj": float(np.mean(s_proj)),
            "mean_target_proj": float(np.mean(t_proj)),
        }

    # Normalize signal strengths to [0,1]
    if all_signal_strengths:
        max_sig = max(all_signal_strengths.values())
        for layer in results:
            results[layer]["signal_normalized"] = results[layer]["signal_strength"] / (max_sig + 1e-8)
    return results

# ─── Experiment 7: PCA Low-Rank Structure (Lin-ACT + CHaRS) ─────────────────

def exp7_pca_low_rank(source_acts: Dict[int, np.ndarray],
                      target_acts: Dict[int, np.ndarray]) -> Dict:
    results = {}
    for layer in LAYERS:
        s = source_acts[layer]
        t = target_acts[layer]
        pooled = np.vstack([s, t])

        pca = PCA()
        pca.fit(pooled)
        evr = pca.explained_variance_ratio_

        # Find K for 90%, 95%, 99% variance
        cumsum = np.cumsum(evr)
        k_90 = int(np.searchsorted(cumsum, 0.90) + 1)
        k_95 = int(np.searchsorted(cumsum, 0.95) + 1)
        k_99 = int(np.searchsorted(cumsum, 0.99) + 1)

        # Effective rank
        ev_sum = evr.sum()
        eff_rank = float((np.arange(1, len(evr) + 1) * evr).sum() / ev_sum)

        # Rank bound test: CHaRS proves rank ≤ 2K-2
        # If optimal K from Exp2 is available, check if 2K-2 PCs capture >95%

        results[layer] = {
            "k_for_90pct": k_90,
            "k_for_95pct": k_95,
            "k_for_99pct": k_99,
            "effective_rank": eff_rank,
            "top5_variance": [float(v) for v in evr[:5].tolist()],
            "cumulative_variance_20": [float(v) for v in cumsum[:20].tolist()],
        }
    return results

# ─── Visualization ───────────────────────────────────────────────────────────

def plot_all_results(all_results: Dict):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = list(all_results.keys())
    layer_labels = [str(l) for l in LAYERS]

    # Fig 1: W2 per layer per task
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        r = all_results[task]
        exp1 = r.get("exp1_distribution_overlap", {})
        exp5 = r.get("exp5_asymmetry", {})

        ax = axes[idx]

        # W2
        w2_vals = [exp1.get(str(l), {}).get("w2", 0) for l in LAYERS]
        ax.plot(layer_labels, w2_vals, "o-", label="W2 distance", linewidth=2)

        # Cosine similarity
        ax2 = ax.twinx()
        cos_vals = [exp1.get(str(l), {}).get("cosine_similarity", 0) for l in LAYERS]
        ax2.plot(layer_labels, cos_vals, "s--", label="Cos Sim", color="orange", linewidth=2)

        ax.set_title(f"{task}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("W2 distance")
        ax2.set_ylabel("Cosine similarity")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    axes[len(tasks)].axis("off")
    axes[len(tasks) + 1].axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig1_distribution_overlap.png", dpi=150)
    plt.close()

    # Fig 2: Optimal K per task per layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for task in tasks:
        r = all_results[task]
        exp2 = r.get("exp2_cluster_analysis", {})
        opt_ks = [exp2.get(str(l), {}).get("optimal_k", 0) for l in LAYERS]
        ax1.plot(layer_labels, opt_ks, "o-", label=task, linewidth=2)

    ax1.set_title("Optimal K (CHaRS diagnostic)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Optimal K")
    ax1.legend()

    # Silhouette at K=opt per task (averaged across layers)
    for task in tasks:
        r = all_results[task]
        exp2 = r.get("exp2_cluster_analysis", {})
        sil_vals = []
        for l in LAYERS:
            ld = exp2.get(str(l), {})
            ok = ld.get("optimal_k", 2)
            sils = ld.get("silhouette_scores", {})
            sil_vals.append(sils.get(str(ok), 0))
        ax2.plot(layer_labels, sil_vals, "o-", label=task, linewidth=2)
    ax2.set_title("Silhouette at Optimal K")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Silhouette score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig2_cluster_analysis.png", dpi=150)
    plt.close()

    # Fig 3: Spearman rho + Curvature
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for task in tasks:
        r = all_results[task]
        exp3 = r.get("exp3_curvature", {})
        rho_vals = [exp3.get(str(l), {}).get("spearman_rho", 0) for l in LAYERS]
        ax1.plot(layer_labels, rho_vals, "o-", label=task, linewidth=2)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Spearman ρ (CurveBall diagnostic)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Spearman ρ")
    ax1.legend()

    for task in tasks:
        r = all_results[task]
        exp4 = r.get("exp4_trajectory_curvature", {})
        curv_vals = [exp4.get(str(l), {}).get("curvature", 0) for l in LAYERS]
        ax2.plot(layer_labels, curv_vals, "o-", label=task, linewidth=2)
    ax2.set_title("Trajectory Curvature (FLAS diagnostic)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Curvature (1 - mean cos)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig3_curvature.png", dpi=150)
    plt.close()

    # Fig 4: Asymmetry
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for task in tasks:
        r = all_results[task]
        exp5 = r.get("exp5_asymmetry", {})
        asym_vals = [exp5.get(str(l), {}).get("var_asymmetry_source_over_target", 1) for l in LAYERS]
        ax1.plot(layer_labels, asym_vals, "o-", label=task, linewidth=2)
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Variance Asymmetry (novel)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Var(source)/Var(target) along steer")
    ax1.legend()

    for task in tasks:
        r = all_results[task]
        exp5 = r.get("exp5_asymmetry", {})
        snr_vals = [exp5.get(str(l), {}).get("steer_snr", 0) for l in LAYERS]
        ax2.plot(layer_labels, snr_vals, "o-", label=task, linewidth=2)
    ax2.set_title("Steering SNR (novel)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Steer SNR (mean / std)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig4_asymmetry.png", dpi=150)
    plt.close()

    # Fig 5: Layer Importance
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for task in tasks:
        r = all_results[task]
        exp6 = r.get("exp6_layer_importance", {})
        sig_vals = [exp6.get(str(l), {}).get("signal_normalized", 0) for l in LAYERS]
        ax.plot(layer_labels, sig_vals, "o-", label=task, linewidth=2)
    ax.set_title("Layer Importance (LinEAS diagnostic)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized signal strength")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig5_layer_importance.png", dpi=150)
    plt.close()

    # Fig 6: PCA Low-rank
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for task in tasks:
        r = all_results[task]
        exp7 = r.get("exp7_pca_low_rank", {})
        eff_ranks = [exp7.get(str(l), {}).get("effective_rank", 0) for l in LAYERS]
        ax.plot(layer_labels, eff_ranks, "o-", label=task, linewidth=2)
    ax.set_title("Effective Rank (PCA diagnostic)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig6_pca_low_rank.png", dpi=150)
    plt.close()

    # Fig 7: PCA reconstruction error along interpolation (CurveBall)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for task in tasks:
        r = all_results[task]
        exp3 = r.get("exp3_curvature", {})
        recon = exp3.get(str(LAYERS[2]), {}).get("pca_recon_error_interp", [])
        if recon:
            ax.plot(np.linspace(0, 1, len(recon)), recon, "o-", label=task, linewidth=2)
    ax.set_title("PCA Recon Error Along Interpolation (CurveBall)")
    ax.set_xlabel("Interpolation t (source→target)")
    ax.set_ylabel("Reconstruction error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "fig7_recon_error.png", dpi=150)
    plt.close()

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run():
    model = load_model()
    print(f"Model loaded: {MODEL_NAME}")

    all_results = {}

    for task_name, cfg in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")

        # Load data
        source_texts, target_texts = load_task_data(task_name, N_SAMPLES)
        n_actual = min(len(source_texts), len(target_texts))
        source_texts = source_texts[:n_actual]
        target_texts = target_texts[:n_actual]
        print(f"  Source samples: {len(source_texts)}")
        print(f"  Target samples: {len(target_texts)}")

        # Collect source activations
        print("  Collecting SOURCE activations...")
        source_acts_raw = collect_activations(model, source_texts, LAYERS)
        source_acts = {l: source_acts_raw[l].cpu().numpy() for l in LAYERS}

        # Collect target activations
        print("  Collecting TARGET activations...")
        target_acts_raw = collect_activations(model, target_texts, LAYERS)
        target_acts = {l: target_acts_raw[l].cpu().numpy() for l in LAYERS}

        # Convert float64 for stability
        for l in LAYERS:
            source_acts[l] = source_acts[l].astype(np.float64)
            target_acts[l] = target_acts[l].astype(np.float64)

        # Run experiments
        print("  Running Experiment 1: Distribution Overlap...")
        exp1 = exp1_distribution_overlap(source_acts, target_acts)

        print("  Running Experiment 2: Cluster Analysis...")
        exp2 = exp2_cluster_analysis(source_acts, target_acts)

        print("  Running Experiment 3: Curvature Diagnostics...")
        exp3 = exp3_curvature_diagnostics(source_acts, target_acts)

        print("  Running Experiment 4: Trajectory Curvature...")
        exp4 = exp4_trajectory_curvature(source_acts, target_acts)

        print("  Running Experiment 5: Asymmetry Analysis...")
        exp5 = exp5_asymmetry(source_acts, target_acts)

        print("  Running Experiment 6: Layer Importance...")
        exp6 = exp6_layer_importance(source_acts, target_acts)

        print("  Running Experiment 7: PCA Low-Rank Structure...")
        exp7 = exp7_pca_low_rank(source_acts, target_acts)

        # Store results (convert int keys to str for JSON)
        def convert_keys(d):
            if isinstance(d, dict):
                return {str(k): convert_keys(v) for k, v in d.items()}
            return d

        all_results[task_name] = {
            "exp1_distribution_overlap": convert_keys(exp1),
            "exp2_cluster_analysis": convert_keys(exp2),
            "exp3_curvature": convert_keys(exp3),
            "exp4_trajectory_curvature": convert_keys(exp4),
            "exp5_asymmetry": convert_keys(exp5),
            "exp6_layer_importance": convert_keys(exp6),
            "exp7_pca_low_rank": convert_keys(exp7),
        }

        # Clean up
        del source_acts_raw, target_acts_raw
        del source_acts, target_acts
        torch.cuda.empty_cache()

    # Save all results to JSON
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SAVE_DIR / "activation_geometry_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    print("Generating plots...")
    plot_all_results(all_results)
    print(f"Plots saved to {SAVE_DIR}/")

if __name__ == "__main__":
    run()
