"""
Target-vs-Contrast Tail Comparison (Extraction-Level).

Ask: Do source (contrast) and target activations have different tail properties
in off-manifold dimensions? Is the contrastive steering signal itself tail-heavy?

Extraction-level only — no inference, no steering. Just analyze the paired
extraction-dataset activations.

Analysis:
1. Extract resid_pre at layer 14 (last token) for target and contrast texts
2. PCA on pooled → partition in-manifold (90% var) vs off-manifold
3. Per-class: off-ratio, per-dim Hill tail index (canonical + PCA), class-mean separation
4. CRITICAL: decompose the steering vector (target_mean - contrast_mean) in PCA space.
   What fraction of its norm is in off-manifold? Does this predict anything?
5. Per-dim MW test: which canonical dims differ most between classes?
   Are the dims with largest class difference in the tail set?
6. Tail-dim agreement: what's the IoU of "Hill α < threshold" dims between classes?

Usage:
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python -m Experiments.TailAnalysis.target_vs_contrast_tail
"""

import json, sys, torch, numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Steering.data.loader import DataLoader
from Steering.utils import collect_dense_activations
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
POOLING = "last"
N_SAMPLES = 500
BATCH_SIZE = 8
TAIL_ALPHA_THRESH = 3.5  # Hill α < 3.5 → heavy tail (bottom ~1% of activation dims)
HILL_K_FRAC = 0.1        # fraction of samples used for Hill upper order statistics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("Experiments/TailAnalysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


TASKS = [
    {"name": "toxic",      "dataset": "toxic_jigsaw",    "target_key": "correct_prompt",   "contrast_key": "false_prompt"},
    {"name": "evil",       "dataset": "evil",            "target_key": "correct_prompt",   "contrast_key": "false_prompt"},
    {"name": "deception",  "dataset": "liarbench",       "target_key": "false_prompt",     "contrast_key": "correct_prompt"},
    {"name": "refusal",    "dataset": "refusal_cast_responses",     "target_key": "correct_prompt",   "contrast_key": "false_prompt"},
]


def main():
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}, Device: {DEVICE}")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=DEVICE, dtype=torch.bfloat16,
    )
    model.to(DEVICE)

    loader = DataLoader()
    all_results = {}

    for task_cfg in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_cfg['name']}")
        print(f"{'='*60}")

        data = loader.load(task_cfg["dataset"], n_samples=N_SAMPLES, format=True, apply_chat_template=True, tokenizer=model.tokenizer)

        target_texts = [d[task_cfg["target_key"]] for d in data]
        contrast_texts = [d[task_cfg["contrast_key"]] for d in data]
        print(f"  Target: {len(target_texts)}, Contrast: {len(contrast_texts)}")

        # ── extract activations ──
        target_acts = collect_dense_activations(
            model, target_texts, layers=[LAYER],
            hook_point="pre", batch_size=BATCH_SIZE,
            pooling=POOLING, device=DEVICE, tokenizer=model.tokenizer,
        )[LAYER].float().cpu().numpy()

        contrast_acts = collect_dense_activations(
            model, contrast_texts, layers=[LAYER],
            hook_point="pre", batch_size=BATCH_SIZE,
            pooling=POOLING, device=DEVICE, tokenizer=model.tokenizer,
        )[LAYER].float().cpu().numpy()

        pooled = np.concatenate([target_acts, contrast_acts], axis=0)
        n, d = pooled.shape
        n_t, n_c = len(target_acts), len(contrast_acts)
        steering_vec = target_acts.mean(axis=0) - contrast_acts.mean(axis=0)

        # ── PCA on pooled ──
        pca = PCA(n_components=min(n, d))
        pooled_pc = pca.fit_transform(pooled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, 0.90) + 1)
        off_dims = slice(n_keep, None)
        print(f"  PCA: {n_keep}/{d} PCs for {cumvar[n_keep-1]:.4f} var (off-manifold: {d - n_keep} dims)")

        # Project each class + steering vector
        target_pc = pca.transform(target_acts)
        contrast_pc = pca.transform(contrast_acts)
        steer_pc = pca.transform(steering_vec.reshape(1, -1)).ravel()  # steering vector in PC space

        # ── Decompose steering vector ──
        steer_norm_in = np.linalg.norm(steer_pc[:n_keep])
        steer_norm_off = np.linalg.norm(steer_pc[n_keep:])
        steer_norm_total = np.linalg.norm(steer_pc)
        steer_off_frac = steer_norm_off / max(steer_norm_total, 1e-12)
        steer_in_frac = steer_norm_in / max(steer_norm_total, 1e-12)

        # ── Reconstruct in/off for each sample ──
        def reconstruct(pc, nk):
            return pca.inverse_transform(np.column_stack([
                pc[:, :nk], np.zeros((pc.shape[0], pc.shape[1] - nk))
            ]))

        t_in = reconstruct(target_pc, n_keep)
        c_in = reconstruct(contrast_pc, n_keep)
        t_off = target_acts - t_in
        c_off = contrast_acts - c_in

        # ── Norm-based metrics ──
        t_total_norm = np.linalg.norm(target_acts, axis=1)
        c_total_norm = np.linalg.norm(contrast_acts, axis=1)
        t_off_norm = np.linalg.norm(t_off, axis=1)
        c_off_norm = np.linalg.norm(c_off, axis=1)
        t_off_ratio = t_off_norm / np.maximum(t_total_norm, 1e-12)
        c_off_ratio = c_off_norm / np.maximum(c_total_norm, 1e-12)

        # ── Per-dim Hill tail index in PCA space ──
        hill_in_t = np.array([hill_estimator(target_pc[:, j])[1] for j in range(n_keep)])
        hill_in_c = np.array([hill_estimator(contrast_pc[:, j])[1] for j in range(n_keep)])
        hill_off_t = np.array([hill_estimator(target_pc[:, j])[1] for j in range(n_keep, target_pc.shape[1])]) if n_keep < target_pc.shape[1] else np.array([])
        hill_off_c = np.array([hill_estimator(contrast_pc[:, j])[1] for j in range(n_keep, contrast_pc.shape[1])]) if n_keep < contrast_pc.shape[1] else np.array([])

        # ── Per-dim Hill tail index in CANONICAL basis ──
        alpha_canon_t = np.array([hill_estimator(target_acts[:, j])[1] for j in range(d)])
        alpha_canon_c = np.array([hill_estimator(contrast_acts[:, j])[1] for j in range(d)])
        tail_dims_t = np.where(alpha_canon_t < TAIL_ALPHA_THRESH)[0]
        tail_dims_c = np.where(alpha_canon_c < TAIL_ALPHA_THRESH)[0]
        tail_dims_both = np.intersect1d(tail_dims_t, tail_dims_c)
        tail_dims_union = np.union1d(tail_dims_t, tail_dims_c)
        tail_iou = len(tail_dims_both) / max(len(tail_dims_union), 1)

        # Is the steering vector concentrated in tail canonical dims?
        steer_norm_tail = np.linalg.norm(steering_vec[tail_dims_union]) if len(tail_dims_union) > 0 else 0.0
        steer_tail_frac = steer_norm_tail / max(steer_norm_total, 1e-12)

        # ── Per-dim MW test: which canonical dims have largest class difference? ──
        p_vals = np.zeros(d)
        for dim in range(d):
            _, p_vals[dim] = mannwhitneyu(target_acts[:, dim], contrast_acts[:, dim], alternative='two-sided')
        # Correct for multiple comparisons (Bonferroni)
        p_corrected = np.minimum(p_vals * d, 1.0)
        sig_dims = np.where(p_corrected < 0.05)[0]
        sig_dims_in_tail = np.intersect1d(sig_dims, tail_dims_union)
        sig_tail_frac = len(sig_dims_in_tail) / max(len(tail_dims_union), 1) if len(tail_dims_union) > 0 else 0.0

        # ── Off-manifold class separation ──
        mu_t_off = target_pc[:, n_keep:].mean(axis=0) if n_keep < d else np.zeros(0)
        mu_c_off = contrast_pc[:, n_keep:].mean(axis=0) if n_keep < d else np.zeros(0)
        off_sep = np.linalg.norm(mu_t_off - mu_c_off) if n_keep < d else 0.0
        mu_t_in = target_pc[:, :n_keep].mean(axis=0)
        mu_c_in = contrast_pc[:, :n_keep].mean(axis=0)
        in_sep = np.linalg.norm(mu_t_in - mu_c_in)

        # What fraction of PC off-manifold dims are individually significant?
        n_pc_total = target_pc.shape[1]  # min(n, d)
        if n_keep < n_pc_total:
            p_vals_off = np.zeros(n_pc_total - n_keep)
            for j, dim in enumerate(range(n_keep, n_pc_total)):
                _, p_vals_off[j] = mannwhitneyu(
                    target_pc[:, dim], contrast_pc[:, dim], alternative='two-sided')
            sig_off_dims = np.sum(p_vals_off * (n_pc_total - n_keep) < 0.05)
            frac_sig_off = sig_off_dims / max(n_pc_total - n_keep, 1)
        else:
            sig_off_dims = 0
            frac_sig_off = 0.0

        # ── MW test on off-ratio ──
        _, mw_p = mannwhitneyu(t_off_ratio, c_off_ratio, alternative='two-sided')

        # ── Results ──
        results = {
            "task": task_cfg["name"],
            "n_samples": n,
            "n_pca_keep": int(n_keep),
            "explained_var_90": float(cumvar[n_keep - 1]),
            "steering_vector_decomposition": {
                "norm_total": float(steer_norm_total),
                "norm_in_manifold": float(steer_norm_in),
                "norm_off_manifold": float(steer_norm_off),
                "frac_in_manifold": float(steer_in_frac),
                "frac_off_manifold": float(steer_off_frac),
                "tail_canonical_dims_fraction": float(steer_tail_frac),
            },
            "target": {
                "off_ratio_mean": float(t_off_ratio.mean()),
                "off_ratio_std": float(t_off_ratio.std()),
                "hill_alpha_in_manifold_mean": float(np.nanmean(hill_in_t)) if len(hill_in_t) else None,
                "hill_alpha_off_manifold_mean": float(np.nanmean(hill_off_t)) if len(hill_off_t) else None,
                "hill_alpha_canonical_mean": float(np.nanmean(alpha_canon_t)),
                "tail_dim_count": int(len(tail_dims_t)),
                "min_hill_alpha_canonical": float(np.nanmin(alpha_canon_t)),
            },
            "contrast": {
                "off_ratio_mean": float(c_off_ratio.mean()),
                "off_ratio_std": float(c_off_ratio.std()),
                "hill_alpha_in_manifold_mean": float(np.nanmean(hill_in_c)) if len(hill_in_c) else None,
                "hill_alpha_off_manifold_mean": float(np.nanmean(hill_off_c)) if len(hill_off_c) else None,
                "hill_alpha_canonical_mean": float(np.nanmean(alpha_canon_c)),
                "tail_dim_count": int(len(tail_dims_c)),
                "min_hill_alpha_canonical": float(np.nanmin(alpha_canon_c)),
            },
            "tail_dim_agreement": {
                "iou_tail_dims": float(tail_iou),
                "tail_dims_target": int(len(tail_dims_t)),
                "tail_dims_contrast": int(len(tail_dims_c)),
                "tail_dims_intersection": int(len(tail_dims_both)),
                "tail_dims_union": int(len(tail_dims_union)),
            },
            "separation": {
                "in_manifold": float(in_sep),
                "off_manifold": float(off_sep),
                "ratio_off_vs_in": float(off_sep / max(in_sep, 1e-12)),
                "frac_sig_off_dims": float(frac_sig_off),
                "n_sig_off_dims": int(sig_off_dims),
                "n_off_dims_total": int(max(n_pc_total - n_keep, 0)),
            },
            "mw_test": {
                "off_ratio_target_vs_contrast_p": float(mw_p),
                "significant_005": bool(mw_p < 0.05),
                "n_sig_canonical_dims": int(len(sig_dims)),
                "sig_dims_in_tail": int(len(sig_dims_in_tail)),
                "sig_dims_in_tail_frac": float(sig_tail_frac),
                "n_sig_dims_total_out_of": d,
            },
        }

        print(f"\n  ── Steering vector decomposition:")
        print(f"     in-manifold: {steer_in_frac:.4f}, off-manifold: {steer_off_frac:.4f}, tail-canon: {steer_tail_frac:.4f}")
        print(f"  ── Off-ratio: target {t_off_ratio.mean():.4f} ± {t_off_ratio.std():.4f}")
        print(f"                contrast {c_off_ratio.mean():.4f} ± {c_off_ratio.std():.4f}")
        print(f"     MW p={mw_p:.6f}")
        print(f"  ── Separation: in={in_sep:.4f} off={off_sep:.4f} off/in={off_sep/max(in_sep,1e-12):.6f}")
        print(f"  ── Hill α dims (α<{TAIL_ALPHA_THRESH}): t={len(tail_dims_t)} c={len(tail_dims_c)} IoU={tail_iou:.4f}")
        print(f"  ── Sig canonical dims: {len(sig_dims)}/{d} (in tail: {len(sig_dims_in_tail)}/{len(tail_dims_union)})")
        print(f"  ── Sig off-manifold PC dims: {sig_off_dims}/{max(n_pc_total-n_keep,0)}")

        all_results[task_cfg["name"]] = results

        torch.save({
            "target_acts": torch.from_numpy(target_acts),
            "contrast_acts": torch.from_numpy(contrast_acts),
            "target_pc": torch.from_numpy(target_pc),
            "contrast_pc": torch.from_numpy(contrast_pc),
            "steering_vec": torch.from_numpy(steering_vec),
            "steer_pc": torch.from_numpy(steer_pc),
            "pca": pca,
            "n_keep": n_keep,
        }, OUTPUT_DIR / f"{task_cfg['name']}_activations.pt")

    save_path = OUTPUT_DIR / "expG_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    print("\n\n=== SUMMARY ===")
    hdr = f"{'Task':>12s}  {'n_keep':>6s}  {'Sv_in':>8s}  {'Sv_off':>8s}  {'T_off_r':>8s}  {'C_off_r':>8s}  {'In_sep':>8s}  {'Off_sep':>8s}  {'Off/In':>8s}  {'Tail_IoU':>8s}  {'SigOff':>6s}  {'MW_p':>8s}"
    print(hdr)
    for name, r in all_results.items():
        sv = r['steering_vector_decomposition']
        print(f"  {name:>12s}  {r['n_pca_keep']:>6d}  {sv['frac_in_manifold']:>8.4f}  {sv['frac_off_manifold']:>8.4f}  {r['target']['off_ratio_mean']:>8.4f}  {r['contrast']['off_ratio_mean']:>8.4f}  {r['separation']['in_manifold']:>8.4f}  {r['separation']['off_manifold']:>8.4f}  {r['separation']['ratio_off_vs_in']:>8.6f}  {r['tail_dim_agreement']['iou_tail_dims']:>8.4f}  {r['separation']['n_sig_off_dims']:>6d}  {r['mw_test']['off_ratio_target_vs_contrast_p']:>8.6f}")


if __name__ == "__main__":
    main()
