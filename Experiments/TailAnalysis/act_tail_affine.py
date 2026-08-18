"""
ACT Affine Parameters (omega, beta) in Tail vs Manifold Dimensions.

Ask: In ACT "linear" mode, are the affine parameters ω (per-dim slope) and β
(per-dim intercept) systematically different in off-manifold (tail) dimensions?

Hypothesis: Off-manifold dims (where tail variance concentrates) have more extreme ω
(further from 1) and larger |β|, because the sorted affine transport must work
harder to align distributions in noisy, degenerate dimensions.

Pipeline:
1. Load ACT LinearAcT config → SteeringPipeline → extract() → get ω, β per dim
2. Extract source (contrast) activations from train data → PCA (90% var)
3. Partition 2304 dims into in-manifold (top-k PCs) vs off-manifold (residual)
4. For each partition: compare magnitude of |ω-1|, |β|, and their spread
5. Per-dim: does ω correlate with off-manifold projection magnitude?
6. Repeat across tasks

Usage:
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python -m Experiments.TailAnalysis.act_tail_affine
"""

import json, sys, torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Steering.config.pipeline import PipelineConfig
from Steering.pipeline import SteeringPipeline
from Steering.utils import collect_dense_activations

OUTPUT_DIR = Path("Experiments/TailAnalysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Each config: (name, config_path)
TASK_CONFIGS = [
    ("toxic",     Path("Configs/Eval/LinearAcT/Gemma/gemma_toxic.json")),
    ("evil",      Path("Configs/Eval/LinearAcT/Gemma/gemma_evil.json")),
    ("deception", Path("Configs/Eval/LinearAcT/Gemma/gemma_deception.json")),
    ("refusal",   Path("Configs/Eval/LinearAcT/Gemma/gemma_refusal_response.json")),
]


def main():
    all_results = {}

    for task_name, config_path in TASK_CONFIGS:
        if not config_path.exists():
            print(f"SKIP: {config_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        # ── Load config and patch to use "linear" act_mode ──
        config = PipelineConfig.load(str(config_path))
        config.extractor.act_mode = "linear"
        config.model.device = DEVICE
        config.model.dtype = "bfloat16"
        config.n_train = 500  # match ExpG

        # ── Create pipeline and run extraction ──
        pipe = SteeringPipeline(config)
        pipe.load_model("tl")
        pipe.authenticate()

        # Load training data the same way pipeline._setup_vector_from_extraction does
        train_data = pipe.load_train_data(
            config.train_dataset, config.n_train,
            apply_chat_template=config.extractor.apply_chat_template,
        )
        target_texts = [d["correct_prompt"] for d in train_data]
        contrast_texts = [d["false_prompt"] for d in train_data]
        if config.extractor.inverse:
            target_texts, contrast_texts = contrast_texts, target_texts

        print(f"  Train samples: target={len(target_texts)}, contrast={len(contrast_texts)}")

        # Run extraction
        pipe.extract(target_data=target_texts, contrast_data=contrast_texts)

        # ── Get omega/beta from extractor metadata ──
        layer = config.extractor.layer[0]
        stats = pipe.extractor.metadata["act_stats"][layer]
        omega = stats["omega"].float().cpu().numpy()   # (2304,)
        beta = stats["beta"].float().cpu().numpy()      # (2304,)
        mu_src = stats["mu_src"].float().cpu().numpy()  # (2304,)
        mu_dst = stats["mu_dst"].float().cpu().numpy()
        print(f"  Omega stats: mean={omega.mean():.6f}, std={omega.std():.6f}, "
              f"min={omega.min():.6f}, max={omega.max():.6f}")
        print(f"  Beta stats: mean={beta.mean():.6f}, std={beta.std():.6f}")

        # ── Extract source activations for PCA ──
        print("  Extracting source (contrast) activations for PCA...")
        source_acts = collect_dense_activations(
            pipe.model, contrast_texts, layers=[layer],
            hook_point=config.extractor.hook_point,
            batch_size=config.extractor.batch_size,
            pooling=config.extractor.position,
            device=DEVICE, tokenizer=pipe.model.tokenizer,
        )[layer].float().cpu().numpy()

        # ── PCA on source activations ──
        n, d = source_acts.shape
        pca = PCA(n_components=min(n, d))
        source_pc = pca.fit_transform(source_acts)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, 0.90) + 1)
        print(f"  PCA: {n_keep}/{d} PCs for {cumvar[n_keep-1]:.4f} var")

        # ── Project omega and beta into PCA space ──
        # omega and beta are in canonical space; rotate to PC space
        omega_pc = pca.transform(omega.reshape(1, -1)).ravel()
        beta_pc = pca.transform(beta.reshape(1, -1)).ravel()
        mu_src_pc = pca.transform(mu_src.reshape(1, -1)).ravel()
        mu_dst_pc = pca.transform(mu_dst.reshape(1, -1)).ravel()

        # ── Per-dim metrics in canonical space ──
        omega_dev = np.abs(omega - 1.0)       # |ω - 1|: deviation from identity
        beta_abs = np.abs(beta)                # |β|: intercept magnitude
        steer_vector = mu_dst - mu_src         # CAA-style mean shift
        steer_norm = np.linalg.norm(steer_vector)

        # Per-dim off-manifold projection magnitude
        # Project each canonical basis vector into off-manifold → get "offness" per dim
        # Method: for each canonical dim e_j, compute off-manifold norm after projection
        # Actually simpler: compute how much of each canonical basis vector
        # lies in off-manifold subspace
        off_proj = np.zeros(d)
        comp_T = pca.components_[:n_keep]  # (n_keep, d) - in-manifold basis
        for j in range(d):
            e_j = np.zeros(d)
            e_j[j] = 1.0
            # Project onto in-manifold
            e_in = comp_T.T @ (comp_T @ e_j)
            # Off-manifold component magnitude
            off_proj[j] = np.linalg.norm(e_j - e_in)

        # ── Partition dims by off-projection magnitude ──
        off_thresh = np.median(off_proj)
        off_mask = off_proj > off_thresh  # dims with strong off-manifold projection
        in_mask = ~off_mask
        n_off = off_mask.sum()
        n_in = in_mask.sum()
        print(f"  By off-projection: {n_in} in-manifold dims, {n_off} off-manifold dims")

        # ── Compare omega/beta in each partition ──
        for metric_name, metric in [("|ω-1|", omega_dev), ("|β|", beta_abs)]:
            m_in = metric[in_mask]
            m_off = metric[off_mask]
            _, mw_p = mannwhitneyu(m_in, m_off, alternative='two-sided')
            print(f"  {metric_name}: in={m_in.mean():.6f}±{m_in.std():.6f}, "
                  f"off={m_off.mean():.6f}±{m_off.std():.6f}, "
                  f"ratio={m_off.mean()/max(m_in.mean(),1e-12):.4f}x, MWp={mw_p:.6f}")

        # ── Spearman: does per-dim off-projection correlate with |ω-1|? ──
        rho_off_omega, p_off_omega = spearmanr(off_proj, omega_dev)
        rho_off_beta, p_off_beta = spearmanr(off_proj, beta_abs)

        # ── Spearman: does per-dim off-projection correlate with steering vector? ──
        rho_off_steer, p_off_steer = spearmanr(off_proj, np.abs(steer_vector))

        # ── PCA-space analysis ──
        # In PC space, in-manifold dims are 0..n_keep-1, off-manifold are n_keep..
        in_slice = slice(0, n_keep)
        off_slice = slice(n_keep, None)

        # Identity in PC space = [1,...,1, 0,...,0] (1 for first d PCs, 0 for rest)
        n_pc = len(omega_pc)
        id_vec_pc = np.zeros(n_pc)
        id_vec_pc[:min(n_pc, d)] = 1.0
        omega_dev_pc = np.abs(omega_pc - id_vec_pc)

        # ── Tail analysis per omega/beta group ──
        # Are dims with high |ω-1| also heavy-tail (low Hill α) dims?
        top_omega = np.argsort(omega_dev)[-max(int(d*0.1), 1):]
        top_off = np.argsort(off_proj)[-max(int(d*0.1), 1):]
        top_overlap = len(np.intersect1d(top_omega, top_off))
        top_iou = top_overlap / max(len(np.union1d(top_omega, top_off)), 1)

        # ── Per-PC-dim analysis in PCA space ──
        # Partition omega_pc into in-manifold vs off-manifold PCs
        omega_dev_in = np.abs(omega_pc[:n_keep] - 1.0)
        omega_dev_off = np.abs(omega_pc[n_keep:] - 1.0) if n_keep < d else np.array([])
        beta_abs_in = np.abs(beta_pc[:n_keep])
        beta_abs_off = np.abs(beta_pc[n_keep:]) if n_keep < d else np.array([])

        results = {
            "task": task_name,
            "n_samples": n,
            "n_dims": d,
            "n_pca_keep": int(n_keep),
            "explained_var": float(cumvar[n_keep - 1]),
            "omega_canonical": {
                "mean": float(omega.mean()),
                "std": float(omega.std()),
                "min": float(omega.min()),
                "max": float(omega.max()),
                "mean_deviation_from_1": float(omega_dev.mean()),
                "median_deviation_from_1": float(np.median(omega_dev)),
            },
            "beta_canonical": {
                "mean": float(beta.mean()),
                "std": float(beta.std()),
                "mean_abs": float(beta_abs.mean()),
            },
            "off_projection_per_dim": {
                "mean": float(off_proj.mean()),
                "median": float(np.median(off_proj)),
                "n_dims_above_median": int(n_off),
            },
            "partition_comparison": {
                "in_manifold_count": int(n_in),
                "off_manifold_count": int(n_off),
                "omega_dev_in_mean": float(omega_dev[in_mask].mean()),
                "omega_dev_off_mean": float(omega_dev[off_mask].mean()),
                "omega_dev_ratio_off_over_in": float(omega_dev[off_mask].mean() / max(omega_dev[in_mask].mean(), 1e-12)),
                "omega_dev_mw_p": float(mannwhitneyu(omega_dev[in_mask], omega_dev[off_mask], alternative='two-sided').pvalue),
                "beta_abs_in_mean": float(beta_abs[in_mask].mean()),
                "beta_abs_off_mean": float(beta_abs[off_mask].mean()),
                "beta_abs_ratio_off_over_in": float(beta_abs[off_mask].mean() / max(beta_abs[in_mask].mean(), 1e-12)),
                "beta_abs_mw_p": float(mannwhitneyu(beta_abs[in_mask], beta_abs[off_mask], alternative='two-sided').pvalue),
            },
            "pca_space": {
                "omega_dev_in_mean": float(omega_dev_in.mean()),
                "omega_dev_off_mean": float(omega_dev_off.mean()) if len(omega_dev_off) else None,
                "beta_abs_in_mean": float(beta_abs_in.mean()),
                "beta_abs_off_mean": float(beta_abs_off.mean()) if len(beta_abs_off) else None,
            },
            "correlations": {
                "spearman_off_proj_vs_omega_dev": {"rho": float(rho_off_omega), "p": float(p_off_omega)},
                "spearman_off_proj_vs_beta_abs": {"rho": float(rho_off_beta), "p": float(p_off_beta)},
                "spearman_off_proj_vs_steer_abs": {"rho": float(rho_off_steer), "p": float(p_off_steer)},
            },
            "tail_dim_overlap": {
                "top10pct_omega_dev_and_off_proj": int(top_overlap),
                "iou_top10pct": float(top_iou),
            },
        }

        print(f"\n  ── Results:")
        print(f"     ω dev in={results['partition_comparison']['omega_dev_in_mean']:.6f} vs "
              f"off={results['partition_comparison']['omega_dev_off_mean']:.6f} "
              f"({results['partition_comparison']['omega_dev_ratio_off_over_in']:.4f}x, "
              f"MW p={results['partition_comparison']['omega_dev_mw_p']:.6f})")
        print(f"     |β| in={results['partition_comparison']['beta_abs_in_mean']:.6f} vs "
              f"off={results['partition_comparison']['beta_abs_off_mean']:.6f} "
              f"({results['partition_comparison']['beta_abs_ratio_off_over_in']:.4f}x, "
              f"MW p={results['partition_comparison']['beta_abs_mw_p']:.6f})")
        print(f"     ρ(off_proj, |ω-1|) = {rho_off_omega:.4f} (p={p_off_omega:.6f})")
        print(f"     ρ(off_proj, |β|)   = {rho_off_beta:.4f} (p={p_off_beta:.6f})")
        print(f"     Top-10% overlap: {top_overlap}/{int(d*0.1)} (IoU={top_iou:.4f})")

        all_results[task_name] = results

        # Clean up to free GPU memory
        del pipe.model, pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    save_path = OUTPUT_DIR / "expJ_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Summary
    print("\n\n=== SUMMARY ===")
    hdr = f"{'Task':>12s}  {'n_keep':>6s}  {'|ω-1|_in':>8s}  {'|ω-1|_off':>10s}  {'ratio_ω':>8s}  {'MW_ω':>8s}  {'|β|_in':>8s}  {'|β|_off':>8s}  {'ratio_β':>8s}  {'MW_β':>8s}"
    print(hdr)
    print("-" * 100)
    for name, r in all_results.items():
        pc = r["partition_comparison"]
        print(f"  {name:>12s}  {r['n_pca_keep']:>6d}  {pc['omega_dev_in_mean']:>8.6f}  {pc['omega_dev_off_mean']:>10.6f}  {pc['omega_dev_ratio_off_over_in']:>8.4f}x  {pc['omega_dev_mw_p']:>8.6f}  {pc['beta_abs_in_mean']:>8.6f}  {pc['beta_abs_off_mean']:>8.6f}  {pc['beta_abs_ratio_off_over_in']:>8.4f}x  {pc['beta_abs_mw_p']:>8.6f}")


if __name__ == "__main__":
    main()
