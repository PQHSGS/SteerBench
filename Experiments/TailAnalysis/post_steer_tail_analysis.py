"""
Post-Steer Tail Analysis (Inference-Level, Deferred).

Ask: Does steering change the tail distribution of activations?
1. Capture pre-steer activations (before hook) and post-steer (after hook) as .pt files
2. Compare tail magnitude (Hill α): does steering amplify or suppress off-manifold components?
3. Per-sample: does change in tail magnitude correlate with steering success/failure?

Saves activations as .pt (not JSON) to avoid bloated result files.

Usage:
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python -m Experiments.TailAnalysis.post_steer_tail_analysis --config Configs/Eval/CAA/example.json --coeff 3.0
"""

import json, sys, torch, numpy as np, argparse
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Steering.config.pipeline import PipelineConfig
from Steering.pipeline import SteeringPipeline
from Steering.utils import get_resid_acts, set_resid_acts

N_SAMPLES = 100
HILL_K_FRAC = 0.1
OUTPUT_DIR = Path("Experiments/TailAnalysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def hill_estimator(x, k=None):
    """Hill estimator for tail index α on absolute values.
    Returns (gamma, alpha) where gamma = extreme-value index, α = 1/γ.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--coeff", type=float, default=1.0)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    task_name = args.task or config_path.stem
    coeff = args.coeff

    # Load pipeline
    config = PipelineConfig.load(str(config_path))
    pipe = SteeringPipeline(config)
    pipe.load_model()
    pipe.setup()

    model = pipe.model
    steer_model = pipe.steer_model
    layer = steer_model.layer  # injection layer(s)

    # Load eval data
    from Steering.data.loader import DataLoader
    loader = DataLoader()
    eval_data = loader.load(
        config.eval.dataset,
        n_samples=N_SAMPLES,
        format=True,
        apply_chat_template=False,
        tokenizer=model.tokenizer,
    )

    # Pre-encode eval data as chat templates
    prompts = []
    ground_truths = []
    for d in eval_data:
        q = d.get("question", "")
        if q:
            full = f"{q} {d.get(config.eval.target_key, '')}"
        else:
            full = d.get(config.eval.target_key, "")
        prompts.append(full)
        ground_truths.append(d.get(config.eval.ground_truth_key, ""))

    # Create hook to capture pre/post steer activations
    captured = {"pre": [], "post": []}

    def make_capture_hook(layer_idx):
        def hook_fn(resid, hook):
            acts = get_resid_acts(resid, "last")
            captured["pre"].append(acts.detach().clone().float().cpu())
            return resid  # no modification (steer_model hook handles that)
        return hook_fn

    def make_post_hook(layer_idx):
        def hook_fn(resid, hook):
            acts = get_resid_acts(resid, "last")
            captured["post"].append(acts.detach().clone().float().cpu())
            return resid
        return hook_fn

    # Set up steering hooks
    steer_model.setup_hooks(coeff)
    hook_name = f"blocks.{layer[0]}.hook_resid_pre"

    # Register capture hooks around the steer hook
    model.add_hook(hook_name, make_capture_hook(layer[0]), "fwd")

    # Run generation
    outputs = []
    is_correct_list = []
    print(f"Running {len(prompts)} prompts with coeff={coeff}...")

    for i, prompt in enumerate(prompts):
        captured["pre"] = []
        captured["post"] = []
        with torch.no_grad():
            output = model.generate(
                prompt,
                max_new_tokens=30,
                do_sample=False,
                prepend_bos=True,
            )
        outputs.append(output)

        # Evaluate using the pipeline evaluator
        # (simplified: use the evaluator from the pipeline)
        if hasattr(pipe, 'evaluator') and pipe.evaluator is not None:
            is_correct, _ = pipe.evaluator.check(
                response=output,
                ground_truth=ground_truths[i] if i < len(ground_truths) else "",
            )
        else:
            is_correct = False

        is_correct_list.append(is_correct)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}]")

    steer_model.cleanup_hooks()
    model.reset_hooks()

    # Stack captured activations
    all_pre = torch.stack(captured["pre"]).numpy() if captured["pre"] else np.array([])
    # For post: we need to capture them differently since the steer hook modifies in-place
    # The post captures above happen at the same hook point before steering
    # We need to capture AFTER the steer hook. Let me use a different approach.
    #
    # Re-run with a post-hook on a downstream layer (L14 resid_post)
    print("\nRe-running with post-steer capture at L14 resid_post...")

    steer_model.setup_hooks(coeff)
    post_captured = []

    def make_downstream_hook(layer_idx):
        def hook_fn(resid, hook):
            acts = get_resid_acts(resid, "last")
            post_captured.append(acts.detach().clone().float().cpu())
            return resid
        return hook_fn

    model.add_hook(f"blocks.{layer[0]}.hook_resid_post", make_downstream_hook(layer[0]), "fwd")

    for i, prompt in enumerate(prompts[:len(outputs)]):
        pos = []
        with torch.no_grad():
            _ = model.generate(prompt, max_new_tokens=1, do_sample=False, prepend_bos=True)
        if (i + 1) % 50 == 0:
            print(f"  post-pass [{i+1}/{len(prompts)}]")

    steer_model.cleanup_hooks()
    model.reset_hooks()

    all_post = torch.stack(post_captured).numpy() if post_captured else np.array([])
    all_pre = torch.stack(captured["pre"]).numpy() if captured["pre"] else np.array([])

    if len(all_pre) == 0 or len(all_post) == 0:
        print("ERROR: No activations captured")
        return

    # Pad to same length
    n_min = min(len(all_pre), len(all_post))
    all_pre = all_pre[:n_min]
    all_post = all_post[:n_min]
    is_correct_arr = np.array(is_correct_list[:n_min], dtype=bool)

    # Save raw activations as .pt
    torch.save({
        "pre_steer_acts": torch.from_numpy(all_pre),
        "post_steer_acts": torch.from_numpy(all_post),
        "is_correct": torch.from_numpy(is_correct_arr),
        "coeff": coeff,
        "task": task_name,
        "config": str(config_path),
    }, OUTPUT_DIR / f"{task_name}_coeff_{coeff}_activations.pt")
    print(f"  Saved activations to {OUTPUT_DIR / f'{task_name}_coeff_{coeff}_activations.pt'}")

    # PCA analysis
    pooled = np.concatenate([all_pre, all_post], axis=0)
    pca = PCA(n_components=min(pooled.shape[0], pooled.shape[1]))
    pooled_pc = pca.fit_transform(pooled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, 0.90) + 1)

    # Project pre and post into same PCA space
    pre_pc = pca.transform(all_pre)
    post_pc = pca.transform(all_post)

    # In-manifold reconstruction
    def in_manifold(acts_pc, n_keep):
        recon = pca.inverse_transform(np.column_stack([
            acts_pc[:, :n_keep],
            np.zeros((acts_pc.shape[0], acts_pc.shape[1] - n_keep))
        ]))
        return recon

    pre_in = in_manifold(pre_pc, n_keep)
    post_in = in_manifold(post_pc, n_keep)
    pre_off = all_pre - pre_in
    post_off = all_post - post_in

    pre_off_norm = np.linalg.norm(pre_off, axis=1)
    post_off_norm = np.linalg.norm(post_off, axis=1)
    pre_total_norm = np.linalg.norm(all_pre, axis=1)
    post_total_norm = np.linalg.norm(all_post, axis=1)

    pre_off_ratio = pre_off_norm / np.maximum(pre_total_norm, 1e-12)
    post_off_ratio = post_off_norm / np.maximum(post_total_norm, 1e-12)
    off_ratio_delta = post_off_ratio - pre_off_ratio

    # Per-dim Hill tail index in PCA space
    n_pc = pre_pc.shape[1]
    hill_pre_in = np.array([hill_estimator(pre_pc[:, j])[1] for j in range(n_keep)])
    hill_post_in = np.array([hill_estimator(post_pc[:, j])[1] for j in range(n_keep)])
    hill_pre_off = np.array([hill_estimator(pre_pc[:, j])[1] for j in range(n_keep, n_pc)]) if n_keep < n_pc else np.array([])
    hill_post_off = np.array([hill_estimator(post_pc[:, j])[1] for j in range(n_keep, n_pc)]) if n_keep < n_pc else np.array([])

    # Pre vs post: difference in off-ratio by success/failure
    succ_mask = is_correct_arr
    fail_mask = ~succ_mask

    delta_succ = off_ratio_delta[succ_mask] if succ_mask.sum() > 0 else np.array([])
    delta_fail = off_ratio_delta[fail_mask] if fail_mask.sum() > 0 else np.array([])

    mw_p = None
    if len(delta_succ) > 2 and len(delta_fail) > 2:
        _, mw_p = mannwhitneyu(delta_succ, delta_fail, alternative='two-sided')

    # Spearman: does pre_off_ratio predict success?
    rho_off, p_off = spearmanr(pre_off_ratio, is_correct_arr.astype(float))

    # Spearman: does off_ratio_delta correlate with success?
    rho_delta, p_delta = spearmanr(off_ratio_delta, is_correct_arr.astype(float))

    results = {
        "task": task_name,
        "config": str(config_path),
        "coeff": coeff,
        "n_samples": n_min,
        "accuracy": float(is_correct_arr.mean()),
        "n_pca_keep": int(n_keep),
        "pre_steer": {
            "off_ratio_mean": float(pre_off_ratio.mean()),
            "off_ratio_std": float(pre_off_ratio.std()),
            "total_norm_mean": float(pre_total_norm.mean()),
            "hill_alpha_in_manifold_mean": float(np.nanmean(hill_pre_in)) if len(hill_pre_in) else None,
            "hill_alpha_off_manifold_mean": float(np.nanmean(hill_pre_off)) if len(hill_pre_off) else None,
        },
        "post_steer": {
            "off_ratio_mean": float(post_off_ratio.mean()),
            "off_ratio_std": float(post_off_ratio.std()),
            "total_norm_mean": float(post_total_norm.mean()),
            "hill_alpha_in_manifold_mean": float(np.nanmean(hill_post_in)) if len(hill_post_in) else None,
            "hill_alpha_off_manifold_mean": float(np.nanmean(hill_post_off)) if len(hill_post_off) else None,
        },
        "delta_post_minus_pre": {
            "off_ratio_mean": float(off_ratio_delta.mean()),
            "off_ratio_std": float(off_ratio_delta.std()),
            "off_ratio_success_mean": float(delta_succ.mean()) if len(delta_succ) > 0 else None,
            "off_ratio_fail_mean": float(delta_fail.mean()) if len(delta_fail) > 0 else None,
        },
        "success_vs_fail": {
            "mw_p_off_ratio_delta": float(mw_p) if mw_p is not None else None,
            "significant_005": bool(mw_p is not None and mw_p < 0.05) if mw_p is not None else None,
        },
        "prediction": {
            "spearman_pre_off_ratio_vs_success": {"rho": float(rho_off), "p": float(p_off)},
            "spearman_delta_off_ratio_vs_success": {"rho": float(rho_delta), "p": float(p_delta)},
        },
    }

    print(f"\n  Results for {task_name} coeff={coeff}:")
    print(f"    Accuracy: {results['accuracy']:.2%}")
    print(f"    Pre off-ratio: {results['pre_steer']['off_ratio_mean']:.4f} → Post off-ratio: {results['post_steer']['off_ratio_mean']:.4f}")
    print(f"    Off-ratio delta: {results['delta_post_minus_pre']['off_ratio_mean']:.4f}")
    print(f"    Spearman ρ(pre_off_ratio, success): {rho_off:.4f} (p={p_off:.4f})")
    print(f"    Spearman ρ(delta_off_ratio, success): {rho_delta:.4f} (p={p_delta:.4f})")

    save_path = OUTPUT_DIR / f"{task_name}_coeff_{coeff}_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {save_path}")


if __name__ == "__main__":
    main()
