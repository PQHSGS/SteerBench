"""
Exp 6: Coupling Mass vs Sample Norm Analysis (Tail-Coupling Diagnostic)

Tests whether tail-region samples (high norm) receive disproportionately poor
Sinkhorn coupling mass in CHARS, confirming the RBF degeneracy mechanism.

Steps:
1. Load source activations from toxic_jigsaw (non-toxic prompts)
2. Load existing CHARS centroids + Sinkhorn coupling from Vector/CHARS/Gemma/toxic/
3. Assign each sample to nearest centroid
4. Bin samples by L2 norm percentile
5. Compute total coupling mass per bin (= sum of P*[i,:] for each centroid weighted by assignment count)
6. Plot + report

Usage:
  conda activate sae_circuit
  unset CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=2
  python -m Experiments.exp6_coupling_mass_analysis
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 500  # match n_train from extraction
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = [
    {
        "name": "toxic",
        "train_dataset": "toxic_jigsaw",
        "source_key": "false_prompt",   # non-toxic
        "target_key": "correct_prompt", # toxic
        "vector_path": "Vector/CHARS/Gemma/toxic/metadata.pt",
    },
    {
        "name": "evil",
        "train_dataset": "evil",
        "source_key": "false_prompt",
        "target_key": "correct_prompt",
        "vector_path": "Vector/CHARS/Gemma/evil/metadata.pt",
    },
    {
        "name": "deception",
        "train_dataset": "liarbench",
        "source_key": "correct_prompt",
        "target_key": "false_prompt",
        "vector_path": "Vector/CHARS/Gemma/deception/metadata.pt",
    },
    {
        "name": "refusal",
        "train_dataset": "refusal_caa",
        "source_key": "false_prompt",
        "target_key": "correct_prompt",
        "vector_path": "Vector/CHARS/Gemma/refusal_response/metadata.pt",
    },
]


def get_source_activations(model, texts, layer=LAYER, batch_size=BATCH_SIZE):
    """Extract resid_pre activations at last token position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [model.tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        ) for t in batch]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre",
                prepend_bos=True,
            )
        acts = cache[f"blocks.{layer}.hook_resid_pre"][:, -1, :]  # last token
        all_acts.append(acts.detach().cpu())
    return torch.cat(all_acts, dim=0).float()


def analyze_task(task_cfg, model):
    print(f"\n{'='*60}")
    print(f"Task: {task_cfg['name']}")
    print(f"{'='*60}")

    # 1. Load data
    print(f"Loading {task_cfg['train_dataset']}...")
    loader = DataLoader()
    data = loader.load(
        task_cfg["train_dataset"],
        n_samples=N_SAMPLES,
        format=True,
        apply_chat_template=False,
    )
    source_texts = [d[task_cfg["source_key"]] for d in data]
    print(f"  {len(source_texts)} source samples")

    # 2. Extract source activations
    print(f"Extracting activations at layer {LAYER}...")
    source_acts = get_source_activations(model, source_texts)
    print(f"  shape: {source_acts.shape}")

    # 3. Load CHARS metadata
    print(f"Loading CHARS centroids from {task_cfg['vector_path']}...")
    md = torch.load(task_cfg["vector_path"], map_location="cpu", weights_only=True)
    layer_key = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    centroids_A = md["chars_centroids_A"][layer_key]     # [K, d_model]
    P_star = md["chars_coupling"][layer_key]              # [K, K]
    K = md["chars_k"][layer_key]

    centroids_A = centroids_A.float()
    P_star = P_star.float()

    print(f"  K={K}, centroids shape={centroids_A.shape}")

    # 4. Assign each sample to nearest centroid
    print("Assigning samples to nearest centroid...")
    # Compute distances: [N, K]
    dists = torch.cdist(source_acts, centroids_A, p=2.0)
    assigned_centroid = dists.argmin(dim=1)  # [N]
    n_per_centroid = torch.bincount(assigned_centroid, minlength=K)
    print(f"  Samples per centroid: {n_per_centroid.tolist()}")

    # 5. Coupling mass per centroid (total Sinkhorn mass originating from each source centroid)
    coupling_mass_per_centroid = P_star.sum(dim=1).numpy()  # [K]

    # 6. Sample norms
    print("Computing per-sample norms and binning...")
    source_norms = source_acts.norm(dim=1).numpy()  # [N]

    # Bin by norm percentile
    percentiles = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100
    norm_percentiles = np.percentile(source_norms, percentiles)
    norm_bin_idx = np.digitize(source_norms, norm_percentiles[1:-1])  # 0=0-10%, 1=10-20%, ..., 9=90-100%

    print("\n=== Coupling Mass per Norm Decile ===")
    print(f"{'Decile':>10s}  {'N':>6s}  {'Norm range':>15s}  {'Avg coupling mass':>18s}  {'Expected (uniform)':>18s}")
    for i in range(10):
        mask = norm_bin_idx == i
        n = mask.sum()
        if n == 0:
            continue
        centroid_ids = assigned_centroid[mask]
        avg_mass = coupling_mass_per_centroid[centroid_ids].mean()
        uniform_mass = 1.0 / K
        norm_range = f"{norm_percentiles[i]:.0f}-{norm_percentiles[i+1]:.0f}"
        print(f"  {i*10:>3d}-{(i+1)*10:>3d}%  {n:>6d}  {norm_range:>15s}  {avg_mass:>18.4f}  {uniform_mass:>18.4f}")

    # 7. Spearman correlation: sample norm vs coupling mass of assigned centroid
    sample_mass = coupling_mass_per_centroid[assigned_centroid.numpy()]
    rho, p = spearmanr(source_norms, sample_mass)
    r_pearson, p_pearson = pearsonr(source_norms, sample_mass)
    print(f"\n  Spearman ρ(sample norm, coupling mass) = {rho:.4f}, p = {p:.6f}")
    print(f"  Pearson r(sample norm, coupling mass) = {r_pearson:.4f}, p = {p_pearson:.6f}")

    # 8. High-norm tail analysis: top 5% vs bottom 95%
    tail_threshold = np.percentile(source_norms, 95)
    tail_mask = source_norms >= tail_threshold
    body_mask = source_norms < tail_threshold
    tail_mass = sample_mass[tail_mask].mean()
    body_mass = sample_mass[body_mask].mean()
    print(f"\n  Top 5% tail N={tail_mask.sum()}: avg coupling mass = {tail_mass:.4f}")
    print(f"  Bottom 95% N={body_mask.sum()}: avg coupling mass = {body_mass:.4f}")
    print(f"  Tail/Body ratio = {tail_mass / body_mass:.4f}")

    # 9. Centroid norms vs coupling mass (centroid-level)
    centroid_norms = centroids_A.norm(dim=1).numpy()
    rho_c, p_c = spearmanr(centroid_norms, coupling_mass_per_centroid)
    print(f"\n  Centroid-level: Spearman ρ(norm, coupling) = {rho_c:.4f}, p = {p_c:.6f}")
    print(f"  Centroid norms: {centroid_norms}")
    print(f"  Centroid CV: {centroid_norms.std() / centroid_norms.mean():.4f}")

    return {
        "task": task_cfg["name"],
        "K": K,
        "centroid_norm_cv": float(centroid_norms.std() / centroid_norms.mean()),
        "spearman_norm_vs_mass_sample": float(rho),
        "spearman_p_sample": float(p),
        "pearson_norm_vs_mass_sample": float(r_pearson),
        "pearson_p_sample": float(p_pearson),
        "spearman_norm_vs_mass_centroid": float(rho_c),
        "spearman_p_centroid": float(p_c),
        "tail_5pct_avg_mass": float(tail_mass),
        "body_95pct_avg_mass": float(body_mass),
        "tail_body_ratio": float(tail_mass / body_mass),
        "n_samples": len(source_texts),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.bfloat16,
        default_padding_side="left",
    )
    model.to(DEVICE)
    print("Model loaded.")

    all_results = {}
    for task_cfg in TASKS:
        try:
            result = analyze_task(task_cfg, model)
            all_results[task_cfg["name"]] = result
        except Exception as e:
            print(f"ERROR on {task_cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path("Experiments/exp6_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'Task':>12s}  {'K':>3s}  {'CV':>6s}  {'ρ_sample':>8s}  {'Tail/Body':>10s}  {'ρ_centroid':>10s}")
    for name, r in all_results.items():
        print(f"  {name:>12s}  {r['K']:>3d}  {r['centroid_norm_cv']:.4f}  {r['spearman_norm_vs_mass_sample']:>8.4f}  {r['tail_body_ratio']:>10.4f}  {r['spearman_norm_vs_mass_centroid']:>10.4f}")


if __name__ == "__main__":
    main()
