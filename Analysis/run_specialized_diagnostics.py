import torch
import numpy as np
import json
import os
from pathlib import Path
from Steering.data.loader import DataLoader
from Steering.utils import collect_dense_activations, get_resid_acts, set_resid_acts
from transformer_lens import HookedTransformer

# Set CUDA device to 4 to avoid VRAM contention with the sweep on GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
HOOK_POINT = "pre"
POSITION = "last"
BATCH_SIZE = 4
N_SAMPLES = 100

def load_data():
    dataloader = DataLoader()
    samples = dataloader.load(
        "liarbench",
        n_samples=N_SAMPLES,
        format=True,
        apply_chat_template=True,
        tokenizer=None,
    )
    # We want to collect activations for the truthful prompts (source)
    source_texts = [s["correct_prompt"] for s in samples]
    return source_texts

def main():
    # 1. Load Model
    print("Loading pretrained Gemma model...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
        default_padding_side="left",
    )
    print("Model loaded successfully.")

    # 2. Load Data and Collect Base Activations
    source_texts = load_data()
    print(f"Collected {len(source_texts)} source prompts. Extracting base activations at Layer {LAYER}...")
    
    # Collect activations using project utilities
    base_acts_dict = collect_dense_activations(
        model=model,
        texts=source_texts,
        layers=[LAYER],
        hook_point=HOOK_POINT,
        batch_size=BATCH_SIZE,
        pooling=POSITION,
        device=device,
        tokenizer=model.tokenizer,
        reduce="none",
        return_key_format="layer",
    )
    base_acts = base_acts_dict[LAYER].to(device=device, dtype=torch.float32) # [N, d_model]
    print(f"Base activations shape: {base_acts.shape}")

    # 3. Load Vector Metadata for ACT, CHARS, and COBRA
    print("\nLoading vector metadata...")
    act_meta = torch.load("Vector/LinearAcT/Gemma/deception/metadata.pt", map_location=device, weights_only=False)
    chars_meta = torch.load("Vector/CHARS/Gemma/deception/metadata.pt", map_location=device, weights_only=False)
    cobra_meta = torch.load("Vector/COBRA/Gemma/deception/metadata.pt", map_location=device, weights_only=False)
    print("Metadata loaded successfully.")

    # 4. Apply ACT Steering (coeff=1.0)
    print("\nApplying ACT steering...")
    stats = act_meta["act_stats"][LAYER]
    omega = stats["omega"].to(device=device, dtype=torch.float32)
    beta = stats["beta"].to(device=device, dtype=torch.float32)
    act_steered = omega * base_acts + beta # transported
    # Update is simply the transported activations
    # Print norm comparison
    print(f"  ACT Steered norm mean: {act_steered.norm(dim=-1).mean().item():.3f}")

    # 5. Apply CHARS Steering (coeff=1.0)
    print("Applying CHARS steering...")
    c_A_chars = chars_meta["chars_centroids_A"][LAYER].to(device=device, dtype=torch.float32)
    c_B_chars = chars_meta["chars_centroids_B"][LAYER].to(device=device, dtype=torch.float32)
    coupling_chars = chars_meta["chars_coupling"][LAYER].to(device=device, dtype=torch.float32)
    K_chars = chars_meta["chars_k"][LAYER]
    
    # CHARS algorithm
    if K_chars == 1:
        v_hat_chars = c_B_chars[0] - c_A_chars[0]
        v_hat_chars = v_hat_chars.unsqueeze(0).expand(base_acts.shape[0], -1)
    else:
        dists = torch.sum((base_acts.unsqueeze(1) - c_A_chars.unsqueeze(0)) ** 2, dim=-1) # [N, K]
        dists_median = torch.median(dists, dim=-1).values.clamp(min=1e-8) # [N]
        kernel = torch.exp(-dists / (2.0 * dists_median.unsqueeze(-1))) # [N, K]
        diffs = c_B_chars.unsqueeze(0) - c_A_chars.unsqueeze(1) # [K, K, d]
        u = torch.einsum("ij,ijd->id", coupling_chars, diffs) # [K, d]
        s = coupling_chars.sum(dim=-1) # [K]
        denom = torch.matmul(kernel, s) # [N]
        num = torch.matmul(kernel, u) # [N, d]
        v_hat_chars = num / (denom.unsqueeze(-1) + 1e-12) # [N, d]
        
    chars_steered = base_acts + v_hat_chars
    print(f"  CHARS Steered norm mean: {chars_steered.norm(dim=-1).mean().item():.3f}")

    # 6. Apply COBRA Steering (coeff=1.0)
    print("Applying COBRA steering...")
    c_A_cobra = cobra_meta["cobra_centroids_A"][LAYER].to(device=device, dtype=torch.float32)
    c_B_cobra = cobra_meta["cobra_centroids_B"][LAYER].to(device=device, dtype=torch.float32)
    coupling_cobra = cobra_meta["cobra_coupling"][LAYER].to(device=device, dtype=torch.float32)
    K_cobra = cobra_meta["cobra_k"][LAYER]
    P_concept = cobra_meta["cobra_P_concept"][LAYER].to(device=device, dtype=torch.float32) # [d, r]
    
    # Project to concept subspace
    z = base_acts @ P_concept # [N, r]
    z_lang = base_acts - z @ P_concept.T # [N, d]
    
    # Barycentric shift in concept space
    if K_cobra == 1:
        v_concept = (c_B_cobra[0] - c_A_cobra[0]).unsqueeze(0)
    else:
        dists = torch.cdist(z, c_A_cobra, p=2) ** 2
        dists_median = torch.median(dists, dim=-1).values.clamp(min=1e-8)
        kernel = torch.exp(-dists / (2.0 * dists_median.unsqueeze(-1)))
        diffs = c_B_cobra.unsqueeze(0) - c_A_cobra.unsqueeze(1)
        u = torch.einsum("ij,ijd->id", coupling_cobra, diffs)
        s = coupling_cobra.sum(dim=-1)
        denom = torch.matmul(kernel, s)
        num = torch.matmul(kernel, u)
        v_concept = num / (denom.unsqueeze(-1) + 1e-12)
        
    z_steered = z + v_concept
    cobra_steered = z_steered @ P_concept.T + z_lang
    print(f"  COBRA Steered norm mean: {cobra_steered.norm(dim=-1).mean().item():.3f}")

    # =========================================================================
    # TEST 2.1: Covariance Distortion Test
    # =========================================================================
    print("\n--- Test 2.1: Covariance Distortion Test ---")
    
    # Baseline Covariance
    acts_centered = base_acts - base_acts.mean(dim=0, keepdim=True)
    cov_base = torch.matmul(acts_centered.T, acts_centered) / (base_acts.shape[0] - 1)
    
    # Steered Covariances
    cov_distortions = {}
    for name, steered in [("ACT", act_steered), ("CHARS", chars_steered), ("COBRA", cobra_steered)]:
        steered_centered = steered - steered.mean(dim=0, keepdim=True)
        cov_steered = torch.matmul(steered_centered.T, steered_centered) / (steered.shape[0] - 1)
        distortion = torch.norm(cov_steered - cov_base, p="fro").item()
        cov_distortions[name] = distortion
        print(f"  {name:6} Covariance Distortion (Frobenius): {distortion:.4f}")

    # =========================================================================
    # TEST 2.2: Subspace Leakage Test
    # =========================================================================
    print("\n--- Test 2.2: Subspace Leakage Test ---")
    
    for name, steered in [("ACT", act_steered), ("CHARS", chars_steered), ("COBRA", cobra_steered)]:
        # Update vector delta = steered - base
        delta = steered - base_acts # [N, d]
        # Project delta to concept subspace P_concept
        delta_proj = (delta @ P_concept) @ P_concept.T
        # Orthogonal component (leakage)
        delta_orth = delta - delta_proj
        
        leakage = (torch.norm(delta_orth, p="fro") / (torch.norm(delta, p="fro") + 1e-8)).item()
        print(f"  {name:6} Subspace Leakage (Orthogonal energy fraction): {leakage * 100:.2f}%")

    # =========================================================================
    # TEST 2.3: Barycentric Cancellation Test in CHARS
    # =========================================================================
    print("\n--- Test 2.3: Barycentric Cancellation Test (CHARS vs COBRA) ---")
    # For CHARS
    diffs_chars = c_B_chars.unsqueeze(0) - c_A_chars.unsqueeze(1) # [K_src, K_dst, d_model]
    u_chars = torch.einsum("ij,ijd->id", coupling_chars, diffs_chars) # [K_src, d_model]
    
    # Calculate pairwise cosine similarities of cluster displacement vectors (b_j - a_i)
    # to see if different transport paths are opposing
    # Let's inspect the coupling matrix and average similarities
    flat_diffs = diffs_chars.reshape(-1, base_acts.shape[-1])
    norm_flat = flat_diffs / (flat_diffs.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(norm_flat, norm_flat.T)
    # Average off-diagonal similarity of transport paths
    off_diag = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=torch.bool)]
    mean_sim = off_diag.mean().item()
    print(f"  CHARS Pairwise Transport Paths Similarity: {mean_sim:.3f}")
    
    # Norm collapse comparison: ||sum_j P_ij * (b_j - a_i)|| vs sum_j P_ij * ||b_j - a_i||
    norms_individual = diffs_chars.norm(dim=-1) # [K_src, K_dst]
    weighted_individual = torch.sum(coupling_chars * norms_individual, dim=-1) # [K_src]
    norm_sum = u_chars.norm(dim=-1) # [K_src]
    cancellation_ratio = (norm_sum / (weighted_individual + 1e-8)).mean().item()
    print(f"  CHARS Barycentric Cancellation Ratio: {cancellation_ratio:.3f} (Lower = more cancellation)")
    
    # For COBRA
    diffs_cobra = c_B_cobra.unsqueeze(0) - c_A_cobra.unsqueeze(1) # [K_src, K_dst, r]
    u_cobra = torch.einsum("ij,ijd->id", coupling_cobra, diffs_cobra) # [K_src, r]
    norms_ind_cobra = diffs_cobra.norm(dim=-1)
    weighted_ind_cobra = torch.sum(coupling_cobra * norms_ind_cobra, dim=-1)
    norm_sum_cobra = u_cobra.norm(dim=-1)
    cancellation_ratio_cobra = (norm_sum_cobra / (weighted_ind_cobra + 1e-8)).mean().item()
    print(f"  COBRA Barycentric Cancellation Ratio: {cancellation_ratio_cobra:.3f}")

if __name__ == "__main__":
    main()
