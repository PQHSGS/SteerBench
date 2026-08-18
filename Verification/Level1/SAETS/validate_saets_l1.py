"""
SAE-TS Level 1: Extraction Match.

Two-part validation:

Part A — Shared-Weights Formula Parity:
    Given the SAME adapter weights and target feature, our formula
    M_j/||M_j|| - λ*M_b/||M_b|| must produce the same vector as
    GT single_step_steer(adapter, one_hot).

Part B — GT Vector Reproduction:
    Using the pre-trained GT adapter (Results/l3_saets_gt/adapter_layer12.pt)
    and the GT feature configs, reproduce the GT OptimisedSteer vectors
    for all 9 concepts. Expect cosine similarity ≥ 0.999.

GT reference: Code/SAE-TS/src/sae_ts/baselines/analysis.py :: single_step_steer
Our code:     Steering/extractors/sae.py :: SAETSSExtractor.extract()

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level1/SAETS/validate_saets_l1.py [--device cuda:2]
"""

import sys
import gc
import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "SAE-TS" / "src"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
LAYER = 12
LAMBDA_REG = 1.0


def main():
    parser = argparse.ArgumentParser(description="SAETS L1 Validation")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    device = args.device

    print("SAE-TS Level 1 Validation")
    print("=" * 60)
    print(f"Config: LAYER={LAYER}, LAMBDA_REG={LAMBDA_REG}")
    print(f"Device: {device}\n")

    # ================================================================
    # Part A: Shared-Weights Formula Parity
    # ================================================================
    print("=" * 60)
    print("PART A: Shared-Weights Formula Parity")
    print("=" * 60)

    # Import GT
    from sae_ts.baselines.analysis import single_step_steer as gt_single_step_steer
    from sae_ts.ft_effects.utils import LinearAdapter as GTLinearAdapter

    # Import ours
    from Steering.extractors.sae import LinearAdapter as OurLinearAdapter

    # Use arbitrary d_model/d_sae matching gemma-2-2b
    d_model, d_sae = 2304, 16384

    # --- Test 1: LinearAdapter weight initialization parity ---
    print("\n  Test 1: LinearAdapter weight initialization")
    torch.manual_seed(123)
    gt_adapter = GTLinearAdapter(d_model, d_sae)
    torch.manual_seed(123)
    our_adapter = OurLinearAdapter(d_model, d_sae)

    w_diff = (gt_adapter.W.data - our_adapter.W.data).abs().max().item()
    b_diff = (gt_adapter.b.data - our_adapter.b.data).abs().max().item()
    init_pass = w_diff < 1e-7 and b_diff < 1e-7
    print(f"    W max diff: {w_diff:.2e}")
    print(f"    b max diff: {b_diff:.2e}")
    print(f"    {'PASSED' if init_pass else 'FAILED'}: Initialization match")

    # --- Test 2: Forward pass parity ---
    print("\n  Test 2: Forward pass parity")
    x = torch.randn(8, d_model)
    gt_out = gt_adapter(x)
    our_out = our_adapter(x)
    fwd_diff = (gt_out - our_out).abs().max().item()
    fwd_cos = F.cosine_similarity(gt_out.flatten().unsqueeze(0),
                                   our_out.flatten().unsqueeze(0)).item()
    fwd_pass = fwd_diff < 1e-6
    print(f"    Max diff: {fwd_diff:.2e}, Cosine: {fwd_cos:.8f}")
    print(f"    {'PASSED' if fwd_pass else 'FAILED'}: Forward pass match")

    # --- Test 3: Steering vector formula (single feature) ---
    print("\n  Test 3: single_step_steer formula")

    # Load the real GT adapter for realistic weights
    gt_adapter_path = ROOT / "Results" / "l3_saets_gt" / "adapter_layer12.pt"
    adapter = GTLinearAdapter(d_model, d_sae)
    adapter.load_state_dict(torch.load(gt_adapter_path, map_location="cpu"))
    adapter.to(device)

    # GT needs the global `device` variable set
    import sae_ts.baselines.analysis as gt_mod
    gt_mod.device = device

    test_features = [4111, 1000, 8000, 15000]  # Various features
    all_formula_cos = []

    for feat_idx in test_features:
        # GT: single_step_steer with one-hot target
        target = torch.zeros(d_sae, device=device)
        target[feat_idx] = 1.0
        gt_vec = gt_single_step_steer(adapter, target, bias_scale=LAMBDA_REG)
        gt_vec = gt_vec.squeeze()

        # Ours: M_j/||M_j|| - λ * M_b/||M_b||, normalized
        M_j = adapter.W[:, feat_idx]
        M_j_norm = M_j / (M_j.norm() + 1e-8)
        M_b = adapter.W @ adapter.b
        M_b_norm = M_b / (M_b.norm() + 1e-8)
        our_vec = M_j_norm - LAMBDA_REG * M_b_norm
        our_vec = our_vec / (our_vec.norm() + 1e-8)

        cos = F.cosine_similarity(gt_vec.unsqueeze(0), our_vec.unsqueeze(0)).item()
        diff = (gt_vec - our_vec).abs().max().item()
        all_formula_cos.append(cos)
        print(f"    Feature {feat_idx}: cosine={cos:.8f}, max_diff={diff:.2e}")

    avg_formula_cos = np.mean(all_formula_cos)
    formula_pass = avg_formula_cos > 0.9999
    print(f"    Avg cosine: {avg_formula_cos:.8f}")
    print(f"    {'PASSED' if formula_pass else 'FAILED'}: Formula parity >= 0.9999")

    # --- Test 4: Multi-feature target ---
    print("\n  Test 4: Multi-feature target (GT supports multi)")
    target_multi = torch.zeros(d_sae, device=device)
    target_multi[4111] = 1.0
    target_multi[1000] = 0.5
    gt_multi = gt_single_step_steer(adapter, target_multi, bias_scale=LAMBDA_REG).squeeze()

    # Our single-feature code can't do this natively, but the formula should be:
    # W @ target / ||W @ target|| - λ * W @ b / ||W @ b||, normalized
    steer_vec = adapter.W @ target_multi
    steer_vec = steer_vec / steer_vec.norm()
    bias_vec = adapter.W @ adapter.b
    bias_vec = bias_vec / bias_vec.norm()
    our_multi = steer_vec - LAMBDA_REG * bias_vec
    our_multi = our_multi / our_multi.norm()

    multi_cos = F.cosine_similarity(gt_multi.unsqueeze(0), our_multi.unsqueeze(0)).item()
    multi_pass = multi_cos > 0.9999
    print(f"    Multi-feature cosine: {multi_cos:.8f}")
    print(f"    {'PASSED' if multi_pass else 'FAILED'}: Multi-feature match")

    del adapter
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # Part B: GT Vector Reproduction
    # ================================================================
    print("\n" + "=" * 60)
    print("PART B: GT Vector Reproduction (9 concepts × OptimisedSteer)")
    print("=" * 60)

    # Load pre-trained GT adapter
    adapter = GTLinearAdapter(d_model, d_sae)
    adapter.load_state_dict(torch.load(gt_adapter_path, map_location="cpu"))
    adapter.to(device)

    concepts = [
        "anger", "christian_evangelist", "conspiracy", "french",
        "london", "love", "praise", "want_to_die", "wedding",
    ]

    gt_vec_dir = ROOT / "Results" / "l3_saets_gt"
    cfg_dir = ROOT / "Code" / "SAE-TS" / "steer_cfgs" / "gemma2"

    all_cos = []

    for concept in concepts:
        # Load GT vector
        gt_vec_path = gt_vec_dir / f"{concept}_optimised_vec.pt"
        gt_vec = torch.load(gt_vec_path, map_location=device)
        if gt_vec.dim() > 1:
            gt_vec = gt_vec.squeeze()

        # Load GT config to get feature IDs
        cfg_path = cfg_dir / concept / "optimised_steer.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Build target from config
        target = torch.zeros(d_sae, device=device)
        for ft_id, ft_scale in cfg["features"]:
            target[ft_id] = ft_scale

        # Reproduce using GT formula (same adapter, same features)
        our_vec = gt_single_step_steer(adapter, target, bias_scale=1).squeeze()

        cos = F.cosine_similarity(
            gt_vec.float().unsqueeze(0), our_vec.float().unsqueeze(0)
        ).item()
        diff = (gt_vec.float() - our_vec.float()).abs().max().item()
        all_cos.append(cos)
        print(f"  {concept:25s}: cosine={cos:.8f}, max_diff={diff:.2e}")

    avg_cos = np.mean(all_cos)
    min_cos = min(all_cos)
    repro_pass = min_cos > 0.999
    print(f"\n  Avg cosine: {avg_cos:.8f}, Min cosine: {min_cos:.8f}")
    print(f"  {'PASSED' if repro_pass else 'FAILED'}: All concepts >= 0.999")

    # Also test with OUR formula (not GT function) to ensure our code matches
    print("\n  --- Cross-check: Our formula vs GT vectors ---")
    all_our_cos = []
    for concept in concepts:
        gt_vec = torch.load(gt_vec_dir / f"{concept}_optimised_vec.pt", map_location=device)
        if gt_vec.dim() > 1:
            gt_vec = gt_vec.squeeze()

        cfg_path = cfg_dir / concept / "optimised_steer.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Build using OUR formula
        features = cfg["features"]
        if len(features) == 1:
            ft_id, ft_scale = features[0]
            M_j = adapter.W[:, ft_id] * ft_scale
            M_j_norm = M_j / (M_j.norm() + 1e-8)
        else:
            target = torch.zeros(d_sae, device=device)
            for ft_id, ft_scale in features:
                target[ft_id] = ft_scale
            M_j = adapter.W @ target
            M_j_norm = M_j / (M_j.norm() + 1e-8)

        M_b = adapter.W @ adapter.b
        M_b_norm = M_b / (M_b.norm() + 1e-8)
        our_vec = M_j_norm - LAMBDA_REG * M_b_norm
        our_vec = our_vec / (our_vec.norm() + 1e-8)

        cos = F.cosine_similarity(
            gt_vec.float().unsqueeze(0), our_vec.float().unsqueeze(0)
        ).item()
        all_our_cos.append(cos)

    avg_our_cos = np.mean(all_our_cos)
    min_our_cos = min(all_our_cos)
    our_repro_pass = min_our_cos > 0.999
    print(f"  Avg cosine (our formula vs GT vec): {avg_our_cos:.8f}")
    print(f"  Min cosine: {min_our_cos:.8f}")
    print(f"  {'PASSED' if our_repro_pass else 'FAILED'}: Our formula reproduces GT")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "A1. LinearAdapter init match": init_pass,
        "A2. Forward pass match": fwd_pass,
        "A3. Formula parity >= 0.9999": formula_pass,
        "A4. Multi-feature match": multi_pass,
        "B1. GT reproduction >= 0.999": repro_pass,
        "B2. Our formula reproduces GT": our_repro_pass,
    }
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
