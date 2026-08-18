"""
CorrSteer Level 1: Feature Extraction Validation.

Two-part validation:

Part A — Accumulator Parity (shared activations):
    Feed the SAME (activation_batch, reward_batch) to GT and our
    StreamingCorrelationAccumulator. Verifies mathematical equivalence
    of Pearson correlation, top-K ranking, and mean coefficient computation.
    Expected: 20/20 feature match, correlation max_diff < 1e-10.

Part B — End-to-End Pipeline (TL vs HF):
    Run our CorrSteerExtractor (TransformerLens) on MMLU data and compare
    against pre-computed GT results (Code/CorrSteer/train.py, HuggingFace).
    Expected: ~17/20 overlap at n=4000 due to TL-vs-HF numerical differences.

GT reference: Code/CorrSteer/train.py StreamingCorrelationAccumulator
Our code:     Steering/extractors/sae.py StreamingCorrelationAccumulator, CorrSteerExtractor

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level1/CorrSteer/validate_corrsteer_l1.py [--n_samples N]
"""

import sys
import json
import gc
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "CorrSteer"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
LAYER = 13
TOP_K = 20


# ============================================================================
# Part A: Accumulator parity on shared data
# ============================================================================

def test_accumulator_parity(n_samples: int = 50):
    """
    Feed identical (activations, rewards) to GT and our accumulator.
    Verifies the math is identical — no model/framework differences.
    """
    print("=" * 60)
    print("PART A: Accumulator Parity (Shared Activations)")
    print("=" * 60)

    # --- Import GT and our accumulators ---
    from train import StreamingCorrelationAccumulator as GTAccumulator
    from Steering.extractors.sae import StreamingCorrelationAccumulator as OurAccumulator

    # --- Load model + SAE ---
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"  Loading google/gemma-2-2b-it on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b-it", device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(DEVICE)
    d_sae = sae.cfg.d_sae
    print(f"  d_sae: {d_sae}")

    # --- Load MMLU data in GT format ---
    mmlu_path = ROOT / "TrainDataset" / "mmlu" / "mmlu_hf_shuffled.json"
    with open(mmlu_path) as f:
        mmlu_data = json.load(f)[:n_samples]

    from corrsteer.utils import build_prompt
    prompts_gts = [build_prompt(s, "mmlu", cot=False, few_shots=None) for s in mmlu_data]
    prompts = [p for p, _ in prompts_gts]
    gt_answers = [g for _, g in prompts_gts]

    print(f"  Loaded {len(prompts)} MMLU prompts")

    # --- Capture SAE activations via generation ---
    all_pooled = []
    all_rewards = []
    batch_size = 4  # Keep small to fit in constrained GPU memory

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gt = gt_answers[i:i + batch_size]

        model.tokenizer.padding_side = "left"
        input_tokens = model.to_tokens(batch_prompts)

        # Hook to capture residual pre → SAE encode at last token per step
        activation_buffer = [None]

        def capture_hook(module, args):
            residual = args[0]
            tokens = residual[:, -1:, :]
            encoded = sae.encode(tokens.to(sae.device, sae.dtype))
            act = encoded.view(residual.shape[0], 1, -1).detach().cpu()
            if activation_buffer[0] is None:
                activation_buffer[0] = act
            else:
                activation_buffer[0] = torch.cat([activation_buffer[0], act], dim=1)

        handle = model.blocks[LAYER].register_forward_pre_hook(capture_hook)
        with torch.no_grad():
            logits = model.forward(input_tokens, return_type="logits")
        handle.remove()

        # Restricted logit argmax for MMLU
        option_tokens = [
            model.tokenizer.encode(opt, add_special_tokens=False)[0]
            for opt in [" A", " B", " C", " D"]
        ]
        restricted = logits[:, -1, :][:, option_tokens]
        predicted_idx = restricted.argmax(dim=-1)
        options = ["A", "B", "C", "D"]
        gen_texts = [options[idx.item()] for idx in predicted_idx]

        rewards = [1.0 if pred.strip() == gold else 0.0
                   for pred, gold in zip(gen_texts, batch_gt)]

        # Pool activations (max over time dim)
        acts = activation_buffer[0]  # [B, T, d_sae]
        pooled = acts.max(dim=1).values  # [B, d_sae]

        all_pooled.append(pooled)
        all_rewards.extend(rewards)
        torch.cuda.empty_cache()

    all_pooled = torch.cat(all_pooled, dim=0)
    all_rewards_t = torch.tensor(all_rewards, dtype=torch.float32)

    accuracy = sum(all_rewards) / len(all_rewards)
    print(f"  Accuracy: {accuracy:.2%} ({int(sum(all_rewards))}/{len(all_rewards)})")
    print(f"  Pooled shape: {all_pooled.shape}")

    # --- Feed to BOTH accumulators in identical batches ---
    gt_acc = GTAccumulator(d_sae, real=False, pos_only=True)
    our_acc = OurAccumulator(d_sae, real=False)

    feed_batch = 32
    for i in range(0, len(all_pooled), feed_batch):
        bx = all_pooled[i:i + feed_batch]
        by = all_rewards_t[i:i + feed_batch]
        gt_acc.update_corr(bx, by)
        gt_acc.update_coeff(bx, by)
        our_acc.update_corr(bx, by)
        our_acc.update_coeff(bx, by)

    # --- Compare correlations ---
    gt_corr = gt_acc.correlations()
    our_corr = our_acc.correlations()

    corr_diff = (gt_corr - our_corr).abs().max().item()
    corr_cos = F.cosine_similarity(
        gt_corr.unsqueeze(0).float(), our_corr.unsqueeze(0).float()
    ).item()

    print(f"\n  Correlation max diff:  {corr_diff:.2e}")
    print(f"  Correlation cosine:    {corr_cos:.10f}")
    corr_pass = corr_diff < 1e-10
    print(f"  {'PASSED' if corr_pass else 'FAILED'}: Correlations match (tol 1e-10)")

    # --- Compare top-K features ---
    # GT: top_features with pos_only=True → returns (top_positive, top_negative)
    gt_top_pos, _ = gt_acc.top_features(TOP_K)
    gt_set = set(t[0] for t in gt_top_pos)

    # Ours: top_features_signed
    our_top = our_acc.top_features_signed(k=TOP_K, pos_only=True)
    our_set = set(d["feature_index"] for d in our_top)

    overlap = len(gt_set & our_set)
    print(f"\n  GT top-{TOP_K}:  {sorted(gt_set)}")
    print(f"  Our top-{TOP_K}: {sorted(our_set)}")
    print(f"  Overlap: {overlap}/{TOP_K}")
    topk_pass = overlap == TOP_K
    print(f"  {'PASSED' if topk_pass else 'FAILED'}: Top-K match")

    # --- Compare correlation values for shared features ---
    shared = sorted(gt_set & our_set)
    if shared:
        gt_corr_vals = torch.tensor([float(gt_corr[i]) for i in shared])
        our_corr_vals = torch.tensor([float(our_corr[i]) for i in shared])
        val_diff = (gt_corr_vals - our_corr_vals).abs().max().item()
        print(f"  Shared feature corr max diff: {val_diff:.2e}")

    # --- Compare mean coefficients ---
    if shared:
        gt_coeffs = torch.tensor([gt_acc.mean_coefficient(i) for i in shared])
        our_coeffs = torch.tensor([our_acc.mean_coefficient(i) for i in shared])
        coeff_diff = (gt_coeffs - our_coeffs).abs().max().item()
        print(f"  Mean coefficient max diff:    {coeff_diff:.2e}")
        coeff_pass = coeff_diff < 1e-8
        print(f"  {'PASSED' if coeff_pass else 'FAILED'}: Coefficients match (tol 1e-8)")
    else:
        coeff_pass = False

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()
    return corr_pass, topk_pass, coeff_pass


# ============================================================================
# Part B: End-to-end pipeline comparison (TL vs HF)
# ============================================================================

def test_pipeline_comparison(n_samples: int = 50):
    """
    Compare features from our CorrSteerExtractor (TransformerLens)
    against pre-computed GT results (HuggingFace) at matching sample size.
    """
    print("\n" + "=" * 60)
    print("PART B: End-to-End Pipeline (TL vs HF)")
    print("=" * 60)

    # --- Find GT results ---
    gt_candidates = [
        ROOT / "Results" / f"l3_corrsteer_gt_{n_samples}_42",
        ROOT / "Results" / "l3_corrsteer_gt_4000_42",
        ROOT / "Results" / "l3_corrsteer_gt_42",
    ]

    gt_dir = None
    for candidate in gt_candidates:
        gt_json = candidate / f"gemma2b_mmlu_{LAYER}_corr.json"
        if gt_json.exists():
            gt_dir = candidate
            break

    if gt_dir is None:
        print("  SKIPPED: No GT results found.")
        print("  Run: cd Code/CorrSteer && python train.py train --model=gemma2b "
              f"--task=mmlu --layer={LAYER} --num_samples={n_samples} "
              f"--topk={TOP_K} --scale=1.0 --pool=max --pos=True --batch_size=8 "
              f"--output_dir=../../Results/l3_corrsteer_gt_{n_samples}_42 --seed=42")
        return None

    gt_json = gt_dir / f"gemma2b_mmlu_{LAYER}_corr.json"
    with open(gt_json) as f:
        gt_data = json.load(f)

    gt_n = gt_data["samples"]
    gt_features = [f["feature_index"]
                   for f in gt_data["results"][str(LAYER)]["top_positive"]]
    gt_corrs = {f["feature_index"]: f["correlation"]
                for f in gt_data["results"][str(LAYER)]["top_positive"]}
    gt_coeffs = {f["feature_index"]: f["coefficient"]
                 for f in gt_data["results"][str(LAYER)]["top_positive"]}

    print(f"  GT source: {gt_dir.name} (n={gt_n})")
    print(f"  GT features: {gt_features}")

    # --- Run our extractor ---
    from Steering.extractors.sae import CorrSteerExtractor
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    print(f"\n  Loading google/gemma-2-2b-it on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b-it", device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(DEVICE)

    # Load MMLU data with our formatter
    mmlu_path = ROOT / "TrainDataset" / "mmlu" / "mmlu_hf_shuffled.json"
    with open(mmlu_path) as f:
        raw_data = json.load(f)

    from Steering.data.formatters import mmlu_corrsteer
    formatted = mmlu_corrsteer(raw_data[:gt_n])

    print(f"  Running CorrSteerExtractor (n={gt_n})...")
    extractor = CorrSteerExtractor(
        model=model,
        sae={LAYER: sae},
        layer=[LAYER],
        batch_size=2,  # Minimal batch to fit constrained GPU memory
        top_k=TOP_K,
        corrsteer_max_new_tokens=1,
        corrsteer_pool="max",
        corrsteer_steer_pool="max",
        corrsteer_pos_only=True,
        corrsteer_selection="correlation",
        hook_point=["pre"],
    )
    vectors = extractor.extract(target_data=formatted)

    our_features = [f["feature_index"] for f in extractor.selected_features]
    our_corrs = {f["feature_index"]: f["correlation"]
                 for f in extractor.selected_features}
    our_coeffs = {f["feature_index"]: f["coefficient"]
                  for f in extractor.selected_features}

    print(f"  Our features: {our_features}")

    # --- Compare ---
    gt_set = set(gt_features)
    our_set = set(our_features)
    overlap = gt_set & our_set
    print(f"\n  Feature overlap: {len(overlap)}/{TOP_K}")
    print(f"  Common: {sorted(overlap)}")
    print(f"  GT only: {sorted(gt_set - our_set)}")
    print(f"  Our only: {sorted(our_set - gt_set)}")

    # Per-feature value comparison
    if overlap:
        print(f"\n  Per-feature comparison (shared):")
        print(f"  {'Feature':>8}  {'GT corr':>10}  {'Our corr':>10}  "
              f"{'corr diff':>10}  {'GT coeff':>10}  {'Our coeff':>10}")
        for idx in sorted(overlap):
            cdiff = abs(gt_corrs[idx] - our_corrs[idx])
            print(f"  {idx:>8}  {gt_corrs[idx]:>10.6f}  {our_corrs[idx]:>10.6f}  "
                  f"{cdiff:>10.2e}  {gt_coeffs[idx]:>10.4f}  {our_coeffs[idx]:>10.4f}")

    # Pass criteria: ~17/20 at n=4000, lower threshold at small n
    if gt_n >= 4000:
        threshold = 17
    elif gt_n >= 500:
        threshold = 15
    else:
        threshold = 15
    pipeline_pass = len(overlap) >= threshold
    print(f"\n  {'PASSED' if pipeline_pass else 'FAILED'}: "
          f"Feature overlap {len(overlap)}/{TOP_K} >= {threshold}")

    del model, sae
    torch.cuda.empty_cache()
    return pipeline_pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CorrSteer L1 Validation")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of MMLU samples (default: 50)")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    device = args.device
    # Update the module-level DEVICE so test functions use it
    globals()["DEVICE"] = device

    print("CorrSteer Level 1 Validation")
    print("=" * 60)
    print(f"Config: LAYER={LAYER}, TOP_K={TOP_K}, N_SAMPLES={args.n_samples}")
    print(f"Device: {device}\n")

    corr_pass, topk_pass, coeff_pass = test_accumulator_parity(args.n_samples)
    pipeline_pass = test_pipeline_comparison(args.n_samples)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "A1. Correlation math match": corr_pass,
        "A2. Top-K feature match (shared data)": topk_pass,
        "A3. Mean coefficient match": coeff_pass,
    }
    if pipeline_pass is not None:
        results["B1. Pipeline feature overlap (TL vs HF)"] = pipeline_pass

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
