"""
SSV Level 2: Steering Inference Match using REAL extracted vector.

Validates that our SSVSteerModel hook produces the same logits as the GT hook
when using the SAME vector on the SAME prompt.

Steps:
1. Load model + SAE
2. Extract a real SSV from sycophancy data using our extractor
3. Apply GT hook (encode → add SSV*scale → decode) with that vector
4. Apply our SSVSteerModel hook with that vector
5. Compare output logits

GT reference: Code/SAE-SSV/saessv-demo.py lines 1182-1197
"""

import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

torch.manual_seed(42)
np.random.seed(42)

from Verification.shared_utils import load_sycophancy_data, load_model_and_sae

DEVICE = "cuda:2"
LAYER = 20
COEFF = 5.0


def main():
    print("=" * 60)
    print("SSV Level 2: Steering Inference Match (Real Vector)")
    print("=" * 60)
    
    # 1. Load model + SAE
    print("\n[1] Loading model + SAE...")
    model, sae, layer = load_model_and_sae(layer=LAYER, device=DEVICE)
    
    # 2. Extract a real SSV vector using our extractor
    print("\n[2] Extracting real SSV vector...")
    target_texts, contrast_texts = load_sycophancy_data(n_per_class=50)
    
    from Steering.extractors.sae import SSVExtractor
    extractor = SSVExtractor(
        model=model,
        sae={layer: sae},
        layer=[layer],
        hook_point=["post"],
        batch_size=8,
        ssv_opt_steps=5,
        ssv_feature_refinement_k=10,
    )
    
    vectors = extractor.extract(target_data=target_texts, contrast_data=contrast_texts)
    ssv_tensor = vectors[layer].float().cpu()
    
    print(f"    SSV shape: {ssv_tensor.shape}, norm: {ssv_tensor.norm():.4f}")
    print(f"    Non-zero dims: {(ssv_tensor.abs() > 1e-8).sum().item()}")
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    prompt = "I believe the government"
    input_ids = model.to_tokens(prompt)
    
    # 4. GT hook (exact logic from saessv-demo.py:1182-1197, but using same
    #    GPU SAE to isolate algorithmic differences from dtype differences)
    print("\n[3] Running GT hook...")
    ssv_device = sae.device if hasattr(sae, 'device') else DEVICE
    ssv_gpu = ssv_tensor.to(device=ssv_device, dtype=torch.bfloat16)

    def gt_modify_activation(act, hook):
        last_token_act = act[0, -1, :].clone()
        latent = sae.encode(last_token_act.unsqueeze(0)).squeeze(0)
        steered_latent = latent + ssv_gpu * COEFF
        steered_act = sae.decode(steered_latent.unsqueeze(0)).squeeze(0)
        act[0, -1, :] = steered_act
        return act
    
    with torch.no_grad():
        logits_gt = model.run_with_hooks(
            input_ids, fwd_hooks=[(hook_name, gt_modify_activation)]
        )[0, -1, :].clone()
    model.reset_hooks()
    
    # 5. Our SSVSteerModel hook
    print("[4] Running our SSVSteerModel hook...")
    from Steering.steer_models.sae import SSVSteerModel
    
    steer_model = SSVSteerModel(
        model=model,
        layer=[layer],
        sae={layer: sae},
        steering_vector={layer: ssv_tensor},
    )
    steer_model.setup_hooks(coeff={layer: COEFF})
    
    with torch.no_grad():
        logits_ours = model(input_ids)[0, -1, :].clone()
    model.reset_hooks()
    
    # 6. Compare
    print("\n" + "=" * 60)
    print("=== Logit Comparison ===")
    print("=" * 60)
    
    probs_gt = F.softmax(logits_gt.float(), dim=-1)
    probs_ours = F.softmax(logits_ours.float(), dim=-1)
    
    max_diff = (probs_gt - probs_ours).abs().max().item()
    cos_sim = F.cosine_similarity(probs_gt.unsqueeze(0), probs_ours.unsqueeze(0)).item()
    
    topk_gt = torch.topk(probs_gt, 5)
    topk_ours = torch.topk(probs_ours, 5)
    
    print(f"\n  GT Top 5:")
    for idx, p in zip(topk_gt.indices, topk_gt.values):
        print(f"    {model.to_string(idx)}: {p.item():.6f}")
    
    print(f"  Our Top 5:")
    for idx, p in zip(topk_ours.indices, topk_ours.values):
        print(f"    {model.to_string(idx)}: {p.item():.6f}")
    
    print(f"\n  Max Prob Diff: {max_diff:.2e}")
    print(f"  Cosine Similarity: {cos_sim:.8f}")
    
    passed = cos_sim > 0.99
    print(f"  {'PASSED' if passed else 'FAILED'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {status}: SSV steering hook match (cosine > 0.99)")
    print(f"\n{'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    
    del sae_float32, model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
