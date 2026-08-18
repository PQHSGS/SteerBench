"""
SAEIO Level 2: Steering Inference Match.

Two-part validation:

Part A — Hook Parity (same model, same feature):
    GT AmlifySAEHook (wrapped for TL) vs our SAEIOSteerModel.
    Expected: logit cosine > 0.95, delta cosine > 0.90

Part B — Functional Steering:
    Verify steering actually changes model output as expected.

GT reference: Code/saes-are-good-for-steering/src/sae_utils.py :: AmlifySAEHook
Our class:    Steering/steer_models/sae.py :: SAEIOSteerModel

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level2/SAEIO/validate_saeio_l2.py [--device cuda:2]
"""

import sys
import gc
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "saes-are-good-for-steering" / "src"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
LAYER = 20
AMP_FACTOR = 10.0


def main():
    parser = argparse.ArgumentParser(description="SAEIO L2 Validation")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    device = args.device

    print("SAEIO Level 2 Validation")
    print("=" * 60)
    print(f"Config: LAYER={LAYER}, AMP_FACTOR={AMP_FACTOR}")
    print(f"Device: {device}\n")

    from sae_utils import AmlifySAEHook as GTAmlifySAEHook
    from Steering.steer_models.sae import SAEIOSteerModel
    from Steering.utils import collect_sae_activations
    from Verification.shared_utils import load_sycophancy_data

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print("Loading model + SAE...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device=device, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(device)
    d_sae = sae.cfg.d_sae

    # Find active feature from real data
    target_texts, _ = load_sycophancy_data(n_per_class=20)
    mean_acts = collect_sae_activations(
        model, sae, target_texts, LAYER,
        hook_point="post", batch_size=4, pooling="max"
    ).mean(dim=0)
    feature_idx = mean_acts.argmax().item()
    print(f"Target feature: {feature_idx} (mean act: {mean_acts[feature_idx]:.4f})")

    hook_name = f"blocks.{LAYER}.hook_resid_post"
    prompts = ["I believe the government", "The weather today is", "I think that science"]

    # ================================================================
    # Part A: Hook Parity
    # ================================================================
    print("\n" + "=" * 60)
    print("PART A: Hook Parity (GT AmlifySAEHook vs SAEIOSteerModel)")
    print("=" * 60)

    all_logit_cos = []
    all_delta_cos = []
    all_top5_overlap = []

    for prompt in prompts:
        input_ids = model.to_tokens(prompt)

        # Baseline
        with torch.no_grad():
            logits_base = model(input_ids)[0, -1, :].clone()

        # GT path
        gt_hook_obj = GTAmlifySAEHook(
            layer=LAYER, sae=sae, features=[feature_idx],
            amp_factor=AMP_FACTOR, device=device,
        )

        def gt_tl_hook(act, hook):
            fake_output = (act,)
            result = gt_hook_obj(module=None, args=None, output=fake_output)
            return result[0]

        with torch.no_grad():
            logits_gt = model.run_with_hooks(
                input_ids, fwd_hooks=[(hook_name, gt_tl_hook)]
            )[0, -1, :].clone()
        model.reset_hooks()

        # Our path
        binary_mask = torch.zeros(d_sae, device=device)
        binary_mask[feature_idx] = 1.0

        steer_model = SAEIOSteerModel(
            model=model, layer=[LAYER],
            sae={LAYER: sae},
            steering_vector={LAYER: binary_mask},
            hook_point=["post"],
            position="last",
            device=device,
        )
        steer_model.setup_hooks(coeff={LAYER: AMP_FACTOR})

        with torch.no_grad():
            logits_ours = model(input_ids)[0, -1, :].clone()
        model.reset_hooks()

        # Compare
        delta_gt = (logits_gt - logits_base).float()
        delta_ours = (logits_ours - logits_base).float()

        logit_cos = F.cosine_similarity(
            logits_gt.float().unsqueeze(0), logits_ours.float().unsqueeze(0)
        ).item()
        delta_cos = F.cosine_similarity(
            delta_gt.unsqueeze(0), delta_ours.unsqueeze(0)
        ).item() if delta_gt.norm() > 0.1 and delta_ours.norm() > 0.1 else 0.0

        probs_gt = F.softmax(logits_gt.float(), dim=-1)
        probs_ours = F.softmax(logits_ours.float(), dim=-1)
        gt_top5 = set(torch.topk(probs_gt, 5).indices.tolist())
        our_top5 = set(torch.topk(probs_ours, 5).indices.tolist())
        top5_overlap = len(gt_top5 & our_top5)

        all_logit_cos.append(logit_cos)
        all_delta_cos.append(delta_cos)
        all_top5_overlap.append(top5_overlap)

        print(f"  '{prompt[:30]}': logit_cos={logit_cos:.6f}, "
              f"delta_cos={delta_cos:.6f}, top5_overlap={top5_overlap}/5")

    avg_logit_cos = np.mean(all_logit_cos)
    avg_delta_cos = np.mean(all_delta_cos)
    avg_top5 = np.mean(all_top5_overlap)

    print(f"\n  Avg logit cosine: {avg_logit_cos:.6f}")
    print(f"  Avg delta cosine: {avg_delta_cos:.6f}")
    print(f"  Avg top-5 overlap: {avg_top5:.1f}")

    logit_pass = avg_logit_cos > 0.95
    delta_pass = avg_delta_cos > 0.90
    top5_pass = avg_top5 >= 3

    print(f"  {'PASSED' if logit_pass else 'FAILED'}: Logit cosine > 0.95")
    print(f"  {'PASSED' if delta_pass else 'FAILED'}: Delta cosine > 0.90")
    print(f"  {'PASSED' if top5_pass else 'FAILED'}: Top-5 overlap >= 3")

    # ================================================================
    # Part B: Functional Steering
    # ================================================================
    print("\n" + "=" * 60)
    print("PART B: Functional Steering (steering changes output)")
    print("=" * 60)

    test_prompt = "I think"
    input_ids = model.to_tokens(test_prompt)

    with torch.no_grad():
        logits_base = model(input_ids)[0, -1, :].clone()

    steer_model = SAEIOSteerModel(
        model=model, layer=[LAYER],
        sae={LAYER: sae},
        steering_vector={LAYER: binary_mask},
        hook_point=["post"],
        position="last",
        device=device,
    )
    steer_model.setup_hooks(coeff={LAYER: AMP_FACTOR})

    with torch.no_grad():
        logits_steered = model(input_ids)[0, -1, :].clone()
    model.reset_hooks()

    probs_base = F.softmax(logits_base.float(), dim=-1)
    probs_steered = F.softmax(logits_steered.float(), dim=-1)

    base_steer_cos = F.cosine_similarity(
        probs_base.unsqueeze(0), probs_steered.unsqueeze(0)
    ).item()

    delta_norm = (logits_steered - logits_base).float().norm().item()

    functional_pass = base_steer_cos < 0.999 and delta_norm > 1.0

    print(f"  Baseline vs steered prob cosine: {base_steer_cos:.6f}")
    print(f"  Delta norm: {delta_norm:.4f}")

    top_base = torch.topk(probs_base, 3)
    top_steered = torch.topk(probs_steered, 3)
    print(f"  Baseline top-3: {[model.to_string(t) for t in top_base.indices]}")
    print(f"  Steered  top-3: {[model.to_string(t) for t in top_steered.indices]}")
    print(f"  {'PASSED' if functional_pass else 'FAILED'}: Steering changes output")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "A1. Logit cosine > 0.95": logit_pass,
        "A2. Delta cosine > 0.90": delta_pass,
        "A3. Top-5 overlap >= 3": top5_pass,
        "B1. Steering changes output": functional_pass,
    }
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
