"""
SAE-TS Level 2: Steering Inference Match.

Two-part validation:

Part A — Hook Parity:
    GT patch_resid vs SAETSSteerModel.
    Same vector, same scale → expect logit cosine > 0.9999

Part B — Functional Steering:
    Verify steering actually changes model output.

GT reference: Code/SAE-TS/src/sae_ts/steering/patch.py :: patch_resid
Our class:    Steering/steer_models/sae.py :: SAETSSteerModel

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level2/SAETS/validate_saets_l2.py [--device cuda:2]
"""

import sys
import gc
import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from functools import partial

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "SAE-TS" / "src"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
LAYER = 12
SCALE = 60.0


def main():
    parser = argparse.ArgumentParser(description="SAETS L2 Validation")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    device = args.device

    print("SAE-TS Level 2 Validation")
    print("=" * 60)
    print(f"Config: LAYER={LAYER}, SCALE={SCALE}")
    print(f"Device: {device}\n")

    # Imports
    from sae_ts.steering.patch import patch_resid as gt_patch_resid
    from sae_ts.ft_effects.utils import LinearAdapter as GTLinearAdapter
    from sae_ts.baselines.analysis import single_step_steer as gt_single_step_steer
    import sae_ts.baselines.analysis as gt_mod
    gt_mod.device = device

    from Steering.steer_models.sae import SAETSSteerModel
    from transformer_lens import HookedTransformer

    # Load model (no SAE needed for L2)
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device=device, dtype=torch.bfloat16
    )
    d_model = model.cfg.d_model  # 2304
    d_sae = 16384

    # Load pre-trained GT adapter
    gt_adapter_path = ROOT / "Results" / "l3_saets_gt" / "adapter_layer12.pt"
    adapter = GTLinearAdapter(d_model, d_sae)
    adapter.load_state_dict(torch.load(gt_adapter_path, map_location="cpu"))
    adapter.to(device)

    # Generate steering vector from GT (anger, feature 4111)
    target = torch.zeros(d_sae, device=device)
    target[4111] = 1.0
    steering_vec = gt_single_step_steer(adapter, target, bias_scale=1).squeeze()
    sv = steering_vec.to(device).to(model.cfg.dtype)
    print(f"Steering vector: norm={sv.float().norm():.6f}, shape={sv.shape}")

    hook_name = f"blocks.{LAYER}.hook_resid_post"
    prompts = ["I think", "The weather today is", "I believe science"]

    # ================================================================
    # Part A: Hook Parity
    # ================================================================
    print("\n" + "=" * 60)
    print("PART A: Hook Parity (GT patch_resid vs SAETSSteerModel)")
    print("=" * 60)

    all_logit_cos = []
    all_delta_cos = []

    for prompt in prompts:
        input_ids = model.to_tokens(prompt)

        # Baseline
        with torch.no_grad():
            logits_base = model(input_ids)[0, -1, :].clone()

        # GT path: patch_resid adds steering*scale to ALL positions
        gt_hook = partial(gt_patch_resid, steering=sv, scale=SCALE)
        with torch.no_grad():
            logits_gt = model.run_with_hooks(
                input_ids, fwd_hooks=[(hook_name, gt_hook)]
            )[0, -1, :].clone()
        model.reset_hooks()

        # Our path: SAETSSteerModel with position="all"
        steer_model = SAETSSteerModel(
            model=model,
            layer=[LAYER],
            sae={LAYER: None},  # Not needed for inference
            steering_vector={LAYER: sv},
            auto_scale=False,
            hook_point=["post"],
            position="all",
        )
        steer_model.setup_hooks(coeff={LAYER: SCALE})
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
        ).item() if delta_gt.norm() > 0.1 else 0.0

        all_logit_cos.append(logit_cos)
        all_delta_cos.append(delta_cos)

        print(f"  '{prompt[:30]}': logit_cos={logit_cos:.8f}, delta_cos={delta_cos:.8f}")

    avg_logit_cos = np.mean(all_logit_cos)
    avg_delta_cos = np.mean(all_delta_cos)
    logit_pass = avg_logit_cos > 0.9999
    delta_pass = avg_delta_cos > 0.9999

    print(f"\n  Avg logit cosine: {avg_logit_cos:.8f}")
    print(f"  Avg delta cosine: {avg_delta_cos:.8f}")
    print(f"  {'PASSED' if logit_pass else 'FAILED'}: Logit cosine > 0.9999")
    print(f"  {'PASSED' if delta_pass else 'FAILED'}: Delta cosine > 0.9999")

    # ================================================================
    # Part B: Functional Steering
    # ================================================================
    print("\n" + "=" * 60)
    print("PART B: Functional Steering")
    print("=" * 60)

    test_prompt = "I think"
    input_ids = model.to_tokens(test_prompt)

    with torch.no_grad():
        logits_base = model(input_ids)[0, -1, :].clone()

    steer_model = SAETSSteerModel(
        model=model, layer=[LAYER], sae={LAYER: None},
        steering_vector={LAYER: sv},
        auto_scale=False, hook_point=["post"], position="all",
    )
    steer_model.setup_hooks(coeff={LAYER: SCALE})
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
    # Part C: Multiple concept vectors (comprehensive check)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART C: Multiple Concepts Hook Parity")
    print("=" * 60)

    concepts = ["anger", "love", "french", "conspiracy"]
    gt_vec_dir = ROOT / "Results" / "l3_saets_gt"
    concept_cos = []

    for concept in concepts:
        vec = torch.load(gt_vec_dir / f"{concept}_optimised_vec.pt", map_location=device)
        if vec.dim() > 1:
            vec = vec.squeeze()
        vec = vec.to(device).to(model.cfg.dtype)

        input_ids = model.to_tokens("I think")

        # GT
        gt_hook = partial(gt_patch_resid, steering=vec, scale=SCALE)
        with torch.no_grad():
            logits_gt = model.run_with_hooks(
                input_ids, fwd_hooks=[(hook_name, gt_hook)]
            )[0, -1, :].clone()
        model.reset_hooks()

        # Ours
        sm = SAETSSteerModel(
            model=model, layer=[LAYER], sae={LAYER: None},
            steering_vector={LAYER: vec},
            auto_scale=False, hook_point=["post"], position="all",
        )
        sm.setup_hooks(coeff={LAYER: SCALE})
        with torch.no_grad():
            logits_ours = model(input_ids)[0, -1, :].clone()
        model.reset_hooks()

        cos = F.cosine_similarity(
            logits_gt.float().unsqueeze(0), logits_ours.float().unsqueeze(0)
        ).item()
        concept_cos.append(cos)
        print(f"  {concept:25s}: logit_cos={cos:.8f}")

    avg_concept_cos = np.mean(concept_cos)
    concept_pass = min(concept_cos) > 0.9999
    print(f"\n  Avg: {avg_concept_cos:.8f}, Min: {min(concept_cos):.8f}")
    print(f"  {'PASSED' if concept_pass else 'FAILED'}: All concepts > 0.9999")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "A1. Logit cosine > 0.9999": logit_pass,
        "A2. Delta cosine > 0.9999": delta_pass,
        "B1. Functional steering": functional_pass,
        "C1. Multi-concept parity": concept_pass,
    }
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
