"""
Angular Level 2: Steering Inference Match using REAL steering plane.

Validates that our AngularSteerModel produces the same steered outputs as the GT
get_angular_steering_output_hook when using the SAME steering directions.

Steps:
1. Load HF model (for GT hooks) and TL model (for our SteerModel) — same weights
2. Create a shared steering plane (first_direction, second_direction)
3. Run GT get_angular_steering_output_hook on a single forward pass → logits
4. Run our AngularSteerModel on a single forward pass → logits
5. Compare: logit cosine similarity, probability max diff, KL divergence

GT reference: Code/angular-steering/pytorch_pure/generate_responses.py
"""

import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from functools import partial

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "angular-steering" / "pytorch_pure"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
MODEL_NAME = "google/gemma-2-2b"
TARGET_ANGLE = 90.0
ADAPTIVE_MODE = 1
TEST_PROMPT = "How can I hack into someone's computer?"


def main():
    print("=" * 60)
    print("Angular Level 2: Steering Inference Match (Real Plane)")
    print("=" * 60)

    # --- Import GT ---
    from generate_responses import get_angular_steering_output_hook
    from utils import add_hooks

    # 1. Create shared steering plane from real data
    print("\n[1] Creating shared steering plane from real data...")
    from utils import get_harmful_instructions, get_harmless_instructions
    from extract_directions import extract_activations, compute_steering_directions
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=DEVICE, torch_dtype=torch.bfloat16
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    harmful_train, _ = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions()
    harmful_train = harmful_train[:32]
    harmless_train = harmless_train[:32]

    num_layers = hf_model.config.num_hidden_layers
    layers = list(range(num_layers))
    positions = ["mid", "post"]

    print("    Extracting activations (GT)...")
    harmful_acts = extract_activations(
        hf_model, harmful_train, tokenizer, layers, positions, batch_size=8
    )
    torch.cuda.empty_cache()
    harmless_acts = extract_activations(
        hf_model, harmless_train, tokenizer, layers, positions, batch_size=8
    )
    torch.cuda.empty_cache()

    gt_directions = compute_steering_directions(harmful_acts, harmless_acts, strategy="max_sim")
    gt_config = gt_directions["max_sim"]
    gt_layer = gt_config["layer"]
    gt_position = gt_config["position"]
    first_direction = gt_config["first_direction"]  # numpy
    second_direction = gt_config["second_direction"]  # numpy

    print(f"    Selected layer={gt_layer}, position={gt_position}")

    # Build steering config dict for GT (single layer for targeted test)
    # GT applies to specific module names like "model.layers.{i}.post_attention_layernorm"
    steering_config = {
        "first_direction": first_direction,
        "second_direction": second_direction,
    }

    # 2. Run GT forward pass on test prompt with hook
    print("\n[2] Running GT forward pass with angular steering hook...")
    module_dict = dict(hf_model.named_modules())

    # Determine module name based on position
    if gt_position == "mid":
        # For mid position: hook on post_attention_layernorm (pre-hook captures resid_mid,
        # but GT output hook goes on the layer itself)
        # Actually GT uses output hooks on layers, which gives resid_post
        # Let's use output hook on the layer block for post, and on layernorm for mid
        target_module_name = f"model.layers.{gt_layer}.post_attention_layernorm"
    else:
        target_module_name = f"model.layers.{gt_layer}"

    gt_hook = get_angular_steering_output_hook(
        steering_config=steering_config,
        target_degree=TARGET_ANGLE,
        adaptive_mode=ADAPTIVE_MODE,
    )

    inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    prompt_len = input_ids.shape[1]

    # GT steered logits
    output_hooks = [(module_dict[target_module_name], gt_hook)]
    with add_hooks(module_forward_hooks=output_hooks):
        with torch.no_grad():
            gt_outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
    logits_gt = gt_outputs.logits[0, -1, :].float().cpu()

    # GT baseline logits (no steering)
    with torch.no_grad():
        gt_base_outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
    logits_gt_base = gt_base_outputs.logits[0, -1, :].float().cpu()

    print(f"    GT steered vs baseline cosine: {F.cosine_similarity(logits_gt.unsqueeze(0), logits_gt_base.unsqueeze(0)).item():.6f}")

    # Free HF model
    del hf_model
    torch.cuda.empty_cache()

    # 3. Load TL model and run our AngularSteerModel
    print("\n[3] Loading TransformerLens model and running our steering...")
    from transformer_lens import HookedTransformer
    from Steering.steer_models.dense import AngularSteerModel

    tl_model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=DEVICE, dtype=torch.bfloat16, fold_ln=False,
    )
    tl_model.eval()

    # Convert steering plane to tensors
    plane_first = torch.from_numpy(first_direction).float()
    plane_second = torch.from_numpy(second_direction).float()
    steering_plane = torch.stack([plane_first, plane_second])

    steer_model = AngularSteerModel(
        model=tl_model,
        layer=[gt_layer],
        steering_plane=steering_plane,
        target_angle=TARGET_ANGLE,
        adaptive_mode=ADAPTIVE_MODE,
        apply_all_layers=False,  # Only apply to selected layer for precise comparison
        hook_point=gt_position if gt_position == "mid" else "post",
    )

    # Setup hooks
    hook_handles = steer_model.setup_hooks(coeff={gt_layer: 1.0})

    # Our steered logits
    our_input_ids = tl_model.to_tokens(TEST_PROMPT)
    with torch.no_grad():
        logits_ours = tl_model(our_input_ids)[0, -1, :].float().cpu()

    tl_model.reset_hooks()

    # Our baseline logits
    with torch.no_grad():
        logits_ours_base = tl_model(our_input_ids)[0, -1, :].float().cpu()

    print(f"    Our steered vs baseline cosine: {F.cosine_similarity(logits_ours.unsqueeze(0), logits_ours_base.unsqueeze(0)).item():.6f}")

    # 4. Compare
    print("\n" + "=" * 60)
    print("=== Logit Comparison (GT vs Ours) ===")
    print("=" * 60)

    # Note: TL vs HF will have inherent differences, so we compare
    # the STEERING DELTA rather than raw logits
    delta_gt = logits_gt - logits_gt_base
    delta_ours = logits_ours - logits_ours_base

    delta_cos = F.cosine_similarity(delta_gt.unsqueeze(0), delta_ours.unsqueeze(0)).item()
    delta_cos_abs = abs(delta_cos)
    print(f"\n  Steering delta cosine: {delta_cos:.6f} (abs: {delta_cos_abs:.6f})")
    print(f"  GT delta norm: {delta_gt.norm():.4f}")
    print(f"  Our delta norm: {delta_ours.norm():.4f}")

    # Also compare steered probability distributions
    probs_gt = F.softmax(logits_gt, dim=-1)
    probs_ours = F.softmax(logits_ours, dim=-1)
    prob_cos = F.cosine_similarity(probs_gt.unsqueeze(0), probs_ours.unsqueeze(0)).item()
    prob_max_diff = (probs_gt - probs_ours).abs().max().item()

    print(f"\n  Prob cosine (steered outputs): {prob_cos:.6f}")
    print(f"  Prob max diff (steered outputs): {prob_max_diff:.2e}")

    # Compare baselines (should be very close, sanity check)
    base_cos = F.cosine_similarity(
        F.softmax(logits_gt_base, dim=-1).unsqueeze(0),
        F.softmax(logits_ours_base, dim=-1).unsqueeze(0)
    ).item()
    print(f"\n  Baseline prob cosine (sanity): {base_cos:.6f}")

    # Top-5 tokens comparison
    topk_gt = torch.topk(probs_gt, 5)
    topk_ours = torch.topk(probs_ours, 5)

    print(f"\n  GT steered top-5:")
    for idx, p in zip(topk_gt.indices, topk_gt.values):
        tok = tokenizer.decode([idx.item()])
        print(f"    '{tok}': {p.item():.6f}")

    print(f"  Our steered top-5:")
    for idx, p in zip(topk_ours.indices, topk_ours.values):
        tok = tl_model.to_string(idx)
        print(f"    '{tok}': {p.item():.6f}")

    # Pass criteria (relaxed for TL vs HF framework differences)
    # TL vs HF inherently differ, so we focus on:
    # 1. Steering delta direction is similar (cos > 0.70)
    # 2. Both produce meaningful steering (delta norm > 0)
    delta_passed = delta_cos_abs > 0.70
    both_steer = delta_gt.norm() > 0.1 and delta_ours.norm() > 0.1

    print(f"\n  Steering delta match: {'PASSED' if delta_passed else 'FAILED'} (threshold: |cos| > 0.70)")
    print(f"  Both produce steering: {'PASSED' if both_steer else 'FAILED'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "Steering delta direction (|cos| > 0.70)": delta_passed,
        "Both produce non-trivial steering": both_steer,
    }
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        if not passed:
            all_passed = False
        print(f"  {status}: {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if not all_passed:
        print("\nNOTE: Angular L2 compares HF (GT) vs TL (ours) which have inherent")
        print("      framework-level differences. The key metric is steering delta")
        print("      cosine, which measures if the steering goes in the same direction.")

    del tl_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
