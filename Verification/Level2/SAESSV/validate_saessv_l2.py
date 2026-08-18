"""
SAE-SSV Level 2 Validation: Pipeline End-to-End + GT Steering Comparison.

Test 1: Pipeline Integration
    - Runs SteeringPipeline.setup() with the registered politics_twinviews dataset
    - Verifies extraction produces a saved vector

Test 2: Steering Output Match (GT vs Pipeline)
    - GT: Code/SAE-SSV/saessv-demo.py test_falseness_ssv_with_hooks (lines 1182-1197)
      encode → add SSV*scale → decode → replace last token activation
    - Ours: SteeringPipeline with SSVSteerModel hook
    - Compares output logits/probabilities on the same prompt

Both use the same model, SAE, and extracted vector.

Usage:
    conda activate sae_circuit
    python validate_saessv_l2.py
"""

import sys
import copy
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from Steering.pipeline import SteeringPipeline
from Steering.config import PipelineConfig, ModelConfig, ExtractorConfig, SteerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:2"


def test_pipeline_integration():
    """Test 1: SSV extraction and steering runs in SteeringPipeline end-to-end."""
    print("=" * 60)
    print("Test 1: SAE-SSV Pipeline Integration")
    print("=" * 60)
    
    layer = 10
    temp_vector_path = Path(__file__).parent / "temp_saessv_vector.pt"
    
    config = PipelineConfig(
        model=ModelConfig(
            name="google/gemma-2-2b",
            device=DEVICE,
            dtype="bfloat16",
            use_compile=False
        ),
        extractor=ExtractorConfig(
            method="SSV",
            layer=[layer],
            ssv_opt_steps=10,
            ssv_feature_refinement_k=10,
        ),
        steer=SteerConfig(
            method="SSV",
            layer=[layer],
            coeff=5.0
        ),
        train_dataset="politics_twinviews",
        n_train=6,
        test_dataset="dummy",
        save_vector=str(temp_vector_path)
    )
    
    try:
        logger.info("Initializing pipeline for extraction...")
        pipeline = SteeringPipeline(config)
        pipeline.setup()
        
        if not temp_vector_path.exists():
            print("  FAILED: Extractor did not save vector.")
            return False, None, None
            
        vector_data = torch.load(str(temp_vector_path), weights_only=False)
        if layer not in vector_data["steering_vector"]:
            print(f"  FAILED: Missing vector for layer {layer}.")
            return False, None, None
        
        print("  PASSED: Extraction completed and vector saved")
        return True, pipeline, temp_vector_path
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return False, None, None


def test_steering_output_match(pipeline, vector_path):
    """Test 2: GT steering hook vs our SSVSteerModel produce matching logits.
    
    GT hook (Code/SAE-SSV/saessv-demo.py lines 1182-1197):
        def modify_activation(act, hook):
            last_token_act = act[0, -1, :].clone()
            act_cpu = last_token_act.cpu().float()
            latent = sae_float32.encode(act_cpu.unsqueeze(0)).squeeze(0).cpu().numpy()
            steered_latent = latent + ssv * scale
            steered_latent_tensor = torch.tensor(steered_latent, dtype=torch.float32)
            steered_act = sae_float32.decode(steered_latent_tensor.unsqueeze(0)).squeeze(0)
            steered_act = steered_act.to(last_token_act.device).to(last_token_act.dtype)
            act[0, -1, :] = steered_act
            return act
    """
    print("\n" + "=" * 60)
    print("Test 2: GT Steering Hook vs SSVSteerModel Output Match")
    print("=" * 60)
    
    if pipeline is None:
        print("  SKIPPED: Pipeline not available (Test 1 failed)")
        return None
    
    model = pipeline.model
    layer = pipeline.config.steer.layer[0]
    coeff = pipeline.config.steer.coeff
    
    # Load the extracted vector using SteeringVector.load() for correct deserialization
    from Steering.config.results import SteeringVector
    sv = SteeringVector.load(str(vector_path), device="cpu")
    
    # sv.vector should be dict[layer -> tensor]
    if isinstance(sv.vector, dict):
        ssv_tensor = sv.vector[layer]
    else:
        ssv_tensor = sv.vector
    
    print(f"  Vector type: {type(ssv_tensor)}, shape: {ssv_tensor.shape}")
    ssv_np = ssv_tensor.numpy().astype(np.float32)
    
    # Load SAE for GT hook
    sae = pipeline._sae_cache.get(layer)
    if sae is None:
        print("  SKIPPED: SAE not in pipeline cache")
        return None
    
    # Create float32 CPU copy of SAE (matching GT exactly)
    sae_float32 = copy.deepcopy(sae).cpu()
    for param in sae_float32.parameters():
        param.data = param.data.to(torch.float32)
    sae_float32.eval()
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    prompt = "I believe the government"
    input_ids = model.to_tokens(prompt)
    
    # --- Method A: GT hook (exact code from saessv-demo.py:1182-1197) ---
    def gt_modify_activation(act, hook):
        last_token_act = act[0, -1, :].clone()
        act_cpu = last_token_act.cpu().float()
        latent = sae_float32.encode(act_cpu.unsqueeze(0)).squeeze(0).cpu().numpy()
        steered_latent = latent + ssv_np * coeff[layer]
        steered_latent_tensor = torch.tensor(steered_latent, dtype=torch.float32)
        steered_act = sae_float32.decode(steered_latent_tensor.unsqueeze(0)).squeeze(0)
        steered_act = steered_act.to(last_token_act.device).to(last_token_act.dtype)
        act[0, -1, :] = steered_act
        return act
    
    with torch.no_grad():
        logits_gt = model.run_with_hooks(
            input_ids, fwd_hooks=[(hook_name, gt_modify_activation)]
        )[0, -1, :].clone()
    model.reset_hooks()
    
    # --- Method B: Our SSVSteerModel via SteeringPipeline ---
    pipeline.steer_model.setup_hooks(coeff=coeff)
    with torch.no_grad():
        logits_ours = model(input_ids)[0, -1, :].clone()
    model.reset_hooks()
    
    # Compare
    probs_gt = F.softmax(logits_gt.float(), dim=-1)
    probs_ours = F.softmax(logits_ours.float(), dim=-1)
    
    max_diff = (probs_gt - probs_ours).abs().max().item()
    cos_sim = F.cosine_similarity(probs_gt.unsqueeze(0), probs_ours.unsqueeze(0)).item()
    
    topk_gt = torch.topk(probs_gt, 5)
    topk_ours = torch.topk(probs_ours, 5)
    
    print(f"\n  GT Hook Top 5:")
    for idx, p in zip(topk_gt.indices, topk_gt.values):
        print(f"    {model.to_string(idx)}: {p.item():.6f}")
    
    print(f"  Our SSVSteerModel Top 5:")
    for idx, p in zip(topk_ours.indices, topk_ours.values):
        print(f"    {model.to_string(idx)}: {p.item():.6f}")
    
    print(f"\n  Max Prob Diff: {max_diff:.2e}")
    print(f"  Cosine Similarity: {cos_sim:.8f}")
    
    # Expect high similarity; small differences may come from float precision
    passed = cos_sim > 0.99
    print(f"  {'PASSED' if passed else 'FAILED'}")
    
    del sae_float32
    torch.cuda.empty_cache()
    
    return passed


def main():
    print("=" * 60)
    print("SAE-SSV Level 2 Validation")
    print("GT: Code/SAE-SSV/saessv-demo.py test_falseness_ssv_with_hooks")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test 1: Pipeline integration
    try:
        passed, pipeline, vector_path = test_pipeline_integration()
        results["pipeline_integration"] = passed
    except Exception as e:
        logger.error(f"Test 1 failed: {e}", exc_info=True)
        results["pipeline_integration"] = False
        pipeline, vector_path = None, None
    
    # Test 2: GT vs Pipeline steering output comparison
    try:
        results["steering_output_match"] = test_steering_output_match(pipeline, vector_path)
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)
        results["steering_output_match"] = False
    
    # Cleanup
    if vector_path and Path(vector_path).exists():
        Path(vector_path).unlink()
    
    if pipeline:
        del pipeline
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {status}: {name}")
    
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
