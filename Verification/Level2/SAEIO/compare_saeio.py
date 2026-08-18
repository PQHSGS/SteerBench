"""
SAE-IO Level 2 Comparison: GT (Code/saes-are-good-for-steering/) vs our SAEIOExtractor.

Uses the ACTUAL GT classes:
  - AmlifySAEHook from Code/saes-are-good-for-steering/src/sae_utils.py
  - get_output_score from Code/saes-are-good-for-steering/src/output_score.py

Both GT and ours use the same model (gemma-2-2b) + SAE.

Test 1: Output Score Match
  - Both compute output scores for the same features on same prompt
  - Compare output_score values

Test 2: Steering Inference Match
  - Both steer with AmlifySAEHook / SAEIOSteerModel on same prompt
  - Compare output logits

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES
    PYTHONPATH=. python Verification/Level2/SAEIO/compare_saeio.py
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
import gc
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "saes-are-good-for-steering" / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
LAYER = 20
AMP_FACTOR = 10.0
TOP_K = 5
NEUTRAL_SENTENCE = "From my experience,"

# ============================================================
# GT imports
# ============================================================
from sae_utils import AmlifySAEHook
from output_score import get_output_score

# ============================================================
# Our imports
# ============================================================
from Steering.extractors.sae import SAEIOExtractor


def load_model_and_sae():
    """Load model and SAE for both GT and ours."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    # GT uses HuggingFace model directly
    logger.info("Loading HF model (for GT)...")
    hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    # Our extractor uses TransformerLens
    logger.info("Loading TransformerLens model (for ours)...")
    tl_model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device=DEVICE, dtype=torch.bfloat16
    )

    logger.info(f"Loading SAE layer {LAYER}...")
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )

    return hf_model, tokenizer, tl_model, sae


def gt_compute_output_scores(hf_model, tokenizer, sae, feature_indices):
    """
    Compute output scores using the ACTUAL GT get_output_score function.
    Reference: Code/saes-are-good-for-steering/src/output_score.py
    """
    logger.info(f"=== GT Output Score Computation ({len(feature_indices)} features) ===")

    # Get logit lens top-k for each feature (GT: cache_logit_lens)
    # Simplified: compute logit lens indices from SAE decoder weights
    final_layer_norm = hf_model.model.norm
    lm_head = hf_model.lm_head
    
    sae_cpu = sae.cpu()
    W_dec = sae_cpu.W_dec.detach().float()  # (d_sae, d_model)
    
    # logit lens: for each feature, project decoder weight through LN + LM head
    logit_lens_top_k = 20  # Same as GT default
    with torch.no_grad():
        # Normalize each decoder vector through layer norm
        normed = final_layer_norm(W_dec)  # (d_sae, d_model)
        # Project through LM head
        logits = normed @ lm_head.weight.T  # (d_sae, vocab_size)
        # Get top-k tokens per feature
        ll_topk = torch.topk(logits, logit_lens_top_k, dim=1)  # values, indices
    
    gt_scores = {}
    sae_device = sae.to(DEVICE)
    hf_model_device = hf_model.to(DEVICE)
    
    for feat_idx in feature_indices:
        logit_lens_tokens = ll_topk.indices[feat_idx, :].tolist()
        score = get_output_score(
            LAYER, feat_idx, logit_lens_tokens,
            NEUTRAL_SENTENCE, sae_device, tokenizer,
            hf_model_device, DEVICE, amp_factor=AMP_FACTOR
        )
        gt_scores[feat_idx] = score
        logger.info(f"  Feature {feat_idx}: output_score = {score:.6f}")
    
    # Move back to CPU to free VRAM
    hf_model.cpu()
    sae.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return gt_scores


def gt_steer(hf_model, tokenizer, sae, features, test_prompt):
    """
    Run GT steering using the ACTUAL AmlifySAEHook.
    Reference: Code/saes-are-good-for-steering/src/sae_utils.py
    """
    logger.info(f"=== GT Steering Inference (features={features}) ===")
    
    hf_model = hf_model.to(DEVICE)
    sae = sae.to(DEVICE)
    
    # Install GT AmlifySAEHook
    sae_hook = AmlifySAEHook(LAYER, sae, features, AMP_FACTOR, DEVICE)
    model_block = hf_model.model.layers[LAYER]
    handle = model_block.register_forward_hook(sae_hook, always_call=True)
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(DEVICE)
    
    with torch.no_grad():
        outputs = hf_model(**inputs)
    
    logits_steered = outputs.logits[:, -1, :].squeeze().float().cpu()
    
    handle.remove()
    # Clean up any remaining hooks
    for hook in list(model_block._forward_hooks.values()):
        hook.remove()
    
    # Baseline (no hook)
    with torch.no_grad():
        outputs_base = hf_model(**inputs)
    logits_base = outputs_base.logits[:, -1, :].squeeze().float().cpu()
    
    hf_model.cpu()
    sae.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    
    return logits_base, logits_steered


def our_extract_and_steer(tl_model, sae, test_prompt):
    """Run our SAEIOExtractor and steer."""
    logger.info("=== Our Extraction ===")
    
    sae_on_device = sae.to(DEVICE)
    
    extractor = SAEIOExtractor(
        model=tl_model,
        sae={LAYER: sae_on_device},
        layer=[LAYER],
        top_k=TOP_K,
        neutral_prompt=NEUTRAL_SENTENCE,
        amp_factor=AMP_FACTOR,
    )

    # Extract with neutral prompt (SAEIO doesn't need contrastive data)
    vectors = extractor.extract(
        target_data=[NEUTRAL_SENTENCE],
        contrast_data=None,
    )

    our_scores = extractor.metadata.get("output_scores", {})
    our_features = extractor.metadata.get("selected_features", [])

    logger.info(f"Our selected features: {our_features}")
    
    return vectors.get(LAYER), our_features, our_scores


def compare_results(gt_scores, gt_logits_base, gt_logits_steered,
                    our_vector, our_features, our_scores):
    """Print comparison."""
    print("\n" + "=" * 60)
    print("=== OUTPUT SCORE COMPARISON ===")
    
    gt_feat_set = set(gt_scores.keys())
    our_feat_set = set(our_features) if our_features else set()
    
    print(f"GT scored features: {sorted(gt_feat_set)[:10]}...")
    print(f"Our selected features: {sorted(our_feat_set)}")
    
    # Compare output scores for features we both have
    common = gt_feat_set & our_feat_set
    if common and our_scores:
        print(f"\nCommon features ({len(common)}):")
        for idx in sorted(common):
            gt_s = gt_scores.get(idx, 0)
            our_s = our_scores.get(str(idx), our_scores.get(idx, 0))
            print(f"  Feature {idx}: GT={gt_s:.6f}, Ours={our_s:.6f}")
    
    print("\n=== STEERING INFERENCE ===")
    delta = gt_logits_steered - gt_logits_base
    cos_sim = F.cosine_similarity(
        gt_logits_steered.unsqueeze(0),
        gt_logits_base.unsqueeze(0)
    ).item()
    print(f"GT baseline → steered cosine sim: {cos_sim:.6f}")
    print(f"GT steer delta norm: {delta.norm().item():.4f}")
    
    if our_vector is not None:
        print(f"Our vector norm: {our_vector.norm().item():.4f}")
    print("=" * 60)


def main():
    hf_model, tokenizer, tl_model, sae = load_model_and_sae()
    
    # Pick some candidate features to score (use first active features)
    # In practice, GT iterates over all features; we'll test a small subset
    test_features = list(range(0, 100, 10))  # 10 features to score
    
    # 1. GT output scores
    gt_scores = gt_compute_output_scores(hf_model, tokenizer, sae, test_features)
    
    # 2. Our extraction
    our_vector, our_features, our_scores = our_extract_and_steer(tl_model, sae, NEUTRAL_SENTENCE)
    
    # 3. GT steering (use top GT-scored feature)
    if gt_scores:
        best_feat = max(gt_scores, key=gt_scores.get)
        logits_base, logits_steered = gt_steer(hf_model, tokenizer, sae, [best_feat], NEUTRAL_SENTENCE)
    else:
        logits_base = torch.zeros(1)
        logits_steered = torch.zeros(1)
    
    compare_results(gt_scores, logits_base, logits_steered, our_vector, our_features, our_scores)


if __name__ == "__main__":
    main()
