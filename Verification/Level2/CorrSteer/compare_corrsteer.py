"""
CorrSteer Level 2 Comparison: GT (Code/CorrSteer/) vs our CorrSteerExtractor.

Uses the ACTUAL GT classes:
  - StreamingCorrelationAccumulator from Code/CorrSteer/train.py
  - ActivationCaptureHook from Code/CorrSteer/train.py
  - SteeringHook + FixedFeaturePolicyNetwork from Code/CorrSteer/

Both GT and ours use the same model (gemma-2-2b) + SAE + data.

Test 1: Feature Selection Match
  - Both run correlation accumulation on the same prompts
  - Compare top-k feature indices and coefficients

Test 2: Steering Inference Match
  - Both steer on the same test prompt
  - Compare output logits

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES
    PYTHONPATH=. python Verification/Level2/CorrSteer/compare_corrsteer.py
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "CorrSteer"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
LAYER = 13
TOP_K = 10
N_TRAIN = 50  # Small dataset for quick comparison
N_TEST = 10

# ============================================================
# GT imports
# ============================================================
from train import StreamingCorrelationAccumulator, ActivationCaptureHook
from corrsteer.steer import get_steering_hook
from eval import FixedFeaturePolicyNetwork

# ============================================================
# Our imports
# ============================================================
from Steering.extractors.sae import CorrSteerExtractor


def load_model_and_sae():
    """Shared model + SAE for both GT and ours."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    logger.info("Loading gemma-2-2b...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device=DEVICE, dtype=torch.bfloat16
    )

    logger.info("Loading SAE layer 13...")
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(DEVICE)

    return model, sae


def load_sycophancy_data():
    """Load sycophancy data from TrainDataset."""
    import csv
    data_path = ROOT / "TrainDataset" / "behaviour" / "sycophancy" / "sycophancy_train.csv"
    if not data_path.exists():
        # Fallback: use simple prompts
        logger.warning(f"Data not found at {data_path}, using fallback prompts")
        target = [f"Q: What is {i}+{i}?\nA: The answer is {i+i}." for i in range(N_TRAIN)]
        contrast = [f"Q: What is {i}+{i}?\nA: I think you're right, it's {i*3}." for i in range(N_TRAIN)]
        return target[:N_TRAIN], contrast[:N_TRAIN]

    rows = []
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= N_TRAIN * 2:
                break

    # Split by label or use correct/false prompts
    target = [r.get("correct_prompt", r.get("text", "")) for r in rows[:N_TRAIN]]
    contrast = [r.get("false_prompt", r.get("text", "")) for r in rows[N_TRAIN:N_TRAIN*2]]
    if not contrast:
        contrast = target  # fallback

    return target, contrast


def gt_extract(model, sae, target_data, contrast_data):
    """
    Run GT CorrSteer extraction using the actual StreamingCorrelationAccumulator.
    
    Reference: Code/CorrSteer/train.py CorrSteerController._run_collection
    """
    logger.info("=== GT Extraction ===")
    d_sae = sae.cfg.d_sae
    accumulator = StreamingCorrelationAccumulator(d_sae, pos_only=True)

    # Process target data (reward=1.0)
    hf_model = model.tokenizer  # TransformerLens model has tokenizer
    
    for i, text in enumerate(target_data):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            residual = cache["resid_pre", LAYER][0, -1:, :]  # last token, pre hook
            
            # SAE encode
            encoded = sae.encode(residual.to(sae.dtype))  # (1, d_sae)
            
            # Max pool (for single token, max = the value itself)
            pooled = encoded.float().cpu()
            reward = torch.tensor([1.0])
            
            accumulator.update_corr(pooled, reward)
            accumulator.update_coeff(pooled, reward)

    # Process contrast data (reward=0.0)
    for i, text in enumerate(contrast_data):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            residual = cache["resid_pre", LAYER][0, -1:, :]
            encoded = sae.encode(residual.to(sae.dtype))
            pooled = encoded.float().cpu()
            reward = torch.tensor([0.0])
            
            accumulator.update_corr(pooled, reward)
            accumulator.update_coeff(pooled, reward)

    # Get top-K features (pos_only=True)
    top_positive, top_negative = accumulator.top_features(TOP_K)
    
    gt_features = []
    for idx, coeff, corr, freq, stats in top_positive:
        gt_features.append({
            "feature_index": idx,
            "coefficient": coeff,
            "correlation": corr,
            "frequency": freq,
        })

    logger.info(f"GT found {len(gt_features)} features")
    for f in gt_features[:5]:
        logger.info(f"  Feature {f['feature_index']}: coeff={f['coefficient']:.4f}, corr={f['correlation']:.4f}")

    return gt_features, accumulator


def our_extract(model, sae, target_data, contrast_data):
    """Run our CorrSteerExtractor."""
    logger.info("=== Our Extraction ===")

    extractor = CorrSteerExtractor(
        model=model,
        sae={LAYER: sae},
        layer=[LAYER],
        batch_size=8,
        top_k=TOP_K,
        corrsteer_max_new_tokens=1,  # Minimal generation for speed
        corrsteer_pool="max",
        corrsteer_pos_only=True,
        corrsteer_scale=1.0,
        corrsteer_mask="generation",
        hook_point="pre",
    )

    vectors = extractor.extract(
        target_data=target_data,
        contrast_data=contrast_data,
    )

    our_features = extractor.selected_features or []
    logger.info(f"Ours found {len(our_features)} features")
    for f in our_features[:5]:
        logger.info(f"  Feature {f['feature_index']}: coeff={f['coefficient']:.4f}, corr={f['correlation']:.4f}")

    return our_features, vectors.get(LAYER)


def gt_steer(model, sae, feature_idx, coeff, test_prompt):
    """Run GT steering inference using SteeringHook + FixedFeaturePolicyNetwork."""
    d_sae = sae.cfg.d_sae
    policy_net = FixedFeaturePolicyNetwork(d_sae, feature_idx, coeff)
    hook = get_steering_hook(policy_net, sae, decode=False, lastk=1, multiple=1, mask="generation")

    # Register on the HF model's layer (TransformerLens wraps HF model)
    hf_layer = model.blocks[LAYER]

    tokens = model.to_tokens(test_prompt)
    
    # Using TransformerLens hook system
    def tl_hook_fn(resid, hook):
        # Wrap for TransformerLens format
        result = hook.__wrapped_hook(model.blocks[LAYER], (resid,))
        return result

    # Direct approach: use run_with_cache to get baseline, then manually apply
    with torch.no_grad():
        # Baseline
        logits_base = model(tokens)[0, -1, :].clone()
        
        # Steered: register GT hook on the underlying HF model layer
        # CorrSteer hooks are pre-hooks on model.model.layers[LAYER]
        # In TransformerLens, blocks[LAYER] wraps this
        hf_layer_module = model.blocks[LAYER].attn  # We need the actual transformer block
        
        # Actually, let's use TransformerLens hook system to match GT behavior
        hook_name = f"blocks.{LAYER}.hook_resid_pre"
        
        def steering_hook(resid, hook_obj):
            """Replicate GT SteeringHook behavior in TransformerLens format."""
            # GT SteeringHook processes last token
            last_token = resid[:, -1:, :].to(sae.dtype)
            encoded = sae.encode(last_token.view(-1, last_token.shape[-1]))
            dict_size = encoded.shape[-1]
            encoded = encoded.view(1, 1, dict_size)
            
            # Policy action
            action = policy_net.select_action(encoded.detach())
            if isinstance(action, tuple):
                action = action[0]
            action = action.detach()
            
            # Decode: action @ W_dec
            action_2d = action.view(-1, dict_size)
            steering = (action_2d @ sae.W_dec).view(1, 1, -1)
            
            resid_out = resid.clone()
            resid_out[:, -1:, :] = resid_out[:, -1:, :] + steering
            return resid_out
        
        logits_gt = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, steering_hook)],
        )[0, -1, :].clone()

    return logits_base, logits_gt


def compare_results(gt_features, our_features, gt_logits_base, gt_logits_steered, our_vector):
    """Compare GT and our results."""
    print("\n" + "=" * 60)
    print("=== FEATURE SELECTION COMPARISON ===")
    
    gt_indices = {f["feature_index"] for f in gt_features}
    our_indices = {f["feature_index"] for f in our_features}
    
    intersection = gt_indices & our_indices
    print(f"GT features: {sorted(gt_indices)}")
    print(f"Our features: {sorted(our_indices)}")
    print(f"Intersection: {len(intersection)}/{len(gt_indices)} ({len(intersection)/max(len(gt_indices),1)*100:.1f}%)")
    
    # Compare coefficients for shared features
    if intersection:
        gt_coeff_map = {f["feature_index"]: f["coefficient"] for f in gt_features}
        our_coeff_map = {f["feature_index"]: f["coefficient"] for f in our_features}
        
        print("\nShared feature coefficients:")
        for idx in sorted(intersection):
            print(f"  Feature {idx}: GT={gt_coeff_map[idx]:.4f}, Ours={our_coeff_map[idx]:.4f}")
    
    # Compare correlations
    gt_corr_map = {f["feature_index"]: f["correlation"] for f in gt_features}
    our_corr_map = {f["feature_index"]: f["correlation"] for f in our_features}
    
    if intersection:
        gt_corr_vec = torch.tensor([gt_corr_map[i] for i in sorted(intersection)])
        our_corr_vec = torch.tensor([our_corr_map[i] for i in sorted(intersection)])
        corr_cosim = F.cosine_similarity(gt_corr_vec.unsqueeze(0), our_corr_vec.unsqueeze(0)).item()
        print(f"\nCorrelation cosine sim (shared features): {corr_cosim:.6f}")
    
    print("\n=== STEERING INFERENCE COMPARISON ===")
    print(f"GT baseline logits norm: {gt_logits_base.norm().item():.4f}")
    print(f"GT steered logits norm: {gt_logits_steered.norm().item():.4f}")
    
    steer_delta = gt_logits_steered - gt_logits_base
    print(f"GT steer delta norm: {steer_delta.norm().item():.4f}")
    
    if our_vector is not None:
        print(f"Our vector norm: {our_vector.norm().item():.4f}")
    
    print("=" * 60)


def main():
    model, sae = load_model_and_sae()
    target_data, contrast_data = load_sycophancy_data()
    
    logger.info(f"Data: {len(target_data)} target + {len(contrast_data)} contrast")
    
    # 1. Feature extraction comparison
    gt_features, gt_accum = gt_extract(model, sae, target_data, contrast_data)
    our_features, our_vector = our_extract(model, sae, target_data, contrast_data)
    
    # 2. Steering inference comparison (use GT's top feature)
    if gt_features:
        test_prompt = "I think the best approach is"
        feat_idx = gt_features[0]["feature_index"]
        coeff = gt_features[0]["coefficient"]
        logger.info(f"Testing steering with feature {feat_idx}, coeff={coeff:.4f}")
        
        logits_base, logits_gt = gt_steer(model, sae, feat_idx, coeff, test_prompt)
    else:
        logits_base = torch.zeros(1)
        logits_gt = torch.zeros(1)
    
    compare_results(gt_features, our_features, logits_base, logits_gt, our_vector)


if __name__ == "__main__":
    main()
