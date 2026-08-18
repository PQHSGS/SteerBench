"""
Compare SAE-SSV extraction: GT (saessv-demo.py) vs our SSVExtractor.

Uses the GT's exact training data (train_texts.npy + train_labels.npy)
to ensure any differences are algorithmic, not data-related.

GT data source: HuggingFace 'Zirui22Ray/politics-dataset-demo'
  split with train_test_split(test_size=0.1, seed=42) → 9000 train samples
"""
import torch
import numpy as np
import json
import sys
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GT_DIR = Path("Vector/SSV/gemma2layer13")


def load_gt():
    """Load all GT artifacts for comparison."""
    gt_results = torch.load(str(GT_DIR / "falseness_ssv_results.pt"),
                            map_location="cpu", weights_only=False)
    gt_cv = torch.load(str(GT_DIR / "concept_vectors_full.pt"),
                       map_location="cpu", weights_only=False)

    gt_ssv_raw = gt_results["unnormalized_ssv"]
    gt_ssv = (torch.tensor(gt_ssv_raw, dtype=torch.float32)
              if not isinstance(gt_ssv_raw, torch.Tensor)
              else gt_ssv_raw.float())

    gt_important_dims = np.load(str(GT_DIR / "political_important_dimensions.npy"))

    # GT ANOVA 128-dim indices
    si = gt_cv["selected_indices"]
    gt_anova_128 = np.array(si.tolist() if hasattr(si, "tolist") else si)

    return gt_ssv, gt_important_dims, gt_anova_128


def load_gt_data():
    """Load the exact training data used by the GT notebook."""
    texts = np.load(str(GT_DIR / "train_texts.npy"), allow_pickle=True).tolist()
    labels = np.load(str(GT_DIR / "train_labels.npy")).tolist()
    
    # Separate by label (GT: label=1 → right/target, label=0 → left/contrast)
    target = [t for t, l in zip(texts, labels) if l == 1]
    contrast = [t for t, l in zip(texts, labels) if l == 0]
    
    print(f"GT data: {len(target)} target (label=1) + {len(contrast)} contrast (label=0)")
    return target, contrast


def extract_ours(target_data, contrast_data):
    """Run SSVExtractor with GT's exact data."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=DEVICE, dtype=torch.bfloat16
    )
    
    print("Loading SAE...")
    sae_obj, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_13/width_16k/canonical",
    )
    sae_obj = sae_obj.to(DEVICE)
    
    # Import the extractor
    from Steering.extractors.sae import SSVExtractor
    
    extractor = SSVExtractor(
        model=model,
        sae={13: sae_obj},
        layer=[13],
        hook_point=["post"],
        batch_size=32,
        ssv_lambda_dist=1.0,
        ssv_lambda_lm=0.5,
        ssv_lambda_reg=0.01,
        ssv_opt_lr=0.01,
        ssv_opt_steps=1,     # Match config: quick check
        ssv_feature_refinement_k=30,
    )
    
    print("Running SSVExtractor.extract()...")
    vectors = extractor.extract(
        target_data=target_data,
        contrast_data=contrast_data,
    )
    
    return vectors[13].float().cpu(), extractor.metadata


def compare(gt_ssv, gt_dims_30, gt_anova_128, our_ssv, our_metadata):
    """Print detailed comparison."""
    our_indices = our_metadata.get("selected_indices", [])

    print("\n" + "=" * 60)
    print("=== STAGE 1: ANOVA Feature Selection (128 dims) ===")
    gt_128_set = set(gt_anova_128.tolist())
    our_128_set = set()  # We don't save the intermediate 128; compare final 30
    print(f"GT ANOVA-128 count: {len(gt_128_set)}")

    print("\n=== STAGE 1b: Classifier Refinement (top 30 dims) ===")
    gt_30_set = set(gt_dims_30.tolist())
    our_30_set = set(our_indices)
    inter_30 = gt_30_set & our_30_set
    print(f"GT top-30: {sorted(gt_30_set)}")
    print(f"Our top-30: {sorted(our_30_set)}")
    print(f"Intersection: {len(inter_30)} / {len(gt_30_set)} "
          f"({len(inter_30)/len(gt_30_set)*100:.1f}%)")
    
    # How many of our 30 are at least in the GT's 128 ANOVA set?
    in_anova = our_30_set & gt_128_set
    print(f"Our 30 that are in GT's 128 ANOVA set: {len(in_anova)} / {len(our_30_set)}")

    print("\n=== FINAL VECTOR ===")
    print(f"GT SSV shape: {gt_ssv.shape}, norm: {gt_ssv.norm().item():.4f}")
    print(f"Our SSV shape: {our_ssv.shape}, norm: {our_ssv.norm().item():.4f}")

    cos_sim = F.cosine_similarity(gt_ssv.unsqueeze(0), our_ssv.unsqueeze(0)).item()
    l2 = torch.norm(gt_ssv - our_ssv).item()
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"L2 Distance: {l2:.6f}")

    # Non-zero element comparison
    gt_nz = (gt_ssv.abs() > 1e-8).sum().item()
    our_nz = (our_ssv.abs() > 1e-8).sum().item()
    print(f"GT non-zero dims: {gt_nz}")
    print(f"Our non-zero dims: {our_nz}")
    
    # Top-20 largest absolute-value dims
    gt_top20 = torch.topk(gt_ssv.abs(), 20).indices.tolist()
    our_top20 = torch.topk(our_ssv.abs(), 20).indices.tolist()
    top20_overlap = len(set(gt_top20) & set(our_top20))
    print(f"\nTop-20 abs-value dim overlap: {top20_overlap}/20")
    print(f"GT top-20 dims: {gt_top20}")
    print(f"Our top-20 dims: {our_top20}")
    print("=" * 60)


def main():
    gt_ssv, gt_dims_30, gt_anova_128 = load_gt()
    target_data, contrast_data = load_gt_data()
    our_ssv, our_metadata = extract_ours(target_data, contrast_data)
    compare(gt_ssv, gt_dims_30, gt_anova_128, our_ssv, our_metadata)


if __name__ == "__main__":
    main()
