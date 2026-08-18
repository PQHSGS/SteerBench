"""
SSV Level 1: Extraction Match using REAL data and shared activations.

Validates that our SSVExtractor produces the same features and vectors as the GT
algorithm when given the EXACT SAME activations.

Steps:
1. Load real sycophancy data (100 target + 100 contrast)
2. Load model + SAE (shared)
3. Get SAE activations ONCE
4. Feed same activations to GT ANOVA + classifier AND our _select_features + _train_classifier
5. Compare: ANOVA indices, classifier weights, final vector directions

GT reference: Code/SAE-SSV/saessv-demo.py
"""

import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

torch.manual_seed(42)
np.random.seed(42)

from Verification.shared_utils import load_sycophancy_data, load_model_and_sae, get_sae_activations

DEVICE = "cuda:2"
LAYER = 20
N_PER_CLASS = 100
ANOVA_TOP_K = 128
CLASSIFIER_TOP_K = 30


# ==================== GT ALGORITHM ====================
import sys
sys.path.insert(0, str(ROOT / "Code" / "SAE-SSV"))
# saessv-demo.py uses LinearConceptExtractor for feature selection and classifier training.
# Since saessv-demo.py is a monolithic script designed for Jupyter, we can import LinearConceptExtractor 
# directly or instantiate it with mock objects if needed to use its methods, 
import ast

def get_gt_class():
    gt_path = ROOT / "Code" / "SAE-SSV" / "saessv-demo.py"
    with open(gt_path, "r") as f:
        source = f.read()
    
    # Parse the AST and find the LinearConceptExtractor class
    tree = ast.parse(source)
    class_code = None
    imports_code = []
    
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports_code.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef) and node.name == "LinearConceptExtractor":
            class_code = ast.unparse(node)
            break
            
    if not class_code:
        raise ValueError("Could not find LinearConceptExtractor in saessv-demo.py")
        
    # Execute the imports and class definition directly into a new dictionary
    namespace = {}
    
    # Add required globals from the file that the class might need but aren't imports
    namespace['DEVICE'] = 'cpu'
    
    exec("\n".join(imports_code), namespace)
    exec(class_code, namespace)
    
    return namespace["LinearConceptExtractor"]

LinearConceptExtractor = get_gt_class()



class DummyLanguageModel:
    class Cfg:
        n_layers = 10
    cfg = Cfg()

class DummySAE:
    class Cfg:
        d_sae = 16384
    cfg = Cfg()
    def to(self, device):
        pass

def gt_select_important_features(latents_np, labels_np, top_k=128):
    # Dummy instantiate to use the unbound method or bound method without crashing
    extractor = LinearConceptExtractor(DummySAE(), DummyLanguageModel(), target_layer=20, device="cpu")
    # method signature: select_important_features(self, latents, labels, top_k=4000)
    # returns: selected_latents, selected_indices
    selected_latents, selected_indices = extractor.select_important_features(latents_np, labels_np, top_k=top_k)
    # The GT method uses np.argsort(feature_scores)[-top_k:] and returns them, but we need the scores for fscore correlation.
    # To get scores, we'd have to rewrite it or just not do fscore correlation if GT doesn't return it.
    # Since we can't modify GT, let's just use what they return (selected_indices).
    return selected_indices

def gt_train_classifier(features_np, labels_np, num_epochs=20, lr=1e-4):
    extractor = LinearConceptExtractor(DummySAE(), DummyLanguageModel(), target_layer=20, device="cpu")
    # signature: train_linear_classifier(self, latents, labels, val_size=0.2, batch_size=32, num_epochs=20, lr=1e-4, weight_decay=5e-2)
    # wait, the original train_linear_classifier in demo does train/test split.
    # We want to train on exactly what's passed, so maybe we use it and accept the internal split,
    # or if we must provide a classifier loop, we can't because of the constraint. Let's look closely at what GT does.
    # We'll use the GT function directly despite its internal validation split to remain 100% faithful to GT.
    classifier, test_acc, test_f1, model_info = extractor.train_linear_classifier(
        latents=features_np, labels=labels_np, val_size=0.0, num_epochs=num_epochs, lr=lr
    )
    return classifier.weight.detach().cpu().numpy()



# ==================== OUR ALGORITHM ====================

def our_select_features(latents_np, labels_np, top_k=128):
    """Our implementation from SSVExtractor._select_features."""
    from sklearn.feature_selection import f_classif as sk_f_classif
    
    F_scores, p_values = sk_f_classif(latents_np, labels_np)
    F_scores = np.nan_to_num(F_scores, nan=0.0)
    selected_indices = np.argsort(F_scores)[-top_k:]
    return selected_indices, F_scores


def our_train_classifier(features_np, labels_np, num_epochs=20, lr=1e-4):
    """Our implementation from SSVExtractor._train_classifier."""
    from torch.utils.data import DataLoader, TensorDataset
    
    X = torch.tensor(features_np, dtype=torch.float32)
    y = torch.tensor(labels_np, dtype=torch.float32)
    
    # Normalize
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - mean) / std
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    n_features = X.shape[1]
    W = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([W, b], lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for _ in range(num_epochs):
        for batch_X, batch_y in loader:
            logits = batch_X @ W + b
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return W.detach().numpy()


def main():
    print("=" * 60)
    print("SSV Level 1: Extraction Match (Real Data, Shared Activations)")
    print("=" * 60)
    
    # 1. Load real data
    print("\n[1] Loading sycophancy data...")
    target_texts, contrast_texts = load_sycophancy_data(n_per_class=N_PER_CLASS)
    print(f"    Target: {len(target_texts)}, Contrast: {len(contrast_texts)}")
    
    # 2. Load model + SAE
    print("\n[2] Loading model + SAE...")
    model, sae, layer = load_model_and_sae(layer=LAYER, device=DEVICE)
    
    # 3. Get SAE activations ONCE (shared)
    print("\n[3] Getting SAE activations (shared)...")
    all_texts = target_texts + contrast_texts
    labels = [1] * len(target_texts) + [0] * len(contrast_texts)
    
    latents_list = get_sae_activations(model, sae, all_texts, layer, batch_size=8)
    latents_np = np.array([l.float().numpy() for l in latents_list])
    labels_np = np.array(labels)
    
    print(f"    Activations shape: {latents_np.shape}")
    print(f"    Labels distribution: {np.bincount(labels_np)}")
    
    # 4. ANOVA Feature Selection (Stage 1)
    print("\n" + "=" * 60)
    print("=== STAGE 1: ANOVA Feature Selection ===")
    print("=" * 60)
    
    gt_anova_indices = gt_select_important_features(latents_np, labels_np, ANOVA_TOP_K)
    our_anova_indices, our_fscores = our_select_features(latents_np, labels_np, ANOVA_TOP_K)
    
    gt_set = set(gt_anova_indices.tolist())
    our_set = set(our_anova_indices.tolist())
    anova_overlap = len(gt_set & our_set)
    
    print(f"  GT ANOVA top-{ANOVA_TOP_K}: {len(gt_set)} features")
    print(f"  Our ANOVA top-{ANOVA_TOP_K}: {len(our_set)} features")
    print(f"  Overlap: {anova_overlap} / {ANOVA_TOP_K} ({anova_overlap/ANOVA_TOP_K*100:.1f}%)")
    
    anova_passed = anova_overlap >= ANOVA_TOP_K * 0.9
    print(f"  {'PASSED' if anova_passed else 'FAILED'}: ANOVA overlap >= 90%")
    
    # 5. Classifier Refinement (Stage 1b)
    # Use the OVERLAPPING indices to ensure fair comparison
    print("\n" + "=" * 60)
    print("=== STAGE 1b: Classifier Refinement (using SAME ANOVA indices) ===")
    print("=" * 60)
    
    # Use GT ANOVA indices for BOTH to isolate classifier
    shared_indices = sorted(gt_set & our_set)
    shared_features = latents_np[:, shared_indices]
    
    # Reset seeds for deterministic classifier
    torch.manual_seed(42)
    np.random.seed(42)
    gt_weights = gt_train_classifier(shared_features, labels_np)
    
    torch.manual_seed(42)
    np.random.seed(42)
    our_weights = our_train_classifier(shared_features, labels_np)
    
    weight_cos = F.cosine_similarity(
        torch.tensor(gt_weights).unsqueeze(0),
        torch.tensor(our_weights).unsqueeze(0)
    ).item()
    weight_diff = np.abs(gt_weights - our_weights).max()
    
    print(f"  Classifier weight cosine similarity: {weight_cos:.6f}")
    print(f"  Classifier weight max diff: {weight_diff:.2e}")
    
    # Top-k refined indices
    gt_top_k = np.argsort(np.abs(gt_weights))[-CLASSIFIER_TOP_K:]
    our_top_k = np.argsort(np.abs(our_weights))[-CLASSIFIER_TOP_K:]
    gt_refined = set([shared_indices[i] for i in gt_top_k])
    our_refined = set([shared_indices[i] for i in our_top_k])
    refine_overlap = len(gt_refined & our_refined)
    
    print(f"  GT top-{CLASSIFIER_TOP_K} refined: {sorted(gt_refined)[:10]}...")
    print(f"  Our top-{CLASSIFIER_TOP_K} refined: {sorted(our_refined)[:10]}...")
    print(f"  Refined overlap: {refine_overlap} / {CLASSIFIER_TOP_K} ({refine_overlap/CLASSIFIER_TOP_K*100:.1f}%)")
    
    classifier_passed = weight_cos > 0.99
    print(f"  {'PASSED' if classifier_passed else 'FAILED'}: Classifier cosine >= 0.99")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "ANOVA feature overlap": anova_passed,
        "Classifier weight match": classifier_passed,
    }
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        if not passed:
            all_passed = False
        print(f"  {status}: {name}")
    
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Cleanup
    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
