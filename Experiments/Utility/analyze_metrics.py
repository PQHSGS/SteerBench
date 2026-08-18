"""Compute numerical metrics for activation separation analysis."""
import os, sys
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 100

TASKS = {
    "toxic": ("toxic_jigsaw", "correct_prompt", "false_prompt"),
    "deception": ("cais_mask", "correct_prompt", "false_prompt"),
    "refusal": ("refusal", "correct_prompt", "false_prompt"),
}

def get_acts(model, texts, layer=14, batch_size=4):
    model.reset_hooks()
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [model.tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        ) for t in batch]
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre")
        acts = cache[f"blocks.{layer}.hook_resid_pre"][:, -1, :]
        all_acts.append(acts.detach().cpu())
    return torch.cat(all_acts, dim=0).float().numpy()

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", torch_dtype=torch.bfloat16)
    model.to("cuda")
    loader = DataLoader()

    for task_name, (ds_name, tgt_key, src_key) in TASKS.items():
        print(f"\n{'='*60}")
        print(f"{task_name.upper()}")
        print(f"{'='*60}")

        data = loader.load(ds_name, n_samples=N_SAMPLES, format=True, apply_chat_template=False)
        targets = [d[tgt_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        contrasts = [d[src_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        print(f"N={len(targets)} pairs")

        acts_tgt = get_acts(model, targets, LAYER)
        acts_src = get_acts(model, contrasts, LAYER)

        # 1. Centroid distance
        centroid_src = acts_src.mean(0)
        centroid_tgt = acts_tgt.mean(0)
        centroid_dist = np.linalg.norm(centroid_tgt - centroid_src)
        print(f"  Centroid distance: {centroid_dist:.3f}")

        # 2. Cosine between centroids
        cos = np.dot(centroid_src, centroid_tgt) / (np.linalg.norm(centroid_src) * np.linalg.norm(centroid_tgt))
        print(f"  Centroid cosine: {cos:.4f}")

        # 3. PCA explained variance
        X = np.vstack([acts_src, acts_tgt])
        pca = PCA().fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"  PCs for 50% var: {int(np.searchsorted(cumvar, 0.50) + 1)}")
        print(f"  PCs for 90% var: {int(np.searchsorted(cumvar, 0.90) + 1)}")
        print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.3%}")
        print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.3%}")
        print(f"  Top-5 cumulative: {sum(pca.explained_variance_ratio_[:5]):.3%}")

        # 4. Class separation in PCA1
        X_pca1 = pca.transform(X)[:, 0]
        src_pca1 = X_pca1[:len(acts_src)]
        tgt_pca1 = X_pca1[len(acts_src):]
        sep = abs(src_pca1.mean() - tgt_pca1.mean()) / (src_pca1.std() + tgt_pca1.std() + 1e-8)
        print(f"  Cohen's d on PC1: {sep:.3f}")

        # 5. Logistic regression accuracy (5-fold CV estimate)
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, np.array([0]*len(acts_src) + [1]*len(acts_tgt)), cv=5)
        print(f"  Logistic regression CV accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

        # 6. 1-NN accuracy (if classes are well-separated into clusters)
        knn = KNeighborsClassifier(n_neighbors=1)
        scores_knn = cross_val_score(knn, X, np.array([0]*len(acts_src) + [1]*len(acts_tgt)), cv=5)
        print(f"  1-NN CV accuracy: {scores_knn.mean()*100:.1f}% ± {scores_knn.std()*100:.1f}%")

    print("\nDone!")

if __name__ == "__main__":
    main()
