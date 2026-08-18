"""PCA plots colored by K-means clusters — see if K-means finds structure."""
import os, sys
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 100
K = 10

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
        print(f"\n=== {task_name} ===")
        data = loader.load(ds_name, n_samples=N_SAMPLES, format=True, apply_chat_template=False)
        targets = [d[tgt_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        contrasts = [d[src_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        print(f"  {len(targets)} pairs")

        acts_tgt = get_acts(model, targets, LAYER)
        acts_src = get_acts(model, contrasts, LAYER)
        X = np.vstack([acts_src, acts_tgt])
        y_class = np.array([0]*len(acts_src) + [1]*len(acts_tgt))

        # K-means on the full 2304D data
        kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Also K-means only on source activations (like CHARS does)
        kmeans_src = KMeans(n_clusters=K, random_state=42, n_init="auto")
        labels_src = kmeans_src.fit_predict(acts_src)
        centroids_src = kmeans_src.cluster_centers_

        # Centroid norms and spread
        norms_src = np.linalg.norm(centroids_src, axis=1)
        print(f"  Source centroid norms: min={norms_src.min():.1f}, max={norms_src.max():.1f}, "
              f"spread={norms_src.max()/norms_src.min():.2f}x, CV={norms_src.std()/norms_src.mean():.3f}")

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(centroids)

        # Also get CAA direction in PCA space
        centroid_src = acts_src.mean(0)
        centroid_tgt = acts_tgt.mean(0)
        caa_dir = centroid_tgt - centroid_src
        caa_pca = pca.transform(caa_dir.reshape(1, -1))[0]
        caa_pca = caa_pca / np.linalg.norm(caa_pca)

        # For each cluster centroid, compute cos to CAA
        cos_to_caa = []
        for c in centroids_src:
            cos = np.dot(c, caa_dir) / (np.linalg.norm(c) * np.linalg.norm(caa_dir) + 1e-12)
            cos_to_caa.append(cos)
        print(f"  Source centroids cos to CAA: min={min(cos_to_caa):.3f}, max={max(cos_to_caa):.3f}, "
              f"mean={np.mean(cos_to_caa):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Panel 1: PCA colored by source/target class
        ax = axes[0, 0]
        ax.scatter(X_pca[y_class==0, 0], X_pca[y_class==0, 1], c="blue", alpha=0.3, s=6, label="Source")
        ax.scatter(X_pca[y_class==1, 0], X_pca[y_class==1, 1], c="red", alpha=0.3, s=6, label="Target")
        ax.set_title(f"{task_name.upper()} — PCA by class")
        ax.legend(fontsize=8)

        # Panel 2: PCA colored by K-means cluster
        ax = axes[0, 1]
        colors = plt.cm.tab10(np.linspace(0, 1, K))
        for k in range(K):
            mask = labels == k
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[k]], alpha=0.4, s=6, label=f"Cluster {k}")
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=100, zorder=5)
        ax.set_title(f"{task_name.upper()} — PCA colored by K-means (K={K})")
        ax.legend(fontsize=6, ncol=2)

        # Panel 3: Centroids only, with CAA direction
        ax = axes[1, 0]
        for k in range(K):
            ax.scatter([centroids_pca[k, 0]], [centroids_pca[k, 1]], c=[colors[k]], marker="X", s=200, edgecolors="black", zorder=5)
        # Plot CAA direction from the origin of PCA
        mid = centroids_pca.mean(0)
        ax.arrow(mid[0], mid[1], caa_pca[0]*50, caa_pca[1]*50, head_width=5, head_length=5, fc="black", ec="black", label="CAA dir")
        ax.set_title(f"{task_name.upper()} — {K} centroids + CAA direction")
        ax.legend(fontsize=8)

        # Panel 4: Centroid norms histogram
        ax = axes[1, 1]
        ax.bar(range(K), norms_src, color=[colors[k] for k in range(K)])
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Centroid norm")
        ax.set_title(f"{task_name.upper()} — Source centroid norms (CV={norms_src.std()/norms_src.mean():.3f})")

        plt.tight_layout()
        path = f"Analysis/{task_name}_kmeans_clusters.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close()

    print("\nDone!")

if __name__ == "__main__":
    main()
