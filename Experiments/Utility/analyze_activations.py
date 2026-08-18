"""PCA / t-SNE visualization of contrast pair activations."""
import os, sys
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 200

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
    os.makedirs("analysis_plots", exist_ok=True)

    for task_name, (ds_name, tgt_key, src_key) in TASKS.items():
        print(f"\n=== {task_name} ({ds_name}) ===")
        data = loader.load(ds_name, n_samples=N_SAMPLES, format=True, apply_chat_template=False)

        targets = [d[tgt_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        contrasts = [d[src_key] for d in data if d.get(tgt_key) and d.get(src_key)]
        print(f"  {len(targets)} valid pairs (target={tgt_key}, contrast={src_key})")

        if len(targets) < 10:
            print("  Not enough data, skipping")
            continue

        acts_tgt = get_acts(model, targets, LAYER)
        acts_src = get_acts(model, contrasts, LAYER)

        X = np.vstack([acts_src, acts_tgt])
        y = np.array([0]*len(acts_src) + [1]*len(acts_tgt))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ax = axes[0, 0]
        ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c="blue", alpha=0.4, s=8, label="Source")
        ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c="red", alpha=0.4, s=8, label="Target")
        ax.set_title(f"{task_name.upper()} — PCA\nvar={pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

        # Centroids
        centroid_src = pca.transform(acts_src.mean(0, keepdims=True))[0]
        centroid_tgt = pca.transform(acts_tgt.mean(0, keepdims=True))[0]
        ax = axes[0, 1]
        ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c="blue", alpha=0.15, s=4)
        ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c="red", alpha=0.15, s=4)
        ax.scatter([centroid_src[0]], [centroid_src[1]], c="blue", marker="X", s=200, edgecolors="black", zorder=5)
        ax.scatter([centroid_tgt[0]], [centroid_tgt[1]], c="red", marker="X", s=200, edgecolors="black", zorder=5)
        ax.plot([centroid_src[0], centroid_tgt[0]], [centroid_src[1], centroid_tgt[1]], "k--", lw=2, label="CAA direction")
        ax.set_title(f"{task_name.upper()} — Centroids")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
        ax = axes[1, 0]
        ax.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c="blue", alpha=0.4, s=8, label="Source")
        ax.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c="red", alpha=0.4, s=8, label="Target")
        ax.set_title(f"{task_name.upper()} — t-SNE (perp={min(30, len(X)-1)})")
        ax.legend(fontsize=8)

        # Cumulative variance
        pca_full = PCA().fit(X)
        ax = axes[1, 1]
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        ax.plot(range(1, len(cumvar)+1), cumvar, "b-", lw=2)
        ax.axhline(0.9, color="r", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(0.95, color="g", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Number of PCs")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(f"{task_name.upper()} — variance curve")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 50)

        plt.tight_layout()
        path = f"Analysis/{task_name}_activation_2d.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close()

    print("\nDone! Plots in Analysis/")

if __name__ == "__main__":
    main()
