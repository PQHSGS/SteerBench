"""Inspect P* coupling matrix: which centroids get coupling mass vs norm."""
import sys, json, torch
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
LAYER = 14

def inspect(task, K):
    if task == "toxic" and K == 10:
        path = Path("Vector/CHARS/Gemma/toxic/metadata.pt")
    elif task == "evil" and K == 20:
        path = Path("Vector/CHARS/Gemma/evil/metadata.pt")
    else:
        path = Path(f"Vector/CHARS/Gemma/{task}_K{K}/metadata.pt")
    if not path.exists():
        return None
    md = torch.load(str(path), map_location="cpu", weights_only=True)
    lk = LAYER if LAYER in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    cA = md["chars_centroids_A"][lk].float()
    cB = md["chars_centroids_B"][lk].float()
    P = md["chars_coupling"][lk].float()
    KA = P.shape[0]
    KB = P.shape[1]
    
    norms_A = cA.norm(dim=1).numpy()
    norms_B = cB.norm(dim=1).numpy()
    mass_from = P.sum(dim=1).numpy()   # [KA]
    mass_to = P.sum(dim=0).numpy()     # [KB]
    
    # Entropy of coupling = how uniform is mass distribution?
    p = mass_from / mass_from.sum()
    entropy = -np.sum(p * np.log(p + 1e-10)) / np.log(KA)
    
    # Sort by norm
    idx = np.argsort(norms_A)
    print(f"\n  {task} K={K} (A={KA}, B={KB}) | total mass={mass_from.sum():.2f} | entropy={entropy:.3f}")
    print(f"  {'#':>3s} {'norm':>8s} {'mass_from':>12s} {'mass_to':>12s} {'dormant':>8s}")
    for i in idx:
        dormant = "YES" if (i == idx[-1] and mass_from[i] < 1.0/KA) else ""
        print(f"  {i:>3d} {norms_A[i]:>8.1f} {mass_from[i]:>12.6f} {mass_to[i]:>12.6f} {dormant:>8s}")
    
    rho_f, _ = spearmanr(norms_A, mass_from)
    rho_t, _ = spearmanr(norms_B, mass_to)
    cv = norms_A.std() / norms_A.mean()
    print(f"  CV={cv:.4f} | ρ(norm, mass_from)={rho_f:.4f} | ρ(norm_B, mass_to)={rho_t:.4f}")
    
    return {"task": task, "K": K, "cv": cv, "rho_from": rho_f, "rho_to": rho_t,
            "entropy": entropy, "mass_from": mass_from.tolist(), "norms_A": norms_A.tolist()}

results = []
for task in ["toxic", "evil"]:
    for K in [3, 5, 10, 20]:
        r = inspect(task, K)
        if r:
            results.append(r)

print("\n\n=== SUMMARY ===")
print(f"  {'task':>5s} K={K:>2s}  {'CV':>6s}  {'entropy':>8s}  {'ρ(mass_from,norm)':>18s}  {'ρ(mass_to,norm_B)':>18s}")
for r in results:
    print(f"  {r['task']:>5s} K={r['K']:>2d}  {r['cv']:.4f}  {r['entropy']:.3f}  {r['rho_from']:>18.4f}  {r['rho_to']:>18.4f}")

# Save for analysis
Path("Experiments/coupling_results/p_star_analysis.json").write_text(json.dumps(results, indent=2))
print("\nSaved to Experiments/coupling_results/p_star_analysis.json")
