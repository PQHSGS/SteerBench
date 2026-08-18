"""
Clean tail analysis: per-eval-sample sum distance to tail clusters.
Check separation visually before comparing accuracy.
"""
import json, torch, numpy as np, glob, re
from pathlib import Path
from transformer_lens import HookedTransformer

LAYER = 14
TAIL_ALPHA_THRESH = 3.5
PCTL = 99

def hill(x, k=None):
    n = len(x); k = max(2, int(n * 0.1)); k = min(k, n // 4)
    s = np.sort(np.abs(x))
    if k < 2 or len(s) <= k or s[-k-1] <= 0: return np.nan
    g = (np.log(s[-k:]) - np.log(max(s[-k-1], 1e-12))).mean()
    return 1.0/g if g > 1e-12 else np.inf

def model_l14(prompts, bs=16):
    m = HookedTransformer.from_pretrained("google/gemma-2-2b-it", device="cpu")
    m.to("cuda")
    acts = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i+bs]
        _, c = m.run_with_cache(b, names_filter=lambda n: n == f"blocks.{LAYER}.hook_resid_pre")
        acts.append(c[f"blocks.{LAYER}.hook_resid_pre"].cpu().float()[:, -1, :])
    del m; torch.cuda.empty_cache()
    return torch.cat(acts, dim=0).numpy()

CHARS_PATHS = {
    "deception": {10: "Vector/CHARS/Gemma/deception_k10/metadata.pt"},
    "toxic": {10: "Vector/CHARS/Gemma/toxic/metadata.pt"},
    "refusal": {10: "Vector/CHARS/Gemma/refusal_K10/metadata.pt"},
    "evil": {10: "Vector/CHARS/Gemma/evil_K10/metadata.pt"},
}

def run(task):
    print(f"\n{'='*60}")
    print(f"  {task}")
    print(f"{'='*60}")

    # Load extraction activations
    pt = Path(f"Experiments/TailAnalysis/{task}_activations.pt")
    if not pt.exists(): print("No activations.pt"); return
    t = torch.load(str(pt), map_location="cpu", weights_only=False)
    extr = np.concatenate([t["target_acts"].float().numpy(), t["contrast_acts"].float().numpy()], axis=0)
    print(f"  Extraction: {len(extr)}")

    # Load CHARS K=10 centroids if in same space (full 2304D)
    md = torch.load(CHARS_PATHS[task][10], map_location="cpu", weights_only=True)
    cA = md["chars_centroids_A"][LAYER].float().numpy()
    pca_k = md.get("chars_pca_k", 0)
    print(f"  Centroids: {cA.shape} (pca_k={pca_k})")

    # Use full 2304D always. If CHARS centroids are PCA (Toxic), run KMeans instead.
    extr_w = extr
    if pca_k > 0 and cA.shape[1] != extr.shape[1]:
        print("  CHARS centroids in PCA space. Running KMeans in full 2304D.")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=10, random_state=42, n_init=10)
        km.fit(extr_w)
        cA_w = km.cluster_centers_
        print(f"  KMeans centroids: {cA_w.shape}")
    else:
        cA_w = cA
    print(f"  Working space: {extr_w.shape[1]}D")

    # Tail dims
    alphas = np.array([hill(extr_w[:, j]) for j in range(extr_w.shape[1])])
    td = np.nan_to_num(alphas, nan=999) < TAIL_ALPHA_THRESH
    ntd = td.sum()
    print(f"  Tail dims: {ntd}/{extr_w.shape[1]}")
    if ntd < 2: print("  Too few. SKIP"); return

    # Tail samples in extraction
    tv = np.abs(extr_w[:, td]); th = np.percentile(tv, PCTL, axis=0)
    is_t = (tv >= th[None, :]).any(axis=1)
    print(f"  Tail samples: {is_t.sum()}/{len(extr_w)}")

    # Assign extraction to centroids (L2)
    d2 = ((extr_w[:, None, :] - cA_w[None, :, :]) ** 2).sum(axis=-1)
    assign = d2.argmin(axis=1)
    K = cA_w.shape[0]
    tf = np.zeros(K)
    print(f"\n  Centroid tail_frac (raw, unsorted):")
    for i in range(K):
        m = assign == i
        if m.sum() > 0: tf[i] = is_t[m].mean()
        print(f"    C{i:2d}: n={m.sum():4d}  tail_frac={tf[i]:.4f}")
    print(f"  Sorted: {np.array2string(np.sort(tf), precision=4, suppress_small=True)}")

    # Tail clusters = top 3
    tail_set = set(np.argsort(tf)[-3:])
    print(f"  Tail clusters: {sorted(tail_set)}  tf={[tf[i] for i in tail_set]}")

    # Load eval prompts
    # Handle refusal's different filename pattern
    if task == "refusal":
        ev_fs = sorted(glob.glob(f"Results/chars/eval_gemma_refusal_response_coeff_*.json") + glob.glob(f"Results/chars/eval_gemma_refusal_coeff_*.json"))
    else:
        ev_fs = sorted(glob.glob(f"Results/chars/eval_gemma_{task}_coeff_*.json"))
    seen, all_p, all_f, all_ic = [], [], [], []
    for f in ev_fs:
        try:
            with open(f) as fp: d = json.load(fp)
        except: continue
        for s in d["result"]["samples"]:
            p = s["prompt"]
            if p not in seen:
                seen.append(p)
                all_p.append(p)
                all_f.append(f)
                all_ic.append(s["is_correct"])
    all_ic = np.array(all_ic, dtype=bool)
    print(f"  Unique eval prompts: {len(all_p)}")

    # Get L14 activations for all eval prompts (full 2304D)
    ev = model_l14(all_p)

    # Distances eval to centroids
    d2_eval = ((ev[:, None, :] - cA_w[None, :, :]) ** 2).sum(axis=-1)
    med_all = np.median(d2_eval, axis=1, keepdims=True).clip(min=1e-8)
    # Sum kernel weights to tail clusters (higher = closer to tail)
    tail_k = np.zeros(len(ev))
    for i in tail_set:
        tail_k += np.exp(-d2_eval[:, i] / (2.0 * med_all[:, 0]))
    # Also sum distance to tail clusters (lower = closer)
    tail_d = d2_eval[:, list(tail_set)].sum(axis=1)

    # Check separation: sort by tail_k, print extremes
    o = np.argsort(tail_k)
    print(f"\n  Tail kernel sum (k_eval):")
    print(f"    Top 5: {tail_k[o[-5:]]}")
    print(f"    Bot 5: {tail_k[o[:5]]}")
    print(f"    Mean: {tail_k.mean():.4f}  Std: {tail_k.std():.4f}")
    # Percentiles
    for p in [1, 10, 25, 50, 75, 90, 99]:
        print(f"    P{p:2d}: {np.percentile(tail_k, p):.6f}")

    # Same for tail distance
    print(f"\n  Sum distance to tail clusters (d_eval):")
    print(f"    Nearest 5: {tail_d[o[:5]]}")
    print(f"    Farthest 5: {tail_d[o[-5:]]}")
    print(f"    Mean: {tail_d.mean():.4f}  Std: {tail_d.std():.4f}")
    for p in [1, 10, 25, 50, 75, 90, 99]:
        print(f"    P{p:2d}: {np.percentile(tail_d, p):.4f}")

    # Compare: top 20% by tail_k vs bottom 80%
    th80 = np.percentile(tail_k, 80)
    high = tail_k >= th80
    n_high, n_low = high.sum(), (~high).sum()
    if n_high > 0 and n_low > 0:
        ha = all_ic[high].mean(); la = all_ic[~high].mean()
        print(f"\n  Comparison (top 20% tail_k vs rest):")
        print(f"    High: {n_high} acc={ha:.4f}")
        print(f"    Low:  {n_low} acc={la:.4f}")
        print(f"    Delta: {ha-la:+.4f}")

    # Per eval file
    print(f"\n  Per eval file:")
    print(f"  {'coeff':>6s}  {'acc':>5s}  {'n_high':>6s}  {'high_acc':>7s}  {'low_acc':>7s}  {'delta':>6s}")
    print(f"  {'-'*48}")
    for f in ev_fs:
        try:
            with open(f) as fp: d = json.load(fp)
        except: continue
        ps = [s["prompt"] for s in d["result"]["samples"]]
        ic = np.array([s["is_correct"] for s in d["result"]["samples"]], dtype=bool)
        idx = [all_p.index(p) for p in ps]
        h = high[idx]; l = ~h
        nh = h.sum(); nl = l.sum()
        ha = ic[h].mean() if nh > 0 else float('nan')
        la = ic[l].mean() if nl > 0 else float('nan')
        fa = d["result"]["accuracy"]
        m = re.search(r'coeff_(\d+(?:p\d+)?)', Path(f).stem)
        cs = m.group(1) if m else "?"
        d = ha - la if (not np.isnan(ha) and not np.isnan(la)) else float('nan')
        print(f"  {cs:>6s}  {fa:.3f}  {nh:6d}  {ha:7.3f}  {la:7.3f}  {d:+.3f}")

if __name__ == "__main__":
    for t in ["deception", "refusal", "toxic", "evil"]:
        run(t)
