"""
Tail proximity at steered generated tokens.
Uses SteeringPipeline + CHARS. Captures post-steer activations
at 1st, 3rd, 10th generated token (hook on L14 resid_pre AFTER steer hook).
Metric: RBF kernel sum to nearest tail clusters.

Usage:
  conda activate sae_circuit
  CUDA_VISIBLE_DEVICES=2 python -m Experiments.TailAnalysis.eval_tail_genpos
"""

import json, torch, numpy as np, glob, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Steering.config.pipeline import PipelineConfig
from Steering.pipeline import SteeringPipeline

LAYER = 14
TAIL_ALPHA_THRESH = 3.5
PCTL = 99
POSITIONS = [1, 3, 10]  # forward-pass steps for 1st, 3rd, 10th generated token
MAX_NEW_TOKENS = 15      # need >= 10 for 10th token

CHARS_CFG = {
    "deception": "Configs/Eval/CHARS/Gemma/gemma_deception.json",
    "toxic":     "Configs/Eval/CHARS/Gemma/gemma_toxic.json",
    "evil":      "Configs/Eval/CHARS/Gemma/gemma_evil.json",
    "refusal":   "Configs/Eval/CHARS/Gemma/gemma_refusal_response.json",
}

K_METADATA = {
    # (K, metadata_path) — full 2304D entries only (no pca_k)
    "toxic": [
        (5,  "Vector/CHARS/Gemma/toxic_K5/metadata.pt"),
        (10, "Vector/CHARS/Gemma/toxic_K10/metadata.pt"),
        (20, "Vector/CHARS/Gemma/toxic_K20/metadata.pt"),
    ],
    "evil": [
        (3,  "Vector/CHARS/Gemma/evil_K3/metadata.pt"),
        (5,  "Vector/CHARS/Gemma/evil_K5/metadata.pt"),
        (10, "Vector/CHARS/Gemma/evil_K10/metadata.pt"),
        (20, "Vector/CHARS/Gemma/evil_K20/metadata.pt"),
    ],
    "deception": [
        (5,  "Vector/CHARS/Gemma/deception_K5/metadata.pt"),
        (10, "Vector/CHARS/Gemma/deception_k10/metadata.pt"),
        (20, "Vector/CHARS/Gemma/deception_K20/metadata.pt"),
        (50, "Vector/CHARS/Gemma/deception_K50/metadata.pt"),
    ],
    "refusal": [
        (5,  "Vector/CHARS/Gemma/refusal_K5/metadata.pt"),
        (10, "Vector/CHARS/Gemma/refusal_K10/metadata.pt"),
        (20, "Vector/CHARS/Gemma/refusal_K20/metadata.pt"),
        (50, "Vector/CHARS/Gemma/refusal_K50/metadata.pt"),
    ],
}

OUTPUT_DIR = Path("Experiments/TailAnalysis")

def hill(x):
    n = len(x); k = max(2, int(n * 0.1)); k = min(k, n // 4)
    s = np.sort(np.abs(x))
    if k < 2 or len(s) <= k or s[-k-1] <= 0: return np.nan
    g = (np.log(s[-k:]) - np.log(max(s[-k-1], 1e-12))).mean()
    return 1.0/g if g > 1e-12 else np.inf


def load_extraction(task):
    """Load pooled target+contrast activations for a task."""
    pt = OUTPUT_DIR / f"{task}_activations.pt"
    if not pt.exists():
        print(f"  ERROR: no extraction activations at {pt}")
        return None, None
    t = torch.load(str(pt), map_location="cpu", weights_only=False)
    extr = np.concatenate([t["target_acts"].float().numpy(),
                           t["contrast_acts"].float().numpy()], axis=0)
    return extr, t


def compute_tail_clusters(extr, cA_w):
    """Hill tail dims → tail samples → per-centroid tail_frac → top-3 tail clusters."""
    if extr is None or cA_w is None:
        return None

    # Hill tail dims on extraction
    alphas = np.array([hill(extr[:, j]) for j in range(extr.shape[1])])
    td = np.nan_to_num(alphas, nan=999) < TAIL_ALPHA_THRESH
    ntd = td.sum()
    print(f"  Tail dims: {ntd}/{extr.shape[1]}")
    if ntd < 2: return None

    # Tail samples
    tv = np.abs(extr[:, td]); th = np.percentile(tv, PCTL, axis=0)
    is_t = (tv >= th[None, :]).any(axis=1)
    print(f"  Tail samples: {is_t.sum()}/{len(extr)}")

    # Assign extraction→centroids, compute tail_frac per centroid
    d2 = ((extr[:, None, :] - cA_w[None, :, :]) ** 2).sum(axis=-1)
    assign = d2.argmin(axis=1)
    K = cA_w.shape[0]
    tf = np.array([is_t[assign == i].mean() if (assign == i).sum() > 0 else 0.0 for i in range(K)])
    tail_set = set(np.argsort(tf)[-3:])
    print(f"  Tail clusters: {sorted(tail_set)}  tf={[tf[i] for i in tail_set]}")

    return {"cA_w": cA_w, "tail_set": tail_set, "K": K, "tail_frac": tf}


def load_centroids_from_metadata(metadata_path, layer=LAYER):
    """Load CHARS centroids from a metadata file. Returns (cA, cB, coupling, pca_k) or None."""
    p = Path(metadata_path)
    if not p.exists():
        print(f"  SKIP: metadata not found: {metadata_path}")
        return None
    md = torch.load(str(p), map_location="cpu", weights_only=True)
    lk = layer if layer in md["chars_centroids_A"] else list(md["chars_centroids_A"].keys())[0]
    cA = md["chars_centroids_A"][lk].float().numpy()
    cB = md["chars_centroids_B"][lk].float().numpy()
    P = md["chars_coupling"][lk].float().numpy()
    pca_k = md.get("chars_pca_k", 0)
    return cA, cB, P, pca_k


def get_eval_prompts_and_accuracy(task):
    """Load unique eval prompts + their accuracy per coeff from existing results."""
    if task == "refusal":
        ev_fs = sorted(
            glob.glob("Results/chars/eval_gemma_refusal_response_coeff_*.json") +
            glob.glob("Results/chars/eval_gemma_refusal_coeff_*.json")
        )
    else:
        ev_fs = sorted(glob.glob(f"Results/chars/eval_gemma_{task}_coeff_*.json"))

    # Build unique prompt index
    seen = {}
    prompt_list = []
    coeff_data = {}  # coeff_str -> {prompt_idx -> is_correct}
    for f in ev_fs:
        m = re.search(r'coeff_(\d+(?:p\d+)?)', Path(f).stem)
        if not m: continue
        cs = m.group(1)
        try:
            with open(f) as fp: d = json.load(fp)
        except: continue
        if cs not in coeff_data:
            coeff_data[cs] = {}
        for s in d["result"]["samples"]:
            p = s["prompt"]
            if p not in seen:
                seen[p] = len(prompt_list)
                prompt_list.append(p)
            coeff_data[cs][seen[p]] = s["is_correct"]
    print(f"  Unique prompts: {len(prompt_list)}")
    print(f"  Coeff files: {sorted(coeff_data.keys())}")
    return prompt_list, coeff_data


def run_generation_and_capture(pipeline, prompts, coeff_val):
    """
    For each prompt, generate with CHARS steering and capture
    POST-steer activations at generated token positions 1, 3, 10.
    Hook on L14 resid_pre AFTER steer hook (sees steered value).
    """
    model = pipeline.model
    steer = pipeline.steer_model
    captured = {p: [] for p in POSITIONS}

    for i, prompt in enumerate(prompts):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(prompts)}]")

        cdict = {LAYER: coeff_val}
        step_ctr = [0]
        pos_acts = {}

        def make_hook(sc, pa):
            def hook(resid, hook):
                s = sc[0]; sc[0] += 1
                if s in POSITIONS:
                    # KV cache: s=1+ → resid shape (1, 1, d_model)
                    # resid[0, 0] = newly generated token (steered)
                    pa[s] = resid[0, 0].detach().clone().float().cpu()
                return resid
            return hook

        # 1) Add steer hooks (adds to blocks.14.hook_resid_pre)
        steer.setup_hooks(cdict)
        # 2) Add capture on SAME hook point AFTER steer hook → sees steered resid
        model.add_hook(f"blocks.{LAYER}.hook_resid_pre", make_hook(step_ctr, pos_acts), "fwd")
        # 3) Generate with steering
        try:
            _ = model.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                prepend_bos=True,
            )
        finally:
            model.reset_hooks()

        for p in POSITIONS:
            if p in pos_acts:
                captured[p].append(pos_acts[p])
            else:
                d = steer.kwargs["chars_centroids_A"][LAYER].shape[1]
                captured[p].append(torch.zeros(d))

    result = {}
    for p in POSITIONS:
        if captured[p]:
            result[p] = torch.stack(captured[p]).numpy()
        else:
            result[p] = np.array([])
    return result


def compute_rbf(ev_acts, cA_w, tail_set):
    """RBF kernel sum to tail clusters. Higher = closer to tail."""
    d2 = ((ev_acts[:, None, :] - cA_w[None, :, :]) ** 2).sum(axis=-1)
    med = np.median(d2, axis=1, keepdims=True).clip(min=1e-8)
    rbf_sum = np.zeros(len(ev_acts))
    for i in tail_set:
        rbf_sum += np.exp(-d2[:, i] / (2.0 * med[:, 0]))
    return rbf_sum


def score_split(scores, acc):
    """Split top 20% (highest RBF = closest to tail) vs rest."""
    th = np.percentile(scores, 80)
    high = scores >= th
    n_h, n_l = high.sum(), (~high).sum()
    if n_h < 1 or n_l < 1:
        return 0.0, 0.0, 0.0, n_h, n_l
    ha = acc[high].mean()
    la = acc[~high].mean()
    return ha, la, ha - la, n_h, n_l


def override_centroids_from_metadata(steer, metadata_result):
    """Override steer kwargs with centroids from a loaded metadata file."""
    cA, cB, P, pca_k = metadata_result
    steer.kwargs["chars_centroids_A"][LAYER] = torch.from_numpy(cA)
    steer.kwargs["chars_centroids_B"][LAYER] = torch.from_numpy(cB)
    steer.kwargs["chars_coupling"][LAYER] = torch.from_numpy(P)
    if pca_k > 0:
        steer.kwargs["chars_pca_k"] = pca_k
    else:
        # Clear any legacy concept projection matrices from the original config
        # to prevent dim mismatch during forward pass with full-dimensional centroids
        steer.kwargs.pop("chars_P_concept", None)
        steer.kwargs.pop("chars_X_mean", None)
        steer.kwargs.pop("chars_pca_k", None)
    return cA, cB, P


def run(task):
    print(f"\n{'='*70}")
    print(f"  {task}")
    print(f"{'='*70}")

    cfg_path = CHARS_CFG.get(task)
    if not cfg_path:
        print(f"  No config for {task}, SKIP"); return
    if not Path(cfg_path).exists():
        print(f"  Config not found: {cfg_path}, SKIP"); return

    # Setup pipeline once (model loading + auth)
    config = PipelineConfig.load(cfg_path)
    config.model.device = "cuda"
    config.model.max_new_tokens = MAX_NEW_TOKENS
    pipe = SteeringPipeline(config)
    pipe.setup()
    steer = pipe.steer_model
    print(f"  Model: {config.model.name}")

    # Load extraction activations once per task (used for tail-cluster definition)
    extr, _ = load_extraction(task)
    if extr is None:
        print("  FAILED: no extraction activations. SKIP")
        return

    # Load eval prompts + accuracy (shared across K variants)
    prompt_list, coeff_data = get_eval_prompts_and_accuracy(task)

    # Iterate K variants
    k_entries = K_METADATA.get(task, [])
    if not k_entries:
        print("  No K variants defined, SKIP"); return

    for k_val, meta_path in k_entries:
        print(f"\n  --- K={k_val} ({meta_path}) ---")

        # Load centroids for this K
        meta_result = load_centroids_from_metadata(meta_path)
        if meta_result is None:
            print(f"  SKIP K={k_val}: metadata not found"); continue

        cA, cB, P_ot, pca_k = meta_result
        print(f"  Centroids: {cA.shape} (pca_k={pca_k})")

        # Skip PCA-space centroids (dim mismatch with activations)
        if pca_k > 0 and cA.shape[1] != extr.shape[1]:
            print(f"  SKIP K={k_val}: PCA-space centroids (dim {cA.shape[1]} != {extr.shape[1]})")
            continue

        # Override pipeline centroids for this K
        override_centroids_from_metadata(steer, meta_result)

        # Compute tail clusters from these centroids + extraction data
        tc = compute_tail_clusters(extr, cA)
        if tc is None:
            print(f"  SKIP K={k_val}: no tail clusters found")
            continue

        cA_w, tail_set = tc["cA_w"], tc["tail_set"]

        # For each coeff, generate with steering + capture steered activations
        # Keep coeff=3.0 only
        for coeff_str in coeff_data.keys():
            coeff_val = float(coeff_str.replace("p", "."))
            if coeff_val != 3.0:
                continue
            
            print(f"\n    coeff={coeff_val} (K={k_val})")

            cd = coeff_data[coeff_str]
            idxs = sorted(cd.keys())
            acc = np.array([cd[i] for i in idxs], dtype=bool)
            prompts = [prompt_list[i] for i in idxs]
            print(f"    N={len(prompts)}  acc={acc.mean():.3f}")

            # Generate + capture steered activations at gen token positions
            gen_acts = run_generation_and_capture(pipe, prompts, coeff_val)

            # RBF metric per position
            for pos in [1, 3, 10]:
                acts = gen_acts.get(pos)
                if acts is None or len(acts) == 0: continue

                rbf = compute_rbf(acts, cA_w, tail_set)
                spread = (rbf.max() - rbf.min()) / max(rbf.mean(), 1e-12) * 100
                ha, la, delta, n_h, n_l = score_split(rbf, acc)
                pos_name = {1: "1st", 3: "3rd", 10: "10th"}[pos]
                print(f"    {pos_name} gen token: "
                      f"rbf spread={spread:>5.1f}%  close={n_h} acc={ha:.4f}  "
                      f"far={n_l} acc={la:.4f}  delta={delta:+.4f}")

    del steer; torch.cuda.empty_cache()


if __name__ == "__main__":
    for task in ["toxic", "deception", "evil", "refusal"]:
        run(task)
