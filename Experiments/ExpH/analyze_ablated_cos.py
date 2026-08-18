"""
Compare cos(h,r̂) for ORIGINAL vs APPROACH 1 (ablated) vectors.
Same multi-layer activation analysis, both vector sets.
"""
import sys; sys.path.insert(0, ".")
import torch, json, numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from Steering.data.loader import EvalDataLoader

device = "cuda:0"   # CUDA_VISIBLE_DEVICES=2 → maps to GPU 2
dtype = torch.bfloat16
model_name = "google/gemma-2-2b-it"
STEER_LAYERS = [14, 18, 22, 25]
TARGET_LAYERS = [14, 18, 22, 25]
N_SAMPLES = 50
BATCH_SIZE = 8

TASKS = {
    "evil":   ("Vector/CAA/Gemma/evil",        "Vector/CAA/Gemma_ablated/evil",        "toxic",      5),
    "toxic":  ("Vector/CAA/Gemma/toxic",       "Vector/CAA/Gemma_ablated/toxic",       "evil",       5),
    "deception": ("Vector/CAA/Gemma/deception", "Vector/CAA/Gemma_ablated/deception", "toxic",      5),
}

print("Loading model...")
model = HookedTransformer.from_pretrained(model_name, dtype=dtype, device=device)
model.eval()

# === Load r̂ vectors from saved file ===
r_vecs = torch.load("Experiments/ExpH/results/refusal_vecs.pt", map_location=device)
r_vecs = {int(k): v.to(dtype=dtype) for k, v in r_vecs.items()}
for L in TARGET_LAYERS:
    print(f"  r̂ layer {L:2d}: norm={r_vecs[L].norm():.4f}")

# === Load test prompts ===
eval_loader = EvalDataLoader()
test_data_dict = {}
for name, (_, _, ds_name, _) in TASKS.items():
    data = eval_loader.load(ds_name, n_samples=N_SAMPLES, apply_chat_template=True, tokenizer=model.tokenizer)
    prompts = [d["question"] for d in data if isinstance(d, dict) and "question" in d]
    test_data_dict[name] = prompts
    print(f"{name}: {len(prompts)} prompts from {ds_name}")

# === Outer loop: steer at each STEER_LAYER independently ===
all_results = {}
for steer_layer in STEER_LAYERS:
    print(f"\n{'='*70}")
    print(f"STEER AT LAYER {steer_layer}")
    print(f"{'='*70}")

    results = {}
    for task_name, (orig_path, ablate_path, _, coeff) in TASKS.items():
        # Load both vector sets
        vec_orig_raw = torch.load(orig_path + "/vector.pt", map_location=device)
        vec_ablate_raw = torch.load(ablate_path + "/vector.pt", map_location=device)
        v_orig = vec_orig_raw[steer_layer].to(dtype=dtype)
        v_ablate = vec_ablate_raw[steer_layer].to(dtype=dtype)

        prompts = test_data_dict[task_name]

        # Per-target-layer accumulators for ORIGINAL and ABLATED
        base_cos_all = {L: [] for L in TARGET_LAYERS}
        steer_cos_orig_all = {L: [] for L in TARGET_LAYERS}
        steer_cos_ablate_all = {L: [] for L in TARGET_LAYERS}

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            tokens = model.to_tokens(batch)
            B = tokens.size(0)

            # -- Baseline: unsteered activations at all TARGET_LAYERS --
            base_cache = {}
            model.reset_hooks()
            for L in TARGET_LAYERS:
                def make_cap(L):
                    def cap(act, **kw):
                        if L not in base_cache: base_cache[L] = []
                        base_cache[L].append(act[:, -1, :].clone().detach())
                        return act
                    return cap
                model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
            with torch.no_grad(): _ = model(tokens)
            model.reset_hooks()

            # -- Steered with ORIGINAL vector --
            steer_cache_orig = {}
            model.reset_hooks()
            def steer_orig(act, **kw):
                act[:, -1, :] += coeff * v_orig.to(dtype=act.dtype)
                return act
            model.add_hook(f"blocks.{steer_layer}.hook_resid_pre", steer_orig, "fwd")
            for L in TARGET_LAYERS:
                def make_cap(L):
                    def cap(act, **kw):
                        if L not in steer_cache_orig: steer_cache_orig[L] = []
                        steer_cache_orig[L].append(act[:, -1, :].clone().detach())
                        return act
                    return cap
                model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
            with torch.no_grad(): _ = model(tokens)
            model.reset_hooks()

            # -- Steered with ABLATED vector --
            steer_cache_ablate = {}
            model.reset_hooks()
            def steer_ablate(act, **kw):
                act[:, -1, :] += coeff * v_ablate.to(dtype=act.dtype)
                return act
            model.add_hook(f"blocks.{steer_layer}.hook_resid_pre", steer_ablate, "fwd")
            for L in TARGET_LAYERS:
                def make_cap(L):
                    def cap(act, **kw):
                        if L not in steer_cache_ablate: steer_cache_ablate[L] = []
                        steer_cache_ablate[L].append(act[:, -1, :].clone().detach())
                        return act
                    return cap
                model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
            with torch.no_grad(): _ = model(tokens)
            model.reset_hooks()

            # Compute cos at each target layer
            for L in TARGET_LAYERS:
                r_n = r_vecs[L].norm().cpu()
                r_cpu = r_vecs[L].float().cpu()
                hb = base_cache[L][0].float().cpu()
                ho = steer_cache_orig[L][0].float().cpu()
                ha = steer_cache_ablate[L][0].float().cpu()
                for b in range(B):
                    c_base = torch.dot(hb[b], r_cpu) / (hb[b].norm() * r_n + 1e-8)
                    c_orig = torch.dot(ho[b], r_cpu) / (ho[b].norm() * r_n + 1e-8)
                    c_ablate = torch.dot(ha[b], r_cpu) / (ha[b].norm() * r_n + 1e-8)
                    base_cos_all[L].append(c_base.item())
                    steer_cos_orig_all[L].append(c_orig.item())
                    steer_cos_ablate_all[L].append(c_ablate.item())

        # Report
        print(f"\n=== {task_name} (coeff={coeff}, steer at {steer_layer}) ===")
        header = f"{'Layer':>5s} {'cos(v_orig,r̂)':>14s} {'cos(v_abl,r̂)':>14s} {'cos(base)':>10s} {'cos(orig)':>10s} {'Δorig':>8s} {'cos(abl)':>10s} {'Δabl':>8s}"
        print(header)
        print("-" * 82)

        task_entry = {
            "coeff": coeff, "steer_layer": steer_layer,
            "layers": {},
            "vector_orig_path": orig_path,
            "vector_ablate_path": ablate_path,
        }
        for L in TARGET_LAYERS:
            v_orig_cos = (torch.dot(v_orig, r_vecs[L]) / (v_orig.norm() * r_vecs[L].norm() + 1e-8)).item()
            v_ablate_cos = (torch.dot(v_ablate, r_vecs[L]) / (v_ablate.norm() * r_vecs[L].norm() + 1e-8)).item()
            bc = np.array(base_cos_all[L])
            oc = np.array(steer_cos_orig_all[L])
            ac = np.array(steer_cos_ablate_all[L])
            delta_orig = oc.mean() - bc.mean()
            delta_ablate = ac.mean() - bc.mean()
            print(f"  {L:2d}   {v_orig_cos:+.4f}     {v_ablate_cos:+.4f}     {bc.mean():+.4f}   {oc.mean():+.4f}   {delta_orig:+.4f}   {ac.mean():+.4f}   {delta_ablate:+.4f}")
            task_entry["layers"][str(L)] = {
                "vector_cos_orig": v_orig_cos,
                "vector_cos_ablate": v_ablate_cos,
                "cos_h_base_mean": float(bc.mean()),
                "cos_h_steered_orig_mean": float(oc.mean()),
                "cos_h_steered_orig_std": float(oc.std()),
                "delta_cos_orig": delta_orig,
                "cos_h_steered_ablate_mean": float(ac.mean()),
                "cos_h_steered_ablate_std": float(ac.std()),
                "delta_cos_ablate": delta_ablate,
            }
        results[task_name] = task_entry
    all_results[str(steer_layer)] = results

del model; torch.cuda.empty_cache()

out = Path("Experiments/ExpH/results/activation_analysis_ablated_comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {out}")
