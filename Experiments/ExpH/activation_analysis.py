"""
Multi-steer-layer: cos(h_steered_at_layer_S, r̂_L) across layers.
r̂ = h(refusal_text) - h(compliant_text) — response-only, captures refusal execution.
"""
import sys; sys.path.insert(0, ".")
import torch, json, numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from Steering.data.loader import EvalDataLoader

device = "cuda:0"   # CUDA_VISIBLE_DEVICES=2 maps to GPU 2
dtype = torch.bfloat16
model_name = "google/gemma-2-2b-it"
STEER_LAYERS = [14, 18, 22, 25]
TARGET_LAYERS = [14, 18, 22, 25]
N_SAMPLES = 50
BATCH_SIZE = 8

TASKS = {
    "evil":   ("Vector/CAA/Gemma/evil",        "toxic",      5),
    "toxic":  ("Vector/CAA/Gemma/toxic",       "evil",     5),
    "deception": ("Vector/CAA/Gemma/deception", "toxic", 5),
}

print("Loading model...")
model = HookedTransformer.from_pretrained(model_name, dtype=dtype, device=device)
model.eval()

# === Extract r̂ from RESPONSE-ONLY text (refusal vs compliant) ===
print("\nLoading behaviour_refusal.json (response-only)...")
with open("TrainDataset/behaviour/refusal/CAST/behaviour_refusal.json") as f:
    resp_data = json.load(f)
non_compliant = resp_data["non_compliant_responses"]
compliant = resp_data["compliant_responses"]
print(f"Loaded {len(compliant)} compliant + {len(non_compliant)} non-compliant")

def get_mean_activations_layers(texts, layers):
    """Collect mean activations at last token for each target layer."""
    sums = {L: None for L in layers}
    counts = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        tokens = model.to_tokens(batch)
        cache = {}
        model.reset_hooks()
        for L in layers:
            def make_cap(L):
                def cap(act, **kw):
                    if L not in cache:
                        cache[L] = []
                    cache[L].append(act[:, -1, :].clone().detach())
                    return act
                return cap
            model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
        with torch.no_grad(): _ = model(tokens)
        model.reset_hooks()
        B = tokens.size(0)
        for L in layers:
            act_sum = cache[L][0].sum(dim=0)
            sums[L] = act_sum if sums[L] is None else sums[L] + act_sum
        counts += B
    return {L: sums[L] / counts for L in layers}

print("Computing non-compliant (refusal) mean activations per layer...")
h_refuse = get_mean_activations_layers(non_compliant, TARGET_LAYERS)
print("Computing compliant mean activations per layer...")
h_comply = get_mean_activations_layers(compliant, TARGET_LAYERS)

r_vecs = {}
for L in TARGET_LAYERS:
    r = (h_refuse[L] - h_comply[L]).to(dtype=dtype)
    r_vecs[L] = r
    print(f"  r̂ layer {L:2d}: norm={r.norm():.4f}")

# === Load test prompts ===
eval_loader = EvalDataLoader()
test_data_dict = {}
for name, (_, ds_name, _) in TASKS.items():
    data = eval_loader.load(ds_name, n_samples=N_SAMPLES, apply_chat_template=True, tokenizer=model.tokenizer)
    prompts = [d["question"] for d in data if isinstance(d, dict) and "question" in d]
    test_data_dict[name] = prompts
    print(f"{name}: {len(prompts)} prompts from {ds_name}")

# === Outer loop: steer at each STEER_LAYER independently ===
all_results = {}
for steer_layer in STEER_LAYERS:
    print(f"\n{'='*60}")
    print(f"STEER AT LAYER {steer_layer}")
    print(f"{'='*60}")

    results = {}
    for task_name, (vec_path, _, coeff) in TASKS.items():
        vec_raw = torch.load(vec_path + "/vector.pt", map_location=device)
        v = vec_raw[steer_layer].to(dtype=dtype)

        prompts = test_data_dict[task_name]
        base_cos_all = {L: [] for L in TARGET_LAYERS}
        steer_cos_all = {L: [] for L in TARGET_LAYERS}

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            tokens = model.to_tokens(batch)
            B = tokens.size(0)

            # -- Baseline: capture all TARGET_LAYERS --
            base_cache = {}
            model.reset_hooks()
            for L in TARGET_LAYERS:
                def make_cap(L):
                    def cap(act, **kw):
                        if L not in base_cache:
                            base_cache[L] = []
                        base_cache[L].append(act[:, -1, :].clone().detach())
                        return act
                    return cap
                model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
            with torch.no_grad(): _ = model(tokens)
            model.reset_hooks()

            # -- Steered: steer at steer_layer, capture all TARGET_LAYERS --
            steer_cache = {}
            model.reset_hooks()
            def steer(act, **kw):
                act[:, -1, :] += coeff * v.to(dtype=act.dtype)
                return act
            model.add_hook(f"blocks.{steer_layer}.hook_resid_pre", steer, "fwd")
            for L in TARGET_LAYERS:
                def make_cap(L):
                    def cap(act, **kw):
                        if L not in steer_cache:
                            steer_cache[L] = []
                        steer_cache[L].append(act[:, -1, :].clone().detach())
                        return act
                    return cap
                model.add_hook(f"blocks.{L}.hook_resid_pre", make_cap(L), "fwd")
            with torch.no_grad(): _ = model(tokens)
            model.reset_hooks()

            # Compute cos for each layer
            for L in TARGET_LAYERS:
                r_vec_L = r_vecs[L]
                r_cpu = r_vec_L.float().cpu()
                r_n = r_vec_L.norm().cpu()
                hb = base_cache[L][0].float().cpu()
                hs = steer_cache[L][0].float().cpu()
                for b in range(B):
                    c_base = torch.dot(hb[b], r_cpu) / (hb[b].norm() * r_n + 1e-8)
                    c_steer = torch.dot(hs[b], r_cpu) / (hs[b].norm() * r_n + 1e-8)
                    base_cos_all[L].append(c_base.item())
                    steer_cos_all[L].append(c_steer.item())

        # Report per-task per-layer
        print(f"\n=== {task_name} (coeff={coeff}, steer at layer {steer_layer}) ===")
        print(f"{'Layer':>5s} {'cos(v,r̂_L)':>10s} {'cos(base)':>10s} {'cos(steer)':>10s} {'Δcos':>8s}")
        print("-" * 46)
        task_entry = {"coeff": coeff, "steer_layer": steer_layer, "layers": {}}
        for L in TARGET_LAYERS:
            r_vec_L = r_vecs[L]
            v_cos = (torch.dot(vec_raw[steer_layer].to(dtype=dtype), r_vec_L) /
                     (vec_raw[steer_layer].to(dtype=dtype).norm() * r_vec_L.norm() + 1e-8)).item()
            bc = np.array(base_cos_all[L])
            sc = np.array(steer_cos_all[L])
            delta = sc.mean() - bc.mean()
            print(f"  {L:2d}   {v_cos:+.4f}    {bc.mean():+.4f}    {sc.mean():+.4f}    {delta:+.4f}")
            task_entry["layers"][str(L)] = {
                "vector_cos": v_cos,
                "cos_h_base_mean": float(bc.mean()),
                "cos_h_base_std": float(bc.std()),
                "cos_h_steered_mean": float(sc.mean()),
                "cos_h_steered_std": float(sc.std()),
                "delta_cos": delta,
            }
        results[task_name] = task_entry

    all_results[str(steer_layer)] = results

del model; torch.cuda.empty_cache()

out = Path("Experiments/ExpH/results/activation_analysis_multi_layer.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {out}")
