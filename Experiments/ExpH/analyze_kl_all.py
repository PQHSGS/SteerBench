"""
Per-layer KL divergence across ALL methods: Original, Appr1, Appr2, Appr3.
Measures how much steering changes the distribution at each layer.

For each steer_layer × method × task:
  1. Compute unsteered logits per layer at position -1 (via model.unembed)
  2. Compute steered logits per layer at position -1
  3. KL(unsteered || steered) averaged over N prompts

All methods share the same base (unsteered) pass — compute once per batch.
"""
import sys; sys.path.insert(0, ".")
import torch, json, numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from Steering.data.loader import EvalDataLoader

device = "cuda:0"
dtype = torch.bfloat16
model_name = "google/gemma-2-2b-it"
STEER_LAYERS = [14, 18, 22, 25]
N_SAMPLES = 50
BATCH_SIZE = 8

METHODS = [
    ("original", "Vector/CAA/Gemma/",        None),
    ("appr1",    "Vector/CAA/Gemma_ablated/", None),
    ("appr2",    "Vector/CAA/Gemma_appr2/",   None),
    ("appr3",    "Vector/CAA/Gemma/",         "project"),  # original vec + on-the-fly r̂ projection
]

TASKS = {
    "evil":      ("toxic",      5),
    "toxic":     ("evil",       5),
    "deception": ("toxic",      5),
}

print("Loading model...")
model = HookedTransformer.from_pretrained(model_name, dtype=dtype, device=device)
model.eval()
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"  n_layers={n_layers}, d_model={d_model}")

# Gemma-2-2b-it: 26 layers (0-25)
ALL_LAYERS = list(range(0, n_layers))  # all layers

# === Load r̂ vectors ===
r_vecs = torch.load("Experiments/ExpH/results/refusal_vecs.pt", map_location=device)
r_vecs = {int(k): v.to(dtype=dtype) for k, v in r_vecs.items()}
for L in sorted(r_vecs.keys()):
    print(f"  r̂ layer {L:2d}: norm={r_vecs[L].norm():.4f}")

# === Load test prompts ===
eval_loader = EvalDataLoader()
test_data_dict = {}
for name, (ds_name, _) in TASKS.items():
    data = eval_loader.load(ds_name, n_samples=N_SAMPLES, apply_chat_template=True, tokenizer=model.tokenizer)
    prompts = [d["question"] for d in data if isinstance(d, dict) and "question" in d]
    test_data_dict[name] = prompts
    print(f"{name}: {len(prompts)} prompts from {ds_name}")


# ---- Helpers ----

def capture_and_unembed(model, tokens, layers):
    """
    Capture residual at position -1 for each `layer`.
    Apply model.unembed to get logits.
    Returns list of GPU tensors aligned with `layers`.
    """
    cache = {}
    model.reset_hooks()
    for L in layers:
        def make_hook(L):
            def hook(act, **kw):
                if L not in cache:
                    cache[L] = []
                cache[L].append(act[:, -1, :].clone())  # keep on GPU
                return act
            return hook
        model.add_hook(f"blocks.{L}.hook_resid_pre", make_hook(L), "fwd")
    with torch.no_grad():
        _ = model(tokens)  # forward pass triggers hooks
    model.reset_hooks()

    logits_out = {}
    for L in layers:
        h = cache[L][0]  # (B, d_model)
        logits_out[L] = model.unembed(h)  # (B, vocab_size)
    return logits_out


def steer_fn_factory(steer_layer, v, coeff, hook_type, r_vec):
    """Return hook function for given method type."""
    if hook_type == "project":
        # Appr3: h' = (h+v) - proj_{r̂}(h+v), preserve norm
        r = r_vec.to(dtype=dtype, device=device)
        r_norm_sq = r.norm() ** 2
        v_d = v.to(dtype=dtype, device=device)

        def hook(act, **kw):
            # act is (B, seq_len, d_model), we want last token
            h_last = act[:, -1, :]  # (B, d_model)
            h_add = h_last + coeff * v_d  # (B, d_model)
            # Project r̂ out: proj = dot(h_add, r) / ||r||^2 * r
            dots = torch.matmul(h_add, r) / (r_norm_sq + 1e-8)  # (B,)
            h_proj = h_add - dots.unsqueeze(-1) * r  # (B, d_model)
            # Preserve unsteered norm per batch element
            norms_unsteered = h_last.norm(dim=-1, keepdim=True)  # (B, 1)
            norms_proj = h_proj.norm(dim=-1, keepdim=True) + 1e-8
            h_final = h_proj * (norms_unsteered / norms_proj)
            act[:, -1, :] = h_final.to(dtype=act.dtype)
            return act
        return hook
    else:
        # Original / Appr1 / Appr2: simple add
        v_d = v.to(dtype=dtype, device=device)
        def hook(act, **kw):
            act[:, -1, :] += coeff * v_d
            return act
        return hook


def kl_single(p_logits, q_logits):
    """KL(p || q). p/q are (B, vocab). Return per-sample KL vector (B,)."""
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    p_clamp = p.clamp(min=1e-10)
    q_clamp = q.clamp(min=1e-10)
    kl = (p * (torch.log(p_clamp) - torch.log(q_clamp))).sum(dim=-1)  # (B,)
    return kl.detach()


# === MAIN ===

all_results = {}
for steer_layer in STEER_LAYERS:
    print(f"\n{'='*70}")
    print(f"STEER AT LAYER {steer_layer}")
    print(f"{'='*70}")

    steer_results = {}

    for task_name, (ds_name, coeff) in TASKS.items():
        prompts = test_data_dict[task_name]
        print(f"\n--- {task_name} (coeff={coeff}) ---")

        # For each method, accumulate KL per layer
        # Structure: method_kl[meth_name][L] = sum of KL over samples
        method_kl = {m[0]: {L: 0.0 for L in ALL_LAYERS} for m in METHODS}
        sample_counts = {m[0]: 0 for m in METHODS}

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            tokens = model.to_tokens(batch)
            B = tokens.size(0)

            # === Unsteered: compute once ===
            base_logits = capture_and_unembed(model, tokens, ALL_LAYERS)
            # base_logits[L]: (B, vocab) on GPU

            # === For each method, steered ===
            for meth_name, vec_base, hook_type in METHODS:
                # Load vector for this task at steer_layer
                vec_path = Path(vec_base) / task_name / "vector.pt"
                vec_raw = torch.load(str(vec_path), map_location=device)
                v = vec_raw[steer_layer].to(dtype=dtype)

                r_vec_for_method = r_vecs[steer_layer] if hook_type == "project" else None

                # Build steer hook
                steer_hook = steer_fn_factory(steer_layer, v, coeff, hook_type, r_vec_for_method)

                # === Steer + capture in single forward pass ===
                cache_steer = {}
                model.reset_hooks()
                # Steering hook (applied FIRST, then cache hooks)
                model.add_hook(f"blocks.{steer_layer}.hook_resid_pre", steer_hook, "fwd")
                for L in ALL_LAYERS:
                    def make_hook(L):
                        def hook(act, **kw):
                            if L not in cache_steer:
                                cache_steer[L] = []
                            cache_steer[L].append(act[:, -1, :].clone())
                            return act
                        return hook
                    model.add_hook(f"blocks.{L}.hook_resid_pre", make_hook(L), "fwd")
                with torch.no_grad():
                    _ = model(tokens)
                model.reset_hooks()

                steer_logits_2 = {}
                for L in ALL_LAYERS:
                    h = cache_steer[L][0]
                    steer_logits_2[L] = model.unembed(h)

                # Compute KL per layer
                for L in ALL_LAYERS:
                    kl_vals = kl_single(base_logits[L], steer_logits_2[L])  # (B,)
                    method_kl[meth_name][L] += kl_vals.sum().item()
                sample_counts[meth_name] += B

                # Clean up
                del cache_steer, steer_logits_2, v
                torch.cuda.empty_cache()

            # Clean up base
            del base_logits
            torch.cuda.empty_cache()

        # === Report ===
        print(f"  {'Method':>12s}", end="")
        for L in range(0, n_layers, 4):  # every 4th layer
            print(f"  L{L:2d}", end="")
        print()

        for meth_name, _, _ in METHODS:
            vals = []
            for L in ALL_LAYERS:
                avg = method_kl[meth_name][L] / max(sample_counts[meth_name], 1)
                vals.append(avg)
            print(f"  {meth_name:>12s}", end="")
            for L in range(0, n_layers, 4):
                v = vals[L]
                if v < 1e-8:
                    print(f"  {'0':>5s}", end="")
                else:
                    print(f"  {v:.3f}", end="")
            print()

        # Save
        for meth_name, _, _ in METHODS:
            if meth_name not in steer_results:
                steer_results[meth_name] = {"steer_layer": steer_layer, "tasks": {}}
            steer_results[meth_name]["tasks"][task_name] = {
                "coeff": coeff,
                "kl_per_layer": {
                    str(L): round(method_kl[meth_name][L] / max(sample_counts[meth_name], 1), 10)
                    for L in ALL_LAYERS
                },
            }

    all_results[str(steer_layer)] = steer_results

del model; torch.cuda.empty_cache()

out = Path("Experiments/ExpH/results/kl_divergence_all_methods.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {out}")
