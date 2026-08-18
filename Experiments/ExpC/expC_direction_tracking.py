"""
Experiment C: Direction rotation — does perturbation direction persist through layers?

Inject steering vector v at L8, track cosine similarity of the perturbation
(steered - unsteered) to the original v at each layer L8-L25.

Prediction:
- v_toxic: direction rotates away (cos drops toward 0) as cancellation circuit
  pushes it into off-manifold directions
- v_deception: direction persists (cos stays high) — no cancellation
- v_random: intermediate rotation
"""
import json, csv, torch, numpy as np, time
from pathlib import Path
from transformer_lens import HookedTransformer
from torch.nn import functional as F
from collections import defaultdict

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda"
COEFF = 2.0
N_PROMPTS = 30
N_EXTRACT = 100
MAX_TOKENS = 32
INJECT_LAYER = 8
TRACK_LAYERS = [8] + list(range(9, 26))
OUTPUT = "Experiments/expC_results.json"
TOXIC_DATA = "TrainDataset/behaviour/toxic/jigsaw/train.csv"
DECEPTION_DATA = "TrainDataset/behaviour/deception/LiarBench/data.jsonl"

torch.set_grad_enabled(False)

t0 = time.time()
print(f"[{t0:.0f}] Loading model...", flush=True)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16)
model.eval()
print(f"[{time.time():.0f}] Model loaded ({time.time()-t0:.1f}s)", flush=True)

# --- Extract steering vectors (same as expB) ---
def get_mean_activation(model, texts, layer, max_len=MAX_TOKENS):
    acts = []
    for i, text in enumerate(texts):
        tokens = model.to_tokens(text, truncate=True)[:, :max_len]
        _, cache = model.run_with_cache(tokens)
        act = cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :].cpu()
        acts.append(act)
        if (i+1) % 50 == 0:
            print(f"  extracted {i+1}/{len(texts)}", flush=True)
    return torch.stack(acts).mean(dim=0)

print("Extracting toxic steering vector...", flush=True)
toxic_texts, nontoxic_texts = [], []
with open(TOXIC_DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["toxic"]) and len(toxic_texts) < N_EXTRACT:
            toxic_texts.append(row["comment_text"])
        elif not int(row["toxic"]) and len(nontoxic_texts) < N_EXTRACT:
            nontoxic_texts.append(row["comment_text"])
        if len(toxic_texts) >= N_EXTRACT and len(nontoxic_texts) >= N_EXTRACT:
            break

toxic_mean = get_mean_activation(model, toxic_texts, INJECT_LAYER)
nontoxic_mean = get_mean_activation(model, nontoxic_texts, INJECT_LAYER)
v_toxic = F.normalize(toxic_mean - nontoxic_mean, dim=-1)
print(f"  v_toxic norm={v_toxic.norm().item():.4f}", flush=True)

print("Extracting deception steering vector...", flush=True)
deception_texts, honest_texts = [], []
with open(DECEPTION_DATA) as f:
    for line in f:
        d = json.loads(line)
        gt = d.get("ground_truth", "").lower()
        prompt = d.get("user_prompt", "")
        if gt == "false" and len(deception_texts) < N_EXTRACT:
            deception_texts.append(prompt)
        elif gt == "true" and len(honest_texts) < N_EXTRACT:
            honest_texts.append(prompt)
        if len(deception_texts) >= N_EXTRACT and len(honest_texts) >= N_EXTRACT:
            break

deception_mean = get_mean_activation(model, deception_texts, INJECT_LAYER)
honest_mean = get_mean_activation(model, honest_texts, INJECT_LAYER)
v_deception = F.normalize(deception_mean - honest_mean, dim=-1)
print(f"  v_deception norm={v_deception.norm().item():.4f}", flush=True)

v_random = F.normalize(torch.randn(2304), dim=-1)
print(f"  cos(v_toxic, v_decep)={(v_toxic*v_deception).sum().item():.4f}")
print(f"  cos(v_toxic, v_rand)={(v_toxic*v_random).sum().item():.4f}")
print(f"Done extracting ({time.time()-t0:.1f}s)", flush=True)

eval_texts = nontoxic_texts[:N_PROMPTS]

def run_and_track_cos(model, texts, steering_vec, inject_layer, track_layers, coeff, max_len=MAX_TOKENS):
    """Track cosine similarity of perturbation (steered - unsteered) to steering_vec at each layer."""
    all_cos = defaultdict(list)
    all_norms = defaultdict(list)  # also track norms for context
    sv = steering_vec.to(DEVICE)
    for idx, text in enumerate(texts):
        tokens = model.to_tokens(text, truncate=True)[:, :max_len]

        # Unsteered
        _, cache_un = model.run_with_cache(tokens)
        unsteered = {l: cache_un[f"blocks.{l}.hook_resid_pre"][0, -1, :].cuda() for l in track_layers}

        # Steered
        def steering_hook(activations, hook):
            activations[0, -1, :] += coeff * sv
            return activations
        model.add_hook(f"blocks.{inject_layer}.hook_resid_pre", steering_hook)
        _, cache_st = model.run_with_cache(tokens)
        model.reset_hooks()

        steered = {l: cache_st[f"blocks.{l}.hook_resid_pre"][0, -1, :].cuda() for l in track_layers}

        for l in track_layers:
            diff = steered[l] - unsteered[l]
            cos = F.cosine_similarity(diff.unsqueeze(0), sv.unsqueeze(0)).item()
            norm = diff.norm().item()
            all_cos[l].append(cos)
            all_norms[l].append(norm)

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(texts)} prompts done", flush=True)

    return {str(l): float(np.mean(v)) for l, v in all_cos.items()}, \
           {str(l): float(np.std(v)) for l, v in all_cos.items()}, \
           {str(l): float(np.mean(v)) for l, v in all_norms.items()}

print("Running toxic vector direction tracking...", flush=True)
toxic_cos_mean, toxic_cos_std, toxic_norm_mean = run_and_track_cos(
    model, eval_texts, v_toxic, INJECT_LAYER, TRACK_LAYERS, COEFF)

print("Running deception vector direction tracking...", flush=True)
decep_cos_mean, decep_cos_std, decep_norm_mean = run_and_track_cos(
    model, eval_texts, v_deception, INJECT_LAYER, TRACK_LAYERS, COEFF)

print("Running random vector direction tracking...", flush=True)
random_cos_mean, random_cos_std, random_norm_mean = run_and_track_cos(
    model, eval_texts, v_random, INJECT_LAYER, TRACK_LAYERS, COEFF)

results = {
    "config": {"model": MODEL_NAME, "inject_layer": INJECT_LAYER, "track_layers": TRACK_LAYERS,
               "coeff": COEFF, "n_prompts": N_PROMPTS, "max_tokens": MAX_TOKENS},
    "vectors_info": {
        "toxic": {"cos_with_deception": (v_toxic * v_deception).sum().item(),
                  "cos_with_random": (v_toxic * v_random).sum().item()},
    },
    "results": {
        "toxic": {"cos_mean": toxic_cos_mean, "cos_std": toxic_cos_std, "norm_mean": toxic_norm_mean},
        "deception": {"cos_mean": decep_cos_mean, "cos_std": decep_cos_std, "norm_mean": decep_norm_mean},
        "random": {"cos_mean": random_cos_mean, "cos_std": random_cos_std, "norm_mean": random_norm_mean},
    },
}

print(f"\nLayer\tToxic(cos)\tDecep(cos)\tRand(cos)\tToxic(norm)\tDecep(norm)\tRand(norm)")
for l in TRACK_LAYERS:
    ls = str(l)
    tc = toxic_cos_mean.get(ls, 0)
    dc = decep_cos_mean.get(ls, 0)
    rc = random_cos_mean.get(ls, 0)
    tn = toxic_norm_mean.get(ls, 0)
    dn = decep_norm_mean.get(ls, 0)
    rn = random_norm_mean.get(ls, 0)
    print(f"L{l}\t{tc:.4f}\t\t{dc:.4f}\t\t{rc:.4f}\t\t{tn:.2f}\t\t{dn:.2f}\t\t{rn:.2f}")

print(f"\nCos decay ({TRACK_LAYERS[0]}->{TRACK_LAYERS[-1]}):")
for name, means in [("Toxic", toxic_cos_mean), ("Deception", decep_cos_mean), ("Random", random_cos_mean)]:
    first_l, last_l = str(TRACK_LAYERS[0]), str(TRACK_LAYERS[-1])
    ratio = means[last_l] / means[first_l] if means.get(first_l, 0) != 0 else 0
    print(f"  {name}: {ratio:.4f} ({means[first_l]:.4f} -> {means[last_l]:.4f})", flush=True)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT} ({time.time()-t0:.0f}s)", flush=True)
