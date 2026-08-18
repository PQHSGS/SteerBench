"""
Experiment B: Perturbation survival (resistance circuitry).

Inject steering vector v at L8, track ||a_steered - a_unsteered||
through layers L9-L25. Compare v_toxic vs v_deception vs v_random.
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
TRACK_LAYERS = list(range(9, 26))
OUTPUT = "Experiments/expB_results.json"
TOXIC_DATA = "TrainDataset/behaviour/toxic/jigsaw/train.csv"
DECEPTION_DATA = "TrainDataset/behaviour/deception/LiarBench/data.jsonl"

torch.set_grad_enabled(False)

t0 = time.time()
print(f"[{t0:.0f}] Loading model...", flush=True)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16)
model.eval()
print(f"[{time.time():.0f}] Model loaded ({time.time()-t0:.1f}s)", flush=True)

# --- Extract steering vectors ---
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
print(f"  toxic={len(toxic_texts)} nontoxic={len(nontoxic_texts)}", flush=True)

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

# --- Neutral prompts for evaluation ---
eval_texts = nontoxic_texts[:N_PROMPTS]

def run_and_track(model, texts, steering_vec, inject_layer, track_layers, coeff, max_len=MAX_TOKENS):
    all_diffs = defaultdict(list)
    # Move steering vec to device once
    sv = steering_vec.to(DEVICE)
    for idx, text in enumerate(texts):
        tokens = model.to_tokens(text, truncate=True)[:, :max_len]

        # Unsteered
        _, cache_un = model.run_with_cache(tokens)
        unsteered = {l: cache_un[f"blocks.{l}.hook_resid_pre"][0, -1, :].cpu() for l in track_layers}

        # Steered with hook
        def steering_hook(activations, hook):
            activations[0, -1, :] += coeff * sv
            return activations

        hook_name = f"blocks.{inject_layer}.hook_resid_pre"
        model.add_hook(hook_name, steering_hook)
        _, cache_st = model.run_with_cache(tokens)
        model.reset_hooks()

        steered = {l: cache_st[f"blocks.{l}.hook_resid_pre"][0, -1, :].cpu() for l in track_layers}

        for l in track_layers:
            diff = (steered[l] - unsteered[l]).norm().item()
            all_diffs[l].append(diff)

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(texts)} prompts done", flush=True)

    return {str(l): float(np.mean(v)) for l, v in all_diffs.items()}, \
           {str(l): float(np.std(v)) for l, v in all_diffs.items()}

print("Running toxic vector perturbation...", flush=True)
toxic_diffs_mean, toxic_diffs_std = run_and_track(model, eval_texts, v_toxic, INJECT_LAYER, TRACK_LAYERS, COEFF)

print("Running deception vector perturbation...", flush=True)
decep_diffs_mean, decep_diffs_std = run_and_track(model, eval_texts, v_deception, INJECT_LAYER, TRACK_LAYERS, COEFF)

print("Running random vector perturbation...", flush=True)
random_diffs_mean, random_diffs_std = run_and_track(model, eval_texts, v_random, INJECT_LAYER, TRACK_LAYERS, COEFF)

results = {
    "config": {"model": MODEL_NAME, "inject_layer": INJECT_LAYER, "track_layers": TRACK_LAYERS,
               "coeff": COEFF, "n_prompts": N_PROMPTS, "max_tokens": MAX_TOKENS},
    "vectors": {
        "toxic": {"norm": v_toxic.norm().item(),
                  "cos_with_deception": (v_toxic * v_deception).sum().item(),
                  "cos_with_random": (v_toxic * v_random).sum().item()},
        "deception": {"norm": v_deception.norm().item()},
        "random": {"norm": v_random.norm().item()},
    },
    "results": {
        "toxic": {"mean": toxic_diffs_mean, "std": toxic_diffs_std},
        "deception": {"mean": decep_diffs_mean, "std": decep_diffs_std},
        "random": {"mean": random_diffs_mean, "std": random_diffs_std},
    },
}

print(f"\nLayer\tToxic\t\tDeception\tRandom")
for l in sorted(TRACK_LAYERS):
    t = toxic_diffs_mean.get(str(l), 0)
    d = decep_diffs_mean.get(str(l), 0)
    r = random_diffs_mean.get(str(l), 0)
    print(f"L{l}\t{t:.4f}\t\t{d:.4f}\t\t{r:.4f}")

first_l, last_l = str(TRACK_LAYERS[0]), str(TRACK_LAYERS[-1])
print(f"\nDecay ratios ({first_l}->{last_l}):")
for name, means in [("Toxic", toxic_diffs_mean), ("Deception", decep_diffs_mean), ("Random", random_diffs_mean)]:
    ratio = means[last_l] / means[first_l] if means[first_l] > 0 else 0
    print(f"  {name}: {ratio:.4f} ({means[first_l]:.4f} -> {means[last_l]:.4f})", flush=True)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT} ({time.time()-t0:.0f}s)", flush=True)
