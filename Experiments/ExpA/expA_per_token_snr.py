"""
Experiment A: Per-position SNR of contrastive activations (extraction bottleneck).

For each token position, compute:
   SNR(pos) = ||mean_diff(pos)||^2 / trace(avg_cov(pos))

where mean_diff = toxic_mean - nontoxic_mean at that position,
and avg_cov = average of per-class covariance.

Prediction: SNR >> 0 only at positions 1-10, then drops to near-zero.
This proves the contrastive signal used by all steering methods is
concentrated in the harm horizon.

Usage:
    CUDA_VISIBLE_DEVICES=3 python Experiments/expA_per_token_snr.py
"""
import json
import csv
import torch
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda"
N_PROMPTS = 200  # 100 toxic + 100 non-toxic
MAX_TOKENS = 64
LAYER = 14  # typical CAA extraction layer
OUTPUT = "Experiments/expA_results.json"
DATA_PATH = "TrainDataset/behaviour/toxic/jigsaw/train.csv"

torch.set_grad_enabled(False)

print("Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16)
model.eval()

print("Loading jigsaw data...")
toxic_texts, nontoxic_texts = [], []
with open(DATA_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["comment_text"]
        is_toxic = int(row["toxic"])
        if is_toxic and len(toxic_texts) < N_PROMPTS // 2:
            toxic_texts.append(text)
        elif not is_toxic and len(nontoxic_texts) < N_PROMPTS // 2:
            nontoxic_texts.append(text)
        if len(toxic_texts) >= N_PROMPTS // 2 and len(nontoxic_texts) >= N_PROMPTS // 2:
            break
print(f"  Toxic: {len(toxic_texts)}, Non-toxic: {len(nontoxic_texts)}")

hook_name = f"blocks.{LAYER}.hook_resid_pre"

def get_activations_per_position(model, texts, hook_name, max_len=MAX_TOKENS):
    """Return list of (seq_len, d_model) activations, one per text."""
    all_acts = []
    for text in texts:
        tokens = model.to_tokens(text, truncate=True)
        if tokens.shape[1] > max_len:
            tokens = tokens[:, :max_len]
        _, cache = model.run_with_cache(tokens)
        acts = cache[hook_name][0].cpu()  # (seq_len, d_model)
        all_acts.append(acts)
        del cache
        torch.cuda.empty_cache()
    return all_acts

print("Running toxic prompts...")
toxic_acts = get_activations_per_position(model, toxic_texts, hook_name)
print(f"  Got {len(toxic_acts)} activation tensors")

print("Running non-toxic prompts...")
nontoxic_acts = get_activations_per_position(model, nontoxic_texts, hook_name)
print(f"  Got {len(nontoxic_acts)} activation tensors")

# Pad to same length
max_len = max(max(t.shape[0] for t in toxic_acts),
              max(t.shape[0] for t in nontoxic_acts))
d_model = toxic_acts[0].shape[-1]

# Per-position SNR
results = {"config": {"model": MODEL_NAME, "layer": LAYER, "n_toxic": len(toxic_texts),
                       "n_nontoxic": len(nontoxic_texts), "max_len": max_len},
           "snr_per_position": []}

print("\nPer-position SNR:")
print(f"Pos\t||diff||²\tTr(cov)\t\tSNR\t\tVerdict")
for pos in range(max_len):
    toxic_pos_list = [t[pos] for t in toxic_acts if pos < t.shape[0]]
    nontoxic_pos_list = [t[pos] for t in nontoxic_acts if pos < t.shape[0]]

    if len(toxic_pos_list) < 2 or len(nontoxic_pos_list) < 2:
        # Not enough samples to compute SNR
        results["snr_per_position"].append({
            "position": pos,
            "diff_norm_sq": 0.0,
            "avg_cov": 0.0,
            "snr": 0.0,
        })
        print(f"{pos}\t0.000000\t0.0000\t\t0.000000\t\tnoise")
        continue

    toxic_pos = torch.stack(toxic_pos_list).to(DEVICE)   # (N_tox_valid, D)
    nontoxic_pos = torch.stack(nontoxic_pos_list).to(DEVICE)  # (N_nontox_valid, D)

    toxic_mean = toxic_pos.mean(dim=0)
    nontoxic_mean = nontoxic_pos.mean(dim=0)
    mean_diff = toxic_mean - nontoxic_mean
    diff_norm_sq = mean_diff.norm().item() ** 2

    # Covariance: 1/N * sum((x - mu)^2)
    toxic_cov = (toxic_pos - toxic_mean).pow(2).mean(dim=0).sum().item()
    nontoxic_cov = (nontoxic_pos - nontoxic_mean).pow(2).mean(dim=0).sum().item()
    avg_cov = (toxic_cov + nontoxic_cov) / 2

    snr = diff_norm_sq / avg_cov if avg_cov > 1e-12 else 0.0

    verdict = "SIGNAL" if snr > 0.001 else "noise" if pos > 0 else "BOS"
    if pos > 0 and snr > 0.01:
        verdict = "STRONG"
    elif pos > 0 and snr > 0.001:
        verdict = "signal"

    results["snr_per_position"].append({
        "position": pos,
        "diff_norm_sq": diff_norm_sq,
        "avg_cov": avg_cov,
        "snr": snr,
    })

    print(f"{pos}\t{diff_norm_sq:.6f}\t{avg_cov:.4f}\t\t{snr:.6f}\t\t{verdict}")

    torch.cuda.empty_cache()

# Summary: positions 0-10 vs 10+
snr_early = [r["snr"] for r in results["snr_per_position"][1:11]]  # skip BOS
snr_late = [r["snr"] for r in results["snr_per_position"][11:]]
results["summary"] = {
    "mean_snr_pos_1_10": float(np.mean(snr_early)) if snr_early else 0,
    "max_snr_pos_1_10": float(np.max(snr_early)) if snr_early else 0,
    "mean_snr_pos_11_plus": float(np.mean(snr_late)) if snr_late else 0,
    "max_snr_pos_11_plus": float(np.max(snr_late)) if snr_late else 0,
    "ratio_early_late": float(np.mean(snr_early) / max(np.mean(snr_late), 1e-12)),
}
print(f"\nSummary:")
print(f"  Mean SNR pos 1-10:  {results['summary']['mean_snr_pos_1_10']:.6f}")
print(f"  Mean SNR pos 11+:   {results['summary']['mean_snr_pos_11_plus']:.6f}")
print(f"  Ratio early/late:   {results['summary']['ratio_early_late']:.2f}x")

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
