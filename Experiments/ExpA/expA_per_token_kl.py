"""
Experiment A: Per-token KL (extraction bottleneck).

Run model on toxic vs non-toxic prompts, measure per-token KL divergence
of output logit distributions. Tests whether the contrastive signal is
concentrated in the first ~10 tokens (harm horizon).

Usage:
    CUDA_VISIBLE_DEVICES=2 python Experiments/expA_per_token_kl.py
"""
import json
import csv
import torch
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from torch.nn import functional as F

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda"
N_PROMPTS = 200  # 100 toxic + 100 non-toxic
MAX_TOKENS = 64
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

def get_per_token_logprobs(model, texts, max_len=MAX_TOKENS):
    all_logprobs = []
    all_tokens_list = []
    for text in texts:
        tokens = model.to_tokens(text, truncate=True)
        if tokens.shape[1] > max_len:
            tokens = tokens[:, :max_len]
        logits = model(tokens)
        logprobs = F.log_softmax(logits, dim=-1)
        all_logprobs.append(logprobs[0].cpu())  # move to CPU
        all_tokens_list.append(tokens[0].cpu())
        torch.cuda.empty_cache()
    return all_logprobs, all_tokens_list

print("Running toxic prompts...")
toxic_lps, toxic_toks = get_per_token_logprobs(model, toxic_texts)

print("Running non-toxic prompts...")
nontoxic_lps, nontoxic_toks = get_per_token_logprobs(model, nontoxic_texts)

# Per-position KL: average over prompts
max_len = max(
    max(t.shape[0] for t in toxic_lps),
    max(t.shape[0] for t in nontoxic_lps),
)
device = torch.device(DEVICE)

kl_per_pos = []
kl_rev_per_pos = []
num_pairs = min(len(toxic_lps), len(nontoxic_lps))

for pos in range(max_len):
    kls_at_pos = []
    kl_revs_at_pos = []
    for idx in range(num_pairs):
        lp_tox = toxic_lps[idx]
        lp_nontox = nontoxic_lps[idx]
        if pos < lp_tox.shape[0] and pos < lp_nontox.shape[0]:
            p_tox = lp_tox[pos].to(device)
            p_nontox = lp_nontox[pos].to(device)
            # KL(P_tox || P_nontox)
            kl = (p_tox.exp() * (p_tox - p_nontox)).sum()
            # Reverse KL
            kl_rev = (p_nontox.exp() * (p_nontox - p_tox)).sum()
            kls_at_pos.append(kl.item())
            kl_revs_at_pos.append(kl_rev.item())
            
    if kls_at_pos:
        kl_per_pos.append(float(np.mean(kls_at_pos)))
        kl_rev_per_pos.append(float(np.mean(kl_revs_at_pos)))
    else:
        kl_per_pos.append(0.0)
        kl_rev_per_pos.append(0.0)

results = {
    "n_toxic": len(toxic_texts),
    "n_nontoxic": len(nontoxic_texts),
    "max_len": max_len,
    "kl_toxic_vs_nontoxic": kl_per_pos,
    "kl_nontoxic_vs_toxic": kl_rev_per_pos,
    "kl_symmetric": [(a + b) / 2 for a, b in zip(kl_per_pos, kl_rev_per_pos)],
}

print(f"\nResults (first 20 positions):")
print(f"Pos\tKL(tox||nontox)\tKL(nontox||tox)\tSymKL")
for i in range(min(20, max_len)):
    print(f"{i}\t{kl_per_pos[i]:.6f}\t{kl_rev_per_pos[i]:.6f}\t{results['kl_symmetric'][i]:.6f}")

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
