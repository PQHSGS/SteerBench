import json
import math

fpath = "Results/linearact/eval_gemma_toxic_coeff_1p0_20260626_174118.json"
with open(fpath) as f:
    data = json.load(f)

samples = data.get("result", {}).get("samples", [])
print(f"Total samples: {len(samples)}")

for i, s in enumerate(samples[:5]):
    resp = s["response"]
    meta = s.get("metadata", {})
    nlls = meta.get("token_nlls", [])
    tokens = meta.get("token_ids", [])
    
    # Recompute lcs score for this sample
    window_size = 10
    penalty_weight = 1.0
    seq_len = len(tokens)
    penalized_losses = []
    
    for t in range(seq_len):
        raw_loss = nlls[t]
        start_idx = max(0, t - window_size)
        history = tokens[start_idx:t]
        
        lcs = 0
        for idx in range(len(history) - 1):
            length = 0
            while idx - length >= 0 and history[idx - length] == history[-1 - length]:
                length += 1
            if length > lcs:
                lcs = length
        
        if lcs > 2:
            penalty = float(lcs - 2) * penalty_weight
            penalized_loss = raw_loss + penalty
        else:
            penalized_loss = raw_loss
        penalized_losses.append(penalized_loss)
        
    mean_raw = sum(nlls) / seq_len if seq_len > 0 else 0
    mean_penalized = sum(penalized_losses) / seq_len if seq_len > 0 else 0
    
    ppl_raw = math.exp(mean_raw) if seq_len > 0 else 0
    ppl_lsp = math.exp(mean_penalized) if seq_len > 0 else 0
    
    print(f"\nSample {i+1}:")
    print(f"Response snippet: {repr(resp[:100])}")
    print(f"Tokens length: {seq_len}")
    print(f"Raw PPL: {ppl_raw:.4f}, LSP PPL: {ppl_lsp:.4f}")
    
# Find the sample with the highest LSP PPL
all_lsps = []
for i, s in enumerate(samples):
    meta = s.get("metadata", {})
    nlls = meta.get("token_nlls", [])
    tokens = meta.get("token_ids", [])
    if not tokens:
        continue
    seq_len = len(tokens)
    penalized_losses = []
    for t in range(seq_len):
        raw_loss = nlls[t]
        start_idx = max(0, t - 10)
        history = tokens[start_idx:t]
        lcs = 0
        for idx in range(len(history) - 1):
            length = 0
            while idx - length >= 0 and history[idx - length] == history[-1 - length]:
                length += 1
            if length > lcs:
                lcs = length
        if lcs > 2:
            penalized_loss = raw_loss + float(lcs - 2) * 1.0
        else:
            penalized_loss = raw_loss
        penalized_losses.append(penalized_loss)
    mean_penalized = sum(penalized_losses) / seq_len
    all_lsps.append((i, math.exp(mean_penalized), s["response"][:100]))

all_lsps.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 samples by LSP PPL:")
for idx, val, text in all_lsps[:5]:
    print(f"Sample {idx+1}: LSP PPL = {val:.4f}, Text = {repr(text)}")
