# Experiment F: Per-Token KL on Steered Outputs

## Motivation
Qi et al. (2024) showed fine-tuning KL concentrates on first ~5 output tokens.
Does steering KL do the same? This experiment answers whether steering operates
through the same shallow channel as RLHF — or is mechanistically deeper.

## What it tests
**Primary**: Is KL(steered || base) concentrated in early output positions (0-5)?
- YES: steering ≈ shallow channel manipulation. Cancellation filter blocks
  the same early-token window that RLHF uses.
- NO: steering is deeper than fine-tuning. Our ceiling is a different phenomenon.

**Secondary**: Does the KL concentration pattern predict accuracy per sample?
- If correct samples have higher early KL → early logit change drives success
- If no difference → cancellation is independent of logit-change magnitude

## Design
- Step-by-step autoregressive generation with FIXED prefix (base's tokens)
- At each position t: both models condition on the same context
- KL is a clean causal quantity (not confounded by different conditioning)
- Then: free-generation with steering → accuracy via evaluator

## Files
- `expF_per_token_kl.py` — main experiment (CAA on Toxic/Evil/Deception/Refusal)
- `analyze_kl.py` — post-hoc analysis of results
- `expF_results.json` — output

## Run
```bash
conda activate sae_circuit
export CUDA_VISIBLE_DEVICES=2
python Experiments/ExpF/expF_per_token_kl.py
python Experiments/ExpF/analyze_kl.py
```

## Expected time
~30 min (3 tasks × 3 coeffs × 50 prompts × 20 tokens)
