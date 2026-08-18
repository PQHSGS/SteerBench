#!/bin/bash
# Run both diagnostic experiments sequentially
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
export TORCH_HOME="/data/caotue/torch_cache"
export TMPDIR="/data/caotue/tmp"

export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

echo "=== [1/2] Experiment A: Per-position SNR ==="
python Experiments/ExpA/expA_per_token_snr.py
echo "=== Done A at $(date) ==="

echo ""
echo "=== [2/2] Experiment B: Perturbation survival ==="
python Experiments/ExpB/expB_perturbation_survival.py
echo "=== Done B at $(date) ==="

echo ""
echo "=== ALL DONE at $(date) ==="
