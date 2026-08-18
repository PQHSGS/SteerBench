#!/bin/bash

set -euo pipefail

export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"

export TORCH_HOME="/data/caotue/torch_cache"

export TMPDIR="/data/caotue/tmp"

export UV_CACHE_DIR="/data/caotue/uv"
export PIP_CACHE_DIR="/data/caotue/pip"
# Authentication tokens
export HF_TOKEN="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"
export WANDB_API_KEY="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"
    
# Login
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_API_KEY


# Run 1B-activation stream training
# 250,000 steps * 4,096 batch size ~= 1,024,000,000 activations.
# Save checkpoints every 100,000,000 activations (100M).
# Use larger max-documents and stats precompute budget than the 1M baseline run.
python -m Steering.post_process.cli stream \
    --model-name "google/gemma-2-2b-it" \
    --init-ckpt "/data/caotue/GLP/gemma/400M" \
    --layer 14 \
    --phase-switch \
    --offload-device cpu \
    --retain input \
    --dataset-name "HuggingFaceFW/fineweb" \
    --dataset-config sample-10BT \
    --dataset-split train \
    --text-field text \
    --max-documents 3000000 \
    --max-length 2048 \
    --token-idx all \
    --drop-bos \
    --padding-side right \
    --document-batch-size 16 \
    --forward-batch-size 2 \
    --device cuda \
    --torch-dtype bfloat16 \
    --storage-dtype bfloat16 \
    --stream-chunk-size 1024000 \
    --run-name "gemma" \
    --batch-size 4096 \
    --total-steps 150000 \
    --num-epochs 1 \
    --learning-rate 5e-5 \
    --normalization-method "gaussian" \
    --noise-sampling-method "sot" \
    --ot-chunk-size 4096 \
    --u-sampling-method "beta" \
    --split \
    --split-proportion 0.05 \
    --gradient-clipping-threshold 10 \
    --log-every-n-steps 10 \
    --warmup-ratio 0.01 \
    --initial-factor 0.01 \
    --final-factor 0.1 \
    --use-bf16 \
    --shuffle \
    --wandb \
    --wandb-project glp \
    --checkpoint-token-step 200000000 \
    --denoiser-layers 3 \
    --seed 42

# # Push the trained GLP model to Hugging Face
# echo "Pushing model to Hugging Face..."
# # Get the current logged-in user dynamically
# HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami(token='$HF_TOKEN')['name'])")
# REPO_ID="${HF_USER}/glp-gemma"

# python cli/push_to_hf.py \
#     --repo-id "$REPO_ID" \
#     --folder "./100m" \
#     --token $HF_TOKEN \
#     --path-in-repo "100M_log" \
#     --allow-overlap
