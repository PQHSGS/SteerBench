#!/bin/bash
set -euo pipefail

# Standard full-space GLP training.
# Legacy subspace/PCA training has been removed.
# This launcher now trains the normal GLP model on the refusal dataset.
python -m Steering.post_process.cli stream \
  --model-name "google/gemma-2-2b-it" \
  --layer 14 \
  --retain input \
  --dataset-name "refusal_cast_responses" \
  --dataset-split train \
  --text-field text \
  --max-documents 3000000 \
  --max-length 2048 \
  --token-idx all \
  --drop-bos \
  --padding-side right \
  --document-batch-size 16 \
  --forward-batch-size 2 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --storage-dtype bfloat16 \
  --stream-chunk-size 1024000 \
  --run-name "glp-refusal" \
  --batch-size 4096 \
  --total-steps 250000 \
  --num-epochs 1 \
  --learning-rate 1e-4 \
  --normalization-method gaussian \
  --noise-sampling-method sot \
  --ot-chunk-size 4096 \
  --u-sampling-method logit_normal \
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
  --checkpoint-token-step 250000000 \
  --denoiser-layers 3 \
  --d-model-mult 2 \
  --d-mlp-mult 4 \
  --phase-switch \
  --offload-device cpu \
  --seed 42
