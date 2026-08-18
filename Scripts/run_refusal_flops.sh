#!/bin/bash
# run_refusal_flops.sh
set -euo pipefail

# Make sure we use the correct Conda environment python
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

# Default GPU to use
export CUDA_VISIBLE_DEVICES=${1:-7}

echo "=== Running FLOP profiling benchmarks on GPU ${CUDA_VISIBLE_DEVICES} ==="

# Finetuning baseline
# FINETUNE_CFG="Configs/Finetune/gemma_refusal_response.json"
# if [ -f "$FINETUNE_CFG" ]; then
#   echo ""
#   echo "--- Running Finetuning Baseline: $FINETUNE_CFG ---"
#   python -m Steering.finetune.cli --config "$FINETUNE_CFG"
# else
#   echo "Finetuning config not found: $FINETUNE_CFG"
# fi

# List of methods to evaluate
METHODS=(
  "SAECOT"
  "SAEFREE"
  "SAEIO"
  "SAS"
  "SPARE"
  "SPHERICAL"
  "SRPS"
)


# Iterate over each method
for method in "${METHODS[@]}"; do
  # Check if directory exists
  if [ ! -d "Configs/Eval/$method" ]; then
    echo "Directory Configs/Eval/$method does not exist. Skipping."
    continue
  fi

  # Find the refusal response config dynamically, excluding sorrybench and llama
  cfg=$(find "Configs/Eval/$method" -name "*refusal_response.json" ! -name "*sorrybench*" ! -name "*llama*" | head -n 1)

  if [ -z "$cfg" ]; then
    echo "No refusal response configuration file found for method $method. Skipping."
    continue
  fi

  echo ""
  echo "--- Running Steering Evaluation for $method: $cfg ---"
  python -m Steering.cli --task eval --config "$cfg"
done

echo ""
echo "=== All evaluations completed successfully! ==="
