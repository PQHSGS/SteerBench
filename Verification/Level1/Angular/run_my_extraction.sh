#!/bin/bash
# Enable conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sae_circuit

# Clean environment
unset CUDA_VISIBLE_DEVICES

# Run my extraction (using Steering CLI)
# Make sure python path includes current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG_FILE="Verification/Level1/Angular/extract_angular_mine.json"

echo "Running My Extraction with config $CONFIG_FILE..."
python -m Steering.cli --task extract --config $CONFIG_FILE

echo "My extraction complete."
