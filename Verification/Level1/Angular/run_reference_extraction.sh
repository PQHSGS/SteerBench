#!/bin/bash
# Enable conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sae_circuit

# Clean environment
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# Run reference extraction
# Add Code/angular-steering/pytorch_pure to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/aiotlab/mnt/hoplt/Benchmark/Code/angular-steering/pytorch_pure

# Configuration
MODEL="google/gemma-2-2b-it"
OUTPUT_DIR="Verification/Level1/Angular/reference_output"
N_SAMPLES=1000
BATCH_SIZE=4
STRATEGY="max_sim"

echo "Running Reference Extraction for $MODEL..."
python Coded/angular-steering/pytorch_pure/extract_directions.py \
    --model $MODEL \
    --output-dir $OUTPUT_DIR \
    --n-samples $N_SAMPLES \
    --strategy max_sim \
    --batch-size 4

echo "Reference extraction complete. Output in $OUTPUT_DIR"
