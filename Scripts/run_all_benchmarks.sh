#!/bin/bash
set -euo pipefail

# ==========================================================
# GPU Configuration (Edit this list to change target GPUs)
# ==========================================================
GPUS=(2 3 4)
NUM_GPUS=${#GPUS[@]}

LOG_DIR="Logs/benchmarks"
mkdir -p "$LOG_DIR"

METHODS=(
  "angular"
  "caa"
  "cast"
  "cold"
  "corrsteer"
  "curve"
  "feat"
  "fgaa"
  "flas"
  "flow"
  "linearact"
  "manifold"
  "reft"
  "saecot"
  "saefree"
  "saeio"
  "sas"
  "spare"
  "spherical"
  "srps"
  "weightsteer"
)

echo "=== Starting Distributed Benchmarks on GPUs: ${GPUS[*]} ==="
echo "Total target GPUs: $NUM_GPUS"
echo "Running parallel batches of $NUM_GPUS methods..."

idx=0
for method in "${METHODS[@]}"; do
  gpu=${GPUS[$((idx % NUM_GPUS))]}
  
  echo "  [Batch $((idx / NUM_GPUS + 1))] Launching run_${method}_benchmark.sh on GPU ${gpu}..."
  
  # Run in background with assigned GPU
  export CUDA_VISIBLE_DEVICES=${gpu}
  bash Scripts/run_${method}_benchmark.sh > "$LOG_DIR/${method}_benchmark.log" 2>&1 &
  
  idx=$((idx + 1))
  
  # Wait if we have filled the current GPU batch
  if (( idx % NUM_GPUS == 0 )); then
    echo "  --> Waiting for current parallel batch to complete..."
    wait
    echo "  --> Batch complete."
  fi
done

# Wait for any leftover background processes from the final incomplete batch
if (( idx % NUM_GPUS != 0 )); then
  echo "  --> Waiting for final leftover benchmarks to complete..."
  wait
fi

echo "=== All benchmarks completed successfully! ==="
