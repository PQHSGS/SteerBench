#!/bin/bash
# Launch all-layer ablation phase 3 across 4 GPUs (2,3,4,5)
# Each GPU runs 6 of 24 eval tasks

source /data/caotue/anaconda3/etc/profile.d/conda.sh
conda activate sae_circuit

export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
export TORCH_HOME="/data/caotue/torch_cache"
export TMPDIR="/data/caotue/tmp"

SCRIPT="Experiments/ExpH/run_all_layer_ablation.py"
FLAGS="--skip-coskl"
LOG_DIR="Experiments/ExpH/results"
mkdir -p "$LOG_DIR"

echo "Launching 4 parallel eval workers..."
echo ""

for gpu in 2 3 4 5; do
    part=$((gpu - 2))
    log="$LOG_DIR/phase3_gpu${gpu}.log"
    echo "  GPU $gpu (partition $part/4) -> $log"
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u "$SCRIPT" $FLAGS --partition $part 4 > "$log" 2>&1 &
done

echo ""
echo "All 4 launched. Monitor with:"
echo "  tail -f $LOG_DIR/phase3_gpu*.log"
echo "  ps aux | grep run_all_layer"
