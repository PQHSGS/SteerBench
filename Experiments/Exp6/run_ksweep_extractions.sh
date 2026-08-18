#!/usr/bin/env bash
# Parallel CHARS K-sweep extraction launcher
# Distributes 6 extraction configs across GPUs 2-5
set +e

LAUNCH_DIR="/home/caotue/SAESteeringBench"
CONFIG_DIR="Configs/Eval/CHARS/Gemma"
LOG_DIR="/tmp/chars_ksweep_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "extract_toxic_K3.json"
  "extract_toxic_K5.json"
  "extract_toxic_K20.json"
  "extract_evil_K3.json"
  "extract_evil_K5.json"
  "extract_evil_K10.json"
)

GPUS=(2 3 4 5)
NUM_GPUS=${#GPUS[@]}

export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
export TORCH_HOME="/data/caotue/torch_cache"
export TMPDIR="/data/caotue/tmp"

echo "Launching ${#CONFIGS[@]} extractions across GPUs ${GPUS[*]}..."
echo ""

PIDS=()
for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  GPU_IDX=$((i % NUM_GPUS))
  GPU="${GPUS[$GPU_IDX]}"
  LOGFILE="$LOG_DIR/${CONFIG%.json}_gpu${GPU}.log"
  CFG_PATH="$LAUNCH_DIR/$CONFIG_DIR/$CONFIG"
  
  echo "[$i] ${CONFIG} → GPU ${GPU} → $LOGFILE"
  
  (cd "$LAUNCH_DIR" && CUDA_VISIBLE_DEVICES=$GPU conda run -n sae_circuit python -m Steering.cli --task extract --config "$CFG_PATH") > "$LOGFILE" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All launched. PIDs: ${PIDS[*]}"
echo "Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "FAILED: ${CONFIGS[$i]}"; FAILED=$((FAILED + 1)); }
done

echo ""
echo "=== DONE ==="
echo "Failed: $FAILED / ${#CONFIGS[@]}"
echo "Logs: $LOG_DIR"
tail -n5 "$LOG_DIR"/*.log 2>/dev/null
