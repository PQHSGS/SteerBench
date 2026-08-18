#!/usr/bin/env bash
# Launch 8 coupling analyses in parallel (cpu-only, no GPU contention)
set +e

LAUNCH_DIR="/home/caotue/SAESteeringBench"
RESULTS_DIR="$LAUNCH_DIR/Experiments/Exp6/coupling_results"
mkdir -p "$RESULTS_DIR"

cd "$LAUNCH_DIR"

# Tasks and K values
TASKS=("toxic" "toxic" "toxic" "toxic" "evil" "evil" "evil" "evil")
KS=(3 5 10 20 3 5 10 20)

echo "Launching ${#TASKS[@]} coupling analyses..."
PIDS=()

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    K="${KS[$i]}"
    LOGFILE="/tmp/ksweep/coupling_${TASK}_K${K}.log"
    mkdir -p /tmp/ksweep
    
    echo "[$i] ${TASK} K=${K}"
    CUDA_VISIBLE_DEVICES="" conda run -n sae_circuit python Experiments/Exp6/analyze_coupling.py "$TASK" "$K" > "$LOGFILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "FAILED: ${TASKS[$i]} K=${KS[$i]}"; FAILED=$((FAILED + 1)); }
done

echo ""
echo "=== DONE ==="
echo "Failed: $FAILED / ${#TASKS[@]}"
echo "Results: $RESULTS_DIR"
for f in "$RESULTS_DIR"/*.json; do
    echo "  $(basename $f): $(python -c "import json; d=json.load(open('$f')); print(f'active={d[\"n_active_centroids\"]}/{d[\"K\"]}, CV={d[\"centroid_norm_cv\"]:.3f}, rho={d[\"spearman_norm_vs_mass_sample\"]}, tail_ratio={d[\"tail_body_ratio\"]:.3f}')" 2>/dev/null)"
done
