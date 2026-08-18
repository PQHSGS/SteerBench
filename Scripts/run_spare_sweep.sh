#!/usr/bin/env bash
# Hyperparameter sweep for SPARE on refusal_response
# Edit the sweep values below, then run.
# Runs sequentially (one job at a time), cycling across GPUs 2-5.

set -euo pipefail

# =============================================================================
# SWEEP VALUES — edit here
# =============================================================================
PROP_VALUES=(0.01 0.05 0.1 0.2 0.5)   # top_k_proportion
LOSS_WEIGHT_VALUES=(true false)         # loss_weight
N_NEIGHBORS_VALUES=(1 3 5 10)           # MI n_neighbors

# Fixed defaults used when sweeping other params
DEFAULT_PROP=0.05
DEFAULT_LW=true
DEFAULT_POS=last
DEFAULT_NN=3

GPUS=(2 3 4 5)   # GPUs to cycle through
# =============================================================================

PYTHON=/data/caotue/anaconda3/envs/sae_circuit/bin/python
BASE_CONFIG="Configs/Eval/SPARE/Gemma/gemma_refusal_response.json"
RESULTS_DIR="Results/SPARE"
mkdir -p "$RESULTS_DIR"

GPU_IDX=0

run_spare() {
  local tag="$1"
  local top_k_prop="$2"
  local loss_weight="$3"
  local position="$4"
  local n_neighbors="$5"
  local gpu="${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}"
  GPU_IDX=$((GPU_IDX + 1))

  # Skip if already done
  if [ -f "Vector/SPARE/Gemma/refusal_response_${tag}/metadata.pt" ]; then
    echo "  [SKIP] $tag already exists."
    return
  fi

  local tmp_config="$RESULTS_DIR/gemma_refusal_response_${tag}.json"

  $PYTHON - "$BASE_CONFIG" "$top_k_prop" "$loss_weight" "$position" "$n_neighbors" "$tmp_config" "$tag" <<'PY'
import json, sys
from pathlib import Path

config_path  = Path(sys.argv[1])
top_k_prop   = float(sys.argv[2])
loss_weight  = sys.argv[3].lower() in ("true", "1")
position     = sys.argv[4]
n_neighbors  = int(sys.argv[5])
output_path  = Path(sys.argv[6])
tag          = sys.argv[7]

with config_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["extractor"]["top_k_proportion"] = top_k_prop
cfg["extractor"]["loss_weight"]      = loss_weight
cfg["extractor"]["position"]         = position
cfg["extractor"]["n_neighbors"]      = n_neighbors
cfg["name"]        = f"gemma_refusal_response_{tag}"
cfg["save_vector"] = f"Vector/SPARE/Gemma/refusal_response_{tag}"
cfg["n_test"]      = 5
cfg["include_baseline"] = False

if "model" in cfg:
    cfg["model"]["device"] = "cuda"
cfg.pop("load_vector", None)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

  echo ""
  echo "--- SPARE | tag=$tag | prop=$top_k_prop | lw=$loss_weight | pos=$position | nn=$n_neighbors | GPU=$gpu ---"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m Steering.cli --task extract --config "$tmp_config"
}

# ── 1. top_k_proportion sweep ─────────────────────────────────────────────────
echo "=== [1/3] top_k_proportion sweep ==="
for prop in "${PROP_VALUES[@]}"; do
  tag="prop_$(echo "$prop" | tr '.' 'p')"
  run_spare "$tag" "$prop" "$DEFAULT_LW" "$DEFAULT_POS" "$DEFAULT_NN"
done

# ── 2. loss_weight sweep ──────────────────────────────────────────────────────
echo "=== [2/3] loss_weight sweep ==="
for lw in "${LOSS_WEIGHT_VALUES[@]}"; do
  tag="lw_${lw}_pos_${DEFAULT_POS}_nn${DEFAULT_NN}"
  run_spare "$tag" "$DEFAULT_PROP" "$lw" "$DEFAULT_POS" "$DEFAULT_NN"
done

# ── 3. n_neighbors sweep ──────────────────────────────────────────────────────
echo "=== [3/3] n_neighbors sweep ==="
for nn in "${N_NEIGHBORS_VALUES[@]}"; do
  tag="lw_${DEFAULT_LW}_pos_${DEFAULT_POS}_nn${nn}"
  run_spare "$tag" "$DEFAULT_PROP" "$DEFAULT_LW" "$DEFAULT_POS" "$nn"
done

echo ""
echo "=== SPARE full sweep complete ==="
