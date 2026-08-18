#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
set -euo pipefail

# Mandatory Conda environment setup
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

COEFFS=(1 2 3 5)  # FishBack uses smaller coefficients (iterative method)
CONFIG_DIR="Configs/Eval/FISHBACK/Gemma"

# Order configs such that extraction configs run BEFORE loading configs
CONFIGS=(
  "gemma_evil.json"                    # Extracts evil vector
  "gemma_toxic.json"                   # Extracts toxic vector
  "gemma_deception.json"               # Extracts deception vector
  "gemma_refusal_response.json"        # Extracts refusal vector
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/fishback_cfg_XXXXXX.json)"

    # Inline Python helper to adjust coefficient and prevent redundant extractions
    python - "$config_path" "$coeff" "$idx" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False

# Swap save_vector to load_vector on subsequent sweep iterations to skip extraction
if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running $(basename "$config_path") | coeff=$coeff ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done
